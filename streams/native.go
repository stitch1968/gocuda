package streams

import (
	"errors"
	"fmt"
	"sync"
	"sync/atomic"
	"time"
	"unsafe"

	"github.com/stitch1968/gocuda/internal"
	"github.com/stitch1968/gocuda/memory"
)

// ErrNativeCUDAUnavailable indicates that a native CUDA stream cannot be created.
var ErrNativeCUDAUnavailable = errors.New("native CUDA is unavailable")

// ErrNativeStreamClosed indicates that the stream has already been closed.
var ErrNativeStreamClosed = errors.New("native stream is closed")

// NativeStream is a CUDA stream, distinct from the host callback Stream.
// Call Close after all associated plans and buffers are no longer in use.
type NativeStream struct {
	mu       sync.Mutex
	handle   unsafe.Pointer
	deviceID int
	pending  []*memory.Retention
}

// NewNativeStream creates a CUDA stream on deviceID. It does not silently
// substitute a host queue when CUDA is unavailable.
func NewNativeStream(deviceID int, flags StreamFlags) (*NativeStream, error) {
	if flags != StreamDefault && flags != StreamNonBlocking {
		return nil, fmt.Errorf("invalid CUDA stream flags: %d", flags)
	}
	if !internal.ShouldUseCuda() {
		return nil, ErrNativeCUDAUnavailable
	}
	if deviceID < 0 || deviceID >= internal.GetCudaDeviceCount() {
		return nil, fmt.Errorf("device ID %d out of range", deviceID)
	}
	var handle unsafe.Pointer
	err := internal.RunOnDevice(deviceID, func() error {
		var err error
		handle, err = createNativeStream(flags)
		return err
	})
	if err != nil {
		if handle != nil {
			cleanupErr := internal.RunOnDevice(deviceID, func() error { return destroyNativeStream(handle) })
			return nil, errors.Join(err, cleanupErr)
		}
		return nil, err
	}
	return &NativeStream{handle: handle, deviceID: deviceID}, nil
}

// DeviceID returns the device on which the stream was created.
func (s *NativeStream) DeviceID() int {
	return s.deviceID
}

// WithNativeHandle runs fn on the stream's device while protecting the handle
// from concurrent Close. The handle must not be retained after fn returns.
// The callback must not re-enter methods on the same stream.
func (s *NativeStream) WithNativeHandle(fn func(unsafe.Pointer) error) error {
	if s == nil || fn == nil {
		return fmt.Errorf("stream and callback must not be nil")
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.handle == nil {
		return ErrNativeStreamClosed
	}
	return internal.RunOnDevice(s.deviceID, func() error { return fn(s.handle) })
}

// Synchronize waits for queued device operations and reports CUDA errors.
func (s *NativeStream) Synchronize() error {
	return s.WithNativeHandle(func(handle unsafe.Pointer) error {
		if err := synchronizeNativeStream(handle); err != nil {
			return err
		}
		s.releasePending()
		return nil
	})
}

// EnqueueWithBuffers invokes fn while retaining its buffers until successful
// stream synchronization or closure. Once fn is invoked, buffers remain
// retained even if it reports an error, because work may have been enqueued.
// The callback must not re-enter methods on the same stream.
func (s *NativeStream) EnqueueWithBuffers(buffers []*memory.Memory, fn func(unsafe.Pointer) error) error {
	if s == nil || fn == nil {
		return fmt.Errorf("stream and callback must not be nil")
	}
	retentions := make([]*memory.Retention, 0, len(buffers))
	for _, buffer := range buffers {
		retention, err := buffer.Retain()
		if err != nil {
			for _, acquired := range retentions {
				acquired.Release()
			}
			return err
		}
		retentions = append(retentions, retention)
	}
	invoked := false
	err := s.WithNativeHandle(func(handle unsafe.Pointer) error {
		s.pending = append(s.pending, retentions...)
		invoked = true
		return fn(handle)
	})
	if !invoked {
		for _, retention := range retentions {
			retention.Release()
		}
	}
	return err
}

// CopyAsync queues a copy between CUDA device/managed or pinned host buffers.
// The stream retains both buffers until Synchronize or Close succeeds. The
// caller must not access or modify them until the operation completes.
// Pageable Go memory is not accepted.
func (s *NativeStream) CopyAsync(dst, src *memory.Memory, size int64) error {
	if s == nil || dst == nil || src == nil {
		return fmt.Errorf("stream and copy buffers must not be nil")
	}
	if dst.Ptr() == nil || src.Ptr() == nil {
		return fmt.Errorf("copy buffer has been freed")
	}
	if size < 0 || size > dst.Size() || size > src.Size() {
		return fmt.Errorf("invalid async copy size %d", size)
	}
	for _, buffer := range []*memory.Memory{dst, src} {
		switch buffer.GetType() {
		case memory.TypePinned:
		case memory.TypeDevice, memory.TypeDeviceOnly, memory.TypeUnified:
			if buffer.GetDeviceID() != s.deviceID {
				return fmt.Errorf("buffer device %d does not match stream device %d", buffer.GetDeviceID(), s.deviceID)
			}
		default:
			return fmt.Errorf("async copy requires device, managed, or pinned memory")
		}
	}
	if size == 0 {
		return s.WithNativeHandle(func(unsafe.Pointer) error { return nil })
	}
	return s.EnqueueWithBuffers([]*memory.Memory{dst, src}, func(handle unsafe.Pointer) error {
		return copyNativeAsync(handle, dst.Ptr(), src.Ptr(), size)
	})
}

func (s *NativeStream) releasePending() {
	for _, retention := range s.pending {
		retention.Release()
	}
	s.pending = nil
}

// Close waits for queued device work and destroys the CUDA stream.
func (s *NativeStream) Close() error {
	if s == nil {
		return nil
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.handle == nil {
		return nil
	}
	return internal.RunOnDevice(s.deviceID, func() error {
		if err := synchronizeNativeStream(s.handle); err != nil {
			return err
		}
		s.releasePending()
		if err := destroyNativeStream(s.handle); err != nil {
			return err
		}
		s.handle = nil
		return nil
	})
}

// NativeEventFlags control CUDA event creation.
type NativeEventFlags uint32

const (
	NativeEventDefault       NativeEventFlags = 0
	NativeEventBlockingSync  NativeEventFlags = 1
	NativeEventDisableTiming NativeEventFlags = 2
)

// NativeEvent represents a CUDA event, not a host timestamp.
type NativeEvent struct {
	mu       sync.Mutex
	handle   unsafe.Pointer
	deviceID int
	id       uint64
	recorded bool
}

var nativeEventIDs atomic.Uint64

// NewNativeEvent creates a CUDA event on deviceID.
func NewNativeEvent(deviceID int, flags NativeEventFlags) (*NativeEvent, error) {
	if flags & ^(NativeEventBlockingSync|NativeEventDisableTiming) != 0 {
		return nil, fmt.Errorf("invalid CUDA event flags: %d", flags)
	}
	if !internal.ShouldUseCuda() {
		return nil, ErrNativeCUDAUnavailable
	}
	if deviceID < 0 || deviceID >= internal.GetCudaDeviceCount() {
		return nil, fmt.Errorf("device ID %d out of range", deviceID)
	}
	var handle unsafe.Pointer
	err := internal.RunOnDevice(deviceID, func() error {
		var err error
		handle, err = createNativeEvent(flags)
		return err
	})
	if err != nil {
		if handle != nil {
			cleanupErr := internal.RunOnDevice(deviceID, func() error { return destroyNativeEvent(handle) })
			return nil, errors.Join(err, cleanupErr)
		}
		return nil, err
	}
	return &NativeEvent{handle: handle, deviceID: deviceID, id: nativeEventIDs.Add(1)}, nil
}

// Record enqueues the event after previously submitted work on stream.
func (e *NativeEvent) Record(stream *NativeStream) error {
	if e == nil || stream == nil {
		return fmt.Errorf("event and stream must not be nil")
	}
	if e.deviceID != stream.deviceID {
		return fmt.Errorf("event and stream belong to different devices")
	}
	return stream.WithNativeHandle(func(handle unsafe.Pointer) error {
		e.mu.Lock()
		defer e.mu.Unlock()
		if e.handle == nil {
			return fmt.Errorf("native event is closed")
		}
		if err := recordNativeEvent(e.handle, handle); err != nil {
			return err
		}
		e.recorded = true
		return nil
	})
}

// WaitEvent makes the stream wait for a previously recorded CUDA event.
func (s *NativeStream) WaitEvent(event *NativeEvent) error {
	if s == nil || event == nil {
		return fmt.Errorf("stream and event must not be nil")
	}
	if s.deviceID != event.deviceID {
		return fmt.Errorf("cross-device event waits are not supported")
	}
	return s.WithNativeHandle(func(handle unsafe.Pointer) error {
		event.mu.Lock()
		defer event.mu.Unlock()
		if event.handle == nil || !event.recorded {
			return fmt.Errorf("native event is closed or unrecorded")
		}
		return waitNativeEvent(handle, event.handle)
	})
}

// Query reports whether a recorded event has completed.
func (e *NativeEvent) Query() (bool, error) {
	if e == nil {
		return false, fmt.Errorf("nil native event")
	}
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.handle == nil || !e.recorded {
		return false, fmt.Errorf("native event is closed or unrecorded")
	}
	var complete bool
	err := internal.RunOnDevice(e.deviceID, func() error {
		var err error
		complete, err = queryNativeEvent(e.handle)
		return err
	})
	return complete, err
}

// Synchronize waits for the recorded event to complete.
func (e *NativeEvent) Synchronize() error {
	if e == nil {
		return fmt.Errorf("nil native event")
	}
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.handle == nil || !e.recorded {
		return fmt.Errorf("native event is closed or unrecorded")
	}
	return internal.RunOnDevice(e.deviceID, func() error { return synchronizeNativeEvent(e.handle) })
}

// ElapsedTime returns CUDA device time between two completed events.
func (e *NativeEvent) ElapsedTime(end *NativeEvent) (time.Duration, error) {
	if e == nil || end == nil {
		return 0, fmt.Errorf("events must not be nil")
	}
	if e.deviceID != end.deviceID {
		return 0, fmt.Errorf("events belong to different devices")
	}
	first, second := e, end
	if first.id > second.id {
		first, second = second, first
	}
	first.mu.Lock()
	defer first.mu.Unlock()
	if first != second {
		second.mu.Lock()
		defer second.mu.Unlock()
	}
	if e.handle == nil || end.handle == nil || !e.recorded || !end.recorded {
		return 0, fmt.Errorf("events must be recorded and open")
	}
	var elapsed time.Duration
	err := internal.RunOnDevice(e.deviceID, func() error {
		var err error
		elapsed, err = elapsedNativeEvents(e.handle, end.handle)
		return err
	})
	return elapsed, err
}

// Close destroys the CUDA event.
func (e *NativeEvent) Close() error {
	if e == nil {
		return nil
	}
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.handle == nil {
		return nil
	}
	return internal.RunOnDevice(e.deviceID, func() error {
		if err := destroyNativeEvent(e.handle); err != nil {
			return err
		}
		e.handle = nil
		return nil
	})
}
