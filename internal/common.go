// Package internal contains shared internal utilities for GoCUDA packages.
// This package is not intended for external use.
package internal

import (
	"errors"
	"fmt"
	"sync"
	"time"
	"unsafe"
)

// Stream represents a CUDA stream for asynchronous operations
type Stream struct {
	ptr        unsafe.Pointer
	deviceID   int
	priority   int
	flags      uint32
	tasks      chan func() error
	workerDone chan struct{}
	isBlocking bool
	name       string
	mu         sync.Mutex
	submitMu   sync.RWMutex
	cond       *sync.Cond
	pending    int
	closed     bool
	taskErr    error
}

// NativeStreamHandle exposes the underlying CUDA stream pointer for internal
// package consumers that need to attach library handles to a CUDA stream.
func NativeStreamHandle(stream *Stream) unsafe.Pointer {
	if stream == nil {
		return nil
	}
	return stream.ptr
}

var (
	defaultStream *Stream
	streamOnce    sync.Once
)

// GetDefaultStream returns the default CUDA stream
func GetDefaultStream() *Stream {
	streamOnce.Do(func() {
		defaultStream = newStreamOnDevice("DefaultStream", 0)
	})
	return defaultStream
}

func newStream(name string) *Stream {
	return newStreamOnDevice(name, 0)
}

func newStreamOnDevice(name string, deviceID int) *Stream {
	stream := &Stream{
		deviceID:   deviceID,
		priority:   0,
		flags:      0,
		tasks:      make(chan func() error, 100),
		workerDone: make(chan struct{}),
		isBlocking: false,
		name:       name,
	}
	stream.cond = sync.NewCond(&stream.mu)
	go stream.processor()
	return stream
}

// processor handles stream task processing
func (s *Stream) processor() {
	defer close(s.workerDone)
	err := RunOnDevice(s.deviceID, func() error {
		for task := range s.tasks {
			s.completeTask(task())
		}
		return nil
	})
	if err != nil {
		s.mu.Lock()
		s.taskErr = errors.Join(s.taskErr, err)
		s.mu.Unlock()
	}
	for range s.tasks {
		s.completeTask(err)
	}
}

// Execute queues a host callback on the stream.
func (s *Stream) Execute(task func()) error {
	if task == nil {
		return fmt.Errorf("nil stream callback")
	}
	return s.Submit(func() error {
		task()
		return nil
	})
}

// Submit queues a host callback whose error is returned by Synchronize.
func (s *Stream) Submit(task func() error) error {
	if task == nil {
		return fmt.Errorf("nil stream callback")
	}
	s.submitMu.RLock()
	defer s.submitMu.RUnlock()
	s.mu.Lock()
	if s.closed {
		s.mu.Unlock()
		return fmt.Errorf("stream is closed")
	}
	s.pending++
	s.mu.Unlock()

	s.tasks <- task
	return nil
}

func (s *Stream) completeTask(err error) {
	s.mu.Lock()
	s.taskErr = errors.Join(s.taskErr, err)
	s.pending--
	if s.pending == 0 {
		s.cond.Broadcast()
	}
	s.mu.Unlock()
}

// Synchronize waits for all stream operations to complete
func (s *Stream) Synchronize() error {
	s.mu.Lock()
	for s.pending > 0 {
		s.cond.Wait()
	}
	err := s.taskErr
	s.taskErr = nil
	s.mu.Unlock()
	return err
}

// Close closes the stream
func (s *Stream) Close() error {
	s.submitMu.Lock()
	s.mu.Lock()
	if s.closed {
		s.mu.Unlock()
		s.submitMu.Unlock()
		<-s.workerDone
		return s.Synchronize()
	}
	s.closed = true
	close(s.tasks)
	s.mu.Unlock()
	s.submitMu.Unlock()
	<-s.workerDone
	return s.Synchronize()
}

// IsClosed reports whether the stream has been closed.
func (s *Stream) IsClosed() bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.closed
}

// DeviceID returns the CUDA device associated with this stream.
func (s *Stream) DeviceID() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.deviceID
}

// Error handling utilities

// CudaError represents a CUDA error
type CudaError struct {
	Code    int
	Message string
}

func (e *CudaError) Error() string {
	return fmt.Sprintf("CUDA error %d: %s", e.Code, e.Message)
}

// GetLastError returns the last CUDA error (simulation/wrapper)
func GetLastError() error {
	// For now this is a stub or needs split impl if real CUDA state tracking is needed
	// In the common case, we just return nil as checking last error is complex with Go/CGO
	return nil
}

// Profiling utilities

// Event represents a CUDA event for timing
type Event struct {
	time time.Time
}

// CreateEvent creates a new CUDA event
func CreateEvent() *Event {
	return &Event{time: time.Now()}
}

// Record records the event time
func (e *Event) Record() {
	e.time = time.Now()
}

// ElapsedTime returns the elapsed time between two events in milliseconds
func ElapsedTime(start, end *Event) float32 {
	return float32(end.time.Sub(start.time).Nanoseconds()) / 1000000.0
}

// NewStream creates a new CUDA stream
func NewStream() *Stream {
	return newStream("UserStream")
}

// NewStreamOnDevice creates a new CUDA stream bound to the specified device.
func NewStreamOnDevice(deviceID int) *Stream {
	return newStreamOnDevice("UserStream", deviceID)
}
