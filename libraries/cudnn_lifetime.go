package libraries

import (
	"errors"
	"fmt"
	"math"
	"sync"
	"unsafe"

	"github.com/stitch1968/gocuda/internal"
	"github.com/stitch1968/gocuda/memory"
	"github.com/stitch1968/gocuda/streams"
)

func dnnElementSize(dataType DNNDataType) (uint64, error) {
	switch dataType {
	case DNNDataFloat, DNNDataInt32:
		return 4, nil
	case DNNDataDouble, DNNDataInt64:
		return 8, nil
	case DNNDataHalf, DNNDataBFloat16:
		return 2, nil
	case DNNDataInt8, DNNDataUint8:
		return 1, nil
	default:
		return 0, fmt.Errorf("unsupported cuDNN buffer data type %d", dataType)
	}
}

func validateDNNScalingTypes(types ...DNNDataType) error {
	hasDouble := false
	for _, dataType := range types {
		hasDouble = hasDouble || dataType == DNNDataDouble
	}
	if hasDouble {
		for _, dataType := range types {
			if dataType != DNNDataDouble {
				return fmt.Errorf("mixed double and non-double cuDNN scaling types are unsupported")
			}
		}
	}
	return nil
}

func validateDNNTensorBuffer(desc *TensorDescriptor, buffer *memory.Memory) error {
	elementSize, err := dnnElementSize(desc.dataType)
	if err != nil {
		return err
	}
	if len(desc.dimensions) == 0 || len(desc.dimensions) != len(desc.strides) {
		return fmt.Errorf("native cuDNN tensor dimensions and strides are unavailable")
	}
	maxElements := uint64(math.MaxInt64) / elementSize
	span := uint64(1)
	for i, dimension := range desc.dimensions {
		stride := desc.strides[i]
		if dimension <= 0 || stride <= 0 {
			return fmt.Errorf("native cuDNN tensor dimensions and strides must be positive")
		}
		if uint64(dimension-1) > (maxElements-span)/uint64(stride) {
			return fmt.Errorf("native cuDNN tensor layout exceeds supported size")
		}
		span += uint64(dimension-1) * uint64(stride)
	}
	if buffer == nil || uint64(buffer.Size()) < span*elementSize {
		return fmt.Errorf("native cuDNN tensor buffer is too small")
	}
	return nil
}

func validateDNNFilterBuffer(desc *FilterDescriptor, buffer *memory.Memory) error {
	elementSize, err := dnnElementSize(desc.dataType)
	if err != nil {
		return err
	}
	if len(desc.dimensions) == 0 {
		return fmt.Errorf("native cuDNN filter dimensions are unavailable")
	}
	maxElements := uint64(math.MaxInt64) / elementSize
	count := uint64(1)
	for _, dimension := range desc.dimensions {
		if dimension <= 0 || uint64(dimension) > maxElements/count {
			return fmt.Errorf("invalid native cuDNN filter size")
		}
		count *= uint64(dimension)
	}
	if buffer == nil || uint64(buffer.Size()) < count*elementSize {
		return fmt.Errorf("native cuDNN filter buffer is too small")
	}
	return nil
}

type dnnDescriptorState struct {
	mu        sync.Mutex
	destroyed bool
	users     int
}

// ErrDNNDescriptorInUse indicates that active work retains a cuDNN descriptor.
var ErrDNNDescriptorInUse = errors.New("cuDNN descriptor is in use")

func (state *dnnDescriptorState) lockMutable() error {
	state.mu.Lock()
	if state.destroyed {
		state.mu.Unlock()
		return fmt.Errorf("descriptor has been destroyed")
	}
	if state.users != 0 {
		state.mu.Unlock()
		return ErrDNNDescriptorInUse
	}
	return nil
}

func retainDNNDescriptors(states ...*dnnDescriptorState) ([]*dnnDescriptorState, error) {
	retained := make([]*dnnDescriptorState, 0, len(states))
	for _, state := range states {
		if state == nil {
			releaseDNNDescriptors(retained)
			return nil, fmt.Errorf("nil cuDNN descriptor")
		}
		state.mu.Lock()
		if state.destroyed {
			state.mu.Unlock()
			releaseDNNDescriptors(retained)
			return nil, fmt.Errorf("cuDNN descriptor has been destroyed")
		}
		state.users++
		state.mu.Unlock()
		retained = append(retained, state)
	}
	return retained, nil
}

func releaseDNNDescriptors(states []*dnnDescriptorState) {
	for _, state := range states {
		state.mu.Lock()
		state.users--
		state.mu.Unlock()
	}
}

// Synchronize waits for the handle's native stream and releases resources held
// after failed cuDNN operations. A successfully closed stream is synchronized.
// Simulation operations already complete inline.
func (h *DNNHandle) Synchronize() error {
	if h == nil {
		return fmt.Errorf("nil cuDNN handle")
	}
	h.mu.Lock()
	defer h.mu.Unlock()
	if h.destroyed {
		return fmt.Errorf("cuDNN handle has been destroyed")
	}
	if !h.native {
		return nil
	}
	return h.synchronizeLocked()
}

func (h *DNNHandle) synchronizeLocked() error {
	if h.nativeStream != nil {
		if err := h.nativeStream.Synchronize(); err != nil && !errors.Is(err, streams.ErrNativeStreamClosed) {
			return err
		}
	} else {
		if err := internal.CudaDefaultStreamSynchronizeOnDevice(h.deviceID); err != nil {
			return err
		}
	}
	releaseLibraryBuffers(h.pendingBuffers)
	h.pendingBuffers = nil
	releaseDNNDescriptors(h.pendingDescriptors)
	h.pendingDescriptors = nil
	var errs []error
	remaining := h.pendingWorkspaces[:0]
	for _, workspace := range h.pendingWorkspaces {
		if err := workspace.Free(); err != nil {
			remaining = append(remaining, workspace)
			errs = append(errs, err)
		}
	}
	h.pendingWorkspaces = remaining
	return errors.Join(errs...)
}

func (h *DNNHandle) execute(states []*dnnDescriptorState, buffers []*memory.Memory, run func() (*memory.Memory, error)) error {
	if h == nil {
		return fmt.Errorf("nil cuDNN handle")
	}
	h.mu.Lock()
	defer h.mu.Unlock()
	if h.destroyed {
		return fmt.Errorf("cuDNN handle has been destroyed")
	}
	descriptors, err := retainDNNDescriptors(states...)
	if err != nil {
		return err
	}
	defer func() { releaseDNNDescriptors(descriptors) }()
	retentions, err := retainLibraryBuffers(buffers...)
	if err != nil {
		return err
	}
	defer func() { releaseLibraryBuffers(retentions) }()
	if !h.native {
		workspace, err := run()
		if workspace != nil {
			err = errors.Join(err, workspace.Free())
		}
		return err
	}
	for _, buffer := range buffers {
		switch buffer.GetType() {
		case memory.TypeDevice, memory.TypeDeviceOnly, memory.TypeUnified:
		default:
			return fmt.Errorf("native cuDNN requires device or managed memory")
		}
		if buffer.GetDeviceID() != h.deviceID {
			return fmt.Errorf("cuDNN buffer device %d does not match handle device %d", buffer.GetDeviceID(), h.deviceID)
		}
	}
	if len(h.pendingBuffers) != 0 || len(h.pendingDescriptors) != 0 || len(h.pendingWorkspaces) != 0 {
		if err := h.synchronizeLocked(); err != nil {
			return err
		}
	}
	var workspace *memory.Memory
	synchronized := false
	invoked := false
	runOnStream := func(stream unsafe.Pointer) error {
		invoked = true
		var runErr error
		workspace, runErr = run()
		var syncErr error
		if stream != nil {
			syncErr = internal.CudaStreamSynchronizeOnDevice(h.deviceID, stream)
		} else {
			syncErr = internal.CudaDefaultStreamSynchronizeOnDevice(h.deviceID)
		}
		synchronized = syncErr == nil
		return errors.Join(runErr, syncErr)
	}
	if h.nativeStream != nil {
		err = h.nativeStream.WithNativeHandle(runOnStream)
	} else {
		err = internal.RunOnDevice(h.deviceID, func() error { return runOnStream(nil) })
	}
	if invoked && !synchronized {
		h.pendingBuffers = append(h.pendingBuffers, retentions...)
		h.pendingDescriptors = append(h.pendingDescriptors, descriptors...)
		if workspace != nil {
			h.pendingWorkspaces = append(h.pendingWorkspaces, workspace)
		}
		retentions, descriptors = nil, nil
		return err
	}
	if workspace != nil {
		if cleanupErr := workspace.Free(); cleanupErr != nil {
			h.pendingWorkspaces = append(h.pendingWorkspaces, workspace)
			err = errors.Join(err, cleanupErr)
		}
	}
	return err
}
