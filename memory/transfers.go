package memory

import (
	"fmt"
	"sync"
	"unsafe"

	"github.com/stitch1968/gocuda/internal"
)

// Copy operations between host and device memory

// CopyHostToDevice copies data from host to device memory
func CopyHostToDevice(dst *Memory, src []byte) error {
	return CopyHostToDeviceWithStream(internal.GetDefaultStream(), dst, src)
}

// CopyHostToDeviceWithStream copies data from host to device with a specific stream
func CopyHostToDeviceWithStream(stream *internal.Stream, dst *Memory, src []byte) error {
	if dst == nil {
		return fmt.Errorf("null destination pointer")
	}
	if stream == nil {
		stream = internal.GetDefaultStream()
	}
	if len(src) == 0 || dst.size == 0 {
		return nil
	}

	copySize := min(int64(len(src)), dst.size)

	// Check if we should use real CUDA or simulation
	if internal.ShouldUseCuda() && dst.memType == TypeDevice && dst.data == nil {
		// Use real CUDA memcpy
		return internal.CudaMemcpyOnDevice(dst.ptr, unsafe.Pointer(&src[0]), copySize, internal.MemcpyKindHostToDevice, dst.deviceID)
	}

	// Use simulation
	if dst.data != nil {
		copy(dst.data[:copySize], src[:copySize])
	}

	return nil
}

// CopyDeviceToHost copies data from device to host memory
func CopyDeviceToHost(dst []byte, src *Memory) error {
	return CopyDeviceToHostWithStream(internal.GetDefaultStream(), dst, src)
}

// CopyDeviceToHostWithStream copies data from device to host with a specific stream
func CopyDeviceToHostWithStream(stream *internal.Stream, dst []byte, src *Memory) error {
	if src == nil {
		return fmt.Errorf("null source pointer")
	}
	if stream == nil {
		stream = internal.GetDefaultStream()
	}
	if len(dst) == 0 || src.size == 0 {
		return nil
	}

	copySize := min(int64(len(dst)), src.size)

	// Check if we should use real CUDA or simulation
	if internal.ShouldUseCuda() && src.memType == TypeDevice && src.data == nil {
		// Use real CUDA memcpy
		return internal.CudaMemcpyOnDevice(unsafe.Pointer(&dst[0]), src.ptr, copySize, internal.MemcpyKindDeviceToHost, src.deviceID)
	}

	// Use simulation
	if src.data != nil {
		copy(dst, src.data[:copySize])
	}

	return nil
}

// CopyDeviceToDevice copies data between device memories
func CopyDeviceToDevice(dst, src *Memory) error {
	return CopyDeviceToDeviceWithStream(internal.GetDefaultStream(), dst, src)
}

// CopyDeviceToDevicePeer performs an explicit device-to-device copy across
// devices using peer access when available or a host bounce buffer in simulation.
func CopyDeviceToDevicePeer(dst, src *Memory) error {
	return CopyDeviceToDevicePeerWithStream(internal.GetDefaultStream(), dst, src)
}

// CopyDeviceToDevicePeerWithStream performs an explicit cross-device copy.
func CopyDeviceToDevicePeerWithStream(stream *internal.Stream, dst, src *Memory) error {
	if dst == nil || src == nil {
		return fmt.Errorf("null pointer")
	}
	if stream == nil {
		stream = internal.GetDefaultStream()
	}

	copySize := src.size
	if dst.size < copySize {
		copySize = dst.size
	}
	if copySize == 0 {
		return nil
	}

	if src.deviceID == dst.deviceID {
		return CopyDeviceToDeviceWithStream(stream, dst, src)
	}

	if internal.ShouldUseCuda() && dst.memType == TypeDevice && src.memType == TypeDevice && dst.data == nil && src.data == nil {
		canAccess, err := internal.CanAccessPeer(dst.deviceID, src.deviceID)
		if err != nil {
			return err
		}
		if !canAccess {
			return fmt.Errorf("peer access unavailable between devices %d and %d", src.deviceID, dst.deviceID)
		}
		if err := internal.EnablePeerAccess(dst.deviceID, src.deviceID); err != nil {
			return err
		}
		return internal.CudaMemcpyPeer(dst.ptr, dst.deviceID, src.ptr, src.deviceID, copySize)
	}

	buffer := make([]byte, copySize)
	if err := CopyDeviceToHostWithStream(stream, buffer, src); err != nil {
		return err
	}
	return CopyHostToDeviceWithStream(stream, dst, buffer)
}

// CopyDeviceToDeviceWithStream copies data between devices with a specific stream
func CopyDeviceToDeviceWithStream(stream *internal.Stream, dst, src *Memory) error {
	if dst == nil || src == nil {
		return fmt.Errorf("null pointer")
	}
	if stream == nil {
		stream = internal.GetDefaultStream()
	}
	if dst.size == 0 || src.size == 0 {
		return nil
	}

	copySize := min(dst.size, src.size)

	// Check if we should use real CUDA or simulation
	if internal.ShouldUseCuda() && dst.memType == TypeDevice && src.memType == TypeDevice {
		if dst.deviceID != src.deviceID {
			return fmt.Errorf("cross-device memcpy requires CopyDeviceToDevicePeer: src device %d dst device %d", src.deviceID, dst.deviceID)
		}
		// Use real CUDA memcpy
		return internal.CudaMemcpyOnDevice(dst.ptr, src.ptr, copySize, internal.MemcpyKindDeviceToDevice, dst.deviceID)
	}

	// Use simulation
	if dst.data != nil && src.data != nil {
		copy(dst.data[:copySize], src.data[:copySize])
	}

	return nil
}

// Pool provides memory pooling for efficient allocation
type Pool struct {
	sizes map[int64][]*Memory
	mu    sync.Mutex
}

// NewPool creates a new memory pool
func NewPool() *Pool {
	return &Pool{
		sizes: make(map[int64][]*Memory),
	}
}

// Get gets memory from the pool or allocates new
func (p *Pool) Get(size int64) (*Memory, error) {
	p.mu.Lock()
	defer p.mu.Unlock()

	if memories, exists := p.sizes[size]; exists && len(memories) > 0 {
		// Reuse existing memory
		mem := memories[len(memories)-1]
		p.sizes[size] = memories[:len(memories)-1]
		return mem, nil
	}

	// Allocate new memory
	return Alloc(size)
}

// Put returns memory to the pool
func (p *Pool) Put(mem *Memory) {
	if mem == nil {
		return
	}

	p.mu.Lock()
	defer p.mu.Unlock()

	size := mem.Size()
	p.sizes[size] = append(p.sizes[size], mem)
}
