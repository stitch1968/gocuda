package memory

import (
	"errors"
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

// CopyHostToDeviceWithStream performs a blocking copy. The stream parameter
// does not order this copy with queued work; use NativeStream.CopyAsync for
// CUDA stream ordering.
func CopyHostToDeviceWithStream(stream *internal.Stream, dst *Memory, src []byte) error {
	if dst == nil {
		return fmt.Errorf("null destination pointer")
	}
	retention, err := dst.Retain()
	if err != nil {
		return fmt.Errorf("destination allocation: %w", err)
	}
	defer retention.Release()
	if len(src) == 0 || dst.size == 0 {
		return nil
	}

	copySize := min(int64(len(src)), dst.size)

	// Check if we should use real CUDA or simulation
	if dst.cudaBacked {
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

// CopyDeviceToHostWithStream performs a blocking copy without stream ordering.
func CopyDeviceToHostWithStream(stream *internal.Stream, dst []byte, src *Memory) error {
	if src == nil {
		return fmt.Errorf("null source pointer")
	}
	retention, err := src.Retain()
	if err != nil {
		return fmt.Errorf("source allocation: %w", err)
	}
	defer retention.Release()
	if len(dst) == 0 || src.size == 0 {
		return nil
	}

	copySize := min(int64(len(dst)), src.size)

	// Check if we should use real CUDA or simulation
	if src.cudaBacked {
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

// CopyDeviceToDevicePeerWithStream performs a blocking cross-device copy
// without stream ordering.
func CopyDeviceToDevicePeerWithStream(stream *internal.Stream, dst, src *Memory) error {
	if dst == nil || src == nil {
		return fmt.Errorf("null pointer")
	}
	dstRetention, err := dst.Retain()
	if err != nil {
		return fmt.Errorf("destination allocation: %w", err)
	}
	defer dstRetention.Release()
	srcRetention, err := src.Retain()
	if err != nil {
		return fmt.Errorf("source allocation: %w", err)
	}
	defer srcRetention.Release()

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

	if dst.cudaBacked && src.cudaBacked {
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

// CopyDeviceToDeviceWithStream performs a blocking copy without stream ordering.
func CopyDeviceToDeviceWithStream(stream *internal.Stream, dst, src *Memory) error {
	if dst == nil || src == nil {
		return fmt.Errorf("null pointer")
	}
	dstRetention, err := dst.Retain()
	if err != nil {
		return fmt.Errorf("destination allocation: %w", err)
	}
	defer dstRetention.Release()
	srcRetention, err := src.Retain()
	if err != nil {
		return fmt.Errorf("source allocation: %w", err)
	}
	defer srcRetention.Release()
	if dst.size == 0 || src.size == 0 {
		return nil
	}

	copySize := min(dst.size, src.size)

	// Check if we should use real CUDA or simulation
	if dst.cudaBacked || src.cudaBacked {
		if dst.cudaBacked && src.cudaBacked && dst.deviceID != src.deviceID {
			return fmt.Errorf("cross-device memcpy requires CopyDeviceToDevicePeer: src device %d dst device %d", src.deviceID, dst.deviceID)
		}
		deviceID := dst.deviceID
		if src.cudaBacked {
			deviceID = src.deviceID
		}
		return internal.CudaMemcpyOnDevice(dst.ptr, src.ptr, copySize, internal.MemcpyKindDefault, deviceID)
	}

	// Use simulation
	if dst.data != nil && src.data != nil {
		copy(dst.data[:copySize], src.data[:copySize])
	}

	return nil
}

// Pool provides memory pooling for efficient allocation
type Pool struct {
	sizes    map[int64][]*Memory
	deviceID int
	closed   bool
	mu       sync.Mutex
}

// NewPool creates a new memory pool
func NewPool() *Pool {
	return &Pool{
		sizes:    make(map[int64][]*Memory),
		deviceID: internal.GetDevice(),
	}
}

// Get gets memory from the pool or allocates new
func (p *Pool) Get(size int64) (*Memory, error) {
	if p == nil {
		return nil, fmt.Errorf("nil memory pool")
	}
	p.mu.Lock()
	if p.closed {
		p.mu.Unlock()
		return nil, fmt.Errorf("memory pool is closed")
	}

	if memories, exists := p.sizes[size]; exists && len(memories) > 0 {
		// Reuse existing memory
		mem := memories[len(memories)-1]
		p.sizes[size] = memories[:len(memories)-1]
		mem.mu.Lock()
		mem.pool = nil
		mem.mu.Unlock()
		p.mu.Unlock()
		return mem, nil
	}
	deviceID := p.deviceID
	p.mu.Unlock()

	// Allocate new memory
	return AllocOnDevice(size, deviceID)
}

// Put returns memory to the pool
func (p *Pool) Put(mem *Memory) error {
	if p == nil || mem == nil {
		return fmt.Errorf("pool and memory must not be nil")
	}

	p.mu.Lock()
	defer p.mu.Unlock()
	if p.closed {
		return fmt.Errorf("memory pool is closed")
	}
	mem.mu.Lock()
	defer mem.mu.Unlock()
	if mem.freed || mem.ptr == nil {
		return fmt.Errorf("cannot pool freed memory")
	}
	if mem.pool != nil {
		return fmt.Errorf("memory is already owned by a pool")
	}
	if mem.retentions != 0 {
		return fmt.Errorf("cannot pool memory retained by %d active consumers", mem.retentions)
	}
	if mem.memType != TypeDevice || mem.deviceID != p.deviceID {
		return fmt.Errorf("pool requires device memory on device %d", p.deviceID)
	}
	size := mem.Size()
	mem.pool = p
	p.sizes[size] = append(p.sizes[size], mem)
	return nil
}

// Close frees cached allocations. Checked-out allocations remain caller-owned.
// If an allocation cannot be freed, Close can be retried.
func (p *Pool) Close() error {
	if p == nil {
		return nil
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	p.closed = true
	var errs []error
	for size, memories := range p.sizes {
		remaining := memories[:0]
		for _, mem := range memories {
			mem.mu.Lock()
			mem.pool = nil
			err := mem.freeLocked()
			if err != nil {
				errs = append(errs, err)
				if !mem.freed {
					mem.pool = p
					remaining = append(remaining, mem)
				}
			}
			mem.mu.Unlock()
		}
		if len(remaining) == 0 {
			delete(p.sizes, size)
		} else {
			p.sizes[size] = remaining
		}
	}
	return errors.Join(errs...)
}
