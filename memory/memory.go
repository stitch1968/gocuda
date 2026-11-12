// Package memory provides GPU memory management functionality for GoCUDA.
// This package handles memory allocation, deallocation, and transfers between
// host and device memory.
package memory

import (
	"fmt"
	"sync"
	"sync/atomic"
	"unsafe"

	"github.com/stitch1968/gocuda/internal"
)

// Type represents different CUDA memory types
type Type int

const (
	TypeDevice     Type = iota // Legacy device allocation, managed in CUDA mode
	TypeHost                   // Host (pageable) memory
	TypePinned                 // Host pinned (page-locked) memory
	TypeUnified                // Unified memory (managed)
	TypeShared                 // Shared memory (per-block)
	TypeConstant               // Constant memory
	TypeTexture                // Texture memory
	TypeDeviceOnly             // Device-only memory; not host-accessible
)

// Attribute represents memory attributes
type Attribute struct {
	IsReadOnly      bool
	IsWriteOnly     bool
	IsCoalesced     bool
	IsAligned       bool
	CachePreference int // cudaFuncCachePreferNone, cudaFuncCachePreferShared, etc.
}

// Memory represents allocated GPU memory
type Memory struct {
	ptr        unsafe.Pointer
	size       int64
	stream     *internal.Stream
	data       []byte // Actual data storage for simulation
	cudaBacked bool
	cudaPinned bool
	memType    Type
	attributes Attribute
	pitch      int64 // For 2D/3D memory allocations
	alignment  int   // Memory alignment (typically 256 bytes for CUDA)
	deviceID   int   // Device where memory is allocated
	mu         sync.Mutex
	freed      bool
	retentions int
	pool       *Pool
}

// Retention prevents an allocation from being freed until Release is called.
type Retention struct {
	state *retentionState
}

type retentionState struct {
	memory   *Memory
	released atomic.Bool
}

// Retain prevents Free while a consumer holds the returned retention.
func (m *Memory) Retain() (*Retention, error) {
	if m == nil {
		return nil, fmt.Errorf("nil memory allocation")
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.freed || m.ptr == nil {
		return nil, fmt.Errorf("memory has been freed")
	}
	if m.pool != nil {
		return nil, fmt.Errorf("memory is owned by a pool")
	}
	m.retentions++
	return &Retention{state: &retentionState{memory: m}}, nil
}

// Release relinquishes a retention. It is safe to call more than once.
func (r *Retention) Release() {
	if r == nil || r.state == nil || !r.state.released.CompareAndSwap(false, true) {
		return
	}
	r.state.memory.mu.Lock()
	r.state.memory.retentions--
	r.state.memory.mu.Unlock()
}

// Manager handles GPU memory allocation and deallocation
type Manager struct {
	allocations map[uintptr]*Memory
	mu          sync.RWMutex
	totalBytes  int64
}

var globalManager = &Manager{
	allocations: make(map[uintptr]*Memory),
}

// Alloc allocates memory. Call Free when the allocation is no longer needed.
func Alloc(size int64) (*Memory, error) {
	deviceID, err := internal.CurrentDevice()
	if err != nil {
		return nil, err
	}
	return allocWithTypeAndStreamOnDevice(nil, size, TypeDevice, deviceID)
}

// AllocOnDevice allocates device memory associated with a specific device ID.
func AllocOnDevice(size int64, deviceID int) (*Memory, error) {
	return allocWithTypeAndStreamOnDevice(internal.GetDefaultStream(), size, TypeDevice, deviceID)
}

// AllocDeviceOnly allocates device-only memory. Use copy helpers to access it.
func AllocDeviceOnly(size int64) (*Memory, error) {
	deviceID, err := internal.CurrentDevice()
	if err != nil {
		return nil, err
	}
	return allocWithTypeAndStreamOnDevice(nil, size, TypeDeviceOnly, deviceID)
}

// AllocDeviceOnlyOnDevice allocates device-only memory on the selected device.
func AllocDeviceOnlyOnDevice(size int64, deviceID int) (*Memory, error) {
	devices := internal.GetDevices()
	if deviceID < 0 || deviceID >= len(devices) {
		return nil, fmt.Errorf("device ID %d out of range", deviceID)
	}
	return allocWithTypeAndStreamOnDevice(internal.GetDefaultStream(), size, TypeDeviceOnly, deviceID)
}

// AllocWithStream allocates memory on the GPU with a specific stream
func AllocWithStream(stream *internal.Stream, size int64) (*Memory, error) {
	return AllocWithTypeAndStream(stream, size, TypeDevice)
}

// AllocWithTypeAndStream allocates memory with specific type and stream
func AllocWithTypeAndStream(stream *internal.Stream, size int64, memType Type) (*Memory, error) {
	var deviceID int
	if stream != nil {
		deviceID = stream.DeviceID()
	} else {
		var err error
		deviceID, err = internal.CurrentDevice()
		if err != nil {
			return nil, err
		}
	}
	return allocWithTypeAndStreamOnDevice(stream, size, memType, deviceID)
}

func allocWithTypeAndStreamOnDevice(stream *internal.Stream, size int64, memType Type, deviceID int) (*Memory, error) {
	if size <= 0 {
		return nil, fmt.Errorf("invalid size: %d", size)
	}
	if size > int64(int(^uint(0)>>1))-255 {
		return nil, fmt.Errorf("allocation size too large: %d", size)
	}
	if (memType < TypeDevice || memType > TypeUnified) && memType != TypeDeviceOnly {
		return nil, fmt.Errorf("memory type %d cannot be allocated as a standalone buffer", memType)
	}
	if stream == nil {
		stream = internal.GetDefaultStream()
	}

	var ptr unsafe.Pointer
	var data []byte
	var err error

	// Check if we should use real CUDA or simulation
	cudaBacked := internal.ShouldUseCuda() && (memType == TypeDevice || memType == TypeUnified || memType == TypeDeviceOnly)
	cudaPinned := internal.ShouldUseCuda() && memType == TypePinned
	if cudaBacked {
		if memType == TypeDeviceOnly {
			ptr, err = internal.CudaMallocDeviceOnDevice(size, deviceID)
		} else {
			ptr, err = internal.CudaMallocOnDevice(size, deviceID)
		}
		if err != nil {
			return nil, fmt.Errorf("CUDA malloc failed: %w", err)
		}
	} else if cudaPinned {
		ptr, err = internal.CudaMallocHost(size)
		if err != nil {
			return nil, fmt.Errorf("CUDA pinned allocation failed: %w", err)
		}
		data = unsafe.Slice((*byte)(ptr), int(size))
	} else {
		raw := make([]byte, int(size)+255)
		offset := (256 - uintptr(unsafe.Pointer(&raw[0]))%256) % 256
		data = raw[offset : offset+uintptr(size)]
		ptr = unsafe.Pointer(&data[0])
	}

	mem := &Memory{
		ptr:        ptr,
		size:       size,
		stream:     stream,
		data:       data,
		cudaBacked: cudaBacked,
		cudaPinned: cudaPinned,
		memType:    memType,
		alignment:  256,
		deviceID:   deviceID,
		attributes: Attribute{
			IsAligned: true,
		},
	}

	// Register the allocation
	globalManager.mu.Lock()
	globalManager.allocations[uintptr(ptr)] = mem
	globalManager.totalBytes += size
	globalManager.mu.Unlock()

	return mem, nil
}

// Free releases memory unless an active retention prevents it.
func (m *Memory) Free() error {
	if m == nil {
		return nil
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.pool != nil {
		return fmt.Errorf("memory is owned by a pool")
	}
	return m.freeLocked()
}

func (m *Memory) freeLocked() error {
	if m.freed {
		return nil // Already freed
	}
	if m.retentions != 0 {
		return fmt.Errorf("memory is retained by %d active consumers", m.retentions)
	}
	if m.ptr == nil {
		return nil
	}

	ptr := m.ptr

	var err error
	released := true
	if m.cudaPinned {
		err = internal.CudaFreeHost(ptr)
		released = err == nil
	} else if m.cudaBacked {
		// Use real CUDA memory deallocation
		released, err = internal.CudaFreeOnDeviceResult(ptr, m.deviceID)
	}
	// For simulation, Go's GC will handle the slice cleanup

	if !released {
		return err
	}

	globalManager.mu.Lock()
	delete(globalManager.allocations, uintptr(ptr))
	globalManager.totalBytes -= m.size
	globalManager.mu.Unlock()

	// Mark as freed
	m.ptr = nil
	m.data = nil
	m.freed = true

	return err
}

// Size returns the memory size
func (m *Memory) Size() int64 {
	return m.size
}

// Ptr returns the memory pointer
func (m *Memory) Ptr() unsafe.Pointer {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.pool != nil {
		return nil
	}
	return m.ptr
}

// Data returns host-visible simulation data, or nil for device-only memory.
func (m *Memory) Data() []byte {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.memType == TypeDeviceOnly || m.pool != nil {
		return nil
	}
	return m.data
}

// View returns a typed slice over host-accessible memory. The caller must keep
// the allocation alive and must not use the slice after Free.
func View[T any](m *Memory, length int) ([]T, error) {
	if m == nil {
		return nil, fmt.Errorf("nil memory allocation")
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.pool != nil {
		return nil, fmt.Errorf("memory is owned by a pool")
	}
	if length < 0 {
		return nil, fmt.Errorf("negative view length: %d", length)
	}
	if m.memType == TypeDeviceOnly {
		return nil, fmt.Errorf("device-only memory is not host-accessible; use a copy helper")
	}
	if length == 0 {
		return []T{}, nil
	}
	if m.ptr == nil {
		return nil, fmt.Errorf("memory has been freed")
	}

	var zero T
	elementSize := unsafe.Sizeof(zero)
	if elementSize == 0 {
		return nil, fmt.Errorf("zero-sized element types are not supported")
	}

	if uint64(length) > uint64(m.size)/uint64(elementSize) {
		return nil, fmt.Errorf("requested view exceeds allocation of %d bytes", m.size)
	}

	return unsafe.Slice((*T)(m.ptr), length), nil
}

// GetType returns the memory type
func (m *Memory) GetType() Type {
	return m.memType
}

// GetAlignment returns the memory alignment
func (m *Memory) GetAlignment() int {
	return m.alignment
}

// GetPitch returns the memory pitch for 2D allocations
func (m *Memory) GetPitch() int64 {
	return m.pitch
}

// GetDeviceID returns the device ID where memory is allocated
func (m *Memory) GetDeviceID() int {
	return m.deviceID
}

// GetInfo returns the total and free memory information
func GetInfo() (free, total int64) {
	if internal.ShouldUseCuda() {
		return internal.GetCudaMemoryInfo()
	}

	// For simulation, return simulated memory info
	return 8 * 1024 * 1024 * 1024, 8 * 1024 * 1024 * 1024 // 8GB simulated
}

// GetManager returns the global memory manager
func GetManager() *Manager {
	return globalManager
}

// GetTotalAllocated returns the total bytes allocated
func (m *Manager) GetTotalAllocated() int64 {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return m.totalBytes
}

// GetAllocationCount returns the number of active allocations
func (m *Manager) GetAllocationCount() int {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return len(m.allocations)
}
