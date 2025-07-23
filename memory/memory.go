// Package memory provides GPU memory management functionality for GoCUDA.
// This package handles memory allocation, deallocation, and transfers between
// host and device memory.
package memory

import (
	"fmt"
	"runtime"
	"sync"
	"unsafe"

	"github.com/stitch1968/gocuda/internal"
)

// Type represents different CUDA memory types
type Type int

const (
	TypeDevice   Type = iota // Device (global) memory
	TypeHost                 // Host (pageable) memory
	TypePinned               // Host pinned (page-locked) memory
	TypeUnified              // Unified memory (managed)
	TypeShared               // Shared memory (per-block)
	TypeConstant             // Constant memory
	TypeTexture              // Texture memory
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
	memType    Type
	attributes Attribute
	pitch      int64 // For 2D/3D memory allocations
	alignment  int   // Memory alignment (typically 256 bytes for CUDA)
	deviceID   int   // Device where memory is allocated
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

// Alloc allocates memory on the GPU (simulated with regular memory)
func Alloc(size int64) (*Memory, error) {
	return AllocWithStream(internal.GetDefaultStream(), size)
}

// AllocWithStream allocates memory on the GPU with a specific stream
func AllocWithStream(stream *internal.Stream, size int64) (*Memory, error) {
	return AllocWithTypeAndStream(stream, size, TypeDevice)
}

// AllocWithTypeAndStream allocates memory with specific type and stream
func AllocWithTypeAndStream(stream *internal.Stream, size int64, memType Type) (*Memory, error) {
	if size <= 0 {
		return nil, fmt.Errorf("invalid size: %d", size)
	}

	var ptr unsafe.Pointer
	var data []byte
	var err error

	// Check if we should use real CUDA or simulation
	if internal.ShouldUseCuda() && memType == TypeDevice {
		// Use real CUDA memory allocation
		ptr, err = internal.CudaMalloc(size)
		if err != nil {
			return nil, fmt.Errorf("CUDA malloc failed: %v", err)
		}
		// For real CUDA memory, we don't have direct access to the data
		data = nil
	} else {
		// Use simulation with regular memory
		// CUDA requires 256-byte alignment for optimal performance
		alignment := 256
		alignedSize := (size + int64(alignment) - 1) &^ (int64(alignment) - 1)

		// Allocate regular memory to simulate GPU memory
		data = make([]byte, alignedSize)
		ptr = unsafe.Pointer(&data[0])
	}

	mem := &Memory{
		ptr:       ptr,
		size:      size,
		stream:    stream,
		data:      data,
		memType:   memType,
		alignment: 256,
		deviceID:  0, // Default device
		attributes: Attribute{
			IsAligned: true,
		},
	}

	// Register the allocation
	globalManager.mu.Lock()
	globalManager.allocations[uintptr(ptr)] = mem
	globalManager.totalBytes += size
	globalManager.mu.Unlock()

	// Set up finalizer for automatic cleanup
	runtime.SetFinalizer(mem, (*Memory).finalize)

	return mem, nil
}

// Free releases the memory
func (m *Memory) Free() error {
	if m.ptr == nil {
		return nil // Already freed
	}

	// Unregister the allocation
	globalManager.mu.Lock()
	delete(globalManager.allocations, uintptr(m.ptr))
	globalManager.totalBytes -= m.size
	globalManager.mu.Unlock()

	var err error
	if internal.ShouldUseCuda() && m.memType == TypeDevice && m.data == nil {
		// Use real CUDA memory deallocation
		err = internal.CudaFree(m.ptr)
	}
	// For simulation, Go's GC will handle the slice cleanup

	// Clear the finalizer
	runtime.SetFinalizer(m, nil)

	// Mark as freed
	m.ptr = nil
	m.data = nil

	return err
}

// finalize is called by the garbage collector
func (m *Memory) finalize() {
	if m.ptr != nil {
		m.Free()
	}
}

// Size returns the memory size
func (m *Memory) Size() int64 {
	return m.size
}

// Ptr returns the memory pointer
func (m *Memory) Ptr() unsafe.Pointer {
	return m.ptr
}

// Data returns the simulation data slice
func (m *Memory) Data() []byte {
	return m.data
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
