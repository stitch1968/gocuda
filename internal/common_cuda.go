//go:build cuda
// +build cuda

package internal

import (
	"fmt"
	"sync"
	"unsafe"
)

/*
#cgo linux CFLAGS: -I/usr/local/cuda/include -I/opt/cuda/include
#cgo windows CFLAGS: -I"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.1/include" -I"D:/NVIDIA/include"
#cgo linux LDFLAGS: -L/usr/local/cuda/lib64 -L/opt/cuda/lib64 -lcudart -lcuda
#cgo windows LDFLAGS: -L${SRCDIR}/../lib_mingw -lcudart -lcuda -Wl,--no-as-needed
#cgo darwin LDFLAGS: -lcudart -lcuda

#include <cuda_runtime.h>
#include <cuda.h>
#include <stdlib.h>

// Simple wrapper functions to avoid complex C macros in Go
int getCudaDeviceCount_Internal() {
	int count = 0;
	cudaError_t err = cudaGetDeviceCount(&count);
	if (err != cudaSuccess) {
		return 0;
	}
	return count;
}

cudaError_t cudaMalloc_Internal(void* ptr, size_t size) {
	// Use managed memory so CPU-simulated code paths can safely access buffers
	// even when the package is built with CUDA support.
	return cudaMallocManaged((void**)ptr, size, cudaMemAttachGlobal);
}

cudaError_t cudaFree_Internal(void* ptr) {
	return cudaFree(ptr);
}

cudaError_t cudaMemcpy_Internal(void* dst, void* src, size_t count, int kind) {
	return cudaMemcpy(dst, src, count, (enum cudaMemcpyKind)kind);
}

cudaError_t cudaDeviceSynchronize_Internal() {
	return cudaDeviceSynchronize();
}

cudaError_t cudaSetDevice_Internal(int device) {
	return cudaSetDevice(device);
}

cudaError_t cudaGetDevice_Internal(int* device) {
	return cudaGetDevice(device);
}
*/
import "C"

// CUDA runtime detection

var (
	cudaAvailable   bool
	cudaDeviceCount int
	simulationMode  = false
	initOnce        sync.Once
)

// ShouldUseCuda returns whether real CUDA should be used
func ShouldUseCuda() bool {
	initOnce.Do(func() {
		// Check if CUDA is actually available by getting device count
		count := int(C.getCudaDeviceCount_Internal())
		if count > 0 {
			cudaAvailable = true
			cudaDeviceCount = count
			simulationMode = false
			fmt.Printf("CUDA Hardware Initialized: Found %d device(s)\n", count)
		} else {
			cudaAvailable = false
			simulationMode = true
			fmt.Println("CUDA hardware check failed: No devices found. Fallback to simulation.")
		}
	})
	return !simulationMode && cudaAvailable
}

// IsCudaAvailable returns whether CUDA is available
func IsCudaAvailable() bool {
	ShouldUseCuda() // Ensure initialized
	return cudaAvailable
}

// GetCudaDeviceCount returns the number of CUDA devices
func GetCudaDeviceCount() int {
	ShouldUseCuda() // Ensure initialized
	return cudaDeviceCount
}

// CUDA memory functions

// CudaMalloc allocates CUDA memory
func CudaMalloc(size int64) (unsafe.Pointer, error) {
	if !ShouldUseCuda() {
		return nil, fmt.Errorf("CUDA not available")
	}

	var ptr unsafe.Pointer
	// C.cudaMalloc expects a void** (pointer to pointer), so we cast &ptr
	// In Go, unsafe.Pointer constitutes a *void in C context, so &ptr is void**
	// However, cgo can be specific. Let's cast properly.
	// Cast unsafe.Pointer(&ptr) to *unsafe.Pointer, then to **C.void
	cPtr := unsafe.Pointer(&ptr)
	err := C.cudaMalloc_Internal(cPtr, C.size_t(size))
	if err != 0 {
		return nil, fmt.Errorf("cudaMalloc failed with error code %d", err)
	}
	return ptr, nil
}

// CudaFree frees CUDA memory
func CudaFree(ptr unsafe.Pointer) error {
	if !ShouldUseCuda() {
		return fmt.Errorf("CUDA not available")
	}
	err := C.cudaFree_Internal(ptr)
	if err != 0 {
		return fmt.Errorf("cudaFree failed with error code %d", err)
	}
	return nil
}

// GetCudaMemoryInfo returns CUDA memory information
func GetCudaMemoryInfo() (free, total int64) {
	if !ShouldUseCuda() {
		return 0, 0
	}
	var cFree, cTotal C.size_t
	C.cudaMemGetInfo(&cFree, &cTotal)
	return int64(cFree), int64(cTotal)
}

// CudaMemcpy performs CUDA memory copy
func CudaMemcpy(dst, src unsafe.Pointer, count int64, kind int) error {
	if !ShouldUseCuda() {
		return fmt.Errorf("CUDA not available")
	}
	err := C.cudaMemcpy_Internal(dst, src, C.size_t(count), C.int(kind))
	if err != 0 {
		return fmt.Errorf("cudaMemcpy failed with error code %d", err)
	}
	return nil
}

// Memory copy kind constants
// These must match cudaMemcpyKind enum
const (
	MemcpyKindHostToHost     = 0
	MemcpyKindHostToDevice   = 1
	MemcpyKindDeviceToHost   = 2
	MemcpyKindDeviceToDevice = 3
)

// Synchronization functions

// CudaDeviceSynchronize synchronizes the device
func CudaDeviceSynchronize() error {
	if !ShouldUseCuda() {
		return nil
	}
	err := C.cudaDeviceSynchronize_Internal()
	if err != 0 {
		return fmt.Errorf("cudaDeviceSynchronize failed with error code %d", err)
	}
	return nil
}

// Device management utilities

// Device represents a CUDA device
type Device struct {
	ID   int
	Name string
}

// GetDevices returns available CUDA devices
func GetDevices() []*Device {
	if !ShouldUseCuda() {
		return []*Device{}
	}

	devices := make([]*Device, cudaDeviceCount)
	for i := 0; i < cudaDeviceCount; i++ {
		var prop C.struct_cudaDeviceProp
		C.cudaGetDeviceProperties(&prop, C.int(i))
		name := C.GoString(&prop.name[0])
		devices[i] = &Device{
			ID:   i,
			Name: name,
		}
	}
	return devices
}

// SetDevice sets the active CUDA device
func SetDevice(deviceID int) error {
	if !ShouldUseCuda() {
		return fmt.Errorf("CUDA not available")
	}
	err := C.cudaSetDevice_Internal(C.int(deviceID))
	if err != 0 {
		return fmt.Errorf("cudaSetDevice failed with error code %d", err)
	}
	return nil
}

// GetDevice returns the current device ID
func GetDevice() int {
	if !ShouldUseCuda() {
		return 0
	}
	var device C.int
	C.cudaGetDevice_Internal(&device)
	return int(device)
}
