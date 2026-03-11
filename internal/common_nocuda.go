//go:build !cuda
// +build !cuda

package internal

import (
	"fmt"
	"runtime"
	"sync"
	"time"
	"unsafe"
)

// CUDA runtime detection and simulation functions

var (
	cudaAvailable   bool
	cudaDeviceCount int
	simulationMode  = true
	initOnce        sync.Once
)

// ShouldUseCuda returns whether real CUDA should be used
func ShouldUseCuda() bool {
	initOnce.Do(func() {
		// In this implementation ensure simulation
		simulationMode = true
		cudaAvailable = false
		cudaDeviceCount = 0

		// Print simulation mode message
		fmt.Println("CUDA support not compiled in, using CPU simulation only")
		fmt.Printf("Simulated environment: %d CPU cores available\n", runtime.NumCPU())
		fmt.Println("Using CPU simulation mode")
	})
	return false
}

// IsCudaAvailable returns whether CUDA is available
func IsCudaAvailable() bool {
	return false
}

// GetCudaDeviceCount returns the number of CUDA devices
func GetCudaDeviceCount() int {
	return 0
}

// CUDA memory functions (simulation stubs)

// CudaMalloc allocates CUDA memory (simulation stub)
func CudaMalloc(size int64) (unsafe.Pointer, error) {
	// In simulation mode, we verify the call but return error as
	// pure simulation typically handles memory differently (Go heaps)
	return nil, fmt.Errorf("CUDA not available - using simulation mode")
}

// CudaFree frees CUDA memory (simulation stub)
func CudaFree(ptr unsafe.Pointer) error {
	return fmt.Errorf("CUDA not available - using simulation mode")
}

// GetCudaMemoryInfo returns CUDA memory information (simulation stub)
func GetCudaMemoryInfo() (free, total int64) {
	// Return simulated memory info
	return 8 * 1024 * 1024 * 1024, 8 * 1024 * 1024 * 1024 // 8GB simulated
}

// CudaMemcpy performs CUDA memory copy (simulation stub)
func CudaMemcpy(dst, src unsafe.Pointer, count int64, kind int) error {
	return fmt.Errorf("CUDA not available - using simulation mode")
}

// Memory copy kind constants
const (
	MemcpyKindHostToHost     = 0
	MemcpyKindHostToDevice   = 1
	MemcpyKindDeviceToHost   = 2
	MemcpyKindDeviceToDevice = 3
)

// Synchronization functions

// CudaDeviceSynchronize synchronizes the device (simulation)
func CudaDeviceSynchronize() error {
	// In simulation mode, just add a small delay to simulate work
	time.Sleep(1 * time.Millisecond)
	return nil
}

// Device management utilities

// Device represents a CUDA device
type Device struct {
	ID   int
	Name string
}

// GetDevices returns available CUDA devices (simulation)
func GetDevices() []*Device {
	// Return simulated device
	return []*Device{
		{ID: 0, Name: "Simulated CUDA Device"},
	}
}

// SetDevice sets the active CUDA device (simulation)
func SetDevice(deviceID int) error {
	// In simulation mode, always succeed
	return nil
}

// GetDevice returns the current device ID (simulation)
func GetDevice() int {
	return 0 // Always device 0 in simulation
}
