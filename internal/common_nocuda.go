//go:build !cuda

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

// CudaMallocOnDevice allocates CUDA memory on a specific device (simulation stub).
func CudaMallocOnDevice(size int64, deviceID int) (unsafe.Pointer, error) {
	return CudaMalloc(size)
}

// CudaFree frees CUDA memory (simulation stub)
func CudaFree(ptr unsafe.Pointer) error {
	return fmt.Errorf("CUDA not available - using simulation mode")
}

// CudaFreeOnDevice frees CUDA memory on a specific device (simulation stub).
func CudaFreeOnDevice(ptr unsafe.Pointer, deviceID int) error {
	return CudaFree(ptr)
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

// CudaMemcpyOnDevice performs CUDA memory copy on a specific device (simulation stub).
func CudaMemcpyOnDevice(dst, src unsafe.Pointer, count int64, kind int, deviceID int) error {
	return CudaMemcpy(dst, src, count, kind)
}

// CudaMemsetOnDevice performs CUDA memset on a specific device (simulation stub).
func CudaMemsetOnDevice(dst unsafe.Pointer, value int, count int64, deviceID int) error {
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

// CudaDeviceSynchronizeOnDevice synchronizes a specific device (simulation).
func CudaDeviceSynchronizeOnDevice(deviceID int) error {
	return CudaDeviceSynchronize()
}

// RunOnDevice executes fn within the selected device context (simulation pass-through).
func RunOnDevice(deviceID int, fn func() error) error {
	return fn()
}

// CanAccessPeer reports whether one device can access another (simulation stub).
func CanAccessPeer(deviceID, peerDeviceID int) (bool, error) {
	return false, fmt.Errorf("CUDA not available - using simulation mode")
}

// EnablePeerAccess enables peer access between devices (simulation stub).
func EnablePeerAccess(deviceID, peerDeviceID int) error {
	return fmt.Errorf("CUDA not available - using simulation mode")
}

// CudaMemcpyPeer performs peer memory copies between devices (simulation stub).
func CudaMemcpyPeer(dst unsafe.Pointer, dstDeviceID int, src unsafe.Pointer, srcDeviceID int, count int64) error {
	return fmt.Errorf("CUDA not available - using simulation mode")
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
