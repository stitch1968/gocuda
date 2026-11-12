//go:build cuda

package internal

import (
	"errors"
	"fmt"
	"runtime"
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

cudaError_t cudaMallocDevice_Internal(void* ptr, size_t size) {
	return cudaMalloc((void**)ptr, size);
}

cudaError_t cudaPointerIsDevice_Internal(void* ptr, int* isDevice) {
	struct cudaPointerAttributes attributes;
	cudaError_t status = cudaPointerGetAttributes(&attributes, ptr);
	if (status == cudaSuccess) {
		*isDevice = attributes.type == cudaMemoryTypeDevice;
	}
	return status;
}

cudaError_t cudaMallocHost_Internal(void* ptr, size_t size) {
	return cudaMallocHost((void**)ptr, size);
}

cudaError_t cudaFreeHost_Internal(void* ptr) {
	return cudaFreeHost(ptr);
}

cudaError_t cudaFree_Internal(void* ptr) {
	return cudaFree(ptr);
}

cudaError_t cudaMemcpy_Internal(void* dst, void* src, size_t count, int kind) {
	return cudaMemcpy(dst, src, count, (enum cudaMemcpyKind)kind);
}

cudaError_t cudaMemset_Internal(void* dst, int value, size_t count) {
	return cudaMemset(dst, value, count);
}

cudaError_t cudaDeviceSynchronize_Internal() {
	return cudaDeviceSynchronize();
}

cudaError_t cudaDefaultStreamSynchronize_Internal() {
	return cudaStreamSynchronize((cudaStream_t)0);
}

cudaError_t cudaSetDevice_Internal(int device) {
	return cudaSetDevice(device);
}

cudaError_t cudaGetDevice_Internal(int* device) {
	return cudaGetDevice(device);
}

cudaError_t cudaDeviceCanAccessPeer_Internal(int* canAccessPeer, int device, int peerDevice) {
	return cudaDeviceCanAccessPeer(canAccessPeer, device, peerDevice);
}

cudaError_t cudaDeviceEnablePeerAccess_Internal(int peerDevice) {
	return cudaDeviceEnablePeerAccess(peerDevice, 0);
}

cudaError_t cudaMemcpyPeer_Internal(void* dst, int dstDevice, void* src, int srcDevice, size_t count) {
	return cudaMemcpyPeer(dst, dstDevice, src, srcDevice, count);
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

func currentDeviceLocked() (int, error) {
	var device C.int
	err := C.cudaGetDevice_Internal(&device)
	if err != 0 {
		return 0, fmt.Errorf("cudaGetDevice failed with error code %d", err)
	}
	return int(device), nil
}

func setDeviceLocked(deviceID int) error {
	err := C.cudaSetDevice_Internal(C.int(deviceID))
	if err != 0 {
		return fmt.Errorf("cudaSetDevice failed with error code %d", err)
	}
	return nil
}

func withLockedDevice(deviceID int, fn func() error) (err error) {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	previousDevice, err := currentDeviceLocked()
	if err != nil {
		return err
	}
	// cudaGetDevice may report the default device before this OS thread has
	// initialized its CUDA primary context.
	if err := setDeviceLocked(deviceID); err != nil {
		return err
	}
	if previousDevice != deviceID {
		defer func() {
			if restoreErr := setDeviceLocked(previousDevice); restoreErr != nil {
				err = errors.Join(err, fmt.Errorf("restore CUDA device %d: %w", previousDevice, restoreErr))
			}
		}()
	}

	return fn()
}

func withLockedThread(fn func() error) error {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	return fn()
}

// RunOnDevice executes fn on a locked OS thread after selecting the target CUDA device.
func RunOnDevice(deviceID int, fn func() error) error {
	if !ShouldUseCuda() {
		return fn()
	}
	return withLockedDevice(deviceID, fn)
}

// CanAccessPeer reports whether one CUDA device can access another via peer access.
func CanAccessPeer(deviceID, peerDeviceID int) (bool, error) {
	if !ShouldUseCuda() {
		return false, fmt.Errorf("CUDA not available")
	}
	var canAccess C.int
	err := withLockedThread(func() error {
		status := C.cudaDeviceCanAccessPeer_Internal(&canAccess, C.int(deviceID), C.int(peerDeviceID))
		if status != 0 {
			return fmt.Errorf("cudaDeviceCanAccessPeer failed with error code %d", status)
		}
		return nil
	})
	if err != nil {
		return false, err
	}
	return canAccess != 0, nil
}

// EnablePeerAccess enables peer access from deviceID to peerDeviceID.
func EnablePeerAccess(deviceID, peerDeviceID int) error {
	if !ShouldUseCuda() {
		return fmt.Errorf("CUDA not available")
	}
	return RunOnDevice(deviceID, func() error {
		status := C.cudaDeviceEnablePeerAccess_Internal(C.int(peerDeviceID))
		if status == C.cudaErrorPeerAccessAlreadyEnabled {
			C.cudaGetLastError()
			return nil
		}
		if status != 0 {
			return fmt.Errorf("cudaDeviceEnablePeerAccess failed with error code %d", status)
		}
		return nil
	})
}

// CudaMemcpyPeer performs direct device-to-device copies between different devices.
func CudaMemcpyPeer(dst unsafe.Pointer, dstDeviceID int, src unsafe.Pointer, srcDeviceID int, count int64) error {
	if !ShouldUseCuda() {
		return fmt.Errorf("CUDA not available")
	}
	return RunOnDevice(dstDeviceID, func() error {
		status := C.cudaMemcpyPeer_Internal(dst, C.int(dstDeviceID), src, C.int(srcDeviceID), C.size_t(count))
		if status != 0 {
			return fmt.Errorf("cudaMemcpyPeer failed with error code %d", status)
		}
		return nil
	})
}

// CudaMalloc allocates CUDA memory
func CudaMalloc(size int64) (unsafe.Pointer, error) {
	return CudaMallocOnDevice(size, 0)
}

// CudaMallocOnDevice allocates CUDA memory on a specific device.
func CudaMallocOnDevice(size int64, deviceID int) (unsafe.Pointer, error) {
	if !ShouldUseCuda() {
		return nil, fmt.Errorf("CUDA not available")
	}

	var ptr unsafe.Pointer
	err := RunOnDevice(deviceID, func() error {
		cPtr := unsafe.Pointer(&ptr)
		status := C.cudaMalloc_Internal(cPtr, C.size_t(size))
		if status != 0 {
			return fmt.Errorf("cudaMalloc failed with error code %d", status)
		}
		return nil
	})
	if err != nil {
		if ptr != nil {
			_, cleanupErr := CudaFreeOnDeviceResult(ptr, deviceID)
			return nil, errors.Join(err, cleanupErr)
		}
		return nil, err
	}
	return ptr, nil
}

// CudaMallocDeviceOnDevice allocates device-only memory on the selected device.
func CudaMallocDeviceOnDevice(size int64, deviceID int) (unsafe.Pointer, error) {
	if !ShouldUseCuda() {
		return nil, fmt.Errorf("CUDA not available")
	}
	var ptr unsafe.Pointer
	err := RunOnDevice(deviceID, func() error {
		status := C.cudaMallocDevice_Internal(unsafe.Pointer(&ptr), C.size_t(size))
		if status != 0 {
			return fmt.Errorf("cudaMalloc failed with error code %d", status)
		}
		return nil
	})
	if err != nil && ptr != nil {
		_, cleanupErr := CudaFreeOnDeviceResult(ptr, deviceID)
		return nil, errors.Join(err, cleanupErr)
	}
	return ptr, err
}

// CudaPointerIsDevice reports whether CUDA identifies ptr as device-only memory.
func CudaPointerIsDevice(ptr unsafe.Pointer) (bool, error) {
	var isDevice C.int
	status := C.cudaPointerIsDevice_Internal(ptr, &isDevice)
	if status != 0 {
		return false, fmt.Errorf("cudaPointerGetAttributes failed with error code %d", status)
	}
	return isDevice != 0, nil
}

// CudaMallocHost allocates page-locked host memory.
func CudaMallocHost(size int64) (unsafe.Pointer, error) {
	if !ShouldUseCuda() {
		return nil, fmt.Errorf("CUDA not available")
	}
	var ptr unsafe.Pointer
	status := C.cudaMallocHost_Internal(unsafe.Pointer(&ptr), C.size_t(size))
	if status != 0 {
		return nil, fmt.Errorf("cudaMallocHost failed with error code %d", status)
	}
	return ptr, nil
}

// CudaFreeHost releases page-locked host memory.
func CudaFreeHost(ptr unsafe.Pointer) error {
	status := C.cudaFreeHost_Internal(ptr)
	if status != 0 {
		return fmt.Errorf("cudaFreeHost failed with error code %d", status)
	}
	return nil
}

// CudaFree frees CUDA memory
func CudaFree(ptr unsafe.Pointer) error {
	return CudaFreeOnDevice(ptr, 0)
}

// CudaFreeOnDevice frees CUDA memory on a specific device.
func CudaFreeOnDevice(ptr unsafe.Pointer, deviceID int) error {
	_, err := CudaFreeOnDeviceResult(ptr, deviceID)
	return err
}

// CudaFreeOnDeviceResult distinguishes a successful free from a later device
// restoration error.
func CudaFreeOnDeviceResult(ptr unsafe.Pointer, deviceID int) (bool, error) {
	if !ShouldUseCuda() {
		return false, fmt.Errorf("CUDA not available")
	}
	freed := false
	err := RunOnDevice(deviceID, func() error {
		status := C.cudaFree_Internal(ptr)
		if status != 0 {
			return fmt.Errorf("cudaFree failed with error code %d", status)
		}
		freed = true
		return nil
	})
	return freed, err
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
	return CudaMemcpyOnDevice(dst, src, count, kind, 0)
}

// CudaMemcpyOnDevice performs CUDA memory copy after selecting a specific device.
func CudaMemcpyOnDevice(dst, src unsafe.Pointer, count int64, kind int, deviceID int) error {
	if !ShouldUseCuda() {
		return fmt.Errorf("CUDA not available")
	}
	return RunOnDevice(deviceID, func() error {
		status := C.cudaMemcpy_Internal(dst, src, C.size_t(count), C.int(kind))
		if status != 0 {
			return fmt.Errorf("cudaMemcpy failed with error code %d", status)
		}
		return nil
	})
}

// CudaMemsetOnDevice performs CUDA memset after selecting a specific device.
func CudaMemsetOnDevice(dst unsafe.Pointer, value int, count int64, deviceID int) error {
	if !ShouldUseCuda() {
		return fmt.Errorf("CUDA not available")
	}
	return RunOnDevice(deviceID, func() error {
		status := C.cudaMemset_Internal(dst, C.int(value), C.size_t(count))
		if status != 0 {
			return fmt.Errorf("cudaMemset failed with error code %d", status)
		}
		return nil
	})
}

// Memory copy kind constants
// These must match cudaMemcpyKind enum
const (
	MemcpyKindHostToHost     = 0
	MemcpyKindHostToDevice   = 1
	MemcpyKindDeviceToHost   = 2
	MemcpyKindDeviceToDevice = 3
	MemcpyKindDefault        = 4
)

// Synchronization functions

// CudaDeviceSynchronize synchronizes the device
func CudaDeviceSynchronize() error {
	return CudaDeviceSynchronizeOnDevice(0)
}

// CudaDeviceSynchronizeOnDevice synchronizes the specified device.
func CudaDeviceSynchronizeOnDevice(deviceID int) error {
	if !ShouldUseCuda() {
		return nil
	}
	return RunOnDevice(deviceID, func() error {
		status := C.cudaDeviceSynchronize_Internal()
		if status != 0 {
			return fmt.Errorf("cudaDeviceSynchronize failed with error code %d", status)
		}
		return nil
	})
}

// CudaDefaultStreamSynchronizeOnDevice waits for the CUDA default stream.
func CudaDefaultStreamSynchronizeOnDevice(deviceID int) error {
	if !ShouldUseCuda() {
		return fmt.Errorf("CUDA not available")
	}
	return RunOnDevice(deviceID, func() error {
		status := C.cudaDefaultStreamSynchronize_Internal()
		if status != C.cudaSuccess {
			return fmt.Errorf("cudaStreamSynchronize failed with error code %d", status)
		}
		return nil
	})
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

// SetDevice sets the CUDA device of the calling OS thread.
// The caller must lock its OS thread across related CUDA calls.
func SetDevice(deviceID int) error {
	if !ShouldUseCuda() {
		if deviceID != 0 {
			return fmt.Errorf("device ID %d out of range", deviceID)
		}
		return nil
	}
	return setDeviceLocked(deviceID)
}

// CurrentDevice returns the CUDA device of the calling OS thread.
func CurrentDevice() (int, error) {
	if !ShouldUseCuda() {
		return 0, nil
	}
	return currentDeviceLocked()
}

// GetDevice returns the current device ID
func GetDevice() int {
	if !ShouldUseCuda() {
		return 0
	}
	var device int
	_ = withLockedThread(func() error {
		current, err := CurrentDevice()
		if err != nil {
			return err
		}
		device = current
		return nil
	})
	return device
}
