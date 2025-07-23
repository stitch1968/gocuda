//go:build !cuda
// +build !cuda

package cuda

import (
	"fmt"
	"runtime"
	"sync"
	"unsafe"
)

// CudaRuntime manages the CUDA runtime and detection (simulation mode)
type CudaRuntime struct {
	Available      bool
	DeviceCount    int
	RuntimeVersion int
	DriverVersion  int
	Devices        []*RealDevice
	mu             sync.RWMutex
}

// RealDevice represents a simulated CUDA device
type RealDevice struct {
	ID         int
	Name       string
	Properties RealDeviceProperties
}

// RealDeviceProperties holds simulated CUDA device properties
type RealDeviceProperties struct {
	Major                       int
	Minor                       int
	TotalGlobalMem              int64
	SharedMemPerBlock           int64
	MaxThreadsPerBlock          int
	MaxThreadsDim               [3]int
	MaxGridSize                 [3]int
	WarpSize                    int
	MultiProcessorCount         int
	ClockRate                   int
	MemoryClockRate             int
	MemoryBusWidth              int
	MaxTexture1D                int
	MaxTexture2D                [2]int
	MaxTexture3D                [3]int
	L2CacheSize                 int
	MaxThreadsPerMultiProcessor int
	ComputeCapabilityMajor      int
	ComputeCapabilityMinor      int
}

var (
	globalRuntime        *CudaRuntime
	runtimeInitOnce      sync.Once
	cudaAvailable        bool = false
	fallbackToSimulation bool = true
)

// InitializeCudaRuntime initializes the simulated CUDA runtime
func InitializeCudaRuntime() *CudaRuntime {
	runtimeInitOnce.Do(func() {
		globalRuntime = &CudaRuntime{
			Available:      false,
			DeviceCount:    0,
			RuntimeVersion: 0,
			DriverVersion:  0,
		}

		fmt.Println("CUDA support not compiled in, using CPU simulation only")
		fmt.Printf("Simulated environment: %d CPU cores available\n", runtime.NumCPU())
	})

	return globalRuntime
}

// GetCudaRuntime returns the global CUDA runtime instance
func GetCudaRuntime() *CudaRuntime {
	if globalRuntime == nil {
		return InitializeCudaRuntime()
	}
	return globalRuntime
}

// IsCudaAvailable returns false in simulation mode
func IsCudaAvailable() bool {
	return false
}

// ShouldUseCuda returns false in simulation mode
func ShouldUseCuda() bool {
	return false
}

// ForceFallbackToSimulation is a no-op in simulation mode
func ForceFallbackToSimulation(force bool) {
	// Always in simulation mode
}

// GetCudaDeviceCount returns 0 in simulation mode
func GetCudaDeviceCount() int {
	return 0
}

// GetCudaDevice returns an error in simulation mode
func GetCudaDevice(deviceID int) (*RealDevice, error) {
	return nil, fmt.Errorf("CUDA not available - compiled without CUDA support")
}

// SetCudaDevice is a no-op in simulation mode
func SetCudaDevice(deviceID int) error {
	return nil // No-op for simulation
}

// GetCudaDeviceProperties returns an error in simulation mode
func GetCudaDeviceProperties(deviceID int) (*RealDeviceProperties, error) {
	return nil, fmt.Errorf("CUDA not available - compiled without CUDA support")
}

// CudaMalloc returns an error in simulation mode
func CudaMalloc(size int64) (unsafe.Pointer, error) {
	return nil, fmt.Errorf("CUDA not available - use simulation memory functions instead")
}

// CudaFree is a no-op in simulation mode
func CudaFree(ptr unsafe.Pointer) error {
	return nil // No-op for simulation
}

// CudaMemcpy returns an error in simulation mode
func CudaMemcpy(dst, src unsafe.Pointer, size int64, kind MemcpyKind) error {
	return fmt.Errorf("CUDA not available - use simulation memory functions instead")
}

// PrintCudaInfo prints simulation information
func PrintCudaInfo() {
	fmt.Println("=== CUDA Information ===")
	fmt.Printf("CUDA Available: No (not compiled with CUDA support)\n")
	fmt.Printf("Fallback Mode: CPU Simulation\n")
	fmt.Printf("CPU Cores: %d\n", runtime.NumCPU())
	fmt.Printf("Current Mode: CPU Simulation\n")
	fmt.Println("========================")
}
