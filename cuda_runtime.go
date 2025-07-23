//go:build cuda
// +build cuda

package cuda

/*
#cgo CFLAGS: -I/usr/local/cuda/include -I/opt/cuda/include -IC:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.0/include
#cgo LDFLAGS: -L/usr/local/cuda/lib64 -L/opt/cuda/lib64 -LC:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.0/lib/x64 -lcudart -lcuda
#cgo windows LDFLAGS: -lcudart -lcuda

#include <cuda_runtime.h>
#include <cuda.h>
#include <stdlib.h>

// Helper function to get CUDA runtime version
int getCudaRuntimeVersion() {
    int version = 0;
    cudaError_t err = cudaRuntimeGetVersion(&version);
    if (err != cudaSuccess) {
        return -1;
    }
    return version;
}

// Helper function to get CUDA driver version
int getCudaDriverVersion() {
    int version = 0;
    cudaError_t err = cudaDriverGetVersion(&version);
    if (err != cudaSuccess) {
        return -1;
    }
    return version;
}

// Helper function to get device count
int getCudaDeviceCount() {
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    if (err != cudaSuccess) {
        return -1;
    }
    return count;
}

// Helper function to get device properties
int getCudaDeviceProperties(int device, cudaDeviceProp* prop) {
    cudaError_t err = cudaGetDeviceProperties(prop, device);
    return (int)err;
}

// Helper function to allocate device memory
cudaError_t cudaMallocWrapper(void** ptr, size_t size) {
    return cudaMalloc(ptr, size);
}

// Helper function to free device memory
cudaError_t cudaFreeWrapper(void* ptr) {
    return cudaFree(ptr);
}

// Helper function for memory copy
cudaError_t cudaMemcpyWrapper(void* dst, const void* src, size_t count, cudaMemcpyKind kind) {
    return cudaMemcpy(dst, src, count, kind);
}

// Helper function for async memory copy
cudaError_t cudaMemcpyAsyncWrapper(void* dst, const void* src, size_t count, cudaMemcpyKind kind, cudaStream_t stream) {
    return cudaMemcpyAsync(dst, src, count, kind, stream);
}

// Helper function to create stream
cudaError_t cudaStreamCreateWrapper(cudaStream_t* stream) {
    return cudaStreamCreate(stream);
}

// Helper function to destroy stream
cudaError_t cudaStreamDestroyWrapper(cudaStream_t stream) {
    return cudaStreamDestroy(stream);
}

// Helper function to synchronize stream
cudaError_t cudaStreamSynchronizeWrapper(cudaStream_t stream) {
    return cudaStreamSynchronize(stream);
}

// Helper function to check for errors
const char* cudaGetErrorStringWrapper(cudaError_t error) {
    return cudaGetErrorString(error);
}

*/
import "C"
import (
	"fmt"
	"sync"
	"unsafe"
)

// CudaRuntime manages the CUDA runtime and detection
type CudaRuntime struct {
	Available      bool
	DeviceCount    int
	RuntimeVersion int
	DriverVersion  int
	Devices        []*RealDevice
	mu             sync.RWMutex
}

// RealDevice represents an actual CUDA device
type RealDevice struct {
	ID         int
	Name       string
	Properties RealDeviceProperties
	cudaDevice C.int
}

// RealDeviceProperties holds actual CUDA device properties
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
	cudaAvailable        bool
	fallbackToSimulation bool
)

// InitializeCudaRuntime initializes the CUDA runtime and detects available devices
func InitializeCudaRuntime() *CudaRuntime {
	runtimeInitOnce.Do(func() {
		globalRuntime = &CudaRuntime{}

		// Try to initialize CUDA
		if detectCuda() {
			globalRuntime.Available = true
			cudaAvailable = true
			fallbackToSimulation = false

			// Get runtime and driver versions
			globalRuntime.RuntimeVersion = int(C.getCudaRuntimeVersion())
			globalRuntime.DriverVersion = int(C.getCudaDriverVersion())

			// Get device count
			deviceCount := int(C.getCudaDeviceCount())
			if deviceCount > 0 {
				globalRuntime.DeviceCount = deviceCount
				globalRuntime.Devices = make([]*RealDevice, deviceCount)

				// Initialize each device
				for i := 0; i < deviceCount; i++ {
					device := &RealDevice{
						ID:         i,
						cudaDevice: C.int(i),
					}

					// Get device properties
					var prop C.cudaDeviceProp
					if C.getCudaDeviceProperties(C.int(i), &prop) == 0 {
						device.Name = C.GoString(&prop.name[0])
						device.Properties = RealDeviceProperties{
							Major:                       int(prop.major),
							Minor:                       int(prop.minor),
							TotalGlobalMem:              int64(prop.totalGlobalMem),
							SharedMemPerBlock:           int64(prop.sharedMemPerBlock),
							MaxThreadsPerBlock:          int(prop.maxThreadsPerBlock),
							WarpSize:                    int(prop.warpSize),
							MultiProcessorCount:         int(prop.multiProcessorCount),
							ClockRate:                   int(prop.clockRate),
							MemoryClockRate:             int(prop.memoryClockRate),
							MemoryBusWidth:              int(prop.memoryBusWidth),
							L2CacheSize:                 int(prop.l2CacheSize),
							MaxThreadsPerMultiProcessor: int(prop.maxThreadsPerMultiProcessor),
							ComputeCapabilityMajor:      int(prop.major),
							ComputeCapabilityMinor:      int(prop.minor),
						}

						// Copy array properties
						for j := 0; j < 3; j++ {
							device.Properties.MaxThreadsDim[j] = int(prop.maxThreadsDim[j])
							device.Properties.MaxGridSize[j] = int(prop.maxGridSize[j])
							if j < 2 {
								device.Properties.MaxTexture2D[j] = int(prop.maxTexture2D[j])
							}
							device.Properties.MaxTexture3D[j] = int(prop.maxTexture3D[j])
						}
						device.Properties.MaxTexture1D = int(prop.maxTexture1D)
					}

					globalRuntime.Devices[i] = device
				}

				fmt.Printf("CUDA initialized successfully: %d device(s) found\n", deviceCount)
				for i, dev := range globalRuntime.Devices {
					fmt.Printf("  Device %d: %s (Compute %d.%d)\n",
						i, dev.Name, dev.Properties.Major, dev.Properties.Minor)
				}
			} else {
				fmt.Println("CUDA runtime available but no devices found, falling back to simulation")
				fallbackToSimulation = true
				cudaAvailable = false
			}
		} else {
			fmt.Println("CUDA not available, using CPU simulation")
			globalRuntime.Available = false
			cudaAvailable = false
			fallbackToSimulation = true
		}
	})

	return globalRuntime
}

// detectCuda attempts to detect CUDA availability
func detectCuda() bool {
	defer func() {
		if r := recover(); r != nil {
			// CUDA library not available
		}
	}()

	// Try to get CUDA runtime version
	version := C.getCudaRuntimeVersion()
	return version > 0
}

// GetCudaRuntime returns the global CUDA runtime instance
func GetCudaRuntime() *CudaRuntime {
	if globalRuntime == nil {
		return InitializeCudaRuntime()
	}
	return globalRuntime
}

// IsCudaAvailable returns true if CUDA is available and devices are found
func IsCudaAvailable() bool {
	runtime := GetCudaRuntime()
	return runtime.Available && runtime.DeviceCount > 0
}

// ShouldUseCuda returns true if CUDA should be used instead of simulation
func ShouldUseCuda() bool {
	return cudaAvailable && !fallbackToSimulation
}

// ForceFallbackToSimulation forces the library to use CPU simulation even if CUDA is available
func ForceFallbackToSimulation(force bool) {
	fallbackToSimulation = force
}

// GetCudaDeviceCount returns the number of CUDA devices
func GetCudaDeviceCount() int {
	runtime := GetCudaRuntime()
	if runtime.Available {
		return runtime.DeviceCount
	}
	return 0
}

// GetCudaDevice returns a CUDA device by ID
func GetCudaDevice(deviceID int) (*RealDevice, error) {
	runtime := GetCudaRuntime()
	if !runtime.Available {
		return nil, fmt.Errorf("CUDA not available")
	}

	if deviceID >= len(runtime.Devices) {
		return nil, fmt.Errorf("device ID %d out of range", deviceID)
	}

	return runtime.Devices[deviceID], nil
}

// SetCudaDevice sets the current CUDA device
func SetCudaDevice(deviceID int) error {
	if !ShouldUseCuda() {
		return nil // No-op for simulation
	}

	runtime := GetCudaRuntime()
	if deviceID >= runtime.DeviceCount {
		return fmt.Errorf("device ID %d out of range", deviceID)
	}

	err := C.cudaSetDevice(C.int(deviceID))
	if err != C.cudaSuccess {
		return fmt.Errorf("failed to set CUDA device %d: %s",
			deviceID, C.GoString(C.cudaGetErrorStringWrapper(err)))
	}

	return nil
}

// GetCudaDeviceProperties returns properties for a CUDA device
func GetCudaDeviceProperties(deviceID int) (*RealDeviceProperties, error) {
	device, err := GetCudaDevice(deviceID)
	if err != nil {
		return nil, err
	}

	return &device.Properties, nil
}

// CudaMalloc allocates memory on the GPU using real CUDA
func CudaMalloc(size int64) (unsafe.Pointer, error) {
	if !ShouldUseCuda() {
		return nil, fmt.Errorf("CUDA not available, use simulation instead")
	}

	var ptr unsafe.Pointer
	err := C.cudaMallocWrapper(&ptr, C.size_t(size))
	if err != C.cudaSuccess {
		return nil, fmt.Errorf("CUDA malloc failed: %s",
			C.GoString(C.cudaGetErrorStringWrapper(err)))
	}

	return ptr, nil
}

// CudaFree frees GPU memory using real CUDA
func CudaFree(ptr unsafe.Pointer) error {
	if !ShouldUseCuda() {
		return nil // No-op for simulation
	}

	err := C.cudaFreeWrapper(ptr)
	if err != C.cudaSuccess {
		return fmt.Errorf("CUDA free failed: %s",
			C.GoString(C.cudaGetErrorStringWrapper(err)))
	}

	return nil
}

// CudaMemcpy copies memory using real CUDA
func CudaMemcpy(dst, src unsafe.Pointer, size int64, kind MemcpyKind) error {
	if !ShouldUseCuda() {
		return fmt.Errorf("CUDA not available, use simulation instead")
	}

	var cudaKind C.cudaMemcpyKind
	switch kind {
	case MemcpyKindHostToHost:
		cudaKind = C.cudaMemcpyHostToHost
	case MemcpyKindHostToDevice:
		cudaKind = C.cudaMemcpyHostToDevice
	case MemcpyKindDeviceToHost:
		cudaKind = C.cudaMemcpyDeviceToHost
	case MemcpyKindDeviceToDevice:
		cudaKind = C.cudaMemcpyDeviceToDevice
	case MemcpyKindDefault:
		cudaKind = C.cudaMemcpyDefault
	}

	err := C.cudaMemcpyWrapper(dst, src, C.size_t(size), cudaKind)
	if err != C.cudaSuccess {
		return fmt.Errorf("CUDA memcpy failed: %s",
			C.GoString(C.cudaGetErrorStringWrapper(err)))
	}

	return nil
}

// PrintCudaInfo prints information about the CUDA runtime and devices
func PrintCudaInfo() {
	runtime := GetCudaRuntime()

	fmt.Println("=== CUDA Information ===")
	if runtime.Available {
		fmt.Printf("CUDA Available: Yes\n")
		fmt.Printf("Runtime Version: %d\n", runtime.RuntimeVersion)
		fmt.Printf("Driver Version: %d\n", runtime.DriverVersion)
		fmt.Printf("Device Count: %d\n", runtime.DeviceCount)
		fmt.Println()

		for i, device := range runtime.Devices {
			props := device.Properties
			fmt.Printf("Device %d: %s\n", i, device.Name)
			fmt.Printf("  Compute Capability: %d.%d\n", props.Major, props.Minor)
			fmt.Printf("  Total Global Memory: %d MB\n", props.TotalGlobalMem/(1024*1024))
			fmt.Printf("  Multiprocessors: %d\n", props.MultiProcessorCount)
			fmt.Printf("  CUDA Cores: %d\n", props.MultiProcessorCount*getSPCount(props.Major, props.Minor))
			fmt.Printf("  Max Threads per Block: %d\n", props.MaxThreadsPerBlock)
			fmt.Printf("  Warp Size: %d\n", props.WarpSize)
			fmt.Printf("  Memory Clock Rate: %d MHz\n", props.MemoryClockRate/1000)
			fmt.Printf("  Memory Bus Width: %d bits\n", props.MemoryBusWidth)
			fmt.Println()
		}
	} else {
		fmt.Printf("CUDA Available: No\n")
		fmt.Printf("Fallback Mode: CPU Simulation\n")
		fmt.Printf("Simulated Cores: %d\n", runtime.NumCPU())
	}

	fmt.Printf("Current Mode: ")
	if ShouldUseCuda() {
		fmt.Printf("GPU Execution\n")
	} else {
		fmt.Printf("CPU Simulation\n")
	}
	fmt.Println("========================")
}

// getSPCount returns the number of streaming processors per multiprocessor
func getSPCount(major, minor int) int {
	switch major {
	case 2: // Fermi
		if minor == 1 {
			return 48
		}
		return 32
	case 3: // Kepler
		return 192
	case 5: // Maxwell
		return 128
	case 6: // Pascal
		if minor == 0 {
			return 64
		}
		return 128
	case 7: // Volta/Turing
		if minor == 0 {
			return 64
		}
		return 64
	case 8: // Ampere
		if minor == 0 {
			return 64
		}
		return 128
	case 9: // Hopper
		return 128
	default:
		return 64 // Default estimate
	}
}
