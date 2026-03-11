package cuda

import (
	"fmt"
	"runtime"
	"sync"
	"sync/atomic"

	"github.com/stitch1968/gocuda/internal"
	"github.com/stitch1968/gocuda/memory"
	"github.com/stitch1968/gocuda/streams"
)

// MemcpyKind represents the kind of memory copy operation
type MemcpyKind int

// Memory copy kind constants
const (
	MemcpyKindHostToHost MemcpyKind = iota
	MemcpyKindHostToDevice
	MemcpyKindDeviceToHost
	MemcpyKindDeviceToDevice
	MemcpyKindDefault
)

// CudaError represents a CUDA-like error
type CudaError struct {
	Code    int
	Message string
}

func (e CudaError) Error() string {
	return fmt.Sprintf("CUDA Error %d: %s", e.Code, e.Message)
}

// Device represents a CUDA device (simulated)
type Device struct {
	ID         int
	Name       string
	Properties DeviceProperties
}

// DeviceProperties holds CUDA device properties
type DeviceProperties struct {
	Major               int
	Minor               int
	TotalGlobalMem      int64
	SharedMemPerBlock   int
	MaxThreadsPerBlock  int
	MaxThreadsDim       [3]int
	MaxGridSize         [3]int
	WarpSize            int
	MultiProcessorCount int
	ClockRate           int
	MemoryClockRate     int
	MemoryBusWidth      int
}

// Context represents a CUDA context
type Context struct {
	device *Device
	mu     sync.RWMutex
	active int64 // atomic counter for active operations
}

// Stream represents a CUDA stream
type Stream = internal.Stream

// Memory represents GPU memory allocation - re-exported from memory package
type Memory = memory.Memory

// SimpleKernel is a basic kernel implementation
type SimpleKernel struct {
	Name string
	Func func(args ...interface{}) error
}

var (
	// Global device list
	devices     []*Device
	devicesOnce sync.Once

	// Default context and stream
	defaultContext *Context
	defaultStream  *Stream
	initOnce       sync.Once
	initialized    atomic.Bool
	initErr        error
)

// Initialize initializes the CUDA runtime (real or simulated)
func Initialize() error {
	initOnce.Do(func() {
		// Initialize CUDA runtime detection
		InitializeCudaRuntime()

		// Create default context with detected or simulated device
		defaultContext, initErr = NewContext(0)
		if initErr != nil {
			return
		}

		// Create default stream
		defaultStream = internal.GetDefaultStream()
		initialized.Store(true)
	})
	return initErr
}

// IsInitialized reports whether the CUDA runtime has been initialized successfully.
func IsInitialized() bool {
	return initialized.Load() && initErr == nil && defaultContext != nil && defaultStream != nil
}

// InitializationError returns the initialization error recorded by Initialize,
// or nil if initialization has not failed.
func InitializationError() error {
	return initErr
}

// GetDevices returns all available CUDA devices (real or simulated)
func GetDevices() ([]*Device, error) {
	var err error
	devicesOnce.Do(func() {
		cudaRuntime := GetCudaRuntime()

		if ShouldUseCuda() {
			// Use real CUDA devices
			devices = make([]*Device, len(cudaRuntime.Devices))
			for i, realDevice := range cudaRuntime.Devices {
				devices[i] = &Device{
					ID:   realDevice.ID,
					Name: realDevice.Name,
					Properties: DeviceProperties{
						Major:               realDevice.Properties.Major,
						Minor:               realDevice.Properties.Minor,
						TotalGlobalMem:      realDevice.Properties.TotalGlobalMem,
						SharedMemPerBlock:   int(realDevice.Properties.SharedMemPerBlock),
						MaxThreadsPerBlock:  realDevice.Properties.MaxThreadsPerBlock,
						MaxThreadsDim:       realDevice.Properties.MaxThreadsDim,
						MaxGridSize:         realDevice.Properties.MaxGridSize,
						WarpSize:            realDevice.Properties.WarpSize,
						MultiProcessorCount: realDevice.Properties.MultiProcessorCount,
						ClockRate:           realDevice.Properties.ClockRate,
						MemoryClockRate:     realDevice.Properties.MemoryClockRate,
						MemoryBusWidth:      realDevice.Properties.MemoryBusWidth,
					},
				}
			}
		} else {
			// Simulate devices based on CPU cores
			numCores := runtime.NumCPU()
			devices = make([]*Device, 1) // Simulate one GPU device

			devices[0] = &Device{
				ID:   0,
				Name: fmt.Sprintf("Simulated CUDA Device (CPU cores: %d)", numCores),
				Properties: DeviceProperties{
					Major:               7,
					Minor:               5,
					TotalGlobalMem:      8 * 1024 * 1024 * 1024, // 8GB
					SharedMemPerBlock:   48 * 1024,              // 48KB
					MaxThreadsPerBlock:  1024,
					MaxThreadsDim:       [3]int{1024, 1024, 64},
					MaxGridSize:         [3]int{2147483647, 65535, 65535},
					WarpSize:            32,
					MultiProcessorCount: numCores,
					ClockRate:           1500000, // 1.5 GHz
					MemoryClockRate:     7000000, // 7 GHz
					MemoryBusWidth:      256,
				},
			}
		}
	})
	return devices, err
}

// NewContext creates a new CUDA context for the specified device
func NewContext(deviceID int) (*Context, error) {
	devices, err := GetDevices()
	if err != nil {
		return nil, err
	}

	if deviceID >= len(devices) {
		return nil, fmt.Errorf("device ID %d out of range", deviceID)
	}

	ctx := &Context{
		device: devices[deviceID],
	}

	return ctx, nil
}

// NewStream creates a new CUDA stream in this context - delegates to streams package
func (c *Context) NewStream() (*Stream, error) {
	// Delegate to the streams package
	stream, err := streams.CreateStream()
	if err != nil {
		return nil, err
	}
	return stream.Stream, nil
}

// GetDefaultStream returns the default CUDA stream
func GetDefaultStream() *Stream {
	stream, err := DefaultStream()
	if err != nil {
		panic(err)
	}
	return stream
}

// DefaultStream returns the default CUDA stream without panicking on init failure.
func DefaultStream() (*Stream, error) {
	if err := Initialize(); err != nil {
		return nil, err
	}
	return defaultStream, nil
}

// GetDefaultContext returns the default CUDA context
func GetDefaultContext() *Context {
	ctx, err := DefaultContext()
	if err != nil {
		panic(err)
	}
	return ctx
}

// DefaultContext returns the default CUDA context without panicking on init failure.
func DefaultContext() (*Context, error) {
	if err := Initialize(); err != nil {
		return nil, err
	}
	return defaultContext, nil
}

// Execute implements the Kernel interface for SimpleKernel
func (k *SimpleKernel) Execute(gridDim, blockDim Dim3, sharedMem int, stream *Stream, args ...interface{}) error {
	return k.Func(args...)
}
