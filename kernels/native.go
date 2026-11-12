package kernels

import (
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"strings"
	"sync"
	"unsafe"

	"github.com/stitch1968/gocuda/internal"
	"github.com/stitch1968/gocuda/memory"
	"github.com/stitch1968/gocuda/streams"
)

// NativeModule owns a CUDA driver module loaded from PTX.
// Close synchronizes its direct launches; open graphs retain the module.
type NativeModule struct {
	mu        sync.Mutex
	handle    unsafe.Pointer
	deviceID  int
	graphRefs int
	streams   map[*streams.NativeStream]struct{}
}

// NativeFunction is a function within a NativeModule.
type NativeFunction struct {
	module         *NativeModule
	handle         unsafe.Pointer
	parameterSizes []int
}

// NativeArg represents a scalar or a device buffer kernel argument.
type NativeArg struct {
	bits     uint64
	size     int
	buffer   *memory.Memory
	offset   int64
	isBuffer bool
}

// Int32Arg passes a 32-bit signed integer by value.
func Int32Arg(value int32) NativeArg { return NativeArg{bits: uint64(uint32(value)), size: 4} }

// Uint32Arg passes a 32-bit unsigned integer by value.
func Uint32Arg(value uint32) NativeArg { return NativeArg{bits: uint64(value), size: 4} }

// Float32Arg passes a 32-bit floating-point value.
func Float32Arg(value float32) NativeArg { return Uint32Arg(math.Float32bits(value)) }

// Int64Arg passes a 64-bit signed integer by value.
func Int64Arg(value int64) NativeArg { return NativeArg{bits: uint64(value), size: 8} }

// Uint64Arg passes a 64-bit unsigned integer by value.
func Uint64Arg(value uint64) NativeArg { return NativeArg{bits: value, size: 8} }

// Float64Arg passes a 64-bit floating-point value.
func Float64Arg(value float64) NativeArg { return Uint64Arg(math.Float64bits(value)) }

// BufferArg passes a device or managed allocation with a byte offset.
// The allocation must remain alive until the launch has completed.
func BufferArg(buffer *memory.Memory, offset int64) NativeArg {
	return NativeArg{buffer: buffer, offset: offset, size: 8, isBuffer: true}
}

// LoadPTX loads PTX for deviceID. CUDA must be available; no host fallback is used.
func LoadPTX(deviceID int, ptx string) (*NativeModule, error) {
	if ptx == "" || strings.IndexByte(ptx, 0) >= 0 {
		return nil, fmt.Errorf("PTX must be nonempty and contain no NUL bytes")
	}
	if !internal.ShouldUseCuda() {
		return nil, streams.ErrNativeCUDAUnavailable
	}
	if deviceID < 0 || deviceID >= internal.GetCudaDeviceCount() {
		return nil, fmt.Errorf("device ID %d out of range", deviceID)
	}
	var handle unsafe.Pointer
	err := internal.RunOnDevice(deviceID, func() error {
		var err error
		handle, err = loadNativePTX(ptx)
		return err
	})
	if err != nil {
		if handle != nil {
			cleanupErr := internal.RunOnDevice(deviceID, func() error { return unloadNativeModule(handle) })
			return nil, errors.Join(err, cleanupErr)
		}
		return nil, err
	}
	return &NativeModule{handle: handle, deviceID: deviceID}, nil
}

// Function resolves a named GPU kernel in the module.
func (m *NativeModule) Function(name string) (*NativeFunction, error) {
	if m == nil || name == "" || strings.IndexByte(name, 0) >= 0 {
		return nil, fmt.Errorf("module and valid function name are required")
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.handle == nil {
		return nil, fmt.Errorf("native module is closed")
	}
	var handle unsafe.Pointer
	var parameterSizes []int
	err := internal.RunOnDevice(m.deviceID, func() error {
		var err error
		handle, parameterSizes, err = nativeModuleFunction(m.handle, name)
		return err
	})
	if err != nil {
		return nil, err
	}
	return &NativeFunction{module: m, handle: handle, parameterSizes: parameterSizes}, nil
}

// Launch enqueues a GPU kernel on a native stream. It checks argument count
// and byte sizes, but cannot verify their PTX types or the kernel's memory
// accesses. Asynchronous errors are reported by stream synchronization.
func (f *NativeFunction) Launch(stream *streams.NativeStream, grid, block Dim3, sharedBytes int, args ...NativeArg) error {
	if f == nil || f.module == nil || stream == nil {
		return fmt.Errorf("function, module, and native stream are required")
	}
	if stream.DeviceID() != f.module.deviceID {
		return fmt.Errorf("stream device %d does not match module device %d", stream.DeviceID(), f.module.deviceID)
	}
	if err := f.validateArguments(grid, block, sharedBytes, args); err != nil {
		return err
	}
	var buffers []*memory.Memory
	for _, arg := range args {
		if arg.isBuffer {
			buffers = append(buffers, arg.buffer)
		}
	}
	f.module.mu.Lock()
	defer f.module.mu.Unlock()
	if f.module.handle == nil {
		return fmt.Errorf("native module is closed")
	}
	return stream.EnqueueWithBuffers(buffers, func(handle unsafe.Pointer) error {
		if f.module.streams == nil {
			f.module.streams = make(map[*streams.NativeStream]struct{})
		}
		f.module.streams[stream] = struct{}{}
		values, err := f.prepareArguments(grid, block, sharedBytes, args)
		if err != nil {
			return err
		}
		return launchNativeFunction(f.handle, handle, grid, block, sharedBytes, values)
	})
}

func (f *NativeFunction) validateArguments(grid, block Dim3, sharedBytes int, args []NativeArg) error {
	for _, value := range []int{grid.X, grid.Y, grid.Z, block.X, block.Y, block.Z} {
		if value <= 0 || uint64(value) > math.MaxUint32 {
			return fmt.Errorf("invalid kernel launch dimensions")
		}
	}
	if sharedBytes < 0 || uint64(sharedBytes) > math.MaxUint32 {
		return fmt.Errorf("invalid shared memory size: %d", sharedBytes)
	}
	if len(args) != len(f.parameterSizes) {
		return fmt.Errorf("kernel expects %d arguments, got %d", len(f.parameterSizes), len(args))
	}
	for i, arg := range args {
		if arg.size != 4 && arg.size != 8 {
			return fmt.Errorf("invalid kernel argument %d", i)
		}
		if arg.size != f.parameterSizes[i] {
			return fmt.Errorf("kernel argument %d has size %d, expected %d", i, arg.size, f.parameterSizes[i])
		}
		if arg.isBuffer {
			if arg.buffer == nil || arg.offset < 0 || arg.offset >= arg.buffer.Size() {
				return fmt.Errorf("kernel argument %d has a nil buffer or invalid offset", i)
			}
			switch arg.buffer.GetType() {
			case memory.TypeDevice, memory.TypeDeviceOnly, memory.TypeUnified:
			default:
				return fmt.Errorf("kernel argument %d requires device or managed memory", i)
			}
			if arg.buffer.GetDeviceID() != f.module.deviceID {
				return fmt.Errorf("kernel argument %d belongs to a different device", i)
			}
		}
	}
	return nil
}

func (f *NativeFunction) prepareArguments(grid, block Dim3, sharedBytes int, args []NativeArg) ([][8]byte, error) {
	if err := f.validateArguments(grid, block, sharedBytes, args); err != nil {
		return nil, err
	}
	values := make([][8]byte, len(args))
	for i, arg := range args {
		if arg.isBuffer {
			if arg.buffer.Ptr() == nil {
				return nil, fmt.Errorf("kernel argument %d has been freed", i)
			}
			arg.bits = uint64(uintptr(unsafe.Add(arg.buffer.Ptr(), uintptr(arg.offset))))
		}
		if arg.size == 4 {
			binary.NativeEndian.PutUint32(values[i][:4], uint32(arg.bits))
		} else {
			binary.NativeEndian.PutUint64(values[i][:], arg.bits)
		}
	}
	return values, nil
}

// Close waits for streams that launched its functions and unloads the module.
func (m *NativeModule) Close() error {
	if m == nil {
		return nil
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.handle == nil {
		return nil
	}
	if m.graphRefs != 0 {
		return fmt.Errorf("native module is retained by %d open CUDA graphs", m.graphRefs)
	}
	var syncErrors []error
	for stream := range m.streams {
		if err := stream.Synchronize(); err != nil && !errors.Is(err, streams.ErrNativeStreamClosed) {
			syncErrors = append(syncErrors, err)
		}
	}
	if err := errors.Join(syncErrors...); err != nil {
		return err
	}
	return internal.RunOnDevice(m.deviceID, func() error {
		if err := unloadNativeModule(m.handle); err != nil {
			return err
		}
		m.handle = nil
		m.streams = nil
		return nil
	})
}
