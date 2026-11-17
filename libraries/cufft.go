// Package libraries provides CUDA runtime library bindings for GoCUDA
// This implements cuFFT functionality for Fast Fourier Transform operations
package libraries

import (
	"errors"
	"fmt"
	"math"
	"sync"
	"time"

	cuda "github.com/stitch1968/gocuda"
	"github.com/stitch1968/gocuda/internal"
	"github.com/stitch1968/gocuda/memory"
	"github.com/stitch1968/gocuda/streams"
)

// cuFFT - Fast Fourier Transform Library

// FFTType represents different FFT transform types
type FFTType int

const (
	FFTTypeC2C FFTType = iota // Complex to Complex
	FFTTypeR2C                // Real to Complex
	FFTTypeC2R                // Complex to Real
	FFTTypeD2Z                // Double to Double Complex
	FFTTypeZ2D                // Double Complex to Double
	FFTTypeZ2Z                // Double Complex to Double Complex
)

// FFTDirection represents FFT direction
type FFTDirection int

const (
	FFTForward FFTDirection = -1
	FFTInverse FFTDirection = 1
)

// FFTPlan represents a cuFFT execution plan
type FFTPlan struct {
	handle            uintptr
	fftType           FFTType
	direction         FFTDirection
	nx, ny, nz        int
	batch             int
	destroyed         bool
	native            bool
	deviceID          int
	nativeStream      *streams.NativeStream
	mu                sync.Mutex
	defaultRetentions []*memory.Retention
}

// FFTContext manages cuFFT operations
type FFTContext struct {
	handle      uintptr
	plans       []*FFTPlan
	native      bool
	deviceBound bool
	deviceID    int
	mu          sync.Mutex
	destroyed   bool
}

// Complex64 represents a single-precision complex number
type Complex64 struct {
	Real float32
	Imag float32
}

// Complex128 represents a double-precision complex number
type Complex128 struct {
	Real float64
	Imag float64
}

// CreateFFTContext creates a new cuFFT context
func CreateFFTContext() (*FFTContext, error) {
	if err := ensureCudaReady(); err != nil {
		return nil, err
	}
	if cuda.ShouldUseCuda() {
		return createNativeFFTContext()
	}

	ctx := &FFTContext{
		handle: uintptr(time.Now().UnixNano()), // Simulated handle
		plans:  make([]*FFTPlan, 0),
	}

	err := simulateKernelExecution("cufftCreate", 1, 1)
	if err != nil {
		return nil, err
	}

	return ctx, nil
}

// CreateNativeFFTContextOnDevice creates a native-only cuFFT context bound to
// deviceID. It never falls back to simulation.
func CreateNativeFFTContextOnDevice(deviceID int) (*FFTContext, error) {
	if err := ensureCudaReady(); err != nil {
		return nil, err
	}
	if !cuda.ShouldUseCuda() {
		return nil, streams.ErrNativeCUDAUnavailable
	}
	if deviceID < 0 || deviceID >= cuda.GetCudaDeviceCount() {
		return nil, fmt.Errorf("device ID %d out of range", deviceID)
	}
	ctx, err := createNativeFFTContext()
	if err != nil {
		return nil, err
	}
	ctx.deviceBound = true
	ctx.deviceID = deviceID
	return ctx, nil
}

// Backend reports whether this context uses native CUDA or Go simulation.
func (ctx *FFTContext) Backend() LibraryImplementationMode {
	if ctx == nil {
		return ""
	}
	if ctx.native {
		return ImplementationModeNativeCUDA
	}
	return ImplementationModeValidatedGo
}

func (ctx *FFTContext) createNativePlan(create func() (*FFTPlan, error)) (*FFTPlan, error) {
	if !ctx.deviceBound {
		return create()
	}
	var plan *FFTPlan
	err := internal.RunOnDevice(ctx.deviceID, func() error {
		var createErr error
		plan, createErr = create()
		return createErr
	})
	if err != nil {
		if plan != nil {
			return nil, errors.Join(err, destroyNativeFFTPlan(plan))
		}
		return nil, err
	}
	return plan, nil
}

// CreatePlan1D creates a 1D FFT plan
func (ctx *FFTContext) CreatePlan1D(nx int, fftType FFTType, batch int) (*FFTPlan, error) {
	if ctx == nil {
		return nil, fmt.Errorf("nil FFT context")
	}
	ctx.mu.Lock()
	defer ctx.mu.Unlock()
	if ctx.destroyed {
		return nil, fmt.Errorf("FFT context has been destroyed")
	}
	if nx <= 0 || batch <= 0 {
		return nil, fmt.Errorf("invalid plan parameters: nx=%d, batch=%d", nx, batch)
	}
	if fftType < FFTTypeC2C || fftType > FFTTypeZ2Z {
		return nil, fmt.Errorf("unsupported FFT type %v", fftType)
	}

	if _, err := fftElementCounts(nx, 0, 0, batch); err != nil {
		return nil, err
	}
	if ctx.native {
		plan, err := ctx.createNativePlan(func() (*FFTPlan, error) {
			return createNativeFFTPlan1D(ctx, nx, fftType, batch)
		})
		if err != nil {
			return nil, err
		}
		ctx.plans = append(ctx.plans, plan)
		return plan, nil
	}

	plan := &FFTPlan{
		handle:  uintptr(time.Now().UnixNano()),
		fftType: fftType,
		nx:      nx,
		batch:   batch,
	}

	ctx.plans = append(ctx.plans, plan)

	err := simulateKernelExecution("cufftPlan1d", nx*batch, 2)
	if err != nil {
		return nil, err
	}

	return plan, nil
}

// CreatePlan2D creates a 2D FFT plan
func (ctx *FFTContext) CreatePlan2D(nx, ny int, fftType FFTType) (*FFTPlan, error) {
	if ctx == nil {
		return nil, fmt.Errorf("nil FFT context")
	}
	ctx.mu.Lock()
	defer ctx.mu.Unlock()
	if ctx.destroyed {
		return nil, fmt.Errorf("FFT context has been destroyed")
	}
	if nx <= 0 || ny <= 0 {
		return nil, fmt.Errorf("invalid plan parameters: nx=%d, ny=%d", nx, ny)
	}
	if fftType < FFTTypeC2C || fftType > FFTTypeZ2Z {
		return nil, fmt.Errorf("unsupported FFT type %v", fftType)
	}
	if _, err := fftElementCounts(nx, ny, 0, 1); err != nil {
		return nil, err
	}
	if ctx.native {
		plan, err := ctx.createNativePlan(func() (*FFTPlan, error) {
			return createNativeFFTPlan2D(ctx, nx, ny, fftType)
		})
		if err != nil {
			return nil, err
		}
		ctx.plans = append(ctx.plans, plan)
		return plan, nil
	}

	plan := &FFTPlan{
		handle:  uintptr(time.Now().UnixNano()),
		fftType: fftType,
		nx:      nx,
		ny:      ny,
		batch:   1,
	}

	ctx.plans = append(ctx.plans, plan)

	err := simulateKernelExecution("cufftPlan2d", nx*ny, 3)
	if err != nil {
		return nil, err
	}

	return plan, nil
}

// CreatePlan3D creates a 3D FFT plan
func (ctx *FFTContext) CreatePlan3D(nx, ny, nz int, fftType FFTType) (*FFTPlan, error) {
	if ctx == nil {
		return nil, fmt.Errorf("nil FFT context")
	}
	ctx.mu.Lock()
	defer ctx.mu.Unlock()
	if ctx.destroyed {
		return nil, fmt.Errorf("FFT context has been destroyed")
	}
	if nx <= 0 || ny <= 0 || nz <= 0 {
		return nil, fmt.Errorf("invalid plan parameters: nx=%d, ny=%d, nz=%d", nx, ny, nz)
	}
	if fftType < FFTTypeC2C || fftType > FFTTypeZ2Z {
		return nil, fmt.Errorf("unsupported FFT type %v", fftType)
	}
	if _, err := fftElementCounts(nx, ny, nz, 1); err != nil {
		return nil, err
	}
	if ctx.native {
		plan, err := ctx.createNativePlan(func() (*FFTPlan, error) {
			return createNativeFFTPlan3D(ctx, nx, ny, nz, fftType)
		})
		if err != nil {
			return nil, err
		}
		ctx.plans = append(ctx.plans, plan)
		return plan, nil
	}

	plan := &FFTPlan{
		handle:  uintptr(time.Now().UnixNano()),
		fftType: fftType,
		nx:      nx,
		ny:      ny,
		nz:      nz,
		batch:   1,
	}

	ctx.plans = append(ctx.plans, plan)

	err := simulateKernelExecution("cufftPlan3d", nx*ny*nz, 4)
	if err != nil {
		return nil, err
	}

	return plan, nil
}

// CreatePlanMany creates a contiguous batched FFT plan with one to three
// dimensions. For real transforms, the last dimension uses cuFFT's packed
// Hermitian layout. Advanced strides and embedded layouts are not supported.
func (ctx *FFTContext) CreatePlanMany(dimensions []int, fftType FFTType, batch int) (*FFTPlan, error) {
	if ctx == nil {
		return nil, fmt.Errorf("nil FFT context")
	}
	ctx.mu.Lock()
	defer ctx.mu.Unlock()
	if ctx.destroyed {
		return nil, fmt.Errorf("FFT context has been destroyed")
	}
	if len(dimensions) < 1 || len(dimensions) > 3 || batch <= 0 {
		return nil, fmt.Errorf("FFT plans require one to three dimensions and a positive batch count")
	}
	if fftType < FFTTypeC2C || fftType > FFTTypeZ2Z {
		return nil, fmt.Errorf("unsupported FFT type %v", fftType)
	}
	for _, dimension := range dimensions {
		if dimension <= 0 {
			return nil, fmt.Errorf("FFT dimensions must be positive")
		}
	}
	nx, ny, nz := dimensions[0], 0, 0
	if len(dimensions) > 1 {
		ny = dimensions[1]
	}
	if len(dimensions) > 2 {
		nz = dimensions[2]
	}
	total, err := fftElementCounts(nx, ny, nz, batch)
	if err != nil {
		return nil, err
	}
	if ctx.native {
		plan, err := ctx.createNativePlan(func() (*FFTPlan, error) {
			return createNativeFFTPlanMany(dimensions, fftType, batch)
		})
		if err != nil {
			return nil, err
		}
		ctx.plans = append(ctx.plans, plan)
		return plan, nil
	}
	plan := &FFTPlan{handle: uintptr(time.Now().UnixNano()), fftType: fftType, nx: nx, ny: ny, nz: nz, batch: batch}
	if err := simulateKernelExecution("cufftPlanMany", total, len(dimensions)+1); err != nil {
		return nil, err
	}
	ctx.plans = append(ctx.plans, plan)
	return plan, nil
}

// ExecC2C executes a complex-to-complex FFT. Like cuFFT, inverse transforms
// are not normalized. Native execution is synchronous on the default stream
// and asynchronous when a NativeStream is attached.
func (ctx *FFTContext) ExecC2C(plan *FFTPlan, input, output *memory.Memory, direction FFTDirection) error {
	return ctx.execFFT(plan, input, output, direction, FFTTypeC2C)
}

// ExecR2C executes an out-of-place real-to-complex FFT using cuFFT's packed
// Hermitian layout.
func (ctx *FFTContext) ExecR2C(plan *FFTPlan, input, output *memory.Memory) error {
	return ctx.execFFT(plan, input, output, FFTForward, FFTTypeR2C)
}

// ExecC2R executes an out-of-place, unnormalized complex-to-real FFT.
func (ctx *FFTContext) ExecC2R(plan *FFTPlan, input, output *memory.Memory) error {
	return ctx.execFFT(plan, input, output, FFTInverse, FFTTypeC2R)
}

// ExecZ2Z executes a double-precision complex FFT. Inverse transforms are
// unnormalized, as in cuFFT. In-place execution is supported.
func (ctx *FFTContext) ExecZ2Z(plan *FFTPlan, input, output *memory.Memory, direction FFTDirection) error {
	return ctx.execFFT(plan, input, output, direction, FFTTypeZ2Z)
}

// ExecD2Z executes an out-of-place double-precision real-to-complex FFT using
// cuFFT's packed Hermitian layout.
func (ctx *FFTContext) ExecD2Z(plan *FFTPlan, input, output *memory.Memory) error {
	return ctx.execFFT(plan, input, output, FFTForward, FFTTypeD2Z)
}

// ExecZ2D executes an out-of-place, unnormalized double-precision inverse FFT.
func (ctx *FFTContext) ExecZ2D(plan *FFTPlan, input, output *memory.Memory) error {
	return ctx.execFFT(plan, input, output, FFTInverse, FFTTypeZ2D)
}

func (ctx *FFTContext) execFFT(plan *FFTPlan, input, output *memory.Memory, direction FFTDirection, kind FFTType) error {
	if plan == nil {
		return fmt.Errorf("invalid or destroyed plan")
	}
	plan.mu.Lock()
	defer plan.mu.Unlock()
	if plan.destroyed {
		return fmt.Errorf("invalid or destroyed plan")
	}
	if plan.fftType != kind {
		return fmt.Errorf("plan type mismatch: expected %v, got %v", kind, plan.fftType)
	}
	if direction != FFTForward && direction != FFTInverse {
		return fmt.Errorf("invalid FFT direction: %d", direction)
	}
	if err := ctx.validateFFTBuffers(plan, input, output); err != nil {
		return err
	}
	if kind != FFTTypeC2C && kind != FFTTypeZ2Z && input.Ptr() == output.Ptr() {
		return fmt.Errorf("in-place real FFTs are not supported")
	}
	if plan.native {
		return execNativeFFT(plan, input, output, direction)
	}

	var err error
	var operation string
	switch kind {
	case FFTTypeC2C:
		operation = "cufftExecC2C"
		err = performComplexFFT(plan, input, output, direction)
	case FFTTypeR2C:
		operation = "cufftExecR2C"
		err = performRealToComplexFFT(plan, input, output)
	case FFTTypeC2R:
		operation = "cufftExecC2R"
		err = performComplexToRealFFT(plan, input, output)
	case FFTTypeZ2Z:
		operation = "cufftExecZ2Z"
		err = performComplexFFTDouble(plan, input, output, direction)
	case FFTTypeD2Z:
		operation = "cufftExecD2Z"
		err = performRealToComplexFFTDouble(plan, input, output)
	case FFTTypeZ2D:
		operation = "cufftExecZ2D"
		err = performComplexToRealFFTDouble(plan, input, output)
	default:
		return fmt.Errorf("unsupported FFT type %v", kind)
	}
	if err != nil {
		return err
	}
	return simulateKernelExecution(operation, plan.elementCount(), 3)
}

// SetStream sets the CUDA stream for the plan, synchronizing its prior native
// stream before changing it.
func (plan *FFTPlan) SetStream(stream any) error {
	if plan == nil {
		return fmt.Errorf("plan has been destroyed")
	}
	plan.mu.Lock()
	defer plan.mu.Unlock()
	if plan.destroyed {
		return fmt.Errorf("plan has been destroyed")
	}
	if plan.native {
		if err := plan.synchronizeDefault(); err != nil {
			return err
		}
		if plan.nativeStream != nil {
			if err := plan.nativeStream.Synchronize(); err != nil && !errors.Is(err, streams.ErrNativeStreamClosed) {
				return err
			}
		}
		return setNativeFFTPlanStream(plan, stream)
	}
	if stream != nil {
		return fmt.Errorf("FFT stream attachment is not supported in simulation mode")
	}
	return simulateKernelExecution("cufftSetStream", 1, 1)
}

// DestroyPlan destroys an FFT plan
func (plan *FFTPlan) DestroyPlan() error {
	if plan == nil {
		return fmt.Errorf("plan already destroyed")
	}
	plan.mu.Lock()
	defer plan.mu.Unlock()
	return plan.destroyLocked()
}

func (plan *FFTPlan) synchronizeDefault() error {
	if len(plan.defaultRetentions) == 0 {
		return nil
	}
	if err := internal.CudaDefaultStreamSynchronizeOnDevice(plan.deviceID); err != nil {
		return err
	}
	for _, retention := range plan.defaultRetentions {
		retention.Release()
	}
	plan.defaultRetentions = nil
	return nil
}

func (plan *FFTPlan) destroyLocked() error {
	if plan.destroyed {
		return fmt.Errorf("plan already destroyed")
	}
	if plan.native {
		if err := plan.synchronizeDefault(); err != nil {
			return err
		}
		if plan.nativeStream != nil {
			if err := plan.nativeStream.Synchronize(); err != nil && !errors.Is(err, streams.ErrNativeStreamClosed) {
				return err
			}
		}
		if err := destroyNativeFFTPlan(plan); err != nil {
			return err
		}
	} else if err := simulateKernelExecution("cufftDestroy", 1, 1); err != nil {
		return err
	}
	plan.destroyed = true
	return nil
}

// DestroyContext destroys the cuFFT context
func (ctx *FFTContext) DestroyContext() error {
	if ctx == nil {
		return fmt.Errorf("nil FFT context")
	}
	ctx.mu.Lock()
	defer ctx.mu.Unlock()
	if ctx.destroyed {
		return fmt.Errorf("FFT context already destroyed")
	}
	var errs []error
	for _, plan := range ctx.plans {
		plan.mu.Lock()
		if !plan.destroyed {
			if err := plan.destroyLocked(); err != nil {
				errs = append(errs, err)
			}
		}
		plan.mu.Unlock()
	}
	if err := errors.Join(errs...); err != nil {
		return err
	}
	ctx.plans = nil
	if ctx.native {
		if err := destroyNativeFFTContext(ctx); err != nil {
			return err
		}
	} else if err := simulateKernelExecution("cufftDestroyContext", 1, 1); err != nil {
		return err
	}
	ctx.destroyed = true
	return nil
}

// Helper functions

func fftElementCounts(nx, ny, nz, batch int) (int, error) {
	total := 1
	for _, dim := range []int{nx, max(ny, 1), max(nz, 1), batch} {
		if dim <= 0 || int64(dim) > 1<<31-1 || total > int(^uint(0)>>1)/dim/16 {
			return 0, fmt.Errorf("FFT dimensions exceed supported size")
		}
		total *= dim
	}
	return total, nil
}

func (plan *FFTPlan) dimensions() []int {
	dims := []int{plan.nx}
	if plan.ny > 0 {
		dims = append(dims, plan.ny)
	}
	if plan.nz > 0 {
		dims = append(dims, plan.nz)
	}
	return dims
}

func (plan *FFTPlan) elementCount() int {
	total, _ := fftElementCounts(plan.nx, plan.ny, plan.nz, plan.batch)
	return total
}

func (plan *FFTPlan) packedCount() int {
	dims := plan.dimensions()
	last := dims[len(dims)-1]
	return plan.elementCount() / last * (last/2 + 1)
}

func (ctx *FFTContext) validateFFTBuffers(plan *FFTPlan, input, output *memory.Memory) error {
	if input == nil || output == nil || input.Ptr() == nil || output.Ptr() == nil {
		return fmt.Errorf("input and output must be live memory allocations")
	}
	if plan.native && (input.Data() != nil || output.Data() != nil) {
		return fmt.Errorf("native FFT requires CUDA device or managed memory")
	}
	if plan.native && (input.GetDeviceID() != plan.deviceID || output.GetDeviceID() != plan.deviceID) {
		return fmt.Errorf("native FFT buffers must belong to plan device %d", plan.deviceID)
	}
	inputBytes, outputBytes := ctx.EstimateMemory(plan)
	if input.Size() < inputBytes || output.Size() < outputBytes {
		return fmt.Errorf("FFT buffers too small: need %d input bytes and %d output bytes", inputBytes, outputBytes)
	}
	return nil
}

func performComplexFFT(plan *FFTPlan, input, output *memory.Memory, direction FFTDirection) error {
	size := plan.elementCount()
	inputData, err := readMathTypedMemory[Complex64](input, size)
	if err != nil {
		return err
	}
	outputData := make([]Complex64, size)
	data := make([]complex64, size)
	for i, value := range inputData {
		data[i] = complex(value.Real, value.Imag)
	}
	transformFFT(data, plan.dimensions(), plan.batch, direction == FFTForward)
	for i, value := range data {
		outputData[i] = Complex64{Real: real(value), Imag: imag(value)}
	}
	return writeMathTypedMemory(output, outputData)
}

func performRealToComplexFFT(plan *FFTPlan, input, output *memory.Memory) error {
	size := plan.elementCount()
	inputData, err := readMathTypedMemory[float32](input, size)
	if err != nil {
		return err
	}
	data := make([]complex64, size)
	for i, value := range inputData {
		data[i] = complex(value, 0)
	}
	dims := plan.dimensions()
	transformFFT(data, dims, plan.batch, true)
	packed := packFFTHermitian(data, dims)
	outputData := make([]Complex64, len(packed))
	for i, value := range packed {
		outputData[i] = Complex64{Real: real(value), Imag: imag(value)}
	}
	return writeMathTypedMemory(output, outputData)
}

func performComplexToRealFFT(plan *FFTPlan, input, output *memory.Memory) error {
	size := plan.elementCount()
	inputData, err := readMathTypedMemory[Complex64](input, plan.packedCount())
	if err != nil {
		return err
	}
	dims := plan.dimensions()
	packed := make([]complex64, len(inputData))
	for i, value := range inputData {
		packed[i] = complex(value.Real, value.Imag)
	}
	data := unpackFFTHermitian(packed, dims, plan.batch, size)
	transformFFT(data, dims, plan.batch, false)
	outputData := make([]float32, size)
	for i, value := range data {
		outputData[i] = real(value)
	}
	return writeMathTypedMemory(output, outputData)
}

func performComplexFFTDouble(plan *FFTPlan, input, output *memory.Memory, direction FFTDirection) error {
	inputData, err := readMathTypedMemory[Complex128](input, plan.elementCount())
	if err != nil {
		return err
	}
	data := make([]complex128, len(inputData))
	for i, value := range inputData {
		data[i] = complex(value.Real, value.Imag)
	}
	transformFFT(data, plan.dimensions(), plan.batch, direction == FFTForward)
	for i, value := range data {
		inputData[i] = Complex128{Real: real(value), Imag: imag(value)}
	}
	return writeMathTypedMemory(output, inputData)
}

func performRealToComplexFFTDouble(plan *FFTPlan, input, output *memory.Memory) error {
	inputData, err := readMathTypedMemory[float64](input, plan.elementCount())
	if err != nil {
		return err
	}
	data := make([]complex128, len(inputData))
	for i, value := range inputData {
		data[i] = complex(value, 0)
	}
	dims := plan.dimensions()
	transformFFT(data, dims, plan.batch, true)
	packed := packFFTHermitian(data, dims)
	outputData := make([]Complex128, len(packed))
	for i, value := range packed {
		outputData[i] = Complex128{Real: real(value), Imag: imag(value)}
	}
	return writeMathTypedMemory(output, outputData)
}

func performComplexToRealFFTDouble(plan *FFTPlan, input, output *memory.Memory) error {
	inputData, err := readMathTypedMemory[Complex128](input, plan.packedCount())
	if err != nil {
		return err
	}
	packed := make([]complex128, len(inputData))
	for i, value := range inputData {
		packed[i] = complex(value.Real, value.Imag)
	}
	dims := plan.dimensions()
	data := unpackFFTHermitian(packed, dims, plan.batch, plan.elementCount())
	transformFFT(data, dims, plan.batch, false)
	outputData := make([]float64, len(data))
	for i, value := range data {
		outputData[i] = real(value)
	}
	return writeMathTypedMemory(output, outputData)
}

func packFFTHermitian[T complex64 | complex128](data []T, dims []int) []T {
	last := dims[len(dims)-1]
	packedLast := last/2 + 1
	packed := make([]T, len(data)/last*packedLast)
	for row := range len(data) / last {
		copy(packed[row*packedLast:(row+1)*packedLast], data[row*last:row*last+packedLast])
	}
	return packed
}

func unpackFFTHermitian[T complex64 | complex128](packed []T, dims []int, batch, size int) []T {
	last := dims[len(dims)-1]
	packedLast := last/2 + 1
	perBatch := size / batch
	rows := perBatch / last
	data := make([]T, size)
	for b := range batch {
		for row := range rows {
			for x := range last {
				sourceRow, sourceX := row, x
				conjugate := x >= packedLast
				if conjugate {
					sourceX = last - x
					sourceRow = 0
					remaining, stride := row, 1
					for axis := len(dims) - 2; axis >= 0; axis-- {
						coordinate := remaining % dims[axis]
						remaining /= dims[axis]
						sourceRow += ((dims[axis] - coordinate) % dims[axis]) * stride
						stride *= dims[axis]
					}
				}
				value := packed[(b*rows+sourceRow)*packedLast+sourceX]
				if conjugate {
					z := complex128(value)
					value = T(complex(real(z), -imag(z)))
				}
				data[b*perBatch+row*last+x] = value
			}
		}
	}
	return data
}

func transformFFT[T complex64 | complex128](data []T, dims []int, batch int, forward bool) {
	perBatch := len(data) / batch
	for b := range batch {
		stride := 1
		for axis := len(dims) - 1; axis >= 0; axis-- {
			n := dims[axis]
			for base := 0; base < perBatch; base += n * stride {
				for offset := range stride {
					line := make([]T, n)
					for i := range n {
						line[i] = data[b*perBatch+base+offset+i*stride]
					}
					line = performDFT1D(line, forward)
					for i, value := range line {
						data[b*perBatch+base+offset+i*stride] = value
					}
				}
			}
			stride *= n
		}
	}
}

// DFT implementations using Cooley-Tukey algorithm
func performDFT1D[T complex64 | complex128](data []T, forward bool) []T {
	n := len(data)
	if n <= 1 {
		return data
	}
	if !isPowerOfTwo(n) {
		result := make([]T, n)
		sign := -1.0
		if !forward {
			sign = 1
		}
		for k := range n {
			for j, value := range data {
				angle := sign * 2 * math.Pi * float64(k) * float64(j) / float64(n)
				result[k] += value * T(complex(math.Cos(angle), math.Sin(angle)))
			}
		}
		return result
	}

	// Bit-reverse the data
	result := make([]T, n)
	copy(result, data)
	bitReverse(result)

	// Cooley-Tukey FFT
	for length := 2; length <= n; length *= 2 {
		angle := -2 * math.Pi / float64(length)
		if !forward {
			angle = -angle
		}
		wlen := T(complex(math.Cos(angle), math.Sin(angle)))

		for i := 0; i < n; i += length {
			w := T(1)
			for j := 0; j < length/2; j++ {
				u := result[i+j]
				v := result[i+j+length/2] * w
				result[i+j] = u + v
				result[i+j+length/2] = u - v
				w *= wlen
			}
		}
	}

	return result
}

func bitReverse[T complex64 | complex128](data []T) {
	n := len(data)
	j := 0
	for i := 1; i < n; i++ {
		bit := n >> 1
		for j&bit != 0 {
			j ^= bit
			bit >>= 1
		}
		j ^= bit
		if i < j {
			data[i], data[j] = data[j], data[i]
		}
	}
}

func isPowerOfTwo(n int) bool {
	return n > 0 && (n&(n-1)) == 0
}

// GetFFTSize returns the optimal FFT size for a given input size
func GetFFTSize(size int) int {
	// Find the next power of 2
	if isPowerOfTwo(size) {
		return size
	}

	n := 1
	for n < size {
		n *= 2
	}
	return n
}

// EstimateMemory estimates memory requirements for FFT operations
func (ctx *FFTContext) EstimateMemory(plan *FFTPlan) (inputBytes, outputBytes int64) {
	totalSize := int64(plan.elementCount())
	packedSize := int64(plan.packedCount())

	switch plan.fftType {
	case FFTTypeC2C:
		inputBytes = totalSize * 8
		outputBytes = totalSize * 8
	case FFTTypeZ2Z:
		inputBytes = totalSize * 16
		outputBytes = totalSize * 16
	case FFTTypeR2C:
		// Real input, complex output
		inputBytes = totalSize * 4 // 1 float per real
		outputBytes = packedSize * 8
	case FFTTypeC2R:
		// Complex input, real output
		inputBytes = packedSize * 8
		outputBytes = totalSize * 4 // 1 float per real
	case FFTTypeD2Z:
		// Double real input, double complex output
		inputBytes = totalSize * 8 // 1 double per real
		outputBytes = packedSize * 16
	case FFTTypeZ2D:
		// Double complex input, double real output
		inputBytes = packedSize * 16
		outputBytes = totalSize * 8 // 1 double per real
	}

	return inputBytes, outputBytes
}
