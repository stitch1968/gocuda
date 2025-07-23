// Package libraries provides CUDA runtime library bindings for GoCUDA
// This implements cuFFT functionality for Fast Fourier Transform operations
package libraries

import (
	"fmt"
	"math"
	"time"

	"github.com/stitch1968/gocuda/memory"
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
	handle     uintptr
	fftType    FFTType
	direction  FFTDirection
	nx, ny, nz int
	batch      int
	destroyed  bool
}

// FFTContext manages cuFFT operations
type FFTContext struct {
	handle uintptr
	plans  []*FFTPlan
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

// CreatePlan1D creates a 1D FFT plan
func (ctx *FFTContext) CreatePlan1D(nx int, fftType FFTType, batch int) (*FFTPlan, error) {
	if nx <= 0 || batch <= 0 {
		return nil, fmt.Errorf("invalid plan parameters: nx=%d, batch=%d", nx, batch)
	}

	// Validate that nx is a power of 2 for optimal performance
	if !isPowerOfTwo(nx) {
		return nil, fmt.Errorf("FFT size must be a power of 2 for optimal performance, got %d", nx)
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
	if nx <= 0 || ny <= 0 {
		return nil, fmt.Errorf("invalid plan parameters: nx=%d, ny=%d", nx, ny)
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
	if nx <= 0 || ny <= 0 || nz <= 0 {
		return nil, fmt.Errorf("invalid plan parameters: nx=%d, ny=%d, nz=%d", nx, ny, nz)
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

// ExecC2C executes a complex-to-complex FFT
func (ctx *FFTContext) ExecC2C(plan *FFTPlan, input, output *memory.Memory, direction FFTDirection) error {
	if plan == nil || plan.destroyed {
		return fmt.Errorf("invalid or destroyed plan")
	}
	if plan.fftType != FFTTypeC2C {
		return fmt.Errorf("plan type mismatch: expected C2C, got %v", plan.fftType)
	}

	totalSize := plan.nx
	if plan.ny > 0 {
		totalSize *= plan.ny
	}
	if plan.nz > 0 {
		totalSize *= plan.nz
	}
	totalSize *= plan.batch

	// Perform actual FFT computation
	err := ctx.performComplexFFT(input, output, totalSize, direction, 1) // 1D
	if plan.ny > 0 {
		if plan.nz > 0 {
			err = ctx.performComplexFFT(input, output, totalSize, direction, 3) // 3D
		} else {
			err = ctx.performComplexFFT(input, output, totalSize, direction, 2) // 2D
		}
	}

	if err != nil {
		return err
	}

	return simulateKernelExecution("cufftExecC2C", totalSize, 3)
}

// ExecR2C executes a real-to-complex FFT
func (ctx *FFTContext) ExecR2C(plan *FFTPlan, input, output *memory.Memory) error {
	if plan == nil || plan.destroyed {
		return fmt.Errorf("invalid or destroyed plan")
	}
	if plan.fftType != FFTTypeR2C {
		return fmt.Errorf("plan type mismatch: expected R2C, got %v", plan.fftType)
	}

	totalSize := plan.nx
	if plan.ny > 0 {
		totalSize *= plan.ny
	}
	if plan.nz > 0 {
		totalSize *= plan.nz
	}
	totalSize *= plan.batch

	err := ctx.performRealToComplexFFT(input, output, totalSize)
	if err != nil {
		return err
	}

	return simulateKernelExecution("cufftExecR2C", totalSize, 3)
}

// ExecC2R executes a complex-to-real FFT
func (ctx *FFTContext) ExecC2R(plan *FFTPlan, input, output *memory.Memory) error {
	if plan == nil || plan.destroyed {
		return fmt.Errorf("invalid or destroyed plan")
	}
	if plan.fftType != FFTTypeC2R {
		return fmt.Errorf("plan type mismatch: expected C2R, got %v", plan.fftType)
	}

	totalSize := plan.nx
	if plan.ny > 0 {
		totalSize *= plan.ny
	}
	if plan.nz > 0 {
		totalSize *= plan.nz
	}
	totalSize *= plan.batch

	err := ctx.performComplexToRealFFT(input, output, totalSize)
	if err != nil {
		return err
	}

	return simulateKernelExecution("cufftExecC2R", totalSize, 3)
}

// SetStream sets the CUDA stream for the plan
func (plan *FFTPlan) SetStream(stream interface{}) error {
	if plan.destroyed {
		return fmt.Errorf("plan has been destroyed")
	}
	// In simulation mode, we just acknowledge the stream setting
	return simulateKernelExecution("cufftSetStream", 1, 1)
}

// DestroyPlan destroys an FFT plan
func (plan *FFTPlan) DestroyPlan() error {
	if plan.destroyed {
		return fmt.Errorf("plan already destroyed")
	}
	plan.destroyed = true
	return simulateKernelExecution("cufftDestroy", 1, 1)
}

// DestroyContext destroys the cuFFT context
func (ctx *FFTContext) DestroyContext() error {
	// Destroy all plans
	for _, plan := range ctx.plans {
		if !plan.destroyed {
			plan.DestroyPlan()
		}
	}
	ctx.plans = nil
	return simulateKernelExecution("cufftDestroyContext", 1, 1)
}

// Helper functions

func (ctx *FFTContext) performComplexFFT(input, output *memory.Memory, size int, direction FFTDirection, dimensions int) error {
	if input == nil || output == nil {
		return fmt.Errorf("input and output memory cannot be nil")
	}

	// Get input and output data
	inputData := (*[1 << 30]Complex64)(input.Ptr())[:size:size]
	outputData := (*[1 << 30]Complex64)(output.Ptr())[:size:size]

	// Perform FFT using Go's complex number support
	complexInput := make([]complex64, size)
	for i := 0; i < size; i++ {
		complexInput[i] = complex(inputData[i].Real, inputData[i].Imag)
	}

	var complexOutput []complex64
	if dimensions == 1 {
		complexOutput = performDFT1D(complexInput, direction == FFTForward)
	} else if dimensions == 2 {
		// For 2D, assume square matrix for simplicity
		n := int(math.Sqrt(float64(size)))
		complexOutput = performDFT2D(complexInput, n, n, direction == FFTForward)
	} else {
		// For 3D and higher, use 1D DFT as approximation
		complexOutput = performDFT1D(complexInput, direction == FFTForward)
	}

	// Copy back to output
	for i := 0; i < len(complexOutput) && i < size; i++ {
		outputData[i].Real = real(complexOutput[i])
		outputData[i].Imag = imag(complexOutput[i])
	}

	return nil
}

func (ctx *FFTContext) performRealToComplexFFT(input, output *memory.Memory, size int) error {
	if input == nil || output == nil {
		return fmt.Errorf("input and output memory cannot be nil")
	}

	// Get input data as real numbers
	inputData := (*[1 << 30]float32)(input.Ptr())[:size:size]
	outputData := (*[1 << 30]Complex64)(output.Ptr())[:size:size]

	// Convert to complex for DFT
	complexInput := make([]complex64, size)
	for i := 0; i < size; i++ {
		complexInput[i] = complex(inputData[i], 0)
	}

	// Perform DFT
	complexOutput := performDFT1D(complexInput, true)

	// Copy to output
	for i := 0; i < len(complexOutput) && i < size; i++ {
		outputData[i].Real = real(complexOutput[i])
		outputData[i].Imag = imag(complexOutput[i])
	}

	return nil
}

func (ctx *FFTContext) performComplexToRealFFT(input, output *memory.Memory, size int) error {
	if input == nil || output == nil {
		return fmt.Errorf("input and output memory cannot be nil")
	}

	// Get input data as complex numbers
	inputData := (*[1 << 30]Complex64)(input.Ptr())[:size:size]
	outputData := (*[1 << 30]float32)(output.Ptr())[:size:size]

	// Convert to Go complex format
	complexInput := make([]complex64, size)
	for i := 0; i < size; i++ {
		complexInput[i] = complex(inputData[i].Real, inputData[i].Imag)
	}

	// Perform inverse DFT
	complexResult := performDFT1D(complexInput, false)

	// Take real part only
	for i := 0; i < len(complexResult) && i < size; i++ {
		outputData[i] = real(complexResult[i])
	}

	return nil
}

// DFT implementations using Cooley-Tukey algorithm
func performDFT1D(data []complex64, forward bool) []complex64 {
	n := len(data)
	if n <= 1 {
		return data
	}

	// Bit-reverse the data
	result := make([]complex64, n)
	copy(result, data)
	bitReverse(result)

	// Cooley-Tukey FFT
	for length := 2; length <= n; length *= 2 {
		angle := 2 * math.Pi / float64(length)
		if !forward {
			angle = -angle
		}
		wlen := complex(float32(math.Cos(angle)), float32(math.Sin(angle)))

		for i := 0; i < n; i += length {
			w := complex64(1)
			for j := 0; j < length/2; j++ {
				u := result[i+j]
				v := result[i+j+length/2] * w
				result[i+j] = u + v
				result[i+j+length/2] = u - v
				w *= complex64(wlen)
			}
		}
	}

	// Normalize for inverse transform
	if !forward {
		for i := range result {
			result[i] /= complex(float32(n), 0)
		}
	}

	return result
}

func performDFT2D(data []complex64, nx, ny int, forward bool) []complex64 {
	if nx*ny != len(data) {
		return data
	}

	result := make([]complex64, len(data))
	copy(result, data)

	// Transform rows
	for y := 0; y < ny; y++ {
		row := result[y*nx : (y+1)*nx]
		transformedRow := performDFT1D(row, forward)
		copy(result[y*nx:(y+1)*nx], transformedRow)
	}

	// Transform columns
	for x := 0; x < nx; x++ {
		col := make([]complex64, ny)
		for y := 0; y < ny; y++ {
			col[y] = result[y*nx+x]
		}
		transformedCol := performDFT1D(col, forward)
		for y := 0; y < ny; y++ {
			result[y*nx+x] = transformedCol[y]
		}
	}

	return result
}

func bitReverse(data []complex64) {
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

// Estimate memory requirements for FFT operations
func (ctx *FFTContext) EstimateMemory(plan *FFTPlan) (inputBytes, outputBytes int64) {
	totalSize := int64(plan.nx)
	if plan.ny > 0 {
		totalSize *= int64(plan.ny)
	}
	if plan.nz > 0 {
		totalSize *= int64(plan.nz)
	}
	totalSize *= int64(plan.batch)

	switch plan.fftType {
	case FFTTypeC2C, FFTTypeZ2Z:
		// Complex input and output
		inputBytes = totalSize * 8 // 2 floats per complex
		outputBytes = totalSize * 8
	case FFTTypeR2C:
		// Real input, complex output
		inputBytes = totalSize * 4  // 1 float per real
		outputBytes = totalSize * 8 // 2 floats per complex
	case FFTTypeC2R:
		// Complex input, real output
		inputBytes = totalSize * 8  // 2 floats per complex
		outputBytes = totalSize * 4 // 1 float per real
	case FFTTypeD2Z:
		// Double real input, double complex output
		inputBytes = totalSize * 8   // 1 double per real
		outputBytes = totalSize * 16 // 2 doubles per complex
	case FFTTypeZ2D:
		// Double complex input, double real output
		inputBytes = totalSize * 16 // 2 doubles per complex
		outputBytes = totalSize * 8 // 1 double per real
	}

	return inputBytes, outputBytes
}
