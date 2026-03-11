// Package kernels provides built-in CUDA kernels and operations for GoCUDA.
// This package contains optimized implementations of common GPU operations
// like vector arithmetic, matrix operations, and custom kernel interfaces.
package kernels

import (
	"fmt"

	"github.com/stitch1968/gocuda/internal"
	"github.com/stitch1968/gocuda/memory"
)

// Dim3 represents 3D dimensions for CUDA kernels
type Dim3 struct {
	X, Y, Z int
}

// Kernel interface for CUDA kernels
type Kernel interface {
	Execute(gridDim, blockDim Dim3, sharedMem int, stream *internal.Stream, args ...any) error
}

// VectorAdd performs vector addition
type VectorAdd struct{}

func (k *VectorAdd) Execute(gridDim, blockDim Dim3, sharedMem int, stream *internal.Stream, args ...any) error {
	if len(args) != 4 {
		return fmt.Errorf("vector add requires 4 arguments: a, b, c, n")
	}

	a, ok1 := args[0].(*memory.Memory)
	b, ok2 := args[1].(*memory.Memory)
	c, ok3 := args[2].(*memory.Memory)
	n, ok4 := args[3].(int)

	if !ok1 || !ok2 || !ok3 || !ok4 {
		return fmt.Errorf("invalid argument types for vector add")
	}

	aData, err := memory.View[float32](a, n)
	if err != nil {
		return err
	}
	bData, err := memory.View[float32](b, n)
	if err != nil {
		return err
	}
	cData, err := memory.View[float32](c, n)
	if err != nil {
		return err
	}

	for i := range n {
		cData[i] = aData[i] + bData[i]
	}

	return nil
}

// MatrixMultiply performs matrix multiplication
type MatrixMultiply struct{}

func (mk *MatrixMultiply) Execute(gridDim, blockDim Dim3, sharedMem int, stream *internal.Stream, args ...any) error {
	if len(args) != 6 {
		return fmt.Errorf("matrix multiply requires 6 arguments: a, b, c, m, n, k")
	}

	a, ok1 := args[0].(*memory.Memory)
	b, ok2 := args[1].(*memory.Memory)
	c, ok3 := args[2].(*memory.Memory)
	m, ok4 := args[3].(int)
	n, ok5 := args[4].(int)
	k, ok6 := args[5].(int)

	if !ok1 || !ok2 || !ok3 || !ok4 || !ok5 || !ok6 {
		return fmt.Errorf("invalid argument types for matrix multiply")
	}

	aData, err := memory.View[float32](a, m*k)
	if err != nil {
		return err
	}
	bData, err := memory.View[float32](b, k*n)
	if err != nil {
		return err
	}
	cData, err := memory.View[float32](c, m*n)
	if err != nil {
		return err
	}

	for i := range m {
		for j := range n {
			sum := float32(0)
			for l := range k {
				sum += aData[i*k+l] * bData[l*n+j]
			}
			cData[i*n+j] = sum
		}
	}

	return nil
}

// Convolution2D performs 2D convolution
type Convolution2D struct{}

func (conv *Convolution2D) Execute(gridDim, blockDim Dim3, sharedMem int, stream *internal.Stream, args ...any) error {
	if len(args) != 9 {
		return fmt.Errorf("convolution2D requires 9 arguments: input, filter, output, inputWidth, inputHeight, filterWidth, filterHeight, outputWidth, outputHeight")
	}

	input, ok1 := args[0].(*memory.Memory)
	filter, ok2 := args[1].(*memory.Memory)
	output, ok3 := args[2].(*memory.Memory)
	inputWidth, ok4 := args[3].(int)
	inputHeight, ok5 := args[4].(int)
	filterWidth, ok6 := args[5].(int)
	filterHeight, ok7 := args[6].(int)
	outputWidth, ok8 := args[7].(int)
	outputHeight, ok9 := args[8].(int)

	if !ok1 || !ok2 || !ok3 || !ok4 || !ok5 || !ok6 || !ok7 || !ok8 || !ok9 {
		return fmt.Errorf("invalid argument types for convolution2D")
	}

	inputData, err := memory.View[float32](input, inputWidth*inputHeight)
	if err != nil {
		return err
	}
	filterData, err := memory.View[float32](filter, filterWidth*filterHeight)
	if err != nil {
		return err
	}
	outputData, err := memory.View[float32](output, outputWidth*outputHeight)
	if err != nil {
		return err
	}

	// Perform 2D convolution
	for y := range outputHeight {
		for x := range outputWidth {
			sum := float32(0)
			for fy := range filterHeight {
				for fx := range filterWidth {
					inputY := y + fy
					inputX := x + fx
					if inputY < inputHeight && inputX < inputWidth {
						sum += inputData[inputY*inputWidth+inputX] * filterData[fy*filterWidth+fx]
					}
				}
			}
			outputData[y*outputWidth+x] = sum
		}
	}

	return nil
}

// SAXPY performs single-precision A*X plus Y operation
type SAXPY struct{}

func (saxpy *SAXPY) Execute(gridDim, blockDim Dim3, sharedMem int, stream *internal.Stream, args ...any) error {
	if len(args) != 4 {
		return fmt.Errorf("SAXPY requires 4 arguments: a, x, y, n")
	}

	a, ok1 := args[0].(float32)
	x, ok2 := args[1].(*memory.Memory)
	y, ok3 := args[2].(*memory.Memory)
	n, ok4 := args[3].(int)

	if !ok1 || !ok2 || !ok3 || !ok4 {
		return fmt.Errorf("invalid argument types for SAXPY")
	}

	xData, err := memory.View[float32](x, n)
	if err != nil {
		return err
	}
	yData, err := memory.View[float32](y, n)
	if err != nil {
		return err
	}

	for i := range n {
		yData[i] = a*xData[i] + yData[i]
	}

	return nil
}

// Reduction performs parallel reduction (sum)
type Reduction struct{}

func (red *Reduction) Execute(gridDim, blockDim Dim3, sharedMem int, stream *internal.Stream, args ...any) error {
	if len(args) != 3 {
		return fmt.Errorf("reduction requires 3 arguments: input, output, n")
	}

	input, ok1 := args[0].(*memory.Memory)
	output, ok2 := args[1].(*memory.Memory)
	n, ok3 := args[2].(int)

	if !ok1 || !ok2 || !ok3 {
		return fmt.Errorf("invalid argument types for reduction")
	}

	inputData, err := memory.View[float32](input, n)
	if err != nil {
		return err
	}
	outputData, err := memory.View[float32](output, 1)
	if err != nil {
		return err
	}

	sum := float32(0)
	for i := range n {
		sum += inputData[i]
	}
	outputData[0] = sum

	return nil
}

// Helper functions for common operations

// ExecuteVectorAdd performs vector addition: c = a + b
func ExecuteVectorAdd(a, b, c *memory.Memory, n int) error {
	kernel := &VectorAdd{}
	gridDim := Dim3{X: (n + 255) / 256, Y: 1, Z: 1}
	blockDim := Dim3{X: 256, Y: 1, Z: 1}

	stream := internal.GetDefaultStream()
	return kernel.Execute(gridDim, blockDim, 0, stream, a, b, c, n)
}

// ExecuteVectorAddAsync performs asynchronous vector addition
func ExecuteVectorAddAsync(stream *internal.Stream, a, b, c *memory.Memory, n int) error {
	kernel := &VectorAdd{}
	gridDim := Dim3{X: (n + 255) / 256, Y: 1, Z: 1}
	blockDim := Dim3{X: 256, Y: 1, Z: 1}

	stream.Execute(func() {
		kernel.Execute(gridDim, blockDim, 0, stream, a, b, c, n)
	})
	return nil
}

// ExecuteMatrixMultiply performs matrix multiplication: c = a * b
func ExecuteMatrixMultiply(a, b, c *memory.Memory, m, n, k int) error {
	kernel := &MatrixMultiply{}
	gridDim := Dim3{X: (n + 15) / 16, Y: (m + 15) / 16, Z: 1}
	blockDim := Dim3{X: 16, Y: 16, Z: 1}

	stream := internal.GetDefaultStream()
	return kernel.Execute(gridDim, blockDim, 0, stream, a, b, c, m, n, k)
}

// ExecuteMatrixMultiplyAsync performs asynchronous matrix multiplication
func ExecuteMatrixMultiplyAsync(stream *internal.Stream, a, b, c *memory.Memory, m, n, k int) error {
	kernel := &MatrixMultiply{}
	gridDim := Dim3{X: (n + 15) / 16, Y: (m + 15) / 16, Z: 1}
	blockDim := Dim3{X: 16, Y: 16, Z: 1}

	stream.Execute(func() {
		kernel.Execute(gridDim, blockDim, 0, stream, a, b, c, m, n, k)
	})
	return nil
}

// CreateVector creates a vector memory allocation with optional initial values
func CreateVector(size int, values []float32) (*memory.Memory, error) {
	mem, err := memory.Alloc(int64(size * 4)) // 4 bytes per float32
	if err != nil {
		return nil, err
	}

	if values != nil {
		if len(values) != size {
			mem.Free()
			return nil, fmt.Errorf("values length (%d) doesn't match size (%d)", len(values), size)
		}

		// Copy values to memory
		memData, viewErr := memory.View[float32](mem, size)
		if viewErr != nil {
			_ = mem.Free()
			return nil, viewErr
		}
		copy(memData, values)
	}

	return mem, nil
}

// FillVector fills a vector with a constant value
func FillVector(mem *memory.Memory, size int, value float32) error {
	data, err := memory.View[float32](mem, size)
	if err != nil {
		return err
	}
	for i := range size {
		data[i] = value
	}
	return nil
}

// CreateMatrix creates a matrix memory allocation
func CreateMatrix(rows, cols int, values []float32) (*memory.Memory, error) {
	size := rows * cols
	mem, err := memory.Alloc(int64(size * 4)) // 4 bytes per float32
	if err != nil {
		return nil, err
	}

	if values != nil {
		if len(values) != size {
			mem.Free()
			return nil, fmt.Errorf("values length (%d) doesn't match matrix size (%d)", len(values), size)
		}

		// Copy values to memory
		memData, viewErr := memory.View[float32](mem, size)
		if viewErr != nil {
			_ = mem.Free()
			return nil, viewErr
		}
		copy(memData, values)
	}

	return mem, nil
}

// GetVectorValue gets a value from a vector at the specified index
func GetVectorValue(mem *memory.Memory, index int) float32 {
	data, err := memory.View[float32](mem, index+1)
	if err != nil {
		return 0
	}
	return data[index]
}

// SetVectorValue sets a value in a vector at the specified index
func SetVectorValue(mem *memory.Memory, index int, value float32) {
	data, err := memory.View[float32](mem, index+1)
	if err != nil {
		return
	}
	data[index] = value
}

// Custom kernel support

// CustomKernel allows users to define custom GPU kernels
type CustomKernel struct {
	Name     string
	Function func(gridDim, blockDim Dim3, sharedMem int, stream *internal.Stream, args ...any) error
}

func (ck *CustomKernel) Execute(gridDim, blockDim Dim3, sharedMem int, stream *internal.Stream, args ...any) error {
	return ck.Function(gridDim, blockDim, sharedMem, stream, args...)
}

// NewCustomKernel creates a new custom kernel
func NewCustomKernel(name string, function func(gridDim, blockDim Dim3, sharedMem int, stream *internal.Stream, args ...any) error) *CustomKernel {
	return &CustomKernel{
		Name:     name,
		Function: function,
	}
}

// Math kernels

// ElementwiseAdd performs element-wise addition
func ElementwiseAdd(a, b, c *memory.Memory, n int) error {
	return ExecuteVectorAdd(a, b, c, n)
}

// ElementwiseMultiply performs element-wise multiplication
func ElementwiseMultiply(a, b, c *memory.Memory, n int) error {
	kernel := NewCustomKernel("ElementwiseMultiply", func(gridDim, blockDim Dim3, sharedMem int, stream *internal.Stream, args ...any) error {
		aData, err := memory.View[float32](a, n)
		if err != nil {
			return err
		}
		bData, err := memory.View[float32](b, n)
		if err != nil {
			return err
		}
		cData, err := memory.View[float32](c, n)
		if err != nil {
			return err
		}

		for i := range n {
			cData[i] = aData[i] * bData[i]
		}
		return nil
	})

	stream := internal.GetDefaultStream()
	return kernel.Execute(Dim3{X: 1, Y: 1, Z: 1}, Dim3{X: 1, Y: 1, Z: 1}, 0, stream)
}

// ScalarMultiply performs scalar multiplication: c = a * scalar
func ScalarMultiply(a, c *memory.Memory, scalar float32, n int) error {
	kernel := NewCustomKernel("ScalarMultiply", func(gridDim, blockDim Dim3, sharedMem int, stream *internal.Stream, args ...any) error {
		aData, err := memory.View[float32](a, n)
		if err != nil {
			return err
		}
		cData, err := memory.View[float32](c, n)
		if err != nil {
			return err
		}

		for i := range n {
			cData[i] = aData[i] * scalar
		}
		return nil
	})

	stream := internal.GetDefaultStream()
	return kernel.Execute(Dim3{X: 1, Y: 1, Z: 1}, Dim3{X: 1, Y: 1, Z: 1}, 0, stream)
}
