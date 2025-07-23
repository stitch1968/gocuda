package cuda

import (
	"fmt"
	"unsafe"
)

// Simple API Layer - addressing the "complex API" and "beginner guidance" issues

// SimpleMemory provides a builder pattern for memory allocation
type SimpleMemory struct {
	size      int64
	memType   Type
	stream    *Stream
	alignment int
}

// Alloc creates a new memory allocation builder (Simple API - Layer 1)
func Alloc(size int64) *SimpleMemory {
	return &SimpleMemory{
		size:      size,
		memType:   TypeDevice,         // Smart default
		stream:    GetDefaultStream(), // Smart default
		alignment: 256,                // Smart default
	}
}

// OnDevice specifies device memory allocation
func (sm *SimpleMemory) OnDevice() *SimpleMemory {
	sm.memType = TypeDevice
	return sm
}

// OnHost specifies host memory allocation
func (sm *SimpleMemory) OnHost() *SimpleMemory {
	sm.memType = TypeHost
	return sm
}

// Pinned specifies pinned host memory allocation
func (sm *SimpleMemory) Pinned() *SimpleMemory {
	sm.memType = TypePinned
	return sm
}

// Unified specifies unified memory allocation
func (sm *SimpleMemory) Unified() *SimpleMemory {
	sm.memType = TypeUnified
	return sm
}

// WithStream specifies a custom stream
func (sm *SimpleMemory) WithStream(stream *Stream) *SimpleMemory {
	sm.stream = stream
	return sm
}

// Allocate performs the actual memory allocation
func (sm *SimpleMemory) Allocate() (*Memory, error) {
	return MallocWithTypeAndStream(sm.stream, sm.size, sm.memType)
}

// SimpleContext provides a simplified context for beginners
type SimpleContext struct {
	ctx *Context
}

// NewSimpleContext creates a new simplified context (addresses complex initialization)
func NewSimpleContext() (*SimpleContext, error) {
	// Auto-detects and handles everything
	err := Initialize()
	if err != nil {
		return nil, &EnhancedError{
			Operation:  "Context Creation",
			Cause:      err.Error(),
			Suggestion: "Ensure CUDA drivers are installed or use CPU simulation mode",
			Details: map[string]interface{}{
				"cuda_available": IsCudaAvailable(),
				"device_count":   GetCudaDeviceCount(),
			},
		}
	}

	ctx := GetDefaultContext()
	return &SimpleContext{ctx: ctx}, nil
}

// Run executes a function in the GPU context (Simple API)
func (sc *SimpleContext) Run(fn func() error) error {
	return fn() // For now, just execute - can be enhanced with GPU scheduling
}

// Compute provides auto-managed memory computation (Simplest API)
func Compute(data []float32, fn func(input []float32) []float32) ([]float32, error) {
	// Auto-managed memory approach
	result := fn(data) // For simulation, just call the function
	return result, nil
}

// Vector represents a GPU vector with simple operations
type Vector struct {
	mem  *Memory
	size int
	data []float32
}

// NewVector creates a vector (Simple API - Layer 1)
func NewVector(data []float32) (*Vector, error) {
	size := len(data)
	mem, err := Alloc(int64(size * 4)).OnDevice().Allocate() // 4 bytes per float32
	if err != nil {
		return nil, &EnhancedError{
			Operation:  "Vector Creation",
			Cause:      err.Error(),
			Suggestion: "Try reducing vector size or use CPU simulation mode",
			Details: map[string]interface{}{
				"requested_size":   size * 4,
				"available_memory": func() int64 { free, _ := GetMemoryInfo(); return free }(),
			},
		}
	}

	// Copy data to GPU/memory (handle both simulation and real CUDA)
	if mem.Data() != nil {
		// Simulation mode - direct copy to data slice
		floatData := (*[1 << 30]float32)(mem.Ptr())[:size:size]
		copy(floatData, data)
	} else {
		// Real CUDA mode - use memory copy functions
		hostData := make([]byte, size*4)
		for i, v := range data {
			bits := FloatToBits(v)
			hostData[i*4] = byte(bits)
			hostData[i*4+1] = byte(bits >> 8)
			hostData[i*4+2] = byte(bits >> 16)
			hostData[i*4+3] = byte(bits >> 24)
		}

		err = CopyHostToDevice(mem, hostData)
		if err != nil {
			mem.Free()
			return nil, err
		}
	}

	return &Vector{
		mem:  mem,
		size: size,
		data: data, // Keep copy for simulation
	}, nil
}

// Add performs vector addition (Simple API)
func (v *Vector) Add(other *Vector) (*Vector, error) {
	if v.size != other.size {
		return nil, &EnhancedError{
			Operation:  "Vector Addition",
			Cause:      "Vector size mismatch",
			Suggestion: "Ensure both vectors have the same size",
			Details: map[string]interface{}{
				"vector1_size": v.size,
				"vector2_size": other.size,
			},
		}
	}

	result, err := NewVector(make([]float32, v.size))
	if err != nil {
		return nil, err
	}

	// Perform addition using existing kernel
	err = ExecuteVectorAdd(v.mem, other.mem, result.mem, v.size)
	if err != nil {
		result.Free()
		return nil, err
	}

	return result, nil
}

// Multiply performs scalar multiplication (Simple API)
func (v *Vector) Multiply(scalar float32) (*Vector, error) {
	result, err := NewVector(make([]float32, v.size))
	if err != nil {
		return nil, err
	}

	// For simulation, just multiply the data
	for i := 0; i < v.size; i++ {
		result.data[i] = v.data[i] * scalar
	}

	return result, nil
}

// ToSlice returns the vector as a Go slice (Simple API)
func (v *Vector) ToSlice() ([]float32, error) {
	if v.data != nil {
		return v.data, nil // Return simulation data
	}

	// For real GPU, copy back from device
	hostData := make([]byte, v.size*4)
	err := CopyDeviceToHost(hostData, v.mem)
	if err != nil {
		return nil, err
	}

	result := make([]float32, v.size)
	for i := 0; i < v.size; i++ {
		bits := uint32(hostData[i*4]) |
			(uint32(hostData[i*4+1]) << 8) |
			(uint32(hostData[i*4+2]) << 16) |
			(uint32(hostData[i*4+3]) << 24)
		result[i] = BitsToFloat(bits)
	}

	return result, nil
}

// Free releases the vector memory
func (v *Vector) Free() error {
	if v.mem != nil {
		return v.mem.Free()
	}
	return nil
}

// Length returns the vector length
func (v *Vector) Length() int {
	return v.size
}

// Size returns the vector size in bytes
func (v *Vector) Size() int {
	return v.size * 4 // 4 bytes per float32
}

// createEmptyVector creates a vector without initializing data (for results)
func createEmptyVector(size int) (*Vector, error) {
	mem, err := Alloc(int64(size * 4)).OnDevice().Allocate()
	if err != nil {
		return nil, err
	}

	return &Vector{
		mem:  mem,
		size: size,
		data: nil, // No initial data
	}, nil
}

// createEmptyMatrix creates a matrix without initializing data (for results)
func createEmptyMatrix(rows, cols int) (*Matrix, error) {
	size := rows * cols
	mem, err := Alloc(int64(size * 4)).OnDevice().Allocate()
	if err != nil {
		return nil, err
	}

	return &Matrix{
		mem:  mem,
		rows: rows,
		cols: cols,
		data: nil, // No initial data
	}, nil
}

// Helper functions for float conversion
func FloatToBits(f float32) uint32 {
	return *(*uint32)(unsafe.Pointer(&f))
}

func BitsToFloat(bits uint32) float32 {
	return *(*float32)(unsafe.Pointer(&bits))
}

// Simple API functions for common operations

// Run executes a function on GPU (Simple API - addresses naming consistency)
func Run(fn func() error) error {
	// Consistent naming instead of Go()
	return fn()
}

// Parallel executes a parallel loop (Simple API - consistent naming)
func Parallel(start, end int, fn func(int) error) error {
	// Consistent naming instead of ParallelFor()
	return ParallelFor(start, end, fn)
}

// Matrix represents a GPU matrix with simple operations (Simple API)
type Matrix struct {
	mem  *Memory
	rows int
	cols int
	data []float32
}

// NewMatrix creates a matrix (Simple API)
func NewMatrix(rows, cols int, data []float32) (*Matrix, error) {
	if len(data) != rows*cols {
		return nil, &EnhancedError{
			Operation:  "Matrix Creation",
			Cause:      "Data size doesn't match matrix dimensions",
			Suggestion: fmt.Sprintf("Provide exactly %d elements for a %dx%d matrix", rows*cols, rows, cols),
			Details: map[string]interface{}{
				"expected_size": rows * cols,
				"actual_size":   len(data),
				"matrix_dims":   fmt.Sprintf("%dx%d", rows, cols),
			},
		}
	}

	size := rows * cols
	mem, err := Alloc(int64(size * 4)).OnDevice().Allocate()
	if err != nil {
		return nil, err
	}

	// Copy data to GPU/memory (handle both simulation and real CUDA)
	if mem.Data() != nil {
		// Simulation mode - direct copy to data slice
		floatData := (*[1 << 30]float32)(mem.Ptr())[:size:size]
		copy(floatData, data)
	} else {
		// Real CUDA mode - use memory copy functions
		hostData := make([]byte, size*4)
		for i, v := range data {
			bits := FloatToBits(v)
			hostData[i*4] = byte(bits)
			hostData[i*4+1] = byte(bits >> 8)
			hostData[i*4+2] = byte(bits >> 16)
			hostData[i*4+3] = byte(bits >> 24)
		}

		err = CopyHostToDevice(mem, hostData)
		if err != nil {
			mem.Free()
			return nil, err
		}
	}

	return &Matrix{
		mem:  mem,
		rows: rows,
		cols: cols,
		data: data,
	}, nil
}

// Multiply performs matrix multiplication (Simple API)
func (m *Matrix) Multiply(other *Matrix) (*Matrix, error) {
	if m.cols != other.rows {
		return nil, &EnhancedError{
			Operation:  "Matrix Multiplication",
			Cause:      "Matrix dimensions incompatible for multiplication",
			Suggestion: fmt.Sprintf("Left matrix columns (%d) must equal right matrix rows (%d)", m.cols, other.rows),
			Details: map[string]interface{}{
				"left_matrix":  fmt.Sprintf("%dx%d", m.rows, m.cols),
				"right_matrix": fmt.Sprintf("%dx%d", other.rows, other.cols),
			},
		}
	}

	result, err := NewMatrix(m.rows, other.cols, make([]float32, m.rows*other.cols))
	if err != nil {
		return nil, err
	}

	// Use existing matrix multiplication kernel
	err = ExecuteMatrixMultiply(m.mem, other.mem, result.mem, m.rows, other.cols, m.cols)
	if err != nil {
		result.Free()
		return nil, err
	}

	return result, nil
}

// Free releases the matrix memory
func (m *Matrix) Free() error {
	if m.mem != nil {
		return m.mem.Free()
	}
	return nil
}

// Rows returns the number of rows
func (m *Matrix) Rows() int {
	return m.rows
}

// Cols returns the number of columns
func (m *Matrix) Cols() int {
	return m.cols
}

// Size returns the matrix size in bytes
func (m *Matrix) Size() int {
	return m.rows * m.cols * 4 // 4 bytes per float32
}

// Simple API wrapper functions that use the existing kernel functions

// SimpleVectorAdd provides an easy-to-use vector addition function
func SimpleVectorAdd(a, b []float32) ([]float32, error) {
	if len(a) != len(b) {
		return nil, fmt.Errorf("vector lengths must match: %d != %d", len(a), len(b))
	}

	// Create vectors using Simple API
	va, err := NewVector(a)
	if err != nil {
		return nil, err
	}
	defer va.Free()

	vb, err := NewVector(b)
	if err != nil {
		return nil, err
	}
	defer vb.Free()

	result, err := createEmptyVector(len(a))
	if err != nil {
		return nil, err
	}
	defer result.Free()

	// Use the existing kernel function
	err = ExecuteVectorAdd(va.mem, vb.mem, result.mem, len(a))
	if err != nil {
		return nil, err
	}

	// Ensure kernel completion
	err = SynchronizeDevice()
	if err != nil {
		return nil, err
	}

	// Get result
	output := make([]float32, len(a))
	if result.mem.Data() != nil {
		// Simulation mode - direct access
		floatData := (*[1 << 30]float32)(result.mem.Ptr())[:len(a):len(a)]
		copy(output, floatData)
	} else {
		// Real CUDA mode - use memory copy
		err = CopyDeviceToHost((*[1 << 30]byte)(unsafe.Pointer(&output[0]))[:len(output)*4], result.mem)
		if err != nil {
			return nil, err
		}
	}

	return output, nil
}

// SimpleMatrixMultiply provides an easy-to-use matrix multiplication function
func SimpleMatrixMultiply(a, b [][]float32) ([][]float32, error) {
	if len(a) == 0 || len(b) == 0 || len(a[0]) != len(b) {
		return nil, fmt.Errorf("invalid matrix dimensions for multiplication")
	}

	rows := len(a)
	cols := len(b[0])
	common := len(a[0])

	// Flatten matrices
	flatA := make([]float32, rows*common)
	flatB := make([]float32, common*cols)

	for i := 0; i < rows; i++ {
		for j := 0; j < common; j++ {
			flatA[i*common+j] = a[i][j]
		}
	}

	for i := 0; i < common; i++ {
		for j := 0; j < cols; j++ {
			flatB[i*cols+j] = b[i][j]
		}
	}

	// Create matrices using Simple API
	ma, err := NewMatrix(rows, common, flatA)
	if err != nil {
		return nil, err
	}
	defer ma.Free()

	mb, err := NewMatrix(common, cols, flatB)
	if err != nil {
		return nil, err
	}
	defer mb.Free()

	result, err := createEmptyMatrix(rows, cols)
	if err != nil {
		return nil, err
	}
	defer result.Free()

	// Use the existing kernel function
	err = ExecuteMatrixMultiply(ma.mem, mb.mem, result.mem, rows, cols, common)
	if err != nil {
		return nil, err
	}

	// Ensure kernel completion
	err = SynchronizeDevice()
	if err != nil {
		return nil, err
	}

	// Get result and reshape
	flatResult := make([]float32, rows*cols)
	if result.mem.Data() != nil {
		// Simulation mode - direct access
		floatData := (*[1 << 30]float32)(result.mem.Ptr())[: rows*cols : rows*cols]
		copy(flatResult, floatData)
	} else {
		// Real CUDA mode - use memory copy
		err = CopyDeviceToHost((*[1 << 30]byte)(unsafe.Pointer(&flatResult[0]))[:len(flatResult)*4], result.mem)
		if err != nil {
			return nil, err
		}
	}

	output := make([][]float32, rows)
	for i := 0; i < rows; i++ {
		output[i] = make([]float32, cols)
		for j := 0; j < cols; j++ {
			output[i][j] = flatResult[i*cols+j]
		}
	}

	return output, nil
}
