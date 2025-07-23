package tests

import (
	"testing"

	"github.com/stitch1968/gocuda/internal"
	"github.com/stitch1968/gocuda/kernels"
	"github.com/stitch1968/gocuda/memory"
)

// TestVectorAdd tests vector addition kernel
func TestVectorAdd(t *testing.T) {
	t.Log("Testing vector addition kernel...")

	size := 256

	// Allocate memory for vectors
	a, err := memory.Alloc(int64(size * 4)) // 4 bytes per float32
	if err != nil {
		t.Fatalf("Failed to allocate vector a: %v", err)
	}
	defer a.Free()

	b, err := memory.Alloc(int64(size * 4))
	if err != nil {
		t.Fatalf("Failed to allocate vector b: %v", err)
	}
	defer b.Free()

	c, err := memory.Alloc(int64(size * 4))
	if err != nil {
		t.Fatalf("Failed to allocate vector c: %v", err)
	}
	defer c.Free()

	// Initialize test data
	aData := (*[1 << 30]float32)(a.Ptr())[:size:size]
	bData := (*[1 << 30]float32)(b.Ptr())[:size:size]
	for i := 0; i < size; i++ {
		aData[i] = float32(i)
		bData[i] = float32(i * 2)
	}

	// Execute vector add kernel
	vectorAdd := &kernels.VectorAdd{}
	gridDim := kernels.Dim3{X: 1, Y: 1, Z: 1}
	blockDim := kernels.Dim3{X: 256, Y: 1, Z: 1}

	err = vectorAdd.Execute(gridDim, blockDim, 0, nil, a, b, c, size)
	if err != nil {
		t.Fatalf("Failed to execute vector add: %v", err)
	}

	// Verify results
	cData := (*[1 << 30]float32)(c.Ptr())[:size:size]
	for i := 0; i < size; i++ {
		expected := float32(i + i*2)
		if cData[i] != expected {
			t.Errorf("At index %d: expected %f, got %f", i, expected, cData[i])
		}
	}

	t.Log("✅ Vector addition test passed")
}

// TestVectorAddWithStream tests vector addition with stream
func TestVectorAddWithStream(t *testing.T) {
	t.Log("Testing vector addition with stream...")

	size := 128
	stream := internal.GetDefaultStream()

	// Allocate memory
	a, err := memory.AllocWithStream(stream, int64(size*4))
	if err != nil {
		t.Fatalf("Failed to allocate vector a: %v", err)
	}
	defer a.Free()

	b, err := memory.AllocWithStream(stream, int64(size*4))
	if err != nil {
		t.Fatalf("Failed to allocate vector b: %v", err)
	}
	defer b.Free()

	c, err := memory.AllocWithStream(stream, int64(size*4))
	if err != nil {
		t.Fatalf("Failed to allocate vector c: %v", err)
	}
	defer c.Free()

	// Initialize data
	aData := (*[1 << 30]float32)(a.Ptr())[:size:size]
	bData := (*[1 << 30]float32)(b.Ptr())[:size:size]
	for i := 0; i < size; i++ {
		aData[i] = float32(i + 1)
		bData[i] = float32(i + 2)
	}

	// Execute with stream
	vectorAdd := &kernels.VectorAdd{}
	gridDim := kernels.Dim3{X: 1, Y: 1, Z: 1}
	blockDim := kernels.Dim3{X: 128, Y: 1, Z: 1}

	err = vectorAdd.Execute(gridDim, blockDim, 0, stream, a, b, c, size)
	if err != nil {
		t.Fatalf("Failed to execute vector add with stream: %v", err)
	}

	t.Log("✅ Vector addition with stream test passed")
}

// TestKernelDim3 tests Dim3 structure
func TestKernelDim3(t *testing.T) {
	t.Log("Testing Dim3 structure...")

	dim := kernels.Dim3{X: 32, Y: 16, Z: 8}

	if dim.X != 32 {
		t.Errorf("Expected X=32, got %d", dim.X)
	}

	if dim.Y != 16 {
		t.Errorf("Expected Y=16, got %d", dim.Y)
	}

	if dim.Z != 8 {
		t.Errorf("Expected Z=8, got %d", dim.Z)
	}

	t.Log("✅ Dim3 structure test passed")
}

// TestKernelInterface tests kernel interface compliance
func TestKernelInterface(t *testing.T) {
	t.Log("Testing kernel interface...")

	var kernel kernels.Kernel = &kernels.VectorAdd{}

	if kernel == nil {
		t.Error("Expected non-nil kernel interface")
	}

	t.Log("✅ Kernel interface test passed")
}

// TestKernelErrorHandling tests error conditions
func TestKernelErrorHandling(t *testing.T) {
	t.Log("Testing kernel error handling...")

	vectorAdd := &kernels.VectorAdd{}
	gridDim := kernels.Dim3{X: 1, Y: 1, Z: 1}
	blockDim := kernels.Dim3{X: 1, Y: 1, Z: 1}

	// Test with wrong number of arguments
	err := vectorAdd.Execute(gridDim, blockDim, 0, nil, 1, 2) // Too few args
	if err == nil {
		t.Error("Expected error with insufficient arguments")
	}

	// Test with wrong argument types
	err = vectorAdd.Execute(gridDim, blockDim, 0, nil, "a", "b", "c", "n")
	if err == nil {
		t.Error("Expected error with wrong argument types")
	}

	t.Log("✅ Kernel error handling test passed")
}

// BenchmarkVectorAdd benchmarks vector addition performance
func BenchmarkVectorAdd(b *testing.B) {
	size := 1024

	// Allocate memory once
	a, _ := memory.Alloc(int64(size * 4))
	defer a.Free()
	vecB, _ := memory.Alloc(int64(size * 4))
	defer vecB.Free()
	c, _ := memory.Alloc(int64(size * 4))
	defer c.Free()

	vectorAdd := &kernels.VectorAdd{}
	gridDim := kernels.Dim3{X: 1, Y: 1, Z: 1}
	blockDim := kernels.Dim3{X: 1024, Y: 1, Z: 1}

	for i := 0; i < b.N; i++ {
		err := vectorAdd.Execute(gridDim, blockDim, 0, nil, a, vecB, c, size)
		if err != nil {
			b.Fatal(err)
		}
	}
}
