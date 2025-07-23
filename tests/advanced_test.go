package tests

import (
	"testing"

	"github.com/stitch1968/gocuda/advanced"
	"github.com/stitch1968/gocuda/memory"
)

// TestFFT tests Fast Fourier Transform functionality
func TestFFT(t *testing.T) {
	t.Log("Testing FFT...")

	size := 64                     // Must be power of 2
	complexSize := int64(size * 8) // 2 * 4 bytes per complex number (real + imag)

	input, err := memory.Alloc(complexSize)
	if err != nil {
		t.Fatalf("Failed to allocate input memory: %v", err)
	}
	defer input.Free()

	output, err := memory.Alloc(complexSize)
	if err != nil {
		t.Fatalf("Failed to allocate output memory: %v", err)
	}
	defer output.Free()

	// Initialize input with test data
	inputData := (*[1 << 30]advanced.Complex64)(input.Ptr())[:size:size]
	for i := 0; i < size; i++ {
		inputData[i] = advanced.Complex64{Real: 1.0, Imag: 0.0}
	}

	// Perform forward FFT
	err = advanced.FFT(input, output, size, false)
	if err != nil {
		t.Errorf("FFT forward failed: %v", err)
	}

	// Perform inverse FFT
	result, err := memory.Alloc(complexSize)
	if err != nil {
		t.Fatalf("Failed to allocate result memory: %v", err)
	}
	defer result.Free()

	err = advanced.FFT(output, result, size, true)
	if err != nil {
		t.Errorf("FFT inverse failed: %v", err)
	}

	t.Log("✅ FFT test passed")
}

// TestSortingAlgorithms tests sorting functionality
func TestSortingAlgorithms(t *testing.T) {
	t.Log("Testing sorting algorithms...")

	// Test RadixSort
	size := 256
	data, err := memory.Alloc(int64(size * 4)) // 4 bytes per uint32
	if err != nil {
		t.Fatalf("Failed to allocate memory: %v", err)
	}
	defer data.Free()

	// Initialize with random-ish data
	uintData := (*[1 << 30]uint32)(data.Ptr())[:size:size]
	for i := 0; i < size; i++ {
		uintData[i] = uint32((size-i)*123) % 1000
	}

	// Sort using radix sort
	err = advanced.RadixSort(data, size)
	if err != nil {
		t.Errorf("RadixSort failed: %v", err)
	}

	// Verify sorting
	for i := 1; i < size; i++ {
		if uintData[i] < uintData[i-1] {
			t.Errorf("Array not properly sorted at index %d: %d > %d", i, uintData[i-1], uintData[i])
			break
		}
	}

	t.Log("✅ Radix sort test passed")

	// Test BitonicSort
	floatSize := 128 // Must be power of 2 for bitonic sort
	floatData, err := memory.Alloc(int64(floatSize * 4))
	if err != nil {
		t.Fatalf("Failed to allocate float memory: %v", err)
	}
	defer floatData.Free()

	// Initialize with reverse-sorted data
	fData := (*[1 << 30]float32)(floatData.Ptr())[:floatSize:floatSize]
	for i := 0; i < floatSize; i++ {
		fData[i] = float32(floatSize - i)
	}

	err = advanced.BitonicSort(floatData, floatSize)
	if err != nil {
		t.Errorf("BitonicSort failed: %v", err)
	}

	// Verify sorting
	for i := 1; i < floatSize; i++ {
		if fData[i] < fData[i-1] {
			t.Errorf("Bitonic sort failed at index %d: %f > %f", i, fData[i-1], fData[i])
			break
		}
	}

	t.Log("✅ Bitonic sort test passed")
}

// TestMatrixOperations tests matrix operations
func TestMatrixOperations(t *testing.T) {
	t.Log("Testing matrix operations...")

	// Test GEMM (General Matrix Multiply)
	m, n, k := 4, 4, 4
	matrixSize := int64(m * k * 4) // 4 bytes per float32

	matA, err := memory.Alloc(matrixSize)
	if err != nil {
		t.Fatalf("Failed to allocate matrix A: %v", err)
	}
	defer matA.Free()

	matB, err := memory.Alloc(int64(k * n * 4))
	if err != nil {
		t.Fatalf("Failed to allocate matrix B: %v", err)
	}
	defer matB.Free()

	matC, err := memory.Alloc(int64(m * n * 4))
	if err != nil {
		t.Fatalf("Failed to allocate matrix C: %v", err)
	}
	defer matC.Free()

	// Initialize matrices
	dataA := (*[1 << 30]float32)(matA.Ptr())[: m*k : m*k]
	dataB := (*[1 << 30]float32)(matB.Ptr())[: k*n : k*n]
	dataC := (*[1 << 30]float32)(matC.Ptr())[: m*n : m*n]

	for i := 0; i < m*k; i++ {
		dataA[i] = 2.0
	}
	for i := 0; i < k*n; i++ {
		dataB[i] = 3.0
	}
	for i := 0; i < m*n; i++ {
		dataC[i] = 0.0
	}

	// Perform GEMM: C = alpha*A*B + beta*C
	alpha := float32(1.0)
	beta := float32(0.0)
	err = advanced.GEMM(alpha, matA, matB, beta, matC, m, n, k)
	if err != nil {
		t.Errorf("GEMM failed: %v", err)
	}

	t.Log("✅ Matrix operations test passed")

	// Test Transpose
	input, err := memory.Alloc(int64(m * n * 4))
	if err != nil {
		t.Fatalf("Failed to allocate transpose input: %v", err)
	}
	defer input.Free()

	output, err := memory.Alloc(int64(m * n * 4))
	if err != nil {
		t.Fatalf("Failed to allocate transpose output: %v", err)
	}
	defer output.Free()

	err = advanced.Transpose(input, output, m, n)
	if err != nil {
		t.Errorf("Transpose failed: %v", err)
	}

	t.Log("✅ Transpose test passed")
}

// TestGraphAlgorithms tests graph algorithm functionality
func TestGraphAlgorithms(t *testing.T) {
	t.Log("Testing graph algorithms...")

	vertices := 5
	edges := 6

	// Create adjacency matrix representation
	graphSize := int64(vertices * vertices * 4)
	graph, err := memory.Alloc(graphSize)
	if err != nil {
		t.Fatalf("Failed to allocate graph memory: %v", err)
	}
	defer graph.Free()

	distances, err := memory.Alloc(int64(vertices * 4))
	if err != nil {
		t.Fatalf("Failed to allocate distances memory: %v", err)
	}
	defer distances.Free()

	// Initialize simple graph
	graphData := (*[1 << 30]int32)(graph.Ptr())[: vertices*vertices : vertices*vertices]
	for i := 0; i < vertices*vertices; i++ {
		graphData[i] = 0
	}
	// Add some edges
	graphData[0*vertices+1] = 1
	graphData[1*vertices+2] = 1
	graphData[2*vertices+3] = 1

	// Perform BFS
	startVertex := 0
	err = advanced.BFS(graph, vertices, edges, startVertex, distances)
	if err != nil {
		t.Errorf("BFS failed: %v", err)
	}

	t.Log("✅ BFS test passed")

	// Test PageRank
	ranks, err := memory.Alloc(int64(vertices * 4))
	if err != nil {
		t.Fatalf("Failed to allocate ranks memory: %v", err)
	}
	defer ranks.Free()

	iterations := 10
	damping := float32(0.85)
	err = advanced.PageRank(graph, vertices, iterations, damping, ranks)
	if err != nil {
		t.Errorf("PageRank failed: %v", err)
	}

	t.Log("✅ PageRank test passed")
}

// TestBatchOperations tests batch operation functionality
func TestBatchOperations(t *testing.T) {
	t.Log("Testing batch operations...")

	batch := advanced.NewBatch()
	if batch == nil {
		t.Fatal("Failed to create batch")
	}

	// Add operations to batch
	completed := 0

	batch.Add(func() error {
		completed++
		return nil
	})

	batch.Add(func() error {
		completed++
		return nil
	})

	// Execute batch
	err := batch.Execute()
	if err != nil {
		t.Errorf("Batch execution failed: %v", err)
	}

	if completed != 2 {
		t.Errorf("Expected 2 completed operations, got %d", completed)
	}

	t.Log("✅ Batch operations test passed")
}

// TestErrorHandling tests error conditions
func TestErrorHandling(t *testing.T) {
	t.Log("Testing error handling...")

	// Test FFT with invalid size (not power of 2)
	invalidSize := 63 // Not a power of 2
	input, _ := memory.Alloc(int64(invalidSize * 8))
	defer input.Free()
	output, _ := memory.Alloc(int64(invalidSize * 8))
	defer output.Free()

	err := advanced.FFT(input, output, invalidSize, false)
	if err == nil {
		t.Error("Expected error for invalid FFT size")
	}

	// Test BitonicSort with invalid size
	invalidFloatSize := 100 // Not a power of 2
	floatData, _ := memory.Alloc(int64(invalidFloatSize * 4))
	defer floatData.Free()

	err = advanced.BitonicSort(floatData, invalidFloatSize)
	if err == nil {
		t.Error("Expected error for invalid bitonic sort size")
	}

	t.Log("✅ Error handling test passed")
}

// BenchmarkFFT benchmarks FFT performance
func BenchmarkFFT(b *testing.B) {
	size := 1024
	complexSize := int64(size * 8)

	input, _ := memory.Alloc(complexSize)
	defer input.Free()
	output, _ := memory.Alloc(complexSize)
	defer output.Free()

	// Initialize input
	inputData := (*[1 << 30]advanced.Complex64)(input.Ptr())[:size:size]
	for i := 0; i < size; i++ {
		inputData[i] = advanced.Complex64{Real: float32(i), Imag: 0.0}
	}

	for i := 0; i < b.N; i++ {
		err := advanced.FFT(input, output, size, false)
		if err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkGEMM benchmarks matrix multiplication performance
func BenchmarkGEMM(b *testing.B) {
	m, n, k := 64, 64, 64

	matA, _ := memory.Alloc(int64(m * k * 4))
	defer matA.Free()
	matB, _ := memory.Alloc(int64(k * n * 4))
	defer matB.Free()
	matC, _ := memory.Alloc(int64(m * n * 4))
	defer matC.Free()

	for i := 0; i < b.N; i++ {
		err := advanced.GEMM(1.0, matA, matB, 0.0, matC, m, n, k)
		if err != nil {
			b.Fatal(err)
		}
	}
}
