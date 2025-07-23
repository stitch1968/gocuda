//go:build ignore

package main

import (
	"fmt"
	"log"

	cuda "github.com/stitch1968/gocuda"
)

func main() {
	fmt.Println("🔬 GoCUDA Simple API Integration Test")
	fmt.Println("=====================================")

	// Initialize CUDA
	err := cuda.Initialize()
	if err != nil {
		log.Fatalf("Failed to initialize CUDA: %v", err)
	}

	// Test Simple Vector Addition
	fmt.Println("\n📊 Testing Simple Vector Addition...")
	testSimpleVectorAdd()

	// Test Simple Matrix Multiplication
	fmt.Println("\n📊 Testing Simple Matrix Multiplication...")
	testSimpleMatrixMultiply()

	// Test Builder Pattern Memory Allocation
	fmt.Println("\n📊 Testing Builder Pattern Memory...")
	testBuilderPattern()

	fmt.Println("\n✅ All Simple API integration tests completed successfully!")
}

func testSimpleVectorAdd() {
	// Prepare test data
	a := []float32{1.0, 2.0, 3.0, 4.0, 5.0}
	b := []float32{5.0, 4.0, 3.0, 2.0, 1.0}

	fmt.Printf("   Vector A: %v\n", a)
	fmt.Printf("   Vector B: %v\n", b)

	// Use Simple API
	result, err := cuda.SimpleVectorAdd(a, b)
	if err != nil {
		log.Fatalf("Simple vector addition failed: %v", err)
	}

	fmt.Printf("   Result:   %v\n", result)

	// Verify result
	expected := []float32{6.0, 6.0, 6.0, 6.0, 6.0}
	for i := range result {
		if result[i] != expected[i] {
			log.Fatalf("Unexpected result at index %d: got %f, expected %f", i, result[i], expected[i])
		}
	}

	fmt.Println("   ✅ Simple Vector Addition: PASSED")
}

func testSimpleMatrixMultiply() {
	// Prepare test matrices
	a := [][]float32{
		{1.0, 2.0},
		{3.0, 4.0},
	}
	b := [][]float32{
		{2.0, 0.0},
		{1.0, 2.0},
	}

	fmt.Println("   Matrix A:")
	for _, row := range a {
		fmt.Printf("     %v\n", row)
	}
	fmt.Println("   Matrix B:")
	for _, row := range b {
		fmt.Printf("     %v\n", row)
	}

	// Use Simple API
	result, err := cuda.SimpleMatrixMultiply(a, b)
	if err != nil {
		log.Fatalf("Simple matrix multiplication failed: %v", err)
	}

	fmt.Println("   Result:")
	for _, row := range result {
		fmt.Printf("     %v\n", row)
	}

	// Verify result: [[1*2+2*1, 1*0+2*2], [3*2+4*1, 3*0+4*2]] = [[4, 4], [10, 8]]
	expected := [][]float32{
		{4.0, 4.0},
		{10.0, 8.0},
	}

	for i := range result {
		for j := range result[i] {
			if result[i][j] != expected[i][j] {
				log.Fatalf("Unexpected result at [%d][%d]: got %f, expected %f", i, j, result[i][j], expected[i][j])
			}
		}
	}

	fmt.Println("   ✅ Simple Matrix Multiplication: PASSED")
}

func testBuilderPattern() {
	fmt.Println("   Testing memory allocation builder pattern...")

	// Test vector creation with builder pattern
	data := []float32{1.0, 2.0, 3.0, 4.0, 5.0}
	vector, err := cuda.NewVector(data)
	if err != nil {
		log.Fatalf("Failed to create vector: %v", err)
	}
	defer vector.Free()

	if vector.Length() != len(data) {
		log.Fatalf("Vector length mismatch: got %d, expected %d", vector.Length(), len(data))
	}

	fmt.Printf("   ✅ Vector created with length: %d\n", vector.Length())

	// Test matrix creation with builder pattern
	matrixData := make([]float32, 6)
	for i := range matrixData {
		matrixData[i] = float32(i + 1)
	}

	matrix, err := cuda.NewMatrix(2, 3, matrixData)
	if err != nil {
		log.Fatalf("Failed to create matrix: %v", err)
	}
	defer matrix.Free()

	if matrix.Rows() != 2 || matrix.Cols() != 3 {
		log.Fatalf("Matrix dimensions mismatch: got %dx%d, expected 2x3", matrix.Rows(), matrix.Cols())
	}

	fmt.Printf("   ✅ Matrix created with dimensions: %dx%d\n", matrix.Rows(), matrix.Cols())

	// Test memory builder
	mem, err := cuda.Alloc(1024).OnDevice().Allocate()
	if err != nil {
		log.Fatalf("Failed to allocate memory: %v", err)
	}
	defer mem.Free()

	if mem.Size() != 1024 {
		log.Fatalf("Memory size mismatch: got %d, expected 1024", mem.Size())
	}

	fmt.Printf("   ✅ Memory allocated with size: %d bytes\n", mem.Size())

	fmt.Println("   ✅ Builder Pattern Tests: PASSED")
}
