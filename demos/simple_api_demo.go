//go:build ignore
// +build ignore

package main

import (
	"fmt"
	"sync/atomic"
	"time"

	cuda "github.com/stitch1968/gocuda"
)

func main() {
	fmt.Println("=== GoCUDA Simple API Demo ===")
	fmt.Println("This demo showcases user-friendly patterns and improvements")
	fmt.Println("Works on any system - GPU optional!")

	// Demo 1: Enhanced Error Handling
	fmt.Println("⚠️ Demo 1: Enhanced Error Handling")
	demoErrorHandling()
	fmt.Println()

	// Demo 2: Simple Memory Management
	fmt.Println("💾 Demo 2: Memory Management with Builder Pattern")
	demoMemoryManagement()
	fmt.Println()

	// Demo 3: Working API Examples
	fmt.Println("⚡ Demo 3: Current Working API")
	demoWorkingAPI()
	fmt.Println()

	// Demo 4: Performance Patterns
	fmt.Println("🏁 Demo 4: Performance Patterns")
	demoPerformancePatterns()
	fmt.Println()

	fmt.Println("✅ All demos completed successfully!")
	fmt.Println("💡 This shows improved error handling and patterns")
	fmt.Println("📚 Next: Check docs/TRAINING_GUIDE_BEGINNER.md for a detailed learning path")
}

func demoErrorHandling() {
	fmt.Println("   Demonstrating enhanced error handling...")

	// Create enhanced error examples
	err := cuda.NewMemoryError("GPU memory allocation",
		"Insufficient GPU memory",
		8*1024*1024*1024, // requested: 8GB
		2*1024*1024*1024) // available: 2GB

	fmt.Printf("   📋 Error: %s\n", err.Error())
	fmt.Printf("   📝 Details: %v\n", err.GetDetails())

	if err.IsRecoverable() {
		fmt.Println("   🔄 Error is recoverable")
	}

	// Create kernel error example
	gridDim := cuda.Dim3{X: 1536, Y: 1, Z: 1}
	blockDim := cuda.Dim3{X: 1024, Y: 1024, Z: 64}
	kernelErr := cuda.NewKernelError("Kernel launch",
		"Block size too large for device",
		gridDim, blockDim)

	fmt.Printf("   📋 Kernel Error: %s\n", kernelErr.Error())
	fmt.Printf("   💡 Details: %v\n", kernelErr.GetDetails())

	fmt.Println("   ✅ Enhanced error handling provides actionable guidance")
}

func demoMemoryManagement() {
	fmt.Println("   Demonstrating builder pattern memory allocation...")

	// Pattern 1: Device memory allocation
	fmt.Println("   📝 Pattern 1: Device memory")
	mem1, err := cuda.Alloc(1024).OnDevice().Allocate()
	if err != nil {
		fmt.Printf("   ⚠️  Device allocation failed: %v\n", err)
	} else {
		fmt.Println("   ✅ Device memory allocated successfully")
		defer mem1.Free()
	}

	// Pattern 2: Pinned memory allocation
	fmt.Println("   📝 Pattern 2: Pinned memory")
	mem2, err := cuda.Alloc(512).Pinned().Allocate()
	if err != nil {
		fmt.Printf("   ⚠️  Pinned allocation failed: %v\n", err)
	} else {
		fmt.Println("   ✅ Pinned memory allocated successfully")
		defer mem2.Free()
	}

	// Pattern 3: Unified memory allocation
	fmt.Println("   📝 Pattern 3: Unified memory")
	mem3, err := cuda.Alloc(256).Unified().Allocate()
	if err != nil {
		fmt.Printf("   ⚠️  Unified allocation failed: %v\n", err)
	} else {
		fmt.Println("   ✅ Unified memory allocated successfully")
		defer mem3.Free()
	}

	fmt.Println("   💡 Builder pattern provides clear, chainable memory allocation")
}

func demoWorkingAPI() {
	fmt.Println("   Using the current working API with improvements...")

	// Example 1: Map operation (works)
	fmt.Println("   📝 Map operation:")
	input := []float64{1, 2, 3, 4, 5}
	result, err := cuda.Map(input, func(x float64) float64 {
		return x * x // Square each element
	})
	if err != nil {
		fmt.Printf("   ❌ Map operation failed: %v\n", err)
	} else {
		fmt.Printf("   Input:  %v\n", input)
		fmt.Printf("   Output: %v (squared)\n", result)
		fmt.Println("   ✅ Map operation completed successfully")
	}

	// Example 2: Reduce operation (works)
	fmt.Println("   📝 Reduce operation:")
	numbers := []float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
	sum, err := cuda.Reduce(numbers, func(a, b float64) float64 { return a + b })
	if err != nil {
		fmt.Printf("   ❌ Reduce operation failed: %v\n", err)
	} else {
		fmt.Printf("   Sum of 1-10: %.0f\n", sum)
		fmt.Println("   ✅ Reduce operation completed successfully")
	}

	// Example 3: Parallel processing (works)
	fmt.Println("   📝 Parallel processing:")
	var processed atomic.Int64
	err = cuda.ParallelFor(0, 100, func(i int) error {
		processed.Add(1)
		return nil
	})
	if err != nil {
		fmt.Printf("   ❌ Parallel processing failed: %v\n", err)
	} else {
		_ = cuda.Synchronize()
		fmt.Printf("   ✅ Processed %d elements in parallel\n", processed.Load())
	}
}

func demoPerformancePatterns() {
	fmt.Println("   Demonstrating performance measurement patterns...")

	// Create test data
	size := 1000
	data := make([]float64, size)
	for i := 0; i < size; i++ {
		data[i] = float64(i)
	}

	// CPU timing
	fmt.Printf("   🖥️  CPU computation (%d elements)...\n", size)
	startCPU := time.Now()
	cpuResult := make([]float64, size)
	for i := 0; i < size; i++ {
		cpuResult[i] = data[i] * data[i]
	}
	cpuTime := time.Since(startCPU)

	// GPU timing (using existing Map function)
	fmt.Printf("   🚀 GPU computation (%d elements)...\n", size)
	startGPU := time.Now()
	gpuResult, err := cuda.Map(data, func(x float64) float64 {
		return x * x
	})
	if err == nil {
		err = cuda.Synchronize()
	}
	gpuTime := time.Since(startGPU)

	if err != nil {
		fmt.Printf("   ⚠️  GPU computation failed: %v\n", err)
		fmt.Printf("   ⏱️  CPU time: %v\n", cpuTime)
		return
	}

	// Verify results match
	matches := len(cpuResult) == len(gpuResult)
	if matches {
		for i := 0; i < len(cpuResult) && i < len(gpuResult); i++ {
			if cpuResult[i] != gpuResult[i] {
				matches = false
				break
			}
		}
	}

	// Display results
	fmt.Printf("   ⏱️  CPU time: %v\n", cpuTime)
	fmt.Printf("   ⏱️  GPU time: %v\n", gpuTime)

	if matches {
		fmt.Println("   ✅ Results match between CPU and GPU")
	} else {
		fmt.Println("   ⚠️  Results differ")
	}

	if cpuTime <= 0 || gpuTime <= 0 {
		fmt.Println("   ℹ️  Relative speed comparison skipped because one timing was below timer resolution")
		return
	}

	if gpuTime < cpuTime {
		speedup := float64(cpuTime) / float64(gpuTime)
		fmt.Printf("   🚀 GPU is %.2fx faster than CPU\n", speedup)
	} else {
		slowdown := float64(gpuTime) / float64(cpuTime)
		fmt.Printf("   🐌 GPU is %.2fx slower (overhead for small data)\n", slowdown)
		fmt.Println("   💡 GPU benefits increase with larger datasets")
	}
}
