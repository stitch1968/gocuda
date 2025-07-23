// Advanced Features Demo - Week 7-8 Implementation
// Demonstrates memory optimization, kernel fusion, async pipeline, and multi-GPU support
package main

import (
	"fmt"
	"log"
	"math/rand"
	"time"

	"github.com/stitch1968/gocuda/memory"
	"github.com/stitch1968/gocuda/performance"
)

func main() {
	fmt.Println("🚀 GoCUDA Week 7-8: Advanced Features Demo")
	fmt.Println("==========================================")

	// Demo 1: Memory Bandwidth Optimization
	fmt.Println("\n1. Memory Bandwidth Optimization Demo")
	demoMemoryOptimization()

	// Demo 2: Kernel Fusion
	fmt.Println("\n2. Kernel Fusion Demo")
	demoKernelFusion()

	// Demo 3: Async Pipeline
	fmt.Println("\n3. Async Pipeline Demo")
	demoAsyncPipeline()

	// Demo 4: Multi-GPU Support
	fmt.Println("\n4. Multi-GPU Support Demo")
	demoMultiGPU()

	fmt.Println("\n🎉 All advanced features demonstrated successfully!")
}

// demoMemoryOptimization demonstrates memory bandwidth optimization
func demoMemoryOptimization() {
	fmt.Println("   Analyzing memory access patterns...")

	// Create memory bandwidth optimizer
	optimizer := performance.NewMemoryBandwidthOptimizer(0)

	// Allocate test memory
	testSize := 1024 * 1024 // 1M elements
	testMem, err := memory.Alloc(int64(testSize * 4))
	if err != nil {
		log.Printf("   ❌ Failed to allocate memory: %v", err)
		return
	}
	defer testMem.Free()

	// Analyze different access patterns
	patterns := []string{"sequential", "strided", "random"}

	for _, pattern := range patterns {
		fmt.Printf("   📊 Analyzing %s access pattern...\n", pattern)

		analysis, err := optimizer.AnalyzeAccessPattern(testMem, testSize, pattern)
		if err != nil {
			log.Printf("   ❌ Pattern analysis failed: %v", err)
			continue
		}

		fmt.Printf("      Coalescing Efficiency: %.1f%%\n", analysis.CoalescingEfficiency)
		fmt.Printf("      Bank Conflicts: %d\n", analysis.BankConflicts)
		fmt.Printf("      Cache Hit Rate: %.1f%%\n", analysis.CacheHitRate)

		if len(analysis.Suggestions) > 0 {
			fmt.Println("      Optimization Suggestions:")
			for _, suggestion := range analysis.Suggestions {
				fmt.Printf("      • %s\n", suggestion)
			}
		}

		// Benchmark this pattern
		throughput, err := optimizer.BenchmarkMemoryPattern(testMem, pattern, 100)
		if err != nil {
			log.Printf("   ❌ Benchmark failed: %v", err)
		} else {
			fmt.Printf("      Throughput: %.2f ops/sec\n", throughput)
		}
	}

	// Generate optimization report
	report := optimizer.GetOptimizationReport()
	fmt.Printf("   📋 Summary: %s\n", report.String())
}

// demoKernelFusion demonstrates kernel fusion capabilities
func demoKernelFusion() {
	fmt.Println("   Creating fused kernels for better performance...")

	// Create kernel fusion manager
	fusionManager := performance.NewKernelFusion(0)

	// Allocate test vectors
	vectorSize := 1024 * 512
	x, err := memory.Alloc(int64(vectorSize * 4))
	if err != nil {
		log.Printf("   ❌ Failed to allocate vector X: %v", err)
		return
	}
	defer x.Free()

	y, err := memory.Alloc(int64(vectorSize * 4))
	if err != nil {
		log.Printf("   ❌ Failed to allocate vector Y: %v", err)
		return
	}
	defer y.Free()

	z, err := memory.Alloc(int64(vectorSize * 4))
	if err != nil {
		log.Printf("   ❌ Failed to allocate vector Z: %v", err)
		return
	}
	defer z.Free()

	result, err := memory.Alloc(int64(vectorSize * 4))
	if err != nil {
		log.Printf("   ❌ Failed to allocate result vector: %v", err)
		return
	}
	defer result.Free()

	// Demo 1: Fused SAXPY + Add operation
	fmt.Println("   🔥 Executing fused SAXPY + Add: result = 2.5 * x + y + z")
	start := time.Now()
	err = fusionManager.FusedVectorSaxpyAdd(2.5, x, y, z, result, vectorSize)
	duration := time.Since(start)
	if err != nil {
		log.Printf("   ❌ Fused SAXPY+Add failed: %v", err)
	} else {
		fmt.Printf("   ✅ Fused operation completed in %v\n", duration)
	}

	// Demo 2: Fused matrix operations
	matSize := 512
	A, _ := memory.Alloc(int64(matSize * matSize * 4))
	defer A.Free()
	B, _ := memory.Alloc(int64(matSize * matSize * 4))
	defer B.Free()
	C, _ := memory.Alloc(int64(matSize * matSize * 4))
	defer C.Free()
	D, _ := memory.Alloc(int64(matSize * matSize * 4))
	defer D.Free()

	fmt.Printf("   🔥 Executing fused GEMM + Add: D = A * B + C (%dx%d)\n", matSize, matSize)
	start = time.Now()
	err = fusionManager.FusedMatrixMultiplyAdd(A, B, C, D, matSize, matSize, matSize)
	duration = time.Since(start)
	if err != nil {
		log.Printf("   ❌ Fused GEMM+Add failed: %v", err)
	} else {
		fmt.Printf("   ✅ Fused matrix operation completed in %v\n", duration)
	}

	// Demo 3: Custom fused elementwise operations
	inputs := []*memory.Memory{x, y, z}
	operations := []string{"add", "multiply", "relu"}

	fmt.Println("   🔥 Executing fused elementwise operations: Add + Multiply + ReLU")
	start = time.Now()
	err = fusionManager.FusedElementwiseOperations(inputs, result, operations, vectorSize)
	duration = time.Since(start)
	if err != nil {
		log.Printf("   ❌ Fused elementwise failed: %v", err)
	} else {
		fmt.Printf("   ✅ Fused elementwise operations completed in %v\n", duration)
	}

	// Get performance report
	report := fusionManager.GetPerformanceReport()
	fmt.Printf("   📊 Fusion Report: %s\n", report.String())

	// Demo fusion optimization
	testOps := []performance.Operation{
		{Type: 0, Priority: 3}, // VectorAdd
		{Type: 1, Priority: 2}, // VectorScale
		{Type: 0, Priority: 1}, // VectorAdd
		{Type: 2, Priority: 1}, // VectorMultiply
	}

	strategy := fusionManager.OptimizeFusionStrategy(testOps)
	fmt.Printf("   🎯 Fusion Strategy: %.2fx speedup expected\n", strategy.EstimatedSpeedup)
	for _, reasoning := range strategy.Reasoning {
		fmt.Printf("      • %s\n", reasoning)
	}
}

// demoAsyncPipeline demonstrates async execution pipeline
func demoAsyncPipeline() {
	fmt.Println("   Setting up async pipeline with overlapped compute and transfers...")

	// Create async pipeline with 3 streams
	pipeline := performance.NewAsyncPipeline(3)

	// Define pipeline stages
	stages := []performance.PipelineStage{
		{
			Name:         "data_preprocessing",
			Compute:      preprocessingStage,
			Dependencies: []string{},
			InputSize:    1024 * 1024,
			OutputSize:   1024 * 1024,
			StreamID:     0,
			Priority:     3,
		},
		{
			Name:         "main_computation",
			Compute:      computationStage,
			Dependencies: []string{"data_preprocessing"},
			InputSize:    1024 * 1024,
			OutputSize:   1024 * 1024,
			StreamID:     1,
			Priority:     2,
		},
		{
			Name:         "postprocessing",
			Compute:      postprocessingStage,
			Dependencies: []string{"main_computation"},
			InputSize:    1024 * 1024,
			OutputSize:   512 * 1024, // Reduced output size
			StreamID:     2,
			Priority:     1,
		},
	}

	// Add stages to pipeline
	for _, stage := range stages {
		err := pipeline.AddStage(stage)
		if err != nil {
			log.Printf("   ❌ Failed to add stage %s: %v", stage.Name, err)
			return
		}
		fmt.Printf("   ✅ Added pipeline stage: %s\n", stage.Name)
	}

	// Prepare input data
	inputSize := 1024 * 1024
	inputMem, err := memory.Alloc(int64(inputSize * 4))
	if err != nil {
		log.Printf("   ❌ Failed to allocate input memory: %v", err)
		return
	}
	defer inputMem.Free()

	// Execute pipeline
	fmt.Println("   🚀 Executing async pipeline...")
	start := time.Now()
	resultChan := pipeline.Execute(inputMem, map[string]interface{}{
		"processing_mode": "optimized",
		"batch_size":      1024,
	})

	// Collect results
	stageCount := 0
	for result := range resultChan {
		stageCount++
		if result.Error != nil {
			log.Printf("   ❌ Stage %s failed: %v", result.StageID, result.Error)
		} else {
			fmt.Printf("   ✅ Stage %s completed successfully\n", result.StageID)
		}
	}

	totalDuration := time.Since(start)
	fmt.Printf("   📊 Pipeline executed %d stages in %v\n", stageCount, totalDuration)

	// Get pipeline metrics
	metrics := pipeline.GetMetrics()
	fmt.Printf("   📈 Pipeline Metrics: %s\n", metrics.String())

	// Get optimization suggestions
	suggestions := pipeline.OptimizePipeline()
	fmt.Printf("   💡 Optimization Suggestions (%.2fx improvement potential):\n", suggestions.EstimatedImprovement)
	for _, suggestion := range suggestions.Suggestions {
		fmt.Printf("      • %s\n", suggestion)
	}

	pipeline.Stop()
}

// demoMultiGPU demonstrates multi-GPU support
func demoMultiGPU() {
	fmt.Println("   Initializing multi-GPU system...")

	// Create multi-GPU manager
	multiGPU, err := performance.NewMultiGPU()
	if err != nil {
		log.Printf("   ❌ Failed to initialize multi-GPU: %v", err)
		return
	}

	// Get device information
	devices := multiGPU.GetDeviceInfo()
	fmt.Printf("   🖥️  Detected %d GPU(s):\n", len(devices))
	for _, device := range devices {
		fmt.Printf("      • %s (ID: %d, Memory: %.1f GB, SMs: %d)\n",
			device.Name, device.ID,
			float64(device.TotalMemory)/1024/1024/1024,
			device.MultiProcessors)
	}

	// Enable P2P communication
	fmt.Println("   🔗 Enabling peer-to-peer communication...")
	err = multiGPU.EnableP2P()
	if err != nil {
		log.Printf("   ⚠️  P2P setup failed: %v", err)
	} else {
		fmt.Println("   ✅ P2P communication enabled")
	}

	// Generate test data
	dataSize := 1024 * 1024 * 4 // 4M elements
	testData := make([]float32, dataSize)
	for i := range testData {
		testData[i] = rand.Float32()
	}

	fmt.Printf("   📊 Distributing %.1f MB of data across GPUs...\n", float64(len(testData)*4)/1024/1024)

	// Distribute data with different strategies
	strategies := []performance.DistributionStrategy{
		performance.EvenSplit,
		performance.WeightedByCapability,
		performance.WeightedByMemory,
	}
	strategyNames := []string{"Even Split", "Capability Weighted", "Memory Weighted"}

	for i, strategy := range strategies {
		fmt.Printf("   🎯 Testing %s distribution strategy...\n", strategyNames[i])

		start := time.Now()
		err = multiGPU.DistributeData(testData, strategy)
		distributionTime := time.Since(start)

		if err != nil {
			log.Printf("   ❌ Data distribution failed: %v", err)
			continue
		}

		fmt.Printf("   ✅ Data distributed in %v\n", distributionTime)

		// Create a test kernel for parallel execution
		testKernel := &performance.MultiGPUKernel{
			Name: "vector_processing",
			Implementation: func(deviceID int, input *memory.Memory, params map[string]interface{}) (*memory.Memory, error) {
				// Simulate compute work
				time.Sleep(10 * time.Millisecond)

				// Return processed result (same size for simplicity)
				result, err := memory.Alloc(input.Size())
				return result, err
			},
			MemoryRequired: func(inputSize int64) int64 {
				return inputSize * 2 // Need 2x input size
			},
		}

		// Execute kernel on all GPUs
		fmt.Printf("   🚀 Executing kernel '%s' on all GPUs...\n", testKernel.GetName())
		execStart := time.Now()
		results, err := multiGPU.ParallelExecute(testKernel, map[string]interface{}{
			"algorithm": "optimized",
			"precision": "float32",
		})
		execTime := time.Since(execStart)

		if err != nil {
			log.Printf("   ❌ Parallel execution failed: %v", err)
			continue
		}

		fmt.Printf("   ✅ Parallel execution completed in %v\n", execTime)
		fmt.Printf("   📊 Results from %d devices\n", len(results))

		// Gather results back to host
		fmt.Println("   📥 Gathering results from all devices...")
		gatherStart := time.Now()
		finalResults, err := multiGPU.GatherResults(results)
		gatherTime := time.Since(gatherStart)

		if err != nil {
			log.Printf("   ❌ Result gathering failed: %v", err)
		} else {
			fmt.Printf("   ✅ Gathered %.1f MB of results in %v\n",
				float64(len(finalResults)*4)/1024/1024, gatherTime)
		}

		// Clean up device results
		for _, result := range results {
			result.Free()
		}

		totalTime := distributionTime + execTime + gatherTime
		fmt.Printf("   ⏱️  Total processing time: %v\n", totalTime)
	}

	// Get multi-GPU metrics
	metrics := multiGPU.GetMetrics()
	fmt.Printf("   📈 Multi-GPU Metrics: %s\n", metrics.String())

	fmt.Println("   🎯 Multi-GPU demo completed successfully!")
}

// Pipeline stage implementations

func preprocessingStage(input *memory.Memory, params map[string]interface{}) (*memory.Memory, error) {
	// Simulate preprocessing work
	time.Sleep(5 * time.Millisecond)

	// Return processed data (same size for simplicity)
	result, err := memory.Alloc(input.Size())
	if err != nil {
		return nil, fmt.Errorf("preprocessing allocation failed: %v", err)
	}

	return result, nil
}

func computationStage(input *memory.Memory, params map[string]interface{}) (*memory.Memory, error) {
	// Simulate main computation work
	time.Sleep(15 * time.Millisecond)

	result, err := memory.Alloc(input.Size())
	if err != nil {
		return nil, fmt.Errorf("computation allocation failed: %v", err)
	}

	return result, nil
}

func postprocessingStage(input *memory.Memory, params map[string]interface{}) (*memory.Memory, error) {
	// Simulate postprocessing (output is half the size)
	time.Sleep(3 * time.Millisecond)

	result, err := memory.Alloc(input.Size() / 2)
	if err != nil {
		return nil, fmt.Errorf("postprocessing allocation failed: %v", err)
	}

	return result, nil
}
