package tests

import (
	"testing"

	"github.com/stitch1968/gocuda/advanced"
	"github.com/stitch1968/gocuda/kernels"
	"github.com/stitch1968/gocuda/memory"
	"github.com/stitch1968/gocuda/profiler"
	"github.com/stitch1968/gocuda/streams"
)

// TestIntegrationWorkflow tests a complete GoCUDA workflow
func TestIntegrationWorkflow(t *testing.T) {
	t.Log("Testing complete GoCUDA integration workflow...")

	// Enable profiling for the entire workflow
	profiler.Enable()
	defer profiler.Disable()

	// Step 1: Create stream for asynchronous operations
	profiler.StartEvent("stream_creation")
	stream, err := streams.CreateStream()
	if err != nil {
		t.Fatalf("Failed to create stream: %v", err)
	}
	defer streams.GetManager().DestroyStream(stream)
	profiler.EndEvent("stream_creation", profiler.EventStream)

	// Step 2: Allocate memory
	profiler.StartEvent("memory_allocation")
	mem1, err := memory.AllocWithStream(stream.Stream, 1024*4) // For 1024 floats
	if err != nil {
		t.Fatalf("Failed to allocate memory 1: %v", err)
	}
	defer mem1.Free()

	mem2, err := memory.AllocWithStream(stream.Stream, 1024*4)
	if err != nil {
		t.Fatalf("Failed to allocate memory 2: %v", err)
	}
	defer mem2.Free()

	result, err := memory.AllocWithStream(stream.Stream, 1024*4)
	if err != nil {
		t.Fatalf("Failed to allocate result memory: %v", err)
	}
	defer result.Free()
	profiler.EndEvent("memory_allocation", profiler.EventMemoryAlloc)

	// Step 3: Initialize data
	profiler.StartEvent("data_initialization")
	data1 := (*[1 << 30]float32)(mem1.Ptr())[:1024:1024]
	data2 := (*[1 << 30]float32)(mem2.Ptr())[:1024:1024]
	for i := 0; i < 1024; i++ {
		data1[i] = float32(i)
		data2[i] = float32(i * 2)
	}
	profiler.EndEvent("data_initialization", profiler.EventOther)

	// Step 4: Execute kernel operation
	profiler.StartEvent("kernel_execution")
	vectorAdd := &kernels.VectorAdd{}
	gridDim := kernels.Dim3{X: 1, Y: 1, Z: 1}
	blockDim := kernels.Dim3{X: 1024, Y: 1, Z: 1}

	err = vectorAdd.Execute(gridDim, blockDim, 0, stream.Stream, mem1, mem2, result, 1024)
	if err != nil {
		t.Fatalf("Vector add execution failed: %v", err)
	}
	profiler.EndEvent("kernel_execution", profiler.EventKernel)

	// Step 5: Advanced operations - sorting
	profiler.StartEvent("advanced_sort")
	sortData, err := memory.Alloc(1024 * 4)
	if err != nil {
		t.Fatalf("Failed to allocate sort data: %v", err)
	}
	defer sortData.Free()

	// Initialize with reverse data
	sortArray := (*[1 << 30]uint32)(sortData.Ptr())[:1024:1024]
	for i := 0; i < 1024; i++ {
		sortArray[i] = uint32(1024 - i)
	}

	err = advanced.RadixSort(sortData, 1024)
	if err != nil {
		t.Errorf("RadixSort failed: %v", err)
	}
	profiler.EndEvent("advanced_sort", profiler.EventOther)

	// Step 6: FFT operation
	profiler.StartEvent("fft_operation")
	fftSize := 512
	complexSize := int64(fftSize * 8) // 2 floats per complex number

	fftInput, err := memory.Alloc(complexSize)
	if err != nil {
		t.Fatalf("Failed to allocate FFT input: %v", err)
	}
	defer fftInput.Free()

	fftOutput, err := memory.Alloc(complexSize)
	if err != nil {
		t.Fatalf("Failed to allocate FFT output: %v", err)
	}
	defer fftOutput.Free()

	// Initialize FFT input
	fftData := (*[1 << 30]advanced.Complex64)(fftInput.Ptr())[:fftSize:fftSize]
	for i := 0; i < fftSize; i++ {
		fftData[i] = advanced.Complex64{Real: float32(i), Imag: 0.0}
	}

	err = advanced.FFT(fftInput, fftOutput, fftSize, false)
	if err != nil {
		t.Errorf("FFT failed: %v", err)
	}
	profiler.EndEvent("fft_operation", profiler.EventOther)

	// Step 7: Synchronize stream
	profiler.StartEvent("stream_sync")
	err = stream.Synchronize()
	if err != nil {
		t.Errorf("Stream synchronization failed: %v", err)
	}
	profiler.EndEvent("stream_sync", profiler.EventStream)

	// Step 8: Validate results
	profiler.StartEvent("result_validation")
	resultData := (*[1 << 30]float32)(result.Ptr())[:1024:1024]

	// Verify vector addition results
	for i := 0; i < 10; i++ { // Check first 10 elements
		expected := float32(i + i*2) // i + 2*i = 3*i
		if resultData[i] != expected {
			t.Errorf("Vector add result mismatch at index %d: expected %f, got %f", i, expected, resultData[i])
		}
	}

	// Verify sorting results (check first few elements)
	for i := 1; i < 10; i++ {
		if sortArray[i] < sortArray[i-1] {
			t.Errorf("Sort result not in order at index %d: %d > %d", i, sortArray[i-1], sortArray[i])
		}
	}
	profiler.EndEvent("result_validation", profiler.EventOther)

	// Step 9: Generate profiling report
	p := profiler.GetProfiler()
	stats := p.GetStatistics()

	t.Logf("Integration workflow completed successfully!")
	t.Logf("Total operations: %d", stats.TotalEvents)
	t.Logf("Total execution time: %v", stats.TotalDuration)
	t.Logf("Average operation time: %v", stats.AverageDuration)

	// Verify we have operations of different types
	if len(stats.EventsByType) == 0 {
		t.Error("Expected multiple event types in profiling stats")
	}

	t.Log("✅ Complete GoCUDA integration workflow test passed")
}

// TestConcurrentOperations tests concurrent operations across multiple streams
func TestConcurrentOperations(t *testing.T) {
	t.Log("Testing concurrent operations...")

	profiler.Enable()
	defer profiler.Disable()

	numStreams := 3
	streamList := make([]*streams.Stream, numStreams)
	memories := make([]*memory.Memory, numStreams)

	// Create multiple streams and memories
	for i := 0; i < numStreams; i++ {
		stream, err := streams.CreateStream()
		if err != nil {
			t.Fatalf("Failed to create stream %d: %v", i, err)
		}
		streamList[i] = stream

		mem, err := memory.AllocWithStream(stream.Stream, 1024*4)
		if err != nil {
			t.Fatalf("Failed to allocate memory for stream %d: %v", i, err)
		}
		memories[i] = mem
	}

	// Initialize data in each memory buffer
	for i, mem := range memories {
		data := (*[1 << 30]float32)(mem.Ptr())[:1024:1024]
		for j := 0; j < 1024; j++ {
			data[j] = float32(j * (i + 1))
		}
	}

	// Execute vector add operations on each stream concurrently
	vectorAdd := &kernels.VectorAdd{}
	gridDim := kernels.Dim3{X: 1, Y: 1, Z: 1}
	blockDim := kernels.Dim3{X: 1024, Y: 1, Z: 1}

	// Execute operations on different streams
	for i, stream := range streamList {
		if i < len(memories)-1 {
			// Use adjacent memory buffers for input/output
			err := vectorAdd.Execute(gridDim, blockDim, 0, stream.Stream,
				memories[i], memories[i], memories[i], 1024)
			if err != nil {
				t.Errorf("Vector add execution failed on stream %d: %v", i, err)
			}
		}
	}

	// Synchronize all streams
	manager := streams.GetManager()
	err := manager.SynchronizeAll()
	if err != nil {
		t.Errorf("Failed to synchronize all streams: %v", err)
	}

	// Cleanup
	for i, stream := range streamList {
		memories[i].Free()
		manager.DestroyStream(stream)
	}

	t.Log("✅ Concurrent operations test passed")
} // TestMemoryPoolingWorkflow tests memory pooling and reuse
func TestMemoryPoolingWorkflow(t *testing.T) {
	t.Log("Testing memory pooling workflow...")

	profiler.Enable()
	defer profiler.Disable()

	manager := memory.GetManager()
	initialAllocated := manager.GetTotalAllocated()

	// Allocate, use, and free memory multiple times
	var memories []*memory.Memory

	// First allocation batch
	for i := 0; i < 5; i++ {
		mem, err := memory.Alloc(1024)
		if err != nil {
			t.Fatalf("Allocation %d failed: %v", i, err)
		}
		memories = append(memories, mem)
	}

	midAllocated := manager.GetTotalAllocated()
	if midAllocated <= initialAllocated {
		t.Error("Expected memory usage to increase")
	}

	// Free first batch
	for _, mem := range memories {
		mem.Free()
	}
	memories = nil

	// Second allocation batch (should potentially reuse memory)
	for i := 0; i < 3; i++ {
		mem, err := memory.Alloc(2048) // Different size
		if err != nil {
			t.Fatalf("Second allocation %d failed: %v", i, err)
		}
		memories = append(memories, mem)
	}

	// Cleanup
	for _, mem := range memories {
		mem.Free()
	}

	// Check final memory state
	finalCount := manager.GetAllocationCount()
	t.Logf("Memory operations completed - final allocation count: %d", finalCount)

	t.Log("✅ Memory pooling workflow test passed")
}

// BenchmarkIntegrationWorkflow benchmarks the complete workflow
func BenchmarkIntegrationWorkflow(b *testing.B) {
	profiler.Disable() // Disable profiling for benchmark

	for i := 0; i < b.N; i++ {
		// Simplified workflow
		stream, _ := streams.CreateStream()
		mem1, _ := memory.AllocWithStream(stream.Stream, 1024*4)
		mem2, _ := memory.AllocWithStream(stream.Stream, 1024*4)
		result, _ := memory.AllocWithStream(stream.Stream, 1024*4)

		// Quick kernel execution
		vectorAdd := &kernels.VectorAdd{}
		gridDim := kernels.Dim3{X: 1, Y: 1, Z: 1}
		blockDim := kernels.Dim3{X: 1024, Y: 1, Z: 1}
		vectorAdd.Execute(gridDim, blockDim, 0, stream.Stream, mem1, mem2, result, 1024)

		// Cleanup
		result.Free()
		mem2.Free()
		mem1.Free()
		streams.GetManager().DestroyStream(stream)
	}
}
