package tests

import (
	"testing"
	"time"

	"github.com/stitch1968/gocuda/memory"
	"github.com/stitch1968/gocuda/profiler"
	"github.com/stitch1968/gocuda/streams"
)

// TestProfilerBasicFunctionality tests basic profiler operations
func TestProfilerBasicFunctionality(t *testing.T) {
	t.Log("Testing basic profiler functionality...")

	// Enable profiling
	profiler.Enable()
	defer profiler.Disable()

	p := profiler.GetProfiler()
	if p == nil {
		t.Fatal("Failed to get profiler instance")
	}

	if !p.IsEnabled() {
		t.Error("Expected profiler to be enabled")
	}

	// Clear any existing events
	p.Clear()

	// Record some events
	profiler.StartEvent("memory_allocation")
	mem, err := memory.Alloc(1024)
	if err != nil {
		t.Fatalf("Memory allocation failed: %v", err)
	}
	profiler.EndEvent("memory_allocation", profiler.EventMemoryAlloc)
	defer mem.Free()

	profiler.StartEvent("stream_creation")
	stream, err := streams.CreateStream()
	if err != nil {
		t.Fatalf("Stream creation failed: %v", err)
	}
	profiler.EndEvent("stream_creation", profiler.EventStream)
	defer streams.GetManager().DestroyStream(stream)

	// Get profiling results
	events := p.GetEvents()
	if len(events) < 2 {
		t.Errorf("Expected at least 2 events, got %d", len(events))
	}

	t.Log("✅ Basic profiler functionality test passed")
}

// TestProfilerTimeMeasurement tests time measurement
func TestProfilerTimeMeasurement(t *testing.T) {
	t.Log("Testing profiler time measurement...")

	profiler.Enable()
	defer profiler.Disable()

	p := profiler.GetProfiler()
	p.Clear()

	// Measure a known sleep duration
	sleepDuration := 50 * time.Millisecond
	profiler.StartEvent("sleep_test")
	time.Sleep(sleepDuration)
	profiler.EndEvent("sleep_test", profiler.EventOther)

	events := p.GetEvents()
	if len(events) != 1 {
		t.Errorf("Expected 1 event, got %d", len(events))
	}

	if len(events) > 0 {
		measuredDuration := events[0].Duration
		tolerance := 20 * time.Millisecond // Allow some tolerance

		if measuredDuration < sleepDuration-tolerance || measuredDuration > sleepDuration+tolerance {
			t.Errorf("Time measurement inaccurate: expected ~%v, got %v", sleepDuration, measuredDuration)
		}
	}

	t.Log("✅ Profiler time measurement test passed")
}

// TestProfilerMemoryTracking tests memory usage tracking
func TestProfilerMemoryTracking(t *testing.T) {
	t.Log("Testing profiler memory tracking...")

	profiler.Enable()
	defer profiler.Disable()

	p := profiler.GetProfiler()
	p.Clear()

	// Track memory allocations
	var memories []*memory.Memory
	for i := 0; i < 5; i++ {
		mem, err := memory.Alloc(1024)
		if err != nil {
			t.Fatalf("Memory allocation %d failed: %v", i, err)
		}
		// Track allocation manually
		p.TrackAllocation(uintptr(mem.Ptr()), mem.Size())
		memories = append(memories, mem)
	}

	// Check current memory usage
	current, peak := p.GetMemoryUsage()
	if current == 0 {
		t.Error("Expected non-zero current memory usage")
	}
	if peak == 0 {
		t.Error("Expected non-zero peak memory usage")
	}

	// Clean up
	for _, mem := range memories {
		p.TrackDeallocation(uintptr(mem.Ptr()))
		mem.Free()
	}

	t.Log("✅ Profiler memory tracking test passed")
}

// TestProfilerStatistics tests statistics generation
func TestProfilerStatistics(t *testing.T) {
	t.Log("Testing profiler statistics...")

	profiler.Enable()
	defer profiler.Disable()

	p := profiler.GetProfiler()
	p.Clear()

	// Create some events
	profiler.StartEvent("fast_operation")
	time.Sleep(1 * time.Millisecond)
	profiler.EndEvent("fast_operation", profiler.EventOther)

	profiler.StartEvent("slow_operation")
	time.Sleep(10 * time.Millisecond)
	profiler.EndEvent("slow_operation", profiler.EventOther)

	// Get statistics
	stats := p.GetStatistics()

	if stats.TotalEvents < 2 {
		t.Errorf("Expected at least 2 total events, got %d", stats.TotalEvents)
	}

	if stats.TotalDuration == 0 {
		t.Error("Expected non-zero total duration")
	}

	if stats.AverageDuration == 0 {
		t.Error("Expected non-zero average duration")
	}

	t.Log("✅ Profiler statistics test passed")
}

// TestProfileKernel tests kernel profiling helper
func TestProfileKernel(t *testing.T) {
	t.Log("Testing kernel profiling...")

	profiler.Enable()
	defer profiler.Disable()

	p := profiler.GetProfiler()
	p.Clear()

	// Profile a kernel-like operation
	kernelExecuted := false
	err := profiler.ProfileKernel("test_kernel", func() {
		time.Sleep(5 * time.Millisecond)
		kernelExecuted = true
	})

	if err != nil {
		t.Errorf("Kernel profiling failed: %v", err)
	}

	if !kernelExecuted {
		t.Error("Kernel function was not executed")
	}

	events := p.GetEvents()
	if len(events) != 1 {
		t.Errorf("Expected 1 event, got %d", len(events))
	}

	if len(events) > 0 && events[0].Type != profiler.EventKernel {
		t.Error("Expected kernel event type")
	}

	t.Log("✅ Kernel profiling test passed")
}

// TestKernelProfiles tests kernel profile generation
func TestKernelProfiles(t *testing.T) {
	t.Log("Testing kernel profiles...")

	profiler.Enable()
	defer profiler.Disable()

	p := profiler.GetProfiler()
	p.Clear()

	// Create multiple kernel events
	for i := 0; i < 3; i++ {
		profiler.ProfileKernel("test_kernel", func() {
			time.Sleep(2 * time.Millisecond)
		})
	}

	// Get kernel profiles
	profiles := p.GetKernelProfiles()
	if len(profiles) == 0 {
		t.Error("Expected non-empty kernel profiles")
	}

	// Should find our test kernel
	found := false
	for _, profile := range profiles {
		if profile.Name == "test_kernel" {
			found = true
			if profile.CallCount != 3 {
				t.Errorf("Expected 3 calls, got %d", profile.CallCount)
			}
			break
		}
	}

	if !found {
		t.Error("test_kernel profile not found")
	}

	t.Log("✅ Kernel profiles test passed")
}

// TestProfilerEnableDisable tests enable/disable functionality
func TestProfilerEnableDisable(t *testing.T) {
	t.Log("Testing profiler enable/disable...")

	p := profiler.GetProfiler()

	// Test disable
	profiler.Disable()
	if p.IsEnabled() {
		t.Error("Expected profiler to be disabled")
	}

	// Test enable
	profiler.Enable()
	if !p.IsEnabled() {
		t.Error("Expected profiler to be enabled")
	}

	// Clean up
	profiler.Disable()

	t.Log("✅ Profiler enable/disable test passed")
}

// TestProfilerDisabledMode tests that profiler works when disabled
func TestProfilerDisabledMode(t *testing.T) {
	t.Log("Testing profiler disabled mode...")

	p := profiler.GetProfiler()
	p.Clear() // Clear any previous events
	profiler.Disable()

	// Operations should work but not record anything
	err := profiler.ProfileKernel("disabled_kernel", func() {
		time.Sleep(1 * time.Millisecond)
	})

	if err != nil {
		t.Errorf("Kernel profiling failed when disabled: %v", err)
	}

	events := p.GetEvents()

	// Should have no events when disabled (check after clearing)
	if len(events) > 0 {
		t.Errorf("Expected no events when disabled, got %d", len(events))
	}

	t.Log("✅ Profiler disabled mode test passed")
} // BenchmarkProfilerOverhead benchmarks profiler overhead
func BenchmarkProfilerOverhead(b *testing.B) {
	profiler.Enable()
	defer profiler.Disable()

	for i := 0; i < b.N; i++ {
		profiler.StartEvent("benchmark_event")
		// Minimal work
		_ = i * 2
		profiler.EndEvent("benchmark_event", profiler.EventOther)
	}
}

// BenchmarkProfilerWithoutProfiling benchmarks without profiling
func BenchmarkProfilerWithoutProfiling(b *testing.B) {
	profiler.Disable()

	for i := 0; i < b.N; i++ {
		// Same minimal work as above
		_ = i * 2
	}
}
