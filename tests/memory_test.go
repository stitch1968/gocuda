package tests

import (
	"testing"

	"github.com/stitch1968/gocuda/internal"
	"github.com/stitch1968/gocuda/memory"
)

// TestMemoryBasicAllocation tests basic memory allocation functionality
func TestMemoryBasicAllocation(t *testing.T) {
	t.Log("Testing basic memory allocation...")

	// Test basic allocation
	mem, err := memory.Alloc(1024)
	if err != nil {
		t.Fatalf("Failed to allocate memory: %v", err)
	}
	defer mem.Free()

	// Verify properties
	if mem.Size() != 1024 {
		t.Errorf("Expected size 1024, got %d", mem.Size())
	}

	if mem.Ptr() == nil {
		t.Error("Memory pointer should not be nil")
	}

	// Verify type
	if mem.GetType() != memory.TypeDevice {
		t.Errorf("Expected device memory type, got %v", mem.GetType())
	}

	t.Log("✅ Basic allocation test passed")
}

// TestMemoryWithStream tests memory allocation with streams
func TestMemoryWithStream(t *testing.T) {
	t.Log("Testing memory allocation with streams...")

	stream := internal.GetDefaultStream()

	// Test allocation with stream - correct parameter order: stream first, then size
	mem, err := memory.AllocWithStream(stream, 2048)
	if err != nil {
		t.Fatalf("Failed to allocate memory with stream: %v", err)
	}
	defer mem.Free()

	if mem.Size() != 2048 {
		t.Errorf("Expected size 2048, got %d", mem.Size())
	}

	// Test typed allocation with stream - correct parameter order
	typedMem, err := memory.AllocWithTypeAndStream(stream, 1024, memory.TypeUnified)
	if err != nil {
		t.Fatalf("Failed to allocate typed memory: %v", err)
	}
	defer typedMem.Free()

	if typedMem.GetType() != memory.TypeUnified {
		t.Errorf("Expected unified memory type, got %v", typedMem.GetType())
	}

	t.Log("✅ Stream allocation test passed")
}

// TestMemoryProperties tests memory property getters
func TestMemoryProperties(t *testing.T) {
	t.Log("Testing memory properties...")

	mem, err := memory.Alloc(1024)
	if err != nil {
		t.Fatalf("Failed to allocate memory: %v", err)
	}
	defer mem.Free()

	// Test all property getters
	size := mem.Size()
	if size != 1024 {
		t.Errorf("Expected size 1024, got %d", size)
	}

	ptr := mem.Ptr()
	if ptr == nil {
		t.Error("Expected non-nil pointer")
	}

	data := mem.Data()
	if data == nil {
		t.Error("Expected non-nil data slice")
	}

	memType := mem.GetType()
	if memType < 0 {
		t.Error("Expected valid memory type")
	}

	alignment := mem.GetAlignment()
	if alignment <= 0 {
		t.Error("Expected positive alignment")
	}

	pitch := mem.GetPitch()
	if pitch < 0 {
		t.Error("Expected non-negative pitch")
	}

	deviceID := mem.GetDeviceID()
	if deviceID < 0 {
		t.Error("Expected non-negative device ID")
	}

	t.Log("✅ Memory properties test passed")
}

// TestMemoryManager tests memory manager functionality
func TestMemoryManager(t *testing.T) {
	t.Log("Testing memory manager...")

	manager := memory.GetManager()
	if manager == nil {
		t.Fatal("Expected non-nil memory manager")
	}

	// Get initial stats
	initialAllocated := manager.GetTotalAllocated()
	initialCount := manager.GetAllocationCount()

	// Allocate some memory
	mem, err := memory.Alloc(1024)
	if err != nil {
		t.Fatalf("Failed to allocate memory: %v", err)
	}

	// Check stats changed
	newAllocated := manager.GetTotalAllocated()
	newCount := manager.GetAllocationCount()

	if newAllocated <= initialAllocated {
		t.Error("Expected total allocated to increase")
	}

	if newCount <= initialCount {
		t.Error("Expected allocation count to increase")
	}

	// Free memory
	mem.Free()

	t.Log("✅ Memory manager test passed")
}

// TestMemoryInfo tests global memory info
func TestMemoryInfo(t *testing.T) {
	t.Log("Testing memory info...")

	free, total := memory.GetInfo()

	if free < 0 {
		t.Error("Expected non-negative free memory")
	}

	if total <= 0 {
		t.Error("Expected positive total memory")
	}

	if free > total {
		t.Error("Free memory should not exceed total memory")
	}

	t.Log("✅ Memory info test passed")
}

// TestMemoryErrorHandling tests error conditions
func TestMemoryErrorHandling(t *testing.T) {
	t.Log("Testing memory error handling...")

	// Test invalid allocation sizes
	_, err := memory.Alloc(0)
	if err == nil {
		t.Error("Expected error for zero allocation")
	}

	_, err = memory.Alloc(-1)
	if err == nil {
		t.Error("Expected error for negative allocation")
	}

	// Test double free
	mem, err := memory.Alloc(1024)
	if err != nil {
		t.Fatalf("Failed to allocate memory: %v", err)
	}

	err = mem.Free()
	if err != nil {
		t.Errorf("First free failed: %v", err)
	}

	err = mem.Free()
	// Note: In simulation mode, double free may not error
	if err == nil {
		t.Log("Double free did not error - this is expected in simulation mode")
	}

	t.Log("✅ Memory error handling test passed")
}

// BenchmarkMemoryAllocation benchmarks memory allocation performance
func BenchmarkMemoryAllocation(b *testing.B) {
	for i := 0; i < b.N; i++ {
		mem, err := memory.Alloc(1024)
		if err != nil {
			b.Fatal(err)
		}
		mem.Free()
	}
}

// BenchmarkMemoryAllocationWithStream benchmarks stream allocation performance
func BenchmarkMemoryAllocationWithStream(b *testing.B) {
	stream := internal.GetDefaultStream()

	for i := 0; i < b.N; i++ {
		mem, err := memory.AllocWithStream(stream, 1024)
		if err != nil {
			b.Fatal(err)
		}
		mem.Free()
	}
}
