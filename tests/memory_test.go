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
	if internal.ShouldUseCuda() {
		if data != nil {
			t.Error("Expected CUDA-backed device allocation to hide host-visible data slice")
		}
	} else if data == nil {
		t.Error("Expected simulation allocation to expose data slice")
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

func TestMemoryDoubleFreeIsSafe(t *testing.T) {
	mem, err := memory.Alloc(64)
	if err != nil {
		t.Fatalf("Failed to allocate memory: %v", err)
	}

	if err := mem.Free(); err != nil {
		t.Fatalf("First free failed: %v", err)
	}
	if err := mem.Free(); err != nil {
		t.Fatalf("Second free should be a no-op, got error: %v", err)
	}
}

func TestZeroLengthTransfersAreNoOps(t *testing.T) {
	mem, err := memory.Alloc(16)
	if err != nil {
		t.Fatalf("Failed to allocate memory: %v", err)
	}
	defer mem.Free()

	if err := memory.CopyHostToDevice(mem, nil); err != nil {
		t.Fatalf("Expected zero-length host-to-device copy to succeed: %v", err)
	}
	if err := memory.CopyDeviceToHost(nil, mem); err != nil {
		t.Fatalf("Expected zero-length device-to-host copy to succeed: %v", err)
	}

	empty, err := memory.Alloc(1)
	if err != nil {
		t.Fatalf("Failed to allocate second memory block: %v", err)
	}
	defer empty.Free()

	if err := memory.CopyDeviceToDevice(empty, mem); err != nil {
		// Non-zero device-to-device copies should still work with valid buffers.
		t.Fatalf("Expected device-to-device copy to succeed: %v", err)
	}
	if err := memory.CopyDeviceToDeviceWithStream(internal.GetDefaultStream(), empty, nil); err == nil {
		t.Fatal("Expected null-pointer device-to-device copy to fail")
	}
}

func TestAllocOnDeviceAndPeerCopy(t *testing.T) {
	src, err := memory.AllocOnDevice(4, 0)
	if err != nil {
		t.Fatalf("Failed to allocate source memory: %v", err)
	}
	defer src.Free()

	dst, err := memory.AllocOnDevice(4, 1)
	if err != nil {
		t.Fatalf("Failed to allocate destination memory: %v", err)
	}
	defer dst.Free()

	if src.GetDeviceID() != 0 {
		t.Fatalf("expected source memory on device 0, got %d", src.GetDeviceID())
	}
	if dst.GetDeviceID() != 1 {
		t.Fatalf("expected destination memory on device 1, got %d", dst.GetDeviceID())
	}

	hostData := []byte{1, 2, 3, 4}
	if err := memory.CopyHostToDevice(src, hostData); err != nil {
		t.Fatalf("Failed to copy host data to source device: %v", err)
	}
	if err := memory.CopyDeviceToDevicePeer(dst, src); err != nil {
		t.Fatalf("Expected peer copy to succeed: %v", err)
	}

	out := make([]byte, 4)
	if err := memory.CopyDeviceToHost(out, dst); err != nil {
		t.Fatalf("Failed to read destination device data: %v", err)
	}
	for i, want := range hostData {
		if out[i] != want {
			t.Fatalf("unexpected copied byte at %d: got %d want %d", i, out[i], want)
		}
	}
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

// TestMemoryViewBounds tests bounds-checked typed memory views.
func TestMemoryViewBounds(t *testing.T) {
	t.Log("Testing typed memory view bounds...")

	mem, err := memory.Alloc(16)
	if err != nil {
		t.Fatalf("Failed to allocate memory: %v", err)
	}
	defer mem.Free()

	values, err := memory.View[uint32](mem, 4)
	if err != nil {
		t.Fatalf("Expected valid memory view: %v", err)
	}
	if len(values) != 4 {
		t.Fatalf("Expected view length 4, got %d", len(values))
	}

	_, err = memory.View[uint32](mem, 5)
	if err == nil {
		t.Fatal("Expected out-of-bounds error for oversized memory view")
	}

	if err := mem.Free(); err != nil {
		t.Fatalf("Failed to free memory: %v", err)
	}

	_, err = memory.View[uint32](mem, 1)
	if err == nil {
		t.Fatal("Expected error when viewing freed memory")
	}

	t.Log("✅ Typed memory view bounds test passed")
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
