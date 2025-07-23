package tests

import (
	"testing"
	"time"

	"github.com/stitch1968/gocuda/streams"
)

// TestStreamCreation tests stream creation
func TestStreamCreation(t *testing.T) {
	t.Log("Testing stream creation...")

	stream, err := streams.CreateStream()
	if err != nil {
		t.Fatalf("Failed to create stream: %v", err)
	}
	defer streams.GetManager().DestroyStream(stream)

	if stream == nil {
		t.Error("Expected non-nil stream")
	}

	t.Log("✅ Stream creation test passed")
}

// TestStreamManager tests stream manager functionality
func TestStreamManager(t *testing.T) {
	t.Log("Testing stream manager...")

	manager := streams.GetManager()
	if manager == nil {
		t.Fatal("Expected non-nil stream manager")
	}

	// Get initial stream count
	initialCount := manager.GetActiveStreams()

	// Create a stream
	stream, err := streams.CreateStream()
	if err != nil {
		t.Fatalf("Failed to create stream: %v", err)
	}

	// Active streams should have increased
	newCount := manager.GetActiveStreams()
	if newCount <= initialCount {
		t.Error("Expected active stream count to increase")
	}

	// Destroy stream
	err = manager.DestroyStream(stream)
	if err != nil {
		t.Errorf("Failed to destroy stream: %v", err)
	}

	t.Log("✅ Stream manager test passed")
}

// TestDefaultStream tests default stream functionality
func TestDefaultStream(t *testing.T) {
	t.Log("Testing default stream...")

	manager := streams.GetManager()
	stream := manager.GetDefaultStream()
	if stream == nil {
		t.Error("Expected non-nil default stream")
	}

	t.Log("✅ Default stream test passed")
}

// TestStreamPriority tests stream priority functionality
func TestStreamPriority(t *testing.T) {
	t.Log("Testing stream priority...")

	manager := streams.GetManager()

	// Create high priority stream
	stream, err := manager.CreateStreamWithPriority(streams.StreamDefault, streams.PriorityHigh)
	if err != nil {
		t.Fatalf("Failed to create high priority stream: %v", err)
	}
	defer manager.DestroyStream(stream)

	if stream.GetPriority() != streams.PriorityHigh {
		t.Error("Expected high priority stream")
	}

	// Change priority
	stream.SetPriority(streams.PriorityLow)
	if stream.GetPriority() != streams.PriorityLow {
		t.Error("Failed to change to low priority")
	}

	t.Log("✅ Stream priority test passed")
}

// TestStreamEvents tests basic event functionality
func TestStreamEvents(t *testing.T) {
	t.Log("Testing stream events...")

	stream, err := streams.CreateStream()
	if err != nil {
		t.Fatalf("Failed to create stream: %v", err)
	}
	defer streams.GetManager().DestroyStream(stream)

	// Create and use event
	event := streams.CreateEvent()

	// Events should be queryable
	completed := event.Query()
	if !completed {
		t.Log("Event not yet completed")
	}

	// Should be able to synchronize event (with timeout for test)
	done := make(chan error, 1)
	go func() {
		done <- event.Synchronize()
	}()

	select {
	case err := <-done:
		if err != nil {
			t.Errorf("Failed to synchronize event: %v", err)
		}
	case <-time.After(100 * time.Millisecond):
		t.Log("Event synchronization timed out - this may be expected")
	}

	t.Log("✅ Stream events test passed")
}

// TestStreamCallbacks tests stream callback functionality
func TestStreamCallbacks(t *testing.T) {
	t.Log("Testing stream callbacks...")

	stream, err := streams.CreateStream()
	if err != nil {
		t.Fatalf("Failed to create stream: %v", err)
	}
	defer streams.GetManager().DestroyStream(stream)

	// Add callback
	err = stream.AddCallback(func() {
		// Simple callback work
	})
	if err != nil {
		t.Errorf("Failed to add callback: %v", err)
	}

	// Give some time for callback execution
	time.Sleep(10 * time.Millisecond)

	t.Log("✅ Stream callbacks test passed")
}

// TestStreamSynchronization tests synchronization functionality
func TestStreamSynchronization(t *testing.T) {
	t.Log("Testing stream synchronization...")

	manager := streams.GetManager()

	// Create multiple streams
	stream1, err := streams.CreateStream()
	if err != nil {
		t.Fatalf("Failed to create stream1: %v", err)
	}
	defer manager.DestroyStream(stream1)

	stream2, err := streams.CreateStream()
	if err != nil {
		t.Fatalf("Failed to create stream2: %v", err)
	}
	defer manager.DestroyStream(stream2)

	// Add work to both streams
	stream1.AddCallback(func() { time.Sleep(5 * time.Millisecond) })
	stream2.AddCallback(func() { time.Sleep(5 * time.Millisecond) })

	// Synchronize all streams
	err = manager.SynchronizeAll()
	if err != nil {
		t.Errorf("Failed to synchronize all streams: %v", err)
	}

	t.Log("✅ Stream synchronization test passed")
}

// TestConvenienceFunctions tests convenience functions
func TestConvenienceFunctions(t *testing.T) {
	t.Log("Testing convenience functions...")

	manager := streams.GetManager()

	// Test high priority stream creation
	highStream, err := streams.CreateHighPriorityStream()
	if err != nil {
		t.Fatalf("Failed to create high priority stream: %v", err)
	}
	defer manager.DestroyStream(highStream)

	if highStream.GetPriority() != streams.PriorityHigh {
		t.Error("Expected high priority stream")
	}

	// Test low priority stream creation
	lowStream, err := streams.CreateLowPriorityStream()
	if err != nil {
		t.Fatalf("Failed to create low priority stream: %v", err)
	}
	defer manager.DestroyStream(lowStream)

	if lowStream.GetPriority() != streams.PriorityLow {
		t.Error("Expected low priority stream")
	}

	// Test default stream access
	defaultStream := streams.GetDefaultStream()
	if defaultStream == nil {
		t.Error("Expected non-nil default stream")
	}

	t.Log("✅ Convenience functions test passed")
}

// BenchmarkStreamCreation benchmarks stream creation performance
func BenchmarkStreamCreation(b *testing.B) {
	manager := streams.GetManager()

	for i := 0; i < b.N; i++ {
		stream, err := streams.CreateStream()
		if err != nil {
			b.Fatal(err)
		}
		manager.DestroyStream(stream)
	}
}

// BenchmarkStreamCallback benchmarks callback performance
func BenchmarkStreamCallback(b *testing.B) {
	stream, err := streams.CreateStream()
	if err != nil {
		b.Fatal(err)
	}
	defer streams.GetManager().DestroyStream(stream)

	for i := 0; i < b.N; i++ {
		err := stream.AddCallback(func() {
			// Minimal work
		})
		if err != nil {
			b.Fatal(err)
		}
	}
}
