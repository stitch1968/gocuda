package tests

import (
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stitch1968/gocuda/memory"
	"github.com/stitch1968/gocuda/streams"
)

func TestStreamDestroyWaitsForPendingCallbacks(t *testing.T) {
	manager := streams.GetManager()
	stream, err := streams.CreateStream()
	if err != nil {
		t.Fatalf("failed to create stream: %v", err)
	}

	var completed atomic.Bool
	err = stream.AddCallback(func() {
		time.Sleep(20 * time.Millisecond)
		completed.Store(true)
	})
	if err != nil {
		t.Fatalf("failed to add callback: %v", err)
	}

	start := time.Now()
	err = manager.DestroyStream(stream)
	if err != nil {
		t.Fatalf("failed to destroy stream: %v", err)
	}
	if !completed.Load() {
		t.Fatal("expected pending callback to finish before stream destroy returned")
	}
	if time.Since(start) < 15*time.Millisecond {
		t.Fatal("expected stream destruction to wait for pending callback work")
	}
}

func TestConcurrentStreamAndMemoryLifecycle(t *testing.T) {
	manager := streams.GetManager()
	initialCount := memory.GetManager().GetAllocationCount()

	const goroutines = 8
	const iterations = 16

	var wg sync.WaitGroup
	for worker := 0; worker < goroutines; worker++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := 0; i < iterations; i++ {
				stream, err := streams.CreateStream()
				if err != nil {
					t.Errorf("failed to create stream: %v", err)
					return
				}

				mem, err := memory.AllocWithStream(stream.Stream, 256)
				if err != nil {
					t.Errorf("failed to allocate memory: %v", err)
					_ = manager.DestroyStream(stream)
					return
				}

				if err := stream.AddCallback(func() {
					view, viewErr := memory.View[byte](mem, 16)
					if viewErr != nil {
						t.Errorf("failed to view memory: %v", viewErr)
						return
					}
					for idx := range view {
						view[idx] = byte(idx)
					}
				}); err != nil {
					t.Errorf("failed to schedule callback: %v", err)
				}

				if err := stream.Synchronize(); err != nil {
					t.Errorf("failed to synchronize stream: %v", err)
				}
				if err := mem.Free(); err != nil {
					t.Errorf("failed to free memory: %v", err)
				}
				if err := manager.DestroyStream(stream); err != nil {
					t.Errorf("failed to destroy stream: %v", err)
				}
			}
		}()
	}

	wg.Wait()
	if got := memory.GetManager().GetAllocationCount(); got != initialCount {
		t.Fatalf("expected allocation count to return to %d, got %d", initialCount, got)
	}
}
