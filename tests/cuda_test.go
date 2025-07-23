package tests

import (
	"context"
	"testing"
	"time"

	cuda "github.com/stitch1968/gocuda"
)

func TestInitialize(t *testing.T) {
	err := cuda.Initialize()
	if err != nil {
		t.Fatalf("Failed to initialize CUDA: %v", err)
	}
}

func TestGetDevices(t *testing.T) {
	devices, err := cuda.GetDevices()
	if err != nil {
		t.Fatalf("Failed to get devices: %v", err)
	}

	if len(devices) == 0 {
		t.Fatal("No devices found")
	}

	t.Logf("Found %d device(s)", len(devices))
	for _, device := range devices {
		t.Logf("Device %d: %s", device.ID, device.Name)
	}
}

func TestNewContext(t *testing.T) {
	ctx, err := cuda.NewContext(0)
	if err != nil {
		t.Fatalf("Failed to create context: %v", err)
	}

	if ctx == nil {
		t.Fatal("Context is nil")
	}

	// Note: Context cleanup is handled automatically
}

func TestNewStream(t *testing.T) {
	ctx, err := cuda.NewContext(0)
	if err != nil {
		t.Fatalf("Failed to create context: %v", err)
	}
	// Note: Context cleanup is handled automatically

	stream, err := ctx.NewStream()
	if err != nil {
		t.Fatalf("Failed to create stream: %v", err)
	}

	if stream == nil {
		t.Fatal("Stream is nil")
	}

	// Note: Stream cleanup is handled automatically
}

func TestBasicGo(t *testing.T) {
	cuda.Initialize()

	executed := false
	err := cuda.Go(func(ctx context.Context, args ...interface{}) error {
		executed = true
		if len(args) != 2 {
			t.Errorf("Expected 2 args, got %d", len(args))
		}
		return nil
	}, "test", 42)

	if err != nil {
		t.Fatalf("cuda.Go failed: %v", err)
	}

	err = cuda.Synchronize()
	if err != nil {
		t.Fatalf("Synchronize failed: %v", err)
	}

	if !executed {
		t.Fatal("Kernel was not executed")
	}
}

func TestMemoryAllocation(t *testing.T) {
	cuda.Initialize()

	size := int64(1024)
	mem, err := cuda.Malloc(size)
	if err != nil {
		t.Fatalf("Failed to allocate memory: %v", err)
	}

	if mem.Size() != size {
		t.Fatalf("Expected size %d, got %d", size, mem.Size())
	}

	err = mem.Free()
	if err != nil {
		t.Fatalf("Failed to free memory: %v", err)
	}
}

func TestMemcpy(t *testing.T) {
	cuda.Initialize()

	src, err := cuda.Malloc(1024)
	if err != nil {
		t.Fatalf("Failed to allocate source memory: %v", err)
	}
	defer src.Free()

	dst, err := cuda.Malloc(1024)
	if err != nil {
		t.Fatalf("Failed to allocate destination memory: %v", err)
	}
	defer dst.Free()

	// Test host to device copy
	hostData := []byte{1, 2, 3, 4, 5}
	err = cuda.CopyHostToDevice(src, hostData)
	if err != nil {
		t.Fatalf("Host to device copy failed: %v", err)
	}

	// Test device to device copy
	err = cuda.CopyDeviceToDevice(dst, src)
	if err != nil {
		t.Fatalf("Device to device copy failed: %v", err)
	}

	// Test device to host copy
	result := make([]byte, len(hostData))
	err = cuda.CopyDeviceToHost(result, dst)
	if err != nil {
		t.Fatalf("Device to host copy failed: %v", err)
	}

	cuda.Synchronize()

	for i, v := range hostData {
		if result[i] != v {
			t.Fatalf("Data mismatch at index %d: expected %d, got %d", i, v, result[i])
		}
	}
}

func TestParallelFor(t *testing.T) {
	cuda.Initialize()

	n := 10000
	results := make([]int, n)

	err := cuda.ParallelFor(0, n, func(i int) error {
		results[i] = i * i
		return nil
	})

	if err != nil {
		t.Fatalf("ParallelFor failed: %v", err)
	}

	// Wait for completion
	err = cuda.Synchronize()
	if err != nil {
		t.Fatalf("Synchronize failed: %v", err)
	}

	// Verify results
	for i := 0; i < 100; i++ { // Check first 100 elements
		expected := i * i
		if results[i] != expected {
			t.Fatalf("Result mismatch at index %d: expected %d, got %d", i, expected, results[i])
		}
	}

	t.Log("ParallelFor completed successfully")
}

func TestCudaChannel(t *testing.T) {
	cuda.Initialize()

	ch := cuda.NewCudaChannel(5)
	defer ch.Close()

	// Send data from kernel
	err := cuda.Go(func(ctx context.Context, args ...interface{}) error {
		channel := args[0].(*cuda.CudaChannel)
		for i := 0; i < 3; i++ {
			channel.Send(i)
		}
		return nil
	}, ch)

	if err != nil {
		t.Fatalf("Failed to execute sender kernel: %v", err)
	}

	cuda.Synchronize()

	// Receive data
	for i := 0; i < 3; i++ {
		val := ch.Receive()
		if val.(int) != i {
			t.Fatalf("Expected %d, got %v", i, val)
		}
	}

	t.Log("CUDA channel test completed successfully")
}

func TestMultipleStreams(t *testing.T) {
	cuda.Initialize()

	ctx := cuda.GetDefaultContext()

	stream1, err := ctx.NewStream()
	if err != nil {
		t.Fatalf("Failed to create stream1: %v", err)
	}
	// Note: Stream cleanup is handled automatically

	stream2, err := ctx.NewStream()
	if err != nil {
		t.Fatalf("Failed to create stream2: %v", err)
	}
	// Note: Stream cleanup is handled automatically

	// Execute on different streams
	executed1 := false
	executed2 := false

	err = cuda.GoWithStream(stream1, func(ctx context.Context, args ...interface{}) error {
		time.Sleep(10 * time.Millisecond)
		executed1 = true
		return nil
	})

	if err != nil {
		t.Fatalf("Failed to execute on stream1: %v", err)
	}

	err = cuda.GoWithStream(stream2, func(ctx context.Context, args ...interface{}) error {
		time.Sleep(10 * time.Millisecond)
		executed2 = true
		return nil
	})

	if err != nil {
		t.Fatalf("Failed to execute on stream2: %v", err)
	}

	// Synchronize streams
	stream1.Synchronize()
	stream2.Synchronize()

	if !executed1 || !executed2 {
		t.Fatal("Not all streams executed")
	}

	t.Log("Multiple streams test completed successfully")
}

func TestCudaWaitGroup(t *testing.T) {
	cuda.Initialize()

	var wg cuda.CudaWaitGroup
	count := 0

	for i := 0; i < 5; i++ {
		wg.Add(1)
		cuda.Go(func(ctx context.Context, args ...interface{}) error {
			defer wg.Done()
			count++
			return nil
		})
	}

	wg.Wait()

	if count != 5 {
		t.Fatalf("Expected count 5, got %d", count)
	}

	t.Log("CUDA WaitGroup test completed successfully")
}

func BenchmarkParallelFor(b *testing.B) {
	cuda.Initialize()

	n := 100000

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		cuda.ParallelFor(0, n, func(i int) error {
			_ = i * i
			return nil
		})
	}
}
