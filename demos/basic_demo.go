//go:build ignore
// +build ignore

package main

import (
	"context"
	"fmt"
	"log"

	cuda "github.com/stitch1968/gocuda"
)

func main() {
	// Initialize CUDA
	if err := cuda.Initialize(); err != nil {
		log.Fatal("Failed to initialize CUDA:", err)
	}

	fmt.Println("🚀 GoCUDA - Go routine-like CUDA Interface")
	fmt.Println("===========================================")

	// Show device information
	devices, _ := cuda.GetDevices()
	for _, device := range devices {
		fmt.Printf("📱 Device %d: %s\n", device.ID, device.Name)
		fmt.Printf("   Memory: %d MB\n", device.Properties.TotalGlobalMem/(1024*1024))
		fmt.Printf("   Cores: %d\n", device.Properties.MultiProcessorCount)
	}

	fmt.Println("\n🔥 Demo: Basic cuda.Go() usage (like Go routines)")
	// Execute function on GPU like a Go routine
	cuda.Go(func(ctx context.Context, args ...interface{}) error {
		message := args[0].(string)
		number := args[1].(int)
		fmt.Printf("   ✅ Hello from GPU! Message: %s, Number: %d\n", message, number)
		return nil
	}, "Hello World", 42)
	cuda.Synchronize()

	fmt.Println("\n🧮 Demo: Vector Addition on GPU")
	n := 100000

	// Create input vectors
	a := make([]float32, n)
	b := make([]float32, n)

	// Fill vectors with data
	for i := 0; i < n; i++ {
		a[i] = 1.5
		b[i] = 2.5
	}

	// Perform vector addition using Simple API
	result, err := cuda.SimpleVectorAdd(a, b)
	if err != nil {
		log.Fatal(err)
	}

	fmt.Printf("   ✅ Added %d vectors: 1.5 + 2.5 = %.1f (result length: %d)\n", n, result[0], len(result))

	fmt.Println("\n⚡ Demo: Parallel Processing")
	sum := 0
	cuda.ParallelFor(0, 1000, func(i int) error {
		sum += i // Simulate parallel work
		return nil
	})
	fmt.Printf("   ✅ Processed 1000 elements in parallel\n")

	fmt.Println("\n📡 Demo: CUDA Channels")
	ch := cuda.NewCudaChannel(5)
	defer ch.Close()

	// Send from GPU
	cuda.Go(func(ctx context.Context, args ...interface{}) error {
		channel := args[0].(*cuda.CudaChannel)
		channel.Send("Message from GPU kernel!")
		return nil
	}, ch)
	cuda.Synchronize()

	// Receive on host
	message := ch.Receive()
	fmt.Printf("   ✅ Received: %v\n", message)

	fmt.Println("\n🎯 Demo: Multiple Streams (Concurrent Execution)")
	ctx, err := cuda.DefaultContext()
	if err != nil {
		log.Fatal(err)
	}
	stream1, _ := ctx.NewStream()
	stream2, _ := ctx.NewStream()

	// Execute different tasks concurrently
	cuda.GoWithStream(stream1, func(ctx context.Context, args ...interface{}) error {
		fmt.Printf("   ✅ Task 1 running on Stream 1\n")
		return nil
	})

	cuda.GoWithStream(stream2, func(ctx context.Context, args ...interface{}) error {
		fmt.Printf("   ✅ Task 2 running on Stream 2\n")
		return nil
	})

	stream1.Synchronize()
	stream2.Synchronize()

	fmt.Println("\n💾 Demo: Memory Management")
	free, total := cuda.GetMemoryInfo()
	fmt.Printf("   📊 GPU Memory - Free: %d MB, Total: %d MB\n",
		free/(1024*1024), total/(1024*1024))

	fmt.Println("\n🎉 All demos completed successfully!")
	fmt.Println("\nKey Features:")
	fmt.Println("• cuda.Go() - Execute functions on GPU like Go routines")
	fmt.Println("• cuda.CudaChannel - Channel-like communication")
	fmt.Println("• cuda.ParallelFor() - Parallel processing patterns")
	fmt.Println("• Multiple streams for concurrent execution")
	fmt.Println("• Automatic memory management with finalizers")
	fmt.Println("• Built-in kernels: VectorAdd, MatrixMultiply, Convolution")
	fmt.Println("• cuda.CudaWaitGroup - Like sync.WaitGroup for GPU")
	fmt.Println("\nAPI Comparison:")
	fmt.Println("Go:      go func() { ... }()")
	fmt.Println("GoCUDA:  cuda.Go(func(ctx context.Context, args ...interface{}) error { ... }, args...)")
}
