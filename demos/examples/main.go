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

	fmt.Println("=== GoCUDA Demo ===")

	// Example 1: Basic cuda.Go() usage (like go routines)
	fmt.Println("\n1. Basic cuda.Go() usage:")
	err := cuda.Go(func(ctx context.Context, args ...interface{}) error {
		fmt.Printf("Hello from CUDA kernel! Args: %v\n", args)
		return nil
	}, "arg1", 42, 3.14)

	if err != nil {
		log.Fatal(err)
	}

	cuda.Synchronize()

	// Example 2: Vector Addition
	fmt.Println("\n2. Vector Addition:")
	vectorAddExample()

	// Example 3: Matrix Multiplication
	fmt.Println("\n3. Matrix Multiplication:")
	matrixMulExample()

	// Example 4: Parallel For Loop
	fmt.Println("\n4. Parallel For Loop:")
	parallelForExample()

	// Example 5: CUDA Channels
	fmt.Println("\n5. CUDA Channels:")
	channelExample()

	// Example 6: Memory Management
	fmt.Println("\n6. Memory Management:")
	memoryExample()

	// Example 7: Multiple Streams
	fmt.Println("\n7. Multiple Streams:")
	streamExample()

	// Example 8: CUDA WaitGroup
	fmt.Println("\n8. CUDA WaitGroup:")
	waitGroupExample()

	// Example 9: Map-Reduce Pattern
	fmt.Println("\n9. Map-Reduce Pattern:")
	mapReduceExample()

	fmt.Println("\nAll examples completed!")
}

func vectorAddExample() {
	n := 1000

	// Create input vectors with data
	a := make([]float32, n)
	b := make([]float32, n)

	// Fill vectors with data
	for i := 0; i < n; i++ {
		a[i] = 1.0
		b[i] = 2.0
	}

	// Perform vector addition using Simple API
	result, err := cuda.SimpleVectorAdd(a, b)
	if err != nil {
		log.Fatal(err)
	}

	fmt.Printf("Added two vectors of size %d (first element: %.1f + %.1f = %.1f)\n",
		len(result), a[0], b[0], result[0])
}

func matrixMulExample() {
	m, n, k := 4, 4, 4 // Smaller size for demo

	// Create matrix A (m x k)
	a := make([][]float32, m)
	for i := range a {
		a[i] = make([]float32, k)
		for j := range a[i] {
			a[i][j] = 1.0
		}
	}

	// Create matrix B (k x n)
	b := make([][]float32, k)
	for i := range b {
		b[i] = make([]float32, n)
		for j := range b[i] {
			b[i][j] = 2.0
		}
	}

	// Perform matrix multiplication using Simple API
	result, err := cuda.SimpleMatrixMultiply(a, b)
	if err != nil {
		log.Fatal(err)
	}

	fmt.Printf("Multiplied matrices: %dx%d × %dx%d = %dx%d (result[0][0] = %.1f)\n",
		m, k, k, n, len(result), len(result[0]), result[0][0])
}

func parallelForExample() {
	n := 1000
	fmt.Printf("Processing %d items in parallel...\n", n)

	err := cuda.ParallelFor(0, n, func(i int) error {
		// Simulate some work
		_ = i * i
		return nil
	})

	if err != nil {
		log.Fatal(err)
	}

	cuda.Synchronize()
	fmt.Printf("Parallel processing completed\n")
}

func channelExample() {
	ch := cuda.NewCudaChannel(5)
	defer ch.Close()

	// Send data from GPU
	err := cuda.Go(func(ctx context.Context, args ...interface{}) error {
		channel := args[0].(*cuda.CudaChannel)
		for i := 0; i < 3; i++ {
			channel.Send(fmt.Sprintf("Message %d from GPU", i))
		}
		return nil
	}, ch)

	if err != nil {
		log.Fatal(err)
	}

	cuda.Synchronize()

	// Receive data on host
	for i := 0; i < 3; i++ {
		msg := ch.Receive()
		fmt.Printf("Received: %v\n", msg)
	}
}

func memoryExample() {
	// Allocate device memory
	size := int64(1024)
	mem, err := cuda.Malloc(size)
	if err != nil {
		log.Fatal(err)
	}
	defer mem.Free()

	fmt.Printf("Allocated %d bytes on GPU\n", size)

	// Show memory info
	free, total := cuda.GetMemoryInfo()
	fmt.Printf("GPU Memory - Free: %d MB, Total: %d MB\n",
		free/(1024*1024), total/(1024*1024))
}

func streamExample() {
	ctx := cuda.GetDefaultContext()
	stream1, err := ctx.NewStream()
	if err != nil {
		log.Fatal(err)
	}

	stream2, err := ctx.NewStream()
	if err != nil {
		log.Fatal(err)
	}

	fmt.Println("Running tasks on multiple streams...")

	// Execute tasks on different streams
	err = cuda.GoWithStream(stream1, func(ctx context.Context, args ...interface{}) error {
		fmt.Println("Task 1 on stream 1")
		return nil
	})
	if err != nil {
		log.Fatal(err)
	}

	err = cuda.GoWithStream(stream2, func(ctx context.Context, args ...interface{}) error {
		fmt.Println("Task 2 on stream 2")
		return nil
	})
	if err != nil {
		log.Fatal(err)
	}

	stream1.Synchronize()
	stream2.Synchronize()
}

func waitGroupExample() {
	var wg cuda.CudaWaitGroup
	numTasks := 5

	fmt.Printf("Starting %d concurrent tasks...\n", numTasks)

	for i := 0; i < numTasks; i++ {
		wg.Add(1)
		taskID := i

		err := cuda.Go(func(ctx context.Context, args ...interface{}) error {
			defer wg.Done()
			id := args[0].(int)
			fmt.Printf("Task %d completed\n", id)
			return nil
		}, taskID)

		if err != nil {
			log.Fatal(err)
		}
	}

	wg.Wait()
	fmt.Println("All tasks completed")
}

func mapReduceExample() {
	data := make([]float64, 1000)
	for i := range data {
		data[i] = float64(i + 1)
	}

	// Map: square each number
	squared, err := cuda.Map(data, func(x float64) float64 {
		return x * x
	})
	if err != nil {
		log.Fatal(err)
	}

	// Reduce: sum all squared values
	sum, err := cuda.Reduce(squared, func(a, b float64) float64 {
		return a + b
	})
	if err != nil {
		log.Fatal(err)
	}

	cuda.Synchronize()
	fmt.Printf("Map-Reduce result: sum of squares = %v\n", sum)
}
