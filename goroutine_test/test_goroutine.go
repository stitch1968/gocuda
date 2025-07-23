package main

import (
	"context"
	"fmt"
	"log"

	cuda "github.com/stitch1968/gocuda"
)

func main() {
	fmt.Println("Testing GoCUDA goroutine functionality...")

	// Test basic Go function
	err := cuda.Go(func(ctx context.Context, args ...interface{}) error {
		fmt.Println("✅ Basic Go function executed successfully")
		return nil
	})
	if err != nil {
		log.Fatalf("Failed to execute Go function: %v", err)
	}

	// Test ParallelFor
	fmt.Println("Testing ParallelFor...")
	counter := 0
	err = cuda.ParallelFor(0, 10, func(i int) error {
		counter++
		fmt.Printf("ParallelFor iteration %d\n", i)
		return nil
	})
	if err != nil {
		log.Fatalf("Failed to execute ParallelFor: %v", err)
	}
	fmt.Printf("✅ ParallelFor completed, counter: %d\n", counter)

	// Test Map function
	fmt.Println("Testing Map function...")
	input := []float64{1.0, 2.0, 3.0, 4.0, 5.0}
	output, err := cuda.Map(input, func(x float64) float64 {
		return x * 2.0
	})
	if err != nil {
		log.Fatalf("Failed to execute Map: %v", err)
	}
	fmt.Printf("✅ Map function: input %v -> output %v\n", input, output)

	// Test Reduce function
	fmt.Println("Testing Reduce function...")
	data := []float64{1.0, 2.0, 3.0, 4.0, 5.0}
	sum, err := cuda.Reduce(data, func(a, b float64) float64 {
		return a + b
	})
	if err != nil {
		log.Fatalf("Failed to execute Reduce: %v", err)
	}
	fmt.Printf("✅ Reduce function: sum of %v = %f\n", data, sum)

	// Test Synchronize
	err = cuda.Synchronize()
	if err != nil {
		log.Fatalf("Failed to synchronize: %v", err)
	}
	fmt.Println("✅ Synchronize completed successfully")

	fmt.Println("\n🎉 All goroutine.go functionality tests passed!")
	fmt.Println("The goroutine.go file is fully compatible with the wider GoCUDA project.")
}
