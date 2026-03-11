package main

import (
	"fmt"
	"log"

	"github.com/stitch1968/gocuda"
)

func main() {
	fmt.Println("=== CGO + CUDA Test ===")
	
	// Initialize CUDA runtime
	err := cuda.Initialize()
	if err != nil {
		log.Fatalf("Failed to initialize CUDA: %v", err)
	}

	// Print CUDA information
	fmt.Println("✅ CUDA initialized successfully!")
	fmt.Printf("CUDA Available: %t\n", cuda.IsCudaAvailable())
	fmt.Printf("Using CUDA: %t\n", cuda.ShouldUseCuda())
	fmt.Printf("Device Count: %d\n", cuda.GetCudaDeviceCount())
	
	if cuda.ShouldUseCuda() {
		fmt.Println("🚀 Real CUDA hardware acceleration active!")
		cuda.PrintCudaInfo()
	} else {
		fmt.Println("💻 CPU simulation mode active")
	}
}
