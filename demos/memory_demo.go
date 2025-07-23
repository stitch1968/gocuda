//go:build ignore
// +build ignore

package main

import (
	"fmt"
	"log"

	cuda "github.com/stitch1968/gocuda"
	"github.com/stitch1968/gocuda/memory"
)

func main() {
	// Initialize CUDA
	if err := cuda.Initialize(); err != nil {
		log.Fatal("Failed to initialize CUDA:", err)
	}

	fmt.Println("🔧 CUDA-Compatible Memory Management Features")
	fmt.Println("=============================================")

	// Example 1: Basic Memory Allocation
	fmt.Println("\n1. Basic Memory Allocation:")

	// Device memory (standard GPU memory)
	deviceMem, err := memory.Alloc(1024)
	if err != nil {
		log.Fatal(err)
	}
	defer deviceMem.Free()
	fmt.Printf("   ✅ Device Memory: %d bytes, type: %d\n", deviceMem.Size(), deviceMem.GetType())

	// Example 2: Memory Information
	fmt.Println("\n2. Memory Information:")
	free, total := memory.GetInfo()
	fmt.Printf("   ✅ Free Memory: %d bytes\n", free)
	fmt.Printf("   ✅ Total Memory: %d bytes\n", total)
	if total > 0 {
		fmt.Printf("   ✅ Memory Usage: %.1f%%\n", float64(total-free)/float64(total)*100)
	}

	// Example 3: Memory Manager Statistics
	fmt.Println("\n3. Memory Manager Statistics:")
	manager := memory.GetManager()
	fmt.Printf("   ✅ Total Allocated: %d bytes\n", manager.GetTotalAllocated())
	fmt.Printf("   ✅ Allocation Count: %d\n", manager.GetAllocationCount())

	// Example 4: Memory Properties
	fmt.Println("\n4. Memory Properties:")
	testMem, err := memory.Alloc(256)
	if err != nil {
		log.Fatal(err)
	}
	defer testMem.Free()

	fmt.Printf("   ✅ Allocated Memory: %d bytes\n", testMem.Size())
	fmt.Printf("   ✅ Memory Type: %d\n", testMem.GetType())
	fmt.Printf("   ✅ Memory Alignment: %d\n", testMem.GetAlignment())
	fmt.Printf("   ✅ Device ID: %d\n", testMem.GetDeviceID())
	fmt.Printf("   ✅ Memory Pitch: %d\n", testMem.GetPitch())

	// Example 5: Multiple Allocations
	fmt.Println("\n5. Multiple Memory Allocations:")
	var memories []*memory.Memory
	for i := 0; i < 5; i++ {
		mem, err := memory.Alloc(512)
		if err != nil {
			log.Fatal(err)
		}
		memories = append(memories, mem)
		fmt.Printf("   ✅ Allocation %d: %d bytes\n", i+1, mem.Size())
	}

	// Free all allocations
	for i, mem := range memories {
		mem.Free()
		fmt.Printf("   ✅ Freed allocation %d\n", i+1)
	}

	fmt.Println("\n🎉 Memory management demo completed!")
	fmt.Println("\nFeatures Demonstrated:")
	fmt.Println("• Basic memory allocation and deallocation")
	fmt.Println("• Memory information and statistics")
	fmt.Println("• Memory manager functionality")
	fmt.Println("• Memory properties and attributes")
	fmt.Println("• Multiple memory allocations")
}
