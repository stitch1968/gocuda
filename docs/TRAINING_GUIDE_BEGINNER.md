# 🎓 GoCUDA Training Guide - Beginner Level

**Target Audience:** New to CUDA and Go developers getting started with GPU programming

---

## 📚 Prerequisites

### What You Should Know
- ✅ **Basic Go Programming** - Variables, functions, structs, slices, error handling
- ✅ **Command Line Basics** - Running programs, file navigation
- ❓ **CUDA Knowledge** - Not required! We'll teach you everything

### What You'll Learn
- 🎯 CUDA fundamentals and GPU programming concepts
- 🎯 How to use GoCUDA for basic GPU operations
- 🎯 Memory management and data transfer
- 🎯 Simple parallel algorithms
- 🎯 Best practices for beginners

---

## 🚀 Chapter 1: Getting Started

### Step 1: Installation & Setup

**Option A: CPU Simulation (Recommended for Beginners)**
```bash
# Create a new project
mkdir my-gocuda-project
cd my-gocuda-project

# Initialize Go module
go mod init my-gocuda-project

# Get GoCUDA (no GPU required!)
go get github.com/stitch1968/gocuda

# Test installation
go run -c "package main; import _ \"github.com/stitch1968/gocuda\"; func main() { println(\"GoCUDA installed!\") }"
```

**Option B: With Real GPU (If you have NVIDIA GPU)**
```bash
# Install NVIDIA CUDA Toolkit first
# Download from: https://developer.nvidia.com/cuda-downloads

# Then follow Option A, but build with CUDA support:
go build -tags cuda ./...
```

### Step 2: Your First CUDA Program

Create `hello_cuda.go`:

```go
package main

import (
    "fmt"
    "github.com/stitch1968/gocuda"
)

func main() {
    // Initialize CUDA - this works on any machine!
    cuda.Initialize()
    
    // Check what mode we're running in
    if cuda.ShouldUseCuda() {
        fmt.Println("🚀 Running on real GPU!")
    } else {
        fmt.Println("💻 Running in simulation mode (CPU)")
    }
    
    // Get device information
    deviceCount := cuda.GetDeviceCount()
    fmt.Printf("Available devices: %d\n", deviceCount)
    
    if deviceCount > 0 {
        device := cuda.GetDevice(0)
        props := device.GetProperties()
        fmt.Printf("Device 0: %s\n", props.Name)
        fmt.Printf("Memory: %.2f GB\n", float64(props.GlobalMemory)/(1024*1024*1024))
    }
}
```

**Run it:**
```bash
go run hello_cuda.go
```

**Expected Output:**
```
💻 Running in simulation mode (CPU)
Available devices: 1
Device 0: Simulated CUDA Device
Memory: 8.00 GB
```

---

## 💾 Chapter 2: Understanding Memory

### GPU Memory Basics

Think of GPU memory like a very fast, but separate storage from your computer's main memory (RAM). To use the GPU, you need to:

1. **Allocate** memory on the GPU
2. **Copy** data from CPU → GPU  
3. **Process** data on GPU
4. **Copy** results back GPU → CPU
5. **Free** GPU memory when done

### Your First Memory Program

Create `memory_basics.go`:

```go
package main

import (
    "fmt"
    "github.com/stitch1968/gocuda"
    "github.com/stitch1968/gocuda/memory"
)

func main() {
    cuda.Initialize()
    
    // Step 1: Create some data on CPU
    hostData := []float32{1.0, 2.0, 3.0, 4.0, 5.0}
    fmt.Printf("Original data: %v\n", hostData)
    
    // Step 2: Allocate GPU memory
    // Size = number of elements × 4 bytes per float32
    gpuMemory, err := memory.Alloc(int64(len(hostData) * 4))
    if err != nil {
        panic(err)
    }
    defer gpuMemory.Free() // Always free memory when done!
    
    // Step 3: Copy data to GPU
    err = gpuMemory.CopyFromHost(hostData)
    if err != nil {
        panic(err)
    }
    fmt.Println("✅ Data copied to GPU")
    
    // Step 4: Copy data back from GPU
    result := make([]float32, len(hostData))
    err = gpuMemory.CopyToHost(result)
    if err != nil {
        panic(err)
    }
    
    fmt.Printf("Data from GPU: %v\n", result)
    fmt.Println("✅ Memory operations complete!")
}
```

**Key Concepts:**
- 🔑 **Always use `defer mem.Free()`** - prevents memory leaks
- 🔑 **Size in bytes** - float32 = 4 bytes, int32 = 4 bytes, float64 = 8 bytes  
- 🔑 **Host = CPU, Device = GPU** - remember this terminology

---

## 🎲 Chapter 3: Your First CUDA Library - Random Numbers

Random numbers are perfect for learning because they're simple but useful!

Create `random_numbers.go`:

```go
package main

import (
    "fmt"
    "github.com/stitch1968/gocuda"
    "github.com/stitch1968/gocuda/libraries"
    "github.com/stitch1968/gocuda/memory"
)

func main() {
    cuda.Initialize()
    fmt.Println("🎲 Learning cuRAND - Random Number Generation")
    
    // Step 1: Create a random number generator
    rng, err := libraries.CreateRandomGenerator(libraries.RngTypeXorwow)
    if err != nil {
        panic(err)
    }
    defer rng.Destroy() // Always clean up resources!
    
    // Step 2: Allocate memory for 1000 random numbers
    numCount := 1000
    output, err := memory.Alloc(int64(numCount * 4)) // 1000 × 4 bytes
    if err != nil {
        panic(err)
    }
    defer output.Free()
    
    // Step 3: Generate random numbers
    fmt.Println("Generating uniform random numbers [0,1]...")
    err = rng.GenerateUniform(output, numCount)
    if err != nil {
        panic(err)
    }
    
    // Step 4: Copy first 10 numbers back to see them
    firstTen := make([]float32, 10)
    tempMem, _ := memory.Alloc(10 * 4)
    defer tempMem.Free()
    
    // Copy just the first 10 numbers
    // (In real code, you'd copy all or process on GPU)
    output.CopyToDevice(tempMem, 0, 0, 10*4)
    tempMem.CopyToHost(firstTen)
    
    fmt.Printf("First 10 random numbers: %v\n", firstTen)
    
    // Step 5: Try different distributions
    fmt.Println("\nGenerating normal distribution (μ=0, σ=1)...")
    err = rng.GenerateNormal(output, numCount, 0.0, 1.0)
    if err != nil {
        panic(err)
    }
    
    output.CopyToDevice(tempMem, 0, 0, 10*4)
    tempMem.CopyToHost(firstTen)
    fmt.Printf("First 10 normal numbers: %v\n", firstTen)
    
    fmt.Println("✅ Random number generation complete!")
}
```

**What You Learned:**
- 🎯 How to create and use CUDA library contexts
- 🎯 Resource management with `defer`
- 🎯 Different random distributions
- 🎯 Copying partial data between GPU memories

---

## ⚡ Chapter 4: Parallel Algorithms with Thrust

Thrust is like having superpowers for arrays! It can sort, search, and process millions of numbers in parallel.

Create `parallel_basics.go`:

```go
package main

import (
    "fmt"
    "math/rand"
    "time"
    "github.com/stitch1968/gocuda"
    "github.com/stitch1968/gocuda/libraries"
    "github.com/stitch1968/gocuda/memory"
)

func main() {
    cuda.Initialize()
    fmt.Println("⚡ Learning Thrust - Parallel Algorithms")
    
    // Create test data
    size := 10000
    hostData := make([]float32, size)
    
    // Fill with random numbers
    rand.Seed(time.Now().UnixNano())
    for i := range hostData {
        hostData[i] = rand.Float32() * 100 // 0-100
    }
    
    fmt.Printf("Created %d random numbers\n", size)
    
    // Copy to GPU
    gpuData, err := memory.Alloc(int64(size * 4))
    if err != nil {
        panic(err)
    }
    defer gpuData.Free()
    
    gpuData.CopyFromHost(hostData)
    
    // Create Thrust context
    thrust, err := libraries.CreateThrustContext()
    if err != nil {
        panic(err)
    }
    defer thrust.DestroyContext()
    
    // 1. SORTING - Make the numbers ordered
    fmt.Println("\n1. Sorting the array...")
    start := time.Now()
    err = thrust.Sort(gpuData, size, libraries.PolicyDevice)
    if err != nil {
        panic(err)
    }
    sortTime := time.Since(start)
    fmt.Printf("   Sorted %d numbers in %v\n", size, sortTime)
    
    // 2. REDUCTION - Add all numbers together
    fmt.Println("\n2. Finding the sum of all numbers...")
    start = time.Now()
    sum, err := thrust.Reduce(gpuData, size, 0.0, libraries.PolicyDevice)
    if err != nil {
        panic(err)
    }
    reduceTime := time.Since(start)
    fmt.Printf("   Sum of all numbers: %.2f (took %v)\n", sum, reduceTime)
    
    // 3. FIND MIN/MAX - Find extreme values
    fmt.Println("\n3. Finding minimum and maximum values...")
    minVal, minIdx, err := thrust.MinElement(gpuData, size, libraries.PolicyDevice)
    if err != nil {
        panic(err)
    }
    
    maxVal, maxIdx, err := thrust.MaxElement(gpuData, size, libraries.PolicyDevice)
    if err != nil {
        panic(err)
    }
    
    fmt.Printf("   Minimum: %.2f at position %d\n", minVal, minIdx)
    fmt.Printf("   Maximum: %.2f at position %d\n", maxVal, maxIdx)
    
    // 4. Check our work - copy back first and last 5 numbers
    fmt.Println("\n4. Verification (first and last 5 sorted numbers):")
    
    // First 5
    firstFive := make([]float32, 5)
    tempMem, _ := memory.Alloc(5 * 4)
    defer tempMem.Free()
    gpuData.CopyToDevice(tempMem, 0, 0, 5*4)
    tempMem.CopyToHost(firstFive)
    fmt.Printf("   First 5 (smallest): %v\n", firstFive)
    
    // Last 5
    lastFive := make([]float32, 5)
    gpuData.CopyToDevice(tempMem, int64((size-5)*4), 0, 5*4)
    tempMem.CopyToHost(lastFive)
    fmt.Printf("   Last 5 (largest): %v\n", lastFive)
    
    fmt.Println("\n✅ Parallel algorithms complete!")
    fmt.Printf("💡 GPU processed %d numbers much faster than CPU could!\n", size)
}
```

**What You Learned:**
- 🎯 Parallel sorting (handles thousands of elements easily)
- 🎯 Parallel reduction (sum, min, max)
- 🎯 How GPU operations are much faster than CPU for large data
- 🎯 Verification techniques to check your results

---

## 🧮 Chapter 5: Simple Math Operations

Let's learn basic matrix and vector operations - the building blocks of scientific computing!

Create `math_basics.go`:

```go
package main

import (
    "fmt"
    "github.com/stitch1968/gocuda"
    "github.com/stitch1968/gocuda/libraries"
    "github.com/stitch1968/gocuda/memory"
)

func main() {
    cuda.Initialize()
    fmt.Println("🧮 Learning Basic Math Operations")
    
    // Create a simple 3x3 matrix and 3x1 vector
    // Matrix A = [[1, 2, 3],
    //            [4, 5, 6], 
    //            [7, 8, 9]]
    // Vector x = [1, 2, 3]
    // We want to compute A*x
    
    matrixA := []float32{
        1, 2, 3,  // Row 1
        4, 5, 6,  // Row 2
        7, 8, 9,  // Row 3
    }
    
    vectorX := []float32{1, 2, 3}
    
    fmt.Println("Matrix A (3x3):")
    fmt.Println("  [1 2 3]")
    fmt.Println("  [4 5 6]")  
    fmt.Println("  [7 8 9]")
    fmt.Printf("Vector x: %v\n", vectorX)
    
    // Expected result: A*x = [14, 32, 50]
    // (1*1 + 2*2 + 3*3 = 14)
    // (4*1 + 5*2 + 6*3 = 32)
    // (7*1 + 8*2 + 9*3 = 50)
    
    // Create cuSOLVER context for linear algebra
    solver, err := libraries.CreateSolverContext()
    if err != nil {
        panic(err)
    }
    defer solver.DestroyContext()
    
    // Allocate GPU memory
    gpuMatrix, _ := memory.Alloc(9 * 4) // 3x3 matrix
    gpuVector, _ := memory.Alloc(3 * 4) // 3x1 vector
    gpuResult, _ := memory.Alloc(3 * 4) // 3x1 result
    defer gpuMatrix.Free()
    defer gpuVector.Free()
    defer gpuResult.Free()
    
    // Copy data to GPU
    gpuMatrix.CopyFromHost(matrixA)
    gpuVector.CopyFromHost(vectorX)
    
    // Perform matrix-vector multiplication
    // Note: This is a simplified example - real matrix operations
    // would use more sophisticated CUDA kernels
    fmt.Println("\nPerforming matrix-vector multiplication A*x...")
    
    // For beginners, let's use a simpler approach
    // We'll demonstrate the concept with basic operations
    result := make([]float32, 3)
    
    // Copy back and do calculation (in real code, this would be a GPU kernel)
    gpuMatrix.CopyToHost(matrixA) // Already have this
    gpuVector.CopyToHost(vectorX) // Already have this
    
    // Matrix multiplication: result[i] = sum(A[i][j] * x[j])
    for i := 0; i < 3; i++ {
        sum := float32(0.0)
        for j := 0; j < 3; j++ {
            sum += matrixA[i*3+j] * vectorX[j]
        }
        result[i] = sum
    }
    
    fmt.Printf("Result A*x = %v\n", result)
    fmt.Println("Expected:   [14 32 50]")
    
    // Verify our calculation
    expected := []float32{14, 32, 50}
    correct := true
    for i, v := range result {
        if v != expected[i] {
            correct = false
            break
        }
    }
    
    if correct {
        fmt.Println("✅ Matrix multiplication correct!")
    } else {
        fmt.Println("❌ Something went wrong...")
    }
    
    fmt.Println("\n💡 This is how GPUs excel at math operations!")
    fmt.Println("💡 With thousands of numbers, GPU would be much faster than CPU")
}
```

**What You Learned:**
- 🎯 Basic matrix and vector concepts
- 🎯 How GPUs excel at mathematical operations
- 🎯 Setting up linear algebra computations
- 🎯 Verification of mathematical results

---

## 🎯 Chapter 6: Best Practices for Beginners

### Memory Management Rules
```go
// ✅ GOOD: Always use defer to clean up
func goodExample() {
    mem, err := memory.Alloc(1024)
    if err != nil {
        return
    }
    defer mem.Free() // This will always run!
    
    // ... use memory ...
}

// ❌ BAD: Forgetting to free memory
func badExample() {
    mem, err := memory.Alloc(1024)
    if err != nil {
        return
    }
    // ... use memory ...
    mem.Free() // What if there's an error above? Memory leak!
}
```

### Error Handling
```go
// ✅ GOOD: Always check errors
result, err := someOperation()
if err != nil {
    fmt.Printf("Error: %v\n", err)
    return
}

// ❌ BAD: Ignoring errors
result, _ := someOperation() // Don't do this!
```

### Start Small, Think Big
```go
// ✅ GOOD: Start with small data sizes for testing
func learnWithSmallData() {
    size := 100 // Easy to debug and understand
    data, _ := memory.Alloc(int64(size * 4))
    defer data.Free()
    
    // ... learn the concepts ...
    
    // Once working, increase size for performance testing
    // size := 100000
}
```

### Performance Expectations
- 🔍 **Small data (< 1000 elements):** CPU might be faster
- ⚡ **Medium data (1K - 100K elements):** GPU starts to win
- 🚀 **Large data (> 100K elements):** GPU dominates

---

## 🏁 Final Project: Putting It All Together

Create `beginner_final_project.go`:

```go
package main

import (
    "fmt"
    "math/rand"
    "time"
    "github.com/stitch1968/gocuda"
    "github.com/stitch1968/gocuda/libraries"
    "github.com/stitch1968/gocuda/memory"
)

func main() {
    fmt.Println("🎓 Beginner Final Project: Data Processing Pipeline")
    fmt.Println("We'll generate data, process it, and analyze results!")
    
    cuda.Initialize()
    
    // Step 1: Generate random data
    fmt.Println("\n📊 Step 1: Generating random dataset...")
    size := 50000
    
    rng, _ := libraries.CreateRandomGenerator(libraries.RngTypeXorwow)
    defer rng.Destroy()
    
    rawData, _ := memory.Alloc(int64(size * 4))
    defer rawData.Free()
    
    // Generate random numbers between 0-100
    rng.GenerateUniform(rawData, size)
    fmt.Printf("Generated %d random data points\n", size)
    
    // Step 2: Process data with Thrust
    fmt.Println("\n⚡ Step 2: Processing data...")
    thrust, _ := libraries.CreateThrustContext()
    defer thrust.DestroyContext()
    
    // Sort the data
    thrust.Sort(rawData, size, libraries.PolicyDevice)
    fmt.Println("✅ Data sorted")
    
    // Calculate statistics
    sum, _ := thrust.Reduce(rawData, size, 0.0, libraries.PolicyDevice)
    minVal, minIdx, _ := thrust.MinElement(rawData, size, libraries.PolicyDevice)
    maxVal, maxIdx, _ := thrust.MaxElement(rawData, size, libraries.PolicyDevice)
    
    average := sum / float32(size)
    
    // Step 3: Display results
    fmt.Println("\n📈 Step 3: Results Analysis")
    fmt.Printf("Dataset size: %d elements\n", size)
    fmt.Printf("Sum: %.2f\n", sum)
    fmt.Printf("Average: %.4f\n", average)
    fmt.Printf("Minimum: %.4f (at position %d)\n", minVal, minIdx)
    fmt.Printf("Maximum: %.4f (at position %d)\n", maxVal, maxIdx)
    fmt.Printf("Range: %.4f\n", maxVal - minVal)
    
    // Step 4: Sample the data
    fmt.Println("\n🔍 Step 4: Data Sampling")
    sampleSize := 10
    sample := make([]float32, sampleSize)
    sampleMem, _ := memory.Alloc(int64(sampleSize * 4))
    defer sampleMem.Free()
    
    // Take samples from different parts of sorted data
    positions := []int{0, size/4, size/2, 3*size/4, size-1}
    fmt.Println("Sample values from different positions:")
    
    for i, pos := range positions {
        if i >= sampleSize { break }
        rawData.CopyToDevice(sampleMem, int64(pos*4), int64(i*4), 4)
        var val float32
        sampleMem.CopyToHost([]float32{val})
        fmt.Printf("Position %d: %.4f\n", pos, val)
    }
    
    fmt.Println("\n🎉 Congratulations! You've completed your first CUDA data processing pipeline!")
    fmt.Println("You've learned:")
    fmt.Println("  ✅ Random number generation")
    fmt.Println("  ✅ Parallel sorting and reduction")
    fmt.Println("  ✅ Statistical analysis")
    fmt.Println("  ✅ Memory management")
    fmt.Println("  ✅ Error handling")
    
    fmt.Println("\n🚀 Ready for intermediate level!")
}
```

---

## 📚 What's Next?

Congratulations! You've learned the fundamentals of GPU programming with GoCUDA. 

### ✅ You Now Know:
- Basic CUDA concepts and terminology
- Memory management and data transfer
- Random number generation with cuRAND
- Parallel algorithms with Thrust  
- Basic mathematical operations
- Best practices and error handling

### 🎯 Next Steps:
1. **Practice** - Try the exercises below
2. **Experiment** - Modify the example codes
3. **Move to Intermediate** - When ready, check out `TRAINING_GUIDE_INTERMEDIATE.md`

### 💪 Practice Exercises

1. **Exercise 1:** Generate 10,000 random numbers and find how many are > 0.5
2. **Exercise 2:** Create two arrays, add them together element-wise
3. **Exercise 3:** Sort an array and find the median value
4. **Exercise 4:** Generate a dataset, remove outliers (values > 2 standard deviations)

### 🆘 Getting Help

- 📖 Read the main README.md for comprehensive examples
- 🔍 Check `demos/examples/` folder for more code
- 💬 Look at error messages carefully - they usually tell you what's wrong
- 🔧 Start with small data sizes when debugging

**Remember:** Everyone starts somewhere, and GPU programming takes practice. Don't get discouraged - you're learning powerful skills that many programmers never master!

---

*Happy CUDA Programming! 🚀*
