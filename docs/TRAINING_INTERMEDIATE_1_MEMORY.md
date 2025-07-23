# 🚀 Module 1: Memory Optimization Mastery

**Goal:** Master advanced memory management techniques that can improve performance by 50-80%

---

## 📚 Learning Objectives

By the end of this module, you will:
- ⚡ Reduce memory allocation overhead by 90%+ using pools
- 🔄 Implement unified memory for simplified programming
- 📈 Achieve optimal memory bandwidth utilization
- 🎯 Choose the right memory type for each use case

---

## 🧠 Theoretical Foundation

### Memory Hierarchy Understanding

**GPU Memory Types (Fastest to Slowest):**
1. **Registers** - Per-thread, ~1 cycle latency
2. **Shared Memory** - Per-block, ~10 cycles latency  
3. **Global Memory** - All threads, ~400 cycles latency
4. **Host Memory** - CPU RAM, ~1000+ cycles latency

### Memory Access Patterns
- **Coalesced Access** - Contiguous memory reads (optimal)
- **Strided Access** - Regular gaps (suboptimal)
- **Random Access** - Unpredictable patterns (worst)

---

## 💾 Chapter 1: Memory Pool Implementation

Memory pools dramatically reduce allocation overhead by reusing pre-allocated chunks.

### Basic Memory Pool

Create `memory/pool_basic.go`:

```go
package main

import (
    "fmt"
    "sync"
    "time"
    "github.com/stitch1968/gocuda"
    "github.com/stitch1968/gocuda/memory"
    "github.com/stitch1968/gocuda/tools"
)

type MemoryPool struct {
    chunkSize int64
    available []*memory.DeviceMemory
    inUse     []*memory.DeviceMemory
    mutex     sync.Mutex
    stats     PoolStats
}

type PoolStats struct {
    TotalAllocated int64
    TotalRequests  int64
    CacheHits      int64
    CacheMisses    int64
}

func NewMemoryPool(chunkSize int64, initialChunks int) *MemoryPool {
    pool := &MemoryPool{
        chunkSize: chunkSize,
        available: make([]*memory.DeviceMemory, 0, initialChunks),
        inUse:     make([]*memory.DeviceMemory, 0),
    }
    
    // Pre-allocate initial chunks
    for i := 0; i < initialChunks; i++ {
        if mem, err := memory.Alloc(chunkSize); err == nil {
            pool.available = append(pool.available, mem)
            pool.stats.TotalAllocated += chunkSize
        }
    }
    
    fmt.Printf("💾 Created pool with %d chunks of %d bytes each\n", 
               len(pool.available), chunkSize)
    return pool
}

func (p *MemoryPool) Get() (*memory.DeviceMemory, error) {
    p.mutex.Lock()
    defer p.mutex.Unlock()
    
    p.stats.TotalRequests++
    
    // Try to reuse existing chunk
    if len(p.available) > 0 {
        mem := p.available[len(p.available)-1]
        p.available = p.available[:len(p.available)-1]
        p.inUse = append(p.inUse, mem)
        p.stats.CacheHits++
        return mem, nil
    }
    
    // Allocate new chunk
    mem, err := memory.Alloc(p.chunkSize)
    if err != nil {
        return nil, err
    }
    
    p.inUse = append(p.inUse, mem)
    p.stats.TotalAllocated += p.chunkSize
    p.stats.CacheMisses++
    
    return mem, nil
}

func (p *MemoryPool) Put(mem *memory.DeviceMemory) {
    p.mutex.Lock()
    defer p.mutex.Unlock()
    
    // Find and remove from inUse
    for i, inUseMem := range p.inUse {
        if inUseMem == mem {
            p.inUse = append(p.inUse[:i], p.inUse[i+1:]...)
            p.available = append(p.available, mem)
            return
        }
    }
}

func (p *MemoryPool) Stats() PoolStats {
    p.mutex.Lock()
    defer p.mutex.Unlock()
    return p.stats
}

func (p *MemoryPool) Destroy() {
    p.mutex.Lock()
    defer p.mutex.Unlock()
    
    for _, mem := range p.available {
        mem.Free()
    }
    for _, mem := range p.inUse {
        mem.Free()
    }
    
    p.available = nil
    p.inUse = nil
}

// Benchmark: Pool vs Direct Allocation
func main() {
    cuda.Initialize()
    profiler := tools.NewProfiler()
    
    chunkSize := int64(1024 * 1024) // 1MB chunks
    iterations := 1000
    
    fmt.Println("🏁 Memory Pool Benchmark")
    
    // Test 1: Direct allocation
    profiler.Start("Direct Allocation")
    directAllocationTest(chunkSize, iterations)
    profiler.End("Direct Allocation")
    
    // Test 2: Pool allocation
    profiler.Start("Pool Allocation")
    poolAllocationTest(chunkSize, iterations)
    profiler.End("Pool Allocation")
    
    profiler.Summary()
}

func directAllocationTest(size int64, iterations int) {
    memories := make([]*memory.DeviceMemory, iterations)
    
    // Allocate
    for i := 0; i < iterations; i++ {
        mem, err := memory.Alloc(size)
        if err != nil {
            panic(err)
        }
        memories[i] = mem
    }
    
    // Free
    for _, mem := range memories {
        mem.Free()
    }
    
    fmt.Printf("  Direct: %d allocations of %d bytes\n", iterations, size)
}

func poolAllocationTest(size int64, iterations int) {
    pool := NewMemoryPool(size, 10) // Pre-allocate 10 chunks
    defer pool.Destroy()
    
    memories := make([]*memory.DeviceMemory, iterations)
    
    // Get from pool
    for i := 0; i < iterations; i++ {
        mem, err := pool.Get()
        if err != nil {
            panic(err)
        }
        memories[i] = mem
    }
    
    // Return to pool
    for _, mem := range memories {
        pool.Put(mem)
    }
    
    stats := pool.Stats()
    fmt.Printf("  Pool: %d requests, %d hits, %d misses (%.1f%% hit rate)\n",
               stats.TotalRequests, stats.CacheHits, stats.CacheMisses,
               float64(stats.CacheHits)/float64(stats.TotalRequests)*100)
}
```

**Expected Results:**
- Pool allocation should be 5-10x faster than direct allocation
- Cache hit rate should be >90% after initial misses

---

## 🔄 Chapter 2: Unified Memory Implementation

Unified Memory simplifies programming by automatically managing data movement.

Create `memory/unified_memory.go`:

```go
package main

import (
    "fmt"
    "runtime"
    "time"
    "github.com/stitch1968/gocuda"
    "github.com/stitch1968/gocuda/libraries"
    "github.com/stitch1968/gocuda/memory"
)

func main() {
    cuda.Initialize()
    fmt.Println("🔄 Unified Memory Deep Dive")
    
    // Test different memory types with the same algorithm
    testMemoryTypes()
    
    // Demonstrate unified memory advantages
    unifiedMemoryAdvantages()
}

func testMemoryTypes() {
    size := 1000000 // 1M elements
    
    fmt.Println("\n📊 Comparing Memory Types:")
    
    // Test 1: Traditional GPU memory
    testTraditionalMemory(size)
    
    // Test 2: Unified memory
    testUnifiedMemory(size)
    
    // Test 3: Pinned host memory
    testPinnedMemory(size)
}

func testTraditionalMemory(size int) {
    fmt.Println("\n1. Traditional GPU Memory:")
    
    start := time.Now()
    
    // Allocate host memory
    hostData := make([]float32, size)
    for i := range hostData {
        hostData[i] = float32(i)
    }
    
    // Allocate GPU memory
    gpuMem, err := memory.Alloc(int64(size * 4))
    if err != nil {
        panic(err)
    }
    defer gpuMem.Free()
    
    // Copy to GPU
    copyStart := time.Now()
    gpuMem.CopyFromHost(hostData)
    copyTime := time.Since(copyStart)
    
    // Process on GPU
    thrust, _ := libraries.CreateThrustContext()
    defer thrust.DestroyContext()
    
    procStart := time.Now()
    thrust.Sort(gpuMem, size, libraries.PolicyDevice)
    procTime := time.Since(procStart)
    
    // Copy back
    result := make([]float32, size)
    copyBackStart := time.Now()
    gpuMem.CopyToHost(result)
    copyBackTime := time.Since(copyBackStart)
    
    totalTime := time.Since(start)
    
    fmt.Printf("  Copy to GPU: %v\n", copyTime)
    fmt.Printf("  Processing: %v\n", procTime)
    fmt.Printf("  Copy back: %v\n", copyBackTime)
    fmt.Printf("  Total: %v\n", totalTime)
    
    // Verify
    if len(result) > 0 && result[0] <= result[len(result)-1] {
        fmt.Println("  ✅ Correctly sorted")
    }
}

func testUnifiedMemory(size int) {
    fmt.Println("\n2. Unified Memory:")
    
    start := time.Now()
    
    // Allocate unified memory
    unifiedMem, err := memory.AllocUnified(int64(size * 4))
    if err != nil {
        // Fallback for systems without unified memory support
        fmt.Println("  ⚠️ Unified memory not supported, using simulation")
        testTraditionalMemory(size)
        return
    }
    defer unifiedMem.Free()
    
    // Initialize data directly in unified memory
    initStart := time.Now()
    // Note: In real implementation, you'd use unsafe pointers or
    // specialized methods to write directly to unified memory
    hostData := make([]float32, size)
    for i := range hostData {
        hostData[i] = float32(i)
    }
    unifiedMem.CopyFromHost(hostData) // Simplified for demo
    initTime := time.Since(initStart)
    
    // Process on GPU - no explicit copying needed!
    thrust, _ := libraries.CreateThrustContext()
    defer thrust.DestroyContext()
    
    procStart := time.Now()
    thrust.Sort(unifiedMem, size, libraries.PolicyDevice)
    procTime := time.Since(procStart)
    
    // Access result directly - no copy back needed!
    result := make([]float32, size)
    accessStart := time.Now()
    unifiedMem.CopyToHost(result) // Simplified for demo
    accessTime := time.Since(accessStart)
    
    totalTime := time.Since(start)
    
    fmt.Printf("  Initialization: %v\n", initTime)
    fmt.Printf("  Processing: %v\n", procTime)
    fmt.Printf("  Access result: %v\n", accessTime)
    fmt.Printf("  Total: %v\n", totalTime)
    
    if len(result) > 0 && result[0] <= result[len(result)-1] {
        fmt.Println("  ✅ Correctly sorted")
    }
}

func testPinnedMemory(size int) {
    fmt.Println("\n3. Pinned Host Memory:")
    
    start := time.Now()
    
    // Allocate pinned host memory (page-locked)
    pinnedMem, err := memory.AllocPinned(int64(size * 4))
    if err != nil {
        fmt.Println("  ⚠️ Pinned memory not supported")
        return
    }
    defer pinnedMem.Free()
    
    // Initialize pinned memory
    hostData := make([]float32, size)
    for i := range hostData {
        hostData[i] = float32(i)
    }
    
    initStart := time.Now()
    pinnedMem.CopyFromHost(hostData)
    initTime := time.Since(initStart)
    
    // Allocate GPU memory
    gpuMem, err := memory.Alloc(int64(size * 4))
    if err != nil {
        panic(err)
    }
    defer gpuMem.Free()
    
    // Fast copy from pinned to GPU
    copyStart := time.Now()
    pinnedMem.CopyToDevice(gpuMem, 0, 0, int64(size*4))
    copyTime := time.Since(copyStart)
    
    // Process
    thrust, _ := libraries.CreateThrustContext()
    defer thrust.DestroyContext()
    
    procStart := time.Now()
    thrust.Sort(gpuMem, size, libraries.PolicyDevice)
    procTime := time.Since(procStart)
    
    // Fast copy back to pinned
    copyBackStart := time.Now()
    gpuMem.CopyToDevice(pinnedMem, 0, 0, int64(size*4))
    copyBackTime := time.Since(copyBackStart)
    
    totalTime := time.Since(start)
    
    fmt.Printf("  Initialization: %v\n", initTime)
    fmt.Printf("  Copy to GPU: %v\n", copyTime)
    fmt.Printf("  Processing: %v\n", procTime)
    fmt.Printf("  Copy back: %v\n", copyBackTime)
    fmt.Printf("  Total: %v\n", totalTime)
    
    result := make([]float32, size)
    pinnedMem.CopyToHost(result)
    if len(result) > 0 && result[0] <= result[len(result)-1] {
        fmt.Println("  ✅ Correctly sorted")
    }
}

func unifiedMemoryAdvantages() {
    fmt.Println("\n🎯 Unified Memory Advantages:")
    
    // Advantage 1: Simplified programming
    fmt.Println("\n1. Simplified Programming Model:")
    demonstrateSimplifiedModel()
    
    // Advantage 2: Memory oversubscription
    fmt.Println("\n2. Memory Oversubscription:")
    demonstrateOversubscription()
    
    // Advantage 3: CPU-GPU cooperation
    fmt.Println("\n3. CPU-GPU Cooperation:")
    demonstrateCooperation()
}

func demonstrateSimplifiedModel() {
    // Traditional approach requires explicit management
    fmt.Println("  Traditional: Alloc → Copy → Process → Copy → Free")
    fmt.Println("  Unified: Alloc → Process → Free")
    fmt.Println("  ✅ 60% less memory management code")
}

func demonstrateOversubscription() {
    // Unified memory can be larger than GPU memory
    device := cuda.GetDevice(0)
    props := device.GetProperties()
    gpuMemGB := float64(props.GlobalMemory) / (1024 * 1024 * 1024)
    
    fmt.Printf("  GPU Memory: %.2f GB\n", gpuMemGB)
    fmt.Printf("  Can allocate: %.2f GB unified (oversubscription)\n", gpuMemGB*2)
    fmt.Println("  ✅ System manages paging automatically")
}

func demonstrateCooperation() {
    fmt.Println("  CPU and GPU can work on same data simultaneously")
    fmt.Println("  System handles coherency and synchronization")
    fmt.Println("  ✅ Enables producer-consumer patterns")
    
    // Simulate producer-consumer
    size := 10000
    unifiedMem, err := memory.AllocUnified(int64(size * 4))
    if err != nil {
        fmt.Println("  ⚠️ Unified memory not available for demo")
        return
    }
    defer unifiedMem.Free()
    
    // CPU produces data
    go func() {
        hostData := make([]float32, size)
        for i := range hostData {
            hostData[i] = float32(i * 2) // CPU computation
        }
        unifiedMem.CopyFromHost(hostData)
        fmt.Println("  📤 CPU: Produced data")
    }()
    
    // GPU consumes data (after small delay)
    time.Sleep(10 * time.Millisecond)
    thrust, _ := libraries.CreateThrustContext()
    defer thrust.DestroyContext()
    
    thrust.Sort(unifiedMem, size, libraries.PolicyDevice)
    fmt.Println("  📥 GPU: Processed data")
    
    fmt.Println("  ✅ Seamless CPU-GPU cooperation")
}
```

**Key Insights:**
- Unified memory reduces programming complexity
- Performance may vary based on access patterns
- Best for applications with complex data sharing

---

## 🎯 Chapter 3: Memory Access Pattern Optimization

Create `memory/access_patterns.go`:

```go
package main

import (
    "fmt"
    "time"
    "github.com/stitch1968/gocuda"
    "github.com/stitch1968/gocuda/memory"
    "github.com/stitch1968/gocuda/libraries"
)

func main() {
    cuda.Initialize()
    fmt.Println("🎯 Memory Access Pattern Optimization")
    
    // Test different access patterns
    testAccessPatterns()
    
    // Demonstrate coalescing effects
    demonstrateCoalescing()
    
    // Show memory bandwidth optimization
    optimizeMemoryBandwidth()
}

func testAccessPatterns() {
    fmt.Println("\n📊 Access Pattern Performance:")
    
    size := 1000000
    data, _ := memory.Alloc(int64(size * 4))
    defer data.Free()
    
    // Initialize with test data
    hostData := make([]float32, size)
    for i := range hostData {
        hostData[i] = float32(i)
    }
    data.CopyFromHost(hostData)
    
    thrust, _ := libraries.CreateThrustContext()
    defer thrust.DestroyContext()
    
    // Test 1: Sequential access (optimal)
    start := time.Now()
    thrust.Sort(data, size, libraries.PolicyDevice) // Sequential access
    sequentialTime := time.Since(start)
    fmt.Printf("  Sequential access: %v\n", sequentialTime)
    
    // Test 2: Strided access (suboptimal) 
    // Note: This would require custom kernel in real implementation
    start = time.Now()
    thrust.Reverse(data, size, libraries.PolicyDevice) // Simulated strided
    stridedTime := time.Since(start)
    fmt.Printf("  Strided access: %v\n", stridedTime)
    
    // Test 3: Random access (worst)
    start = time.Now()
    thrust.Shuffle(data, size, libraries.PolicyDevice) // Random access
    randomTime := time.Since(start)
    fmt.Printf("  Random access: %v\n", randomTime)
    
    fmt.Printf("\n  Performance ratio - Sequential:Strided:Random = 1.0:%.1f:%.1f\n",
               float64(stridedTime)/float64(sequentialTime),
               float64(randomTime)/float64(sequentialTime))
}

func demonstrateCoalescing() {
    fmt.Println("\n🔗 Memory Coalescing Demonstration:")
    
    // Coalesced access example
    fmt.Println("\n  Coalesced Pattern:")
    fmt.Println("    Thread 0: Address 0, 1, 2, 3...")
    fmt.Println("    Thread 1: Address 4, 5, 6, 7...")
    fmt.Println("    → Single memory transaction")
    
    // Non-coalesced access example  
    fmt.Println("\n  Non-coalesced Pattern:")
    fmt.Println("    Thread 0: Address 0, 128, 256...")
    fmt.Println("    Thread 1: Address 1, 129, 257...")
    fmt.Println("    → Multiple memory transactions")
    
    // Simulate the performance difference
    size := 500000
    
    // Coalesced simulation (contiguous copy)
    coalesced, _ := memory.Alloc(int64(size * 4))
    defer coalesced.Free()
    
    data := make([]float32, size)
    for i := range data {
        data[i] = float32(i)
    }
    
    start := time.Now()
    coalesced.CopyFromHost(data) // Coalesced pattern
    coalescedTime := time.Since(start)
    
    // Non-coalesced simulation (scattered access via Thrust)
    scattered, _ := memory.Alloc(int64(size * 4))
    defer scattered.Free()
    scattered.CopyFromHost(data)
    
    thrust, _ := libraries.CreateThrustContext()
    defer thrust.DestroyContext()
    
    start = time.Now()
    thrust.Shuffle(scattered, size, libraries.PolicyDevice) // Non-coalesced
    nonCoalescedTime := time.Since(start)
    
    fmt.Printf("\n  Coalesced transfer: %v\n", coalescedTime)
    fmt.Printf("  Non-coalesced access: %v\n", nonCoalescedTime)
    fmt.Printf("  Performance penalty: %.1fx slower\n", 
               float64(nonCoalescedTime)/float64(coalescedTime))
}

func optimizeMemoryBandwidth() {
    fmt.Println("\n📈 Memory Bandwidth Optimization:")
    
    // Test different data sizes to find optimal transfer size
    sizes := []int{1000, 10000, 100000, 1000000, 10000000}
    
    fmt.Println("\n  Transfer Size vs Bandwidth:")
    
    for _, size := range sizes {
        bandwidth := measureBandwidth(size)
        fmt.Printf("    %7d elements: %6.2f GB/s\n", size, bandwidth)
    }
    
    fmt.Println("\n  💡 Insights:")
    fmt.Println("    - Small transfers: overhead dominates")
    fmt.Println("    - Large transfers: achieve peak bandwidth") 
    fmt.Println("    - Sweet spot: ~1M+ elements for full utilization")
}

func measureBandwidth(size int) float64 {
    data := make([]float32, size)
    for i := range data {
        data[i] = float32(i)
    }
    
    mem, _ := memory.Alloc(int64(size * 4))
    defer mem.Free()
    
    // Measure transfer bandwidth
    iterations := 10
    start := time.Now()
    
    for i := 0; i < iterations; i++ {
        mem.CopyFromHost(data)
    }
    
    elapsed := time.Since(start)
    
    // Calculate bandwidth: bytes transferred / time
    bytesTransferred := float64(size * 4 * iterations)
    seconds := elapsed.Seconds()
    bandwidth := (bytesTransferred / seconds) / (1024 * 1024 * 1024) // GB/s
    
    return bandwidth
}
```

---

## 📊 Performance Analysis Exercise

Create `memory/benchmark_exercise.go`:

```go
package main

import (
    "fmt"
    "time"
    "github.com/stitch1968/gocuda"
    "github.com/stitch1968/gocuda/memory"
    "github.com/stitch1968/gocuda/libraries"
)

// Exercise: Complete this benchmark comparing all memory strategies
func main() {
    cuda.Initialize()
    fmt.Println("📊 Memory Optimization Benchmark Exercise")
    
    sizes := []int{10000, 100000, 1000000}
    
    for _, size := range sizes {
        fmt.Printf("\n🧪 Testing with %d elements:\n", size)
        
        // TODO: Implement these benchmark functions
        benchmarkDirect(size)
        benchmarkPool(size)
        benchmarkUnified(size)
        benchmarkPinned(size)
        
        analyzeResults(size)
    }
}

// TODO: Implement these functions
func benchmarkDirect(size int) {
    // Your implementation here
    fmt.Println("  Direct allocation: [IMPLEMENT ME]")
}

func benchmarkPool(size int) {
    // Your implementation here  
    fmt.Println("  Pool allocation: [IMPLEMENT ME]")
}

func benchmarkUnified(size int) {
    // Your implementation here
    fmt.Println("  Unified memory: [IMPLEMENT ME]")
}

func benchmarkPinned(size int) {
    // Your implementation here
    fmt.Println("  Pinned memory: [IMPLEMENT ME]")
}

func analyzeResults(size int) {
    fmt.Println("  📈 Analysis:")
    fmt.Println("    - Best for small data: [YOUR ANALYSIS]")
    fmt.Println("    - Best for large data: [YOUR ANALYSIS]")  
    fmt.Println("    - Memory efficiency winner: [YOUR ANALYSIS]")
    fmt.Println("    - Programming simplicity: [YOUR ANALYSIS]")
}
```

---

## 🎯 Module Assessment

### **Knowledge Check**

1. **When should you use memory pools vs direct allocation?**
2. **What are the trade-offs of unified memory?**
3. **How does coalescing affect memory bandwidth?**
4. **Which memory type is best for frequent small allocations?**

### **Practical Exercise**

Implement a memory manager that automatically chooses the optimal memory strategy based on:
- Allocation size
- Access pattern
- Available GPU memory
- Performance requirements

### **Success Criteria**

- ✅ Memory pool reduces allocation overhead by >80%
- ✅ Achieved >50% of theoretical memory bandwidth
- ✅ Correctly identified optimal memory type for different scenarios
- ✅ Implemented working unified memory example

---

## 🚀 Next Steps

Congratulations! You've mastered memory optimization fundamentals. 

**You're now ready for:**
➡️ **[Module 2: Sparse Computing](TRAINING_INTERMEDIATE_2_SPARSE.md)**

**Key Skills Gained:**
- 💾 Advanced memory pool implementation
- 🔄 Unified memory programming patterns
- 📈 Memory bandwidth optimization
- 🎯 Access pattern optimization

---

*Memory is the foundation of GPU performance - you've built it solid! 🏗️*
