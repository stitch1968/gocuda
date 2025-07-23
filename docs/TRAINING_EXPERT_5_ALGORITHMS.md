# 🧬 Expert Module 5: GPU-Native Algorithms

**Goal:** Design and implement algorithms specifically optimized for GPU architectures, exploiting parallelism, memory hierarchy, and specialized units

---

## 📚 Learning Objectives

By the end of this module, you will:
- 🧬 **Design GPU-native algorithms** from first principles
- 🔀 **Master parallel design patterns** - reduction, scan, sort, graph algorithms
- 🎯 **Exploit GPU-specific features** - warp primitives, tensor cores, specialized units
- 📊 **Analyze algorithmic complexity** in the GPU context
- ⚡ **Achieve theoretical performance limits** for custom algorithms

---

## 🧠 Theoretical Foundation

### GPU Algorithm Design Principles

**Parallel Algorithm Categories:**
```
Data-Parallel Algorithms:
├── Map (element-wise operations)
├── Reduce (aggregation operations)
├── Scan (prefix operations)
├── Histogram (counting operations)
└── Filter (selection operations)

Task-Parallel Algorithms:
├── Fork-Join (divide-and-conquer)
├── Pipeline (streaming operations)
├── Work-Stealing (load balancing)
└── Producer-Consumer (queuing)

Graph Algorithms:
├── BFS/DFS (traversal)
├── Shortest Path (Dijkstra, Floyd-Warshall)
├── Connected Components (Union-Find)
└── Maximum Flow (Push-Relabel)
```

### GPU Memory Hierarchy Exploitation

**Memory Access Patterns:**
- **Coalesced**: Sequential access within warp
- **Strided**: Regular patterns with fixed stride
- **Gather**: Random access with good locality
- **Scatter**: Random write with conflict resolution

**Algorithm-Memory Mapping:**
```
Global Memory → Large datasets, sequential access
Shared Memory → Tile-based algorithms, data reuse
Constant Memory → Read-only parameters, broadcast
Texture Memory → Spatial locality, interpolation
Register Memory → Temporary variables, loop counters
```

### Complexity Analysis for GPUs

**Performance Metrics:**
- **Work**: Total operations O(W)
- **Span**: Critical path length O(S)
- **Parallelism**: P = W/S
- **GPU Efficiency**: Actual speedup / Theoretical speedup

---

## 🧬 Chapter 1: Fundamental Parallel Primitives

### GPU-Native Reduction Algorithms

Create `algorithms/parallel_primitives.go`:

```go
package main

import (
    "fmt"
    "math"
    "time"
    "sync"
    "github.com/stitch1968/gocuda"
    "github.com/stitch1968/gocuda/memory"
    "github.com/stitch1968/gocuda/warp"
)

// GPU-native algorithm implementations
type ParallelPrimitives struct {
    ctx          *cuda.Context
    streams      []*cuda.Stream
    tempBuffers  []*memory.DeviceMemory
    
    // Algorithm-specific optimizations
    warpSize     int
    smCount      int
    maxThreads   int
    
    // Performance metrics
    operationStats map[string]*OperationStats
    mutex         sync.RWMutex
}

type OperationStats struct {
    TotalCalls    int64
    TotalTime     time.Duration
    TotalElements int64
    BestThroughput float64 // Elements/second
}

type ReductionOp func(a, b float32) float32

// Standard reduction operations
var (
    SumOp = func(a, b float32) float32 { return a + b }
    MaxOp = func(a, b float32) float32 { 
        if a > b { return a } else { return b } 
    }
    MinOp = func(a, b float32) float32 { 
        if a < b { return a } else { return b } 
    }
    ProductOp = func(a, b float32) float32 { return a * b }
)

func NewParallelPrimitives() *ParallelPrimitives {
    cuda.Initialize()
    
    device := cuda.GetDevice(0)
    props := device.GetProperties()
    
    pp := &ParallelPrimitives{
        ctx:         cuda.GetDefaultContext(),
        streams:     make([]*cuda.Stream, 4),
        tempBuffers: make([]*memory.DeviceMemory, 4),
        warpSize:    props.WarpSize,
        smCount:     props.MultiprocessorCount,
        maxThreads:  props.MaxThreadsPerBlock,
        operationStats: make(map[string]*OperationStats),
    }
    
    // Initialize streams and temp buffers
    for i := range pp.streams {
        stream, _ := pp.ctx.CreateStream()
        pp.streams[i] = stream
        
        // Allocate temporary buffer for each stream
        tempBuffer, _ := memory.Alloc(1024 * 1024 * 4) // 4MB
        pp.tempBuffers[i] = tempBuffer
    }
    
    fmt.Printf("🧬 Parallel Primitives initialized: %d SMs, %d warp size\n", 
               pp.smCount, pp.warpSize)
    
    return pp
}

// High-performance reduction with multiple algorithms
func (pp *ParallelPrimitives) Reduce(data *memory.DeviceMemory, size int, op ReductionOp) (float32, error) {
    fmt.Printf("🔄 GPU Reduction: %d elements\n", size)
    
    startTime := time.Now()
    defer pp.recordOperationStats("Reduce", startTime, size)
    
    // Choose optimal reduction strategy based on size
    if size <= pp.warpSize {
        return pp.warpReduce(data, size, op)
    } else if size <= pp.maxThreads {
        return pp.blockReduce(data, size, op)
    } else {
        return pp.hierarchicalReduce(data, size, op)
    }
}

// Warp-level reduction for small arrays
func (pp *ParallelPrimitives) warpReduce(data *memory.DeviceMemory, size int, op ReductionOp) (float32, error) {
    // Copy data to host for warp simulation
    hostData := make([]float32, size)
    err := data.CopyToHost(hostData)
    if err != nil {
        return 0, err
    }
    
    // Simulate warp shuffle reduction
    return pp.simulateWarpShuffle(hostData, op), nil
}

func (pp *ParallelPrimitives) simulateWarpShuffle(data []float32, op ReductionOp) float32 {
    // Simulate GPU warp shuffle-based reduction
    // In real implementation: for (int offset = warpSize/2; offset > 0; offset /= 2)
    //                           value = op(value, __shfl_down_sync(0xFFFFFFFF, value, offset));
    
    if len(data) == 0 {
        return 0.0
    }
    
    result := data[0]
    for i := 1; i < len(data); i++ {
        result = op(result, data[i])
    }
    
    return result
}

// Block-level reduction for medium arrays
func (pp *ParallelPrimitives) blockReduce(data *memory.DeviceMemory, size int, op ReductionOp) (float32, error) {
    hostData := make([]float32, size)
    err := data.CopyToHost(hostData)
    if err != nil {
        return 0, err
    }
    
    // Simulate shared memory reduction
    return pp.simulateSharedMemoryReduce(hostData, op), nil
}

func (pp *ParallelPrimitives) simulateSharedMemoryReduce(data []float32, op ReductionOp) float32 {
    // Simulate GPU shared memory reduction with tree-based approach
    blockSize := pp.maxThreads
    if len(data) < blockSize {
        blockSize = len(data)
    }
    
    // Copy to simulate shared memory
    sharedData := make([]float32, blockSize)
    copy(sharedData, data[:min(len(data), blockSize)])
    
    // Tree reduction in shared memory
    for stride := blockSize / 2; stride > 0; stride /= 2 {
        for i := 0; i < stride && i < len(sharedData) && i+stride < len(sharedData); i++ {
            sharedData[i] = op(sharedData[i], sharedData[i+stride])
        }
    }
    
    // Handle remaining elements
    result := sharedData[0]
    for i := blockSize; i < len(data); i++ {
        result = op(result, data[i])
    }
    
    return result
}

// Hierarchical reduction for large arrays
func (pp *ParallelPrimitives) hierarchicalReduce(data *memory.DeviceMemory, size int, op ReductionOp) (float32, error) {
    // Multi-level reduction: Grid -> Block -> Warp -> Element
    
    numBlocks := min((size + pp.maxThreads - 1) / pp.maxThreads, pp.smCount * 4)
    elementsPerBlock := (size + numBlocks - 1) / numBlocks
    
    hostData := make([]float32, size)
    err := data.CopyToHost(hostData)
    if err != nil {
        return 0, err
    }
    
    // First level: block reductions
    blockResults := make([]float32, numBlocks)
    
    for blockId := 0; blockId < numBlocks; blockId++ {
        start := blockId * elementsPerBlock
        end := min(start + elementsPerBlock, size)
        
        if start < end {
            blockData := hostData[start:end]
            blockResults[blockId] = pp.simulateSharedMemoryReduce(blockData, op)
        }
    }
    
    // Final reduction of block results
    return pp.simulateWarpShuffle(blockResults, op), nil
}

// Parallel prefix sum (scan) implementation
func (pp *ParallelPrimitives) Scan(data *memory.DeviceMemory, size int, inclusive bool) (*memory.DeviceMemory, error) {
    fmt.Printf("📊 GPU Scan: %d elements (inclusive=%t)\n", size, inclusive)
    
    startTime := time.Now()
    defer pp.recordOperationStats("Scan", startTime, size)
    
    hostData := make([]float32, size)
    err := data.CopyToHost(hostData)
    if err != nil {
        return nil, err
    }
    
    // Implement Hillis-Steele scan for simplicity
    // In production, would use more memory-efficient Blelloch scan
    result := pp.hillisSteeleScan(hostData, inclusive)
    
    // Copy result back to GPU
    resultBuffer, err := memory.Alloc(int64(size * 4))
    if err != nil {
        return nil, err
    }
    
    err = resultBuffer.CopyFromHost(result)
    if err != nil {
        resultBuffer.Free()
        return nil, err
    }
    
    return resultBuffer, nil
}

func (pp *ParallelPrimitives) hillisSteeleScan(data []float32, inclusive bool) []float32 {
    n := len(data)
    result := make([]float32, n)
    
    // Initialize
    if inclusive {
        copy(result, data)
    } else {
        result[0] = 0
        copy(result[1:], data[:n-1])
    }
    
    // Hillis-Steele scan
    for step := 1; step < n; step *= 2 {
        temp := make([]float32, n)
        copy(temp, result)
        
        for i := step; i < n; i++ {
            result[i] = temp[i] + temp[i-step]
        }
    }
    
    return result
}

// GPU-optimized histogram computation
func (pp *ParallelPrimitives) Histogram(data *memory.DeviceMemory, size int, bins int, minVal, maxVal float32) ([]int32, error) {
    fmt.Printf("📈 GPU Histogram: %d elements, %d bins\n", size, bins)
    
    startTime := time.Now()
    defer pp.recordOperationStats("Histogram", startTime, size)
    
    hostData := make([]float32, size)
    err := data.CopyToHost(hostData)
    if err != nil {
        return nil, err
    }
    
    // Use atomic operations for histogram computation
    return pp.atomicHistogram(hostData, bins, minVal, maxVal), nil
}

func (pp *ParallelPrimitives) atomicHistogram(data []float32, bins int, minVal, maxVal float32) []int32 {
    histogram := make([]int32, bins)
    binWidth := (maxVal - minVal) / float32(bins)
    
    // Simulate atomic additions
    var mutex sync.Mutex
    
    // Process in parallel chunks
    chunkSize := len(data) / pp.smCount
    if chunkSize < 1000 {
        chunkSize = len(data)
    }
    
    var wg sync.WaitGroup
    
    for start := 0; start < len(data); start += chunkSize {
        end := min(start+chunkSize, len(data))
        
        wg.Add(1)
        go func(dataChunk []float32) {
            defer wg.Done()
            
            localHist := make([]int32, bins)
            
            for _, value := range dataChunk {
                if value >= minVal && value <= maxVal {
                    binIndex := int((value - minVal) / binWidth)
                    if binIndex >= bins {
                        binIndex = bins - 1
                    }
                    localHist[binIndex]++
                }
            }
            
            // Merge with global histogram
            mutex.Lock()
            for i := 0; i < bins; i++ {
                histogram[i] += localHist[i]
            }
            mutex.Unlock()
        }(data[start:end])
    }
    
    wg.Wait()
    return histogram
}

// GPU-native sorting algorithm
func (pp *ParallelPrimitives) Sort(data *memory.DeviceMemory, size int) error {
    fmt.Printf("🔀 GPU Sort: %d elements\n", size)
    
    startTime := time.Now()
    defer pp.recordOperationStats("Sort", startTime, size)
    
    hostData := make([]float32, size)
    err := data.CopyToHost(hostData)
    if err != nil {
        return err
    }
    
    // Use bitonic sort for GPU efficiency
    pp.bitonicSort(hostData)
    
    return data.CopyFromHost(hostData)
}

func (pp *ParallelPrimitives) bitonicSort(data []float32) {
    n := len(data)
    
    // Pad to power of 2 if necessary
    paddedSize := 1
    for paddedSize < n {
        paddedSize *= 2
    }
    
    if paddedSize > n {
        padded := make([]float32, paddedSize)
        copy(padded, data)
        for i := n; i < paddedSize; i++ {
            padded[i] = math.MaxFloat32 // Padding value
        }
        pp.bitonicSortRecursive(padded, 0, paddedSize, true)
        copy(data, padded[:n])
    } else {
        pp.bitonicSortRecursive(data, 0, n, true)
    }
}

func (pp *ParallelPrimitives) bitonicSortRecursive(data []float32, start, length int, ascending bool) {
    if length > 1 {
        mid := length / 2
        
        // Sort first half ascending
        pp.bitonicSortRecursive(data, start, mid, true)
        
        // Sort second half descending
        pp.bitonicSortRecursive(data, start+mid, mid, false)
        
        // Merge both halves
        pp.bitonicMerge(data, start, length, ascending)
    }
}

func (pp *ParallelPrimitives) bitonicMerge(data []float32, start, length int, ascending bool) {
    if length > 1 {
        mid := length / 2
        
        for i := start; i < start+mid; i++ {
            if (data[i] > data[i+mid]) == ascending {
                data[i], data[i+mid] = data[i+mid], data[i]
            }
        }
        
        pp.bitonicMerge(data, start, mid, ascending)
        pp.bitonicMerge(data, start+mid, mid, ascending)
    }
}

// Stream compaction (filter) operation
func (pp *ParallelPrimitives) Compact(data *memory.DeviceMemory, size int, predicate func(float32) bool) (*memory.DeviceMemory, int, error) {
    fmt.Printf("🎯 GPU Compact: %d elements\n", size)
    
    startTime := time.Now()
    defer pp.recordOperationStats("Compact", startTime, size)
    
    hostData := make([]float32, size)
    err := data.CopyToHost(hostData)
    if err != nil {
        return nil, 0, err
    }
    
    // Two-phase compaction: 1) mark valid elements, 2) scan and scatter
    validMask := make([]int, size)
    validCount := 0
    
    // Phase 1: Apply predicate
    for i, value := range hostData {
        if predicate(value) {
            validMask[i] = 1
            validCount++
        } else {
            validMask[i] = 0
        }
    }
    
    if validCount == 0 {
        empty, _ := memory.Alloc(4)
        return empty, 0, nil
    }
    
    // Phase 2: Exclusive scan to get output positions
    positions := make([]int, size)
    sum := 0
    for i, mask := range validMask {
        positions[i] = sum
        sum += mask
    }
    
    // Phase 3: Scatter valid elements
    compactData := make([]float32, validCount)
    for i, value := range hostData {
        if validMask[i] == 1 {
            compactData[positions[i]] = value
        }
    }
    
    // Copy result to GPU
    result, err := memory.Alloc(int64(validCount * 4))
    if err != nil {
        return nil, 0, err
    }
    
    err = result.CopyFromHost(compactData)
    if err != nil {
        result.Free()
        return nil, 0, err
    }
    
    return result, validCount, nil
}

func (pp *ParallelPrimitives) recordOperationStats(opName string, startTime time.Time, elements int) {
    elapsed := time.Since(startTime)
    throughput := float64(elements) / elapsed.Seconds()
    
    pp.mutex.Lock()
    defer pp.mutex.Unlock()
    
    stats, exists := pp.operationStats[opName]
    if !exists {
        stats = &OperationStats{}
        pp.operationStats[opName] = stats
    }
    
    stats.TotalCalls++
    stats.TotalTime += elapsed
    stats.TotalElements += int64(elements)
    
    if throughput > stats.BestThroughput {
        stats.BestThroughput = throughput
    }
}

func (pp *ParallelPrimitives) PrintPerformanceReport() {
    fmt.Println("\n📊 Parallel Primitives Performance Report:")
    fmt.Println("Operation\tCalls\tTotal Time\tElements\tAvg Throughput\tBest Throughput")
    
    pp.mutex.RLock()
    defer pp.mutex.RUnlock()
    
    for opName, stats := range pp.operationStats {
        avgThroughput := float64(stats.TotalElements) / stats.TotalTime.Seconds()
        
        fmt.Printf("%-10s\t%d\t%v\t%d\t%.2e elem/s\t%.2e elem/s\n",
                   opName, stats.TotalCalls, stats.TotalTime, stats.TotalElements,
                   avgThroughput, stats.BestThroughput)
    }
}

func (pp *ParallelPrimitives) Cleanup() {
    for _, stream := range pp.streams {
        stream.Destroy()
    }
    for _, buffer := range pp.tempBuffers {
        buffer.Free()
    }
}

func min(a, b int) int {
    if a < b {
        return a
    }
    return b
}

// Demonstration
func main() {
    cuda.Initialize()
    fmt.Println("🧬 Expert GPU-Native Algorithms: Parallel Primitives")
    
    pp := NewParallelPrimitives()
    defer pp.Cleanup()
    
    // Test with different data sizes
    sizes := []int{1000, 10000, 100000, 1000000}
    
    for _, size := range sizes {
        fmt.Printf("\n" + "="*60 + "\n")
        fmt.Printf("Testing with %d elements\n", size)
        fmt.Printf("="*60 + "\n")
        
        // Generate test data
        testData := generateTestData(size)
        defer testData.Free()
        
        // Test reduction operations
        testReduction(pp, testData, size)
        
        // Test scan operation
        testScan(pp, testData, size)
        
        // Test histogram
        testHistogram(pp, testData, size)
        
        // Test sorting
        testSorting(pp, testData, size)
        
        // Test compaction
        testCompaction(pp, testData, size)
    }
    
    // Print performance summary
    pp.PrintPerformanceReport()
    
    // Analyze algorithm characteristics
    analyzeAlgorithmComplexity()
}

func generateTestData(size int) *memory.DeviceMemory {
    // Generate random test data
    hostData := make([]float32, size)
    
    for i := 0; i < size; i++ {
        // Mix of patterns for comprehensive testing
        switch i % 4 {
        case 0:
            hostData[i] = float32(i % 1000) // Sequential
        case 1:
            hostData[i] = float32((i * 1103515245 + 12345) % 2048) // Pseudo-random
        case 2:
            hostData[i] = float32(math.Sin(float64(i) * 0.01) * 100) // Sinusoidal
        case 3:
            hostData[i] = float32(i/1000) + float32(i%10)/10.0 // Stepped
        }
    }
    
    deviceData, err := memory.Alloc(int64(size * 4))
    if err != nil {
        panic(err)
    }
    
    err = deviceData.CopyFromHost(hostData)
    if err != nil {
        panic(err)
    }
    
    return deviceData
}

func testReduction(pp *ParallelPrimitives, data *memory.DeviceMemory, size int) {
    fmt.Println("\n🔄 Testing Reduction Operations:")
    
    // Test different reduction operations
    reductions := []struct {
        name string
        op   ReductionOp
    }{
        {"Sum", SumOp},
        {"Max", MaxOp},
        {"Min", MinOp},
        {"Product", ProductOp},
    }
    
    for _, reduction := range reductions {
        start := time.Now()
        result, err := pp.Reduce(data, size, reduction.op)
        elapsed := time.Since(start)
        
        if err != nil {
            fmt.Printf("  %s: Error - %v\n", reduction.name, err)
        } else {
            throughput := float64(size) / elapsed.Seconds()
            fmt.Printf("  %s: %.6f (%.2e elem/s)\n", reduction.name, result, throughput)
        }
    }
}

func testScan(pp *ParallelPrimitives, data *memory.DeviceMemory, size int) {
    fmt.Println("\n📊 Testing Scan Operations:")
    
    // Test inclusive scan
    start := time.Now()
    inclusiveResult, err := pp.Scan(data, size, true)
    elapsed := time.Since(start)
    
    if err != nil {
        fmt.Printf("  Inclusive Scan: Error - %v\n", err)
    } else {
        defer inclusiveResult.Free()
        throughput := float64(size) / elapsed.Seconds()
        fmt.Printf("  Inclusive Scan: %.2e elem/s\n", throughput)
    }
    
    // Test exclusive scan
    start = time.Now()
    exclusiveResult, err := pp.Scan(data, size, false)
    elapsed = time.Since(start)
    
    if err != nil {
        fmt.Printf("  Exclusive Scan: Error - %v\n", err)
    } else {
        defer exclusiveResult.Free()
        throughput := float64(size) / elapsed.Seconds()
        fmt.Printf("  Exclusive Scan: %.2e elem/s\n", throughput)
    }
}

func testHistogram(pp *ParallelPrimitives, data *memory.DeviceMemory, size int) {
    fmt.Println("\n📈 Testing Histogram:")
    
    bins := 256
    start := time.Now()
    histogram, err := pp.Histogram(data, size, bins, -1000, 1000)
    elapsed := time.Since(start)
    
    if err != nil {
        fmt.Printf("  Histogram: Error - %v\n", err)
    } else {
        throughput := float64(size) / elapsed.Seconds()
        
        // Count non-zero bins
        nonZeroBins := 0
        maxCount := int32(0)
        for _, count := range histogram {
            if count > 0 {
                nonZeroBins++
                if count > maxCount {
                    maxCount = count
                }
            }
        }
        
        fmt.Printf("  Histogram: %d bins, %d non-zero, max count: %d (%.2e elem/s)\n",
                   bins, nonZeroBins, maxCount, throughput)
    }
}

func testSorting(pp *ParallelPrimitives, data *memory.DeviceMemory, size int) {
    fmt.Println("\n🔀 Testing Sorting:")
    
    // Create copy for sorting (preserve original)
    sortData, _ := memory.Alloc(int64(size * 4))
    defer sortData.Free()
    
    hostData := make([]float32, size)
    data.CopyToHost(hostData)
    sortData.CopyFromHost(hostData)
    
    start := time.Now()
    err := pp.Sort(sortData, size)
    elapsed := time.Since(start)
    
    if err != nil {
        fmt.Printf("  Sort: Error - %v\n", err)
    } else {
        throughput := float64(size) / elapsed.Seconds()
        
        // Verify sort correctness (sample)
        sortedData := make([]float32, min(size, 100))
        sortData.CopyToHost(sortedData[:min(size, 100)])
        
        isSorted := true
        for i := 1; i < len(sortedData); i++ {
            if sortedData[i] < sortedData[i-1] {
                isSorted = false
                break
            }
        }
        
        status := "✅"
        if !isSorted {
            status = "❌"
        }
        
        fmt.Printf("  Sort: %s Sorted correctly (%.2e elem/s)\n", status, throughput)
    }
}

func testCompaction(pp *ParallelPrimitives, data *memory.DeviceMemory, size int) {
    fmt.Println("\n🎯 Testing Compaction:")
    
    // Filter for positive values
    predicate := func(x float32) bool { return x > 0 }
    
    start := time.Now()
    compactData, compactSize, err := pp.Compact(data, size, predicate)
    elapsed := time.Since(start)
    
    if err != nil {
        fmt.Printf("  Compact: Error - %v\n", err)
    } else {
        defer compactData.Free()
        throughput := float64(size) / elapsed.Seconds()
        compressionRatio := float64(compactSize) / float64(size) * 100
        
        fmt.Printf("  Compact: %d → %d elements (%.1f%% kept, %.2e elem/s)\n",
                   size, compactSize, compressionRatio, throughput)
    }
}

func analyzeAlgorithmComplexity() {
    fmt.Println("\n🧮 Algorithm Complexity Analysis:")
    
    algorithms := []struct {
        name      string
        workComplexity string
        spanComplexity string
        parallelism    string
        memoryPattern  string
    }{
        {"Reduce", "O(n)", "O(log n)", "O(n/log n)", "Tree-based, shared memory"},
        {"Scan", "O(n)", "O(log n)", "O(n/log n)", "Up-sweep/down-sweep, global memory"},
        {"Histogram", "O(n)", "O(1)", "O(n)", "Scatter/gather, atomic operations"},
        {"Sort (Bitonic)", "O(n log² n)", "O(log² n)", "O(n)", "Data-independent, compare-exchange"},
        {"Compact", "O(n)", "O(log n)", "O(n/log n)", "Predicate + scan + scatter"},
    }
    
    fmt.Printf("%-15s%-15s%-15s%-15s%s\n", "Algorithm", "Work", "Span", "Parallelism", "Memory Pattern")
    fmt.Println(strings.Repeat("-", 80))
    
    for _, alg := range algorithms {
        fmt.Printf("%-15s%-15s%-15s%-15s%s\n", 
                   alg.name, alg.workComplexity, alg.spanComplexity, 
                   alg.parallelism, alg.memoryPattern)
    }
    
    fmt.Println("\n💡 GPU Algorithm Design Guidelines:")
    fmt.Println("1. Maximize parallelism (high work/span ratio)")
    fmt.Println("2. Minimize memory divergence and conflicts")  
    fmt.Println("3. Exploit memory hierarchy (shared memory, coalescing)")
    fmt.Println("4. Use warp-level primitives for fine-grained operations")
    fmt.Println("5. Consider memory bandwidth vs. compute intensity")
    fmt.Println("6. Design for scalability across different GPU sizes")
}

// Helper function for string repetition
func strings.Repeat(s string, count int) string {
    if count < 0 {
        panic("negative repeat count")
    }
    
    result := make([]byte, len(s)*count)
    bp := copy(result, s)
    for bp < len(result) {
        copy(result[bp:], result[:bp])
        bp *= 2
    }
    return string(result)
}
```

---

## 🎯 Module Assessment  

### **GPU Algorithm Design Mastery**

1. **Primitive Implementation**: Successfully implement 5+ parallel primitives
2. **Performance Optimization**: Achieve near-theoretical performance limits
3. **Scalability Analysis**: Demonstrate scaling across different problem sizes
4. **Memory Hierarchy Exploitation**: Optimize for GPU memory patterns

### **Success Criteria**

- ✅ Algorithms scale efficiently with problem size and GPU resources
- ✅ Memory bandwidth utilization >80% for bandwidth-bound operations
- ✅ Correct implementation verified through comprehensive testing
- ✅ Performance analysis demonstrates understanding of GPU constraints

---

## 🚀 Next Steps

**Incredible! You've mastered GPU-native algorithm design.**

**You're now ready for:**
➡️ **[Module 6: Performance Engineering](TRAINING_EXPERT_6_PERFORMANCE.md)**

**Skills Mastered:**
- 🧬 **GPU-Native Algorithm Design** - From first principles optimization
- 🔀 **Parallel Design Patterns** - Reduction, scan, sort, histogram
- 📊 **Complexity Analysis** - Work, span, and parallelism metrics
- ⚡ **Performance Optimization** - Memory hierarchy and architectural exploitation

---

*From sequential thinking to parallel mastery - designing algorithms for the massively parallel future! 🧬⚡*
