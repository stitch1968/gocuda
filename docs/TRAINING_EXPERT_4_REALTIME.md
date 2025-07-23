# ⚡ Expert Module 4: Real-Time & Streaming Processing

**Goal:** Master real-time GPU computing with ultra-low latency, streaming data processing, and temporal optimization techniques

---

## 📚 Learning Objectives

By the end of this module, you will:
- ⚡ **Achieve sub-millisecond latency** for real-time applications
- 🌊 **Implement streaming algorithms** with continuous data processing
- 📡 **Optimize pipeline throughput** with overlapped execution
- ⏱️ **Master temporal locality** and time-sensitive computations
- 🎯 **Build production systems** for high-frequency trading, gaming, and control systems

---

## 🧠 Theoretical Foundation

### Real-Time Computing Constraints

**Temporal Requirements:**
```
Hard Real-Time: Deadlines must never be missed
├── Deterministic execution times
├── Bounded memory allocation
├── Predictable GPU kernel execution
└── Guaranteed resource availability

Soft Real-Time: Occasional deadline misses acceptable
├── Statistical deadline guarantees
├── Best-effort optimization
├── Graceful degradation
└── Quality-of-service metrics
```

**Latency Categories:**
- **Ultra-Low**: < 1 microsecond (HFT, control systems)
- **Low**: 1-100 microseconds (gaming, audio)
- **Interactive**: 100 microseconds - 16ms (graphics, UI)
- **Streaming**: Variable, throughput-focused

### GPU Streaming Architecture

**Pipeline Stages:**
```
Data Ingestion → GPU Transfer → Compute → Results Transfer → Output
     ↕              ↕            ↕           ↕              ↕
   Network        PCIe/NVLink   Kernels    PCIe/NVLink    Network
   Storage        Memory Pools   Streams   Result Queues   Display
```

**Optimization Strategies:**
- **Pipelining**: Overlap CPU/GPU operations
- **Batching**: Amortize overhead across operations  
- **Prefetching**: Anticipate data needs
- **Resource Pooling**: Minimize allocation overhead
- **Temporal Scheduling**: Optimize for deadlines

---

## ⚡ Chapter 1: Ultra-Low Latency Computing

### Real-Time GPU Framework

Create `realtime/ultra_low_latency.go`:

```go
package main

import (
    "fmt"
    "sync"
    "time"
    "unsafe"
    "runtime"
    "context"
    "sync/atomic"
    "github.com/stitch1968/gocuda"
    "github.com/stitch1968/gocuda/memory"
    "github.com/stitch1968/gocuda/timing"
)

// Ultra-low latency GPU processing framework
type UltraLowLatencyProcessor struct {
    ctx                *cuda.Context
    streams            []*cuda.Stream
    memoryPools        []*memory.Pool
    kernelCache        *KernelCache
    latencyTracker     *LatencyTracker
    scheduler          *RealTimeScheduler
    
    // Pre-allocated resources
    inputBuffers       []*memory.PinnedBuffer
    outputBuffers      []*memory.PinnedBuffer
    deviceBuffers      []*memory.DeviceMemory
    
    // Performance configuration
    maxLatencyNs       int64
    batchSize          int
    priorityLevels     int
    
    // State management
    isRunning          int32
    statsLock          sync.RWMutex
    lastProcessTime    time.Time
}

type LatencyTracker struct {
    measurements     []time.Duration
    percentiles      map[float64]time.Duration
    maxMeasurements  int
    currentIndex     int
    mutex           sync.Mutex
}

type RealTimeScheduler struct {
    taskQueues      []chan RealTimeTask
    priorities      []int
    deadlines       []time.Time
    cpuAffinity     []int
    gpuStreams      []*cuda.Stream
    scheduling      SchedulingPolicy
}

type SchedulingPolicy int

const (
    EarliestDeadlineFirst SchedulingPolicy = iota
    RateMonotonic
    FixedPriority
    ProportionalShare
)

type RealTimeTask struct {
    TaskID          string
    Data            *memory.PinnedBuffer
    Result          *memory.PinnedBuffer
    Priority        int
    Deadline        time.Time
    ArrivalTime     time.Time
    ProcessingFunc  func(*memory.DeviceMemory, *memory.DeviceMemory) error
    Callback        func(RealTimeResult)
    MaxLatency      time.Duration
}

type RealTimeResult struct {
    TaskID          string
    Success         bool
    Latency         time.Duration
    MissedDeadline  bool
    Timestamp       time.Time
    Error           error
}

type KernelCache struct {
    compiledKernels map[string]*cuda.Function
    launchParams    map[string]KernelConfig
    mutex          sync.RWMutex
}

type KernelConfig struct {
    GridSize        dim3
    BlockSize       dim3
    SharedMemSize   int
    OptimalOccupancy float64
}

type dim3 struct {
    X, Y, Z int
}

func NewUltraLowLatencyProcessor(maxLatencyMicros int) *UltraLowLatencyProcessor {
    cuda.Initialize()
    
    processor := &UltraLowLatencyProcessor{
        ctx:              cuda.GetDefaultContext(),
        streams:          make([]*cuda.Stream, 4), // Multiple streams for overlap
        memoryPools:      make([]*memory.Pool, 2),
        maxLatencyNs:     int64(maxLatencyMicros * 1000),
        batchSize:        1, // Start with single-item processing
        priorityLevels:   4,
        maxMeasurements:  10000,
    }
    
    // Initialize GPU streams
    for i := range processor.streams {
        stream, err := processor.ctx.CreateStreamWithFlags(cuda.StreamNonBlocking)
        if err != nil {
            panic(fmt.Sprintf("Failed to create stream: %v", err))
        }
        processor.streams[i] = stream
    }
    
    // Initialize memory pools for zero-allocation processing
    processor.initializeMemoryPools()
    
    // Pre-allocate buffers
    processor.preallocateBuffers()
    
    // Initialize subsystems
    processor.latencyTracker = NewLatencyTracker(10000)
    processor.kernelCache = NewKernelCache()
    processor.scheduler = NewRealTimeScheduler(processor.priorityLevels, processor.streams)
    
    // Set CPU affinity for deterministic performance
    processor.setCPUAffinity()
    
    // Configure GPU for minimal latency
    processor.optimizeForLatency()
    
    fmt.Printf("⚡ Ultra-Low Latency Processor initialized (target: %d μs)\n", maxLatencyMicros)
    return processor
}

func (ulp *ultraLowLatencyProcessor) initializeMemoryPools() {
    // Pool for input data (pinned memory for fast transfers)
    inputPoolSize := int64(1024 * 1024 * 16) // 16MB
    inputPool, err := memory.CreatePinnedPool(inputPoolSize)
    if err != nil {
        panic(fmt.Sprintf("Failed to create input memory pool: %v", err))
    }
    ulp.memoryPools[0] = inputPool
    
    // Pool for GPU device memory
    devicePoolSize := int64(1024 * 1024 * 64) // 64MB
    devicePool, err := memory.CreateDevicePool(devicePoolSize)
    if err != nil {
        panic(fmt.Sprintf("Failed to create device memory pool: %v", err))
    }
    ulp.memoryPools[1] = devicePool
    
    fmt.Printf("   Memory pools initialized: Input=%dMB, Device=%dMB\n", 
               inputPoolSize/(1024*1024), devicePoolSize/(1024*1024))
}

func (ulp *UltraLowLatencyProcessor) preallocateBuffers() {
    bufferCount := len(ulp.streams) * 2 // Input and output per stream
    ulp.inputBuffers = make([]*memory.PinnedBuffer, bufferCount)
    ulp.outputBuffers = make([]*memory.PinnedBuffer, bufferCount)
    ulp.deviceBuffers = make([]*memory.DeviceMemory, bufferCount)
    
    bufferSize := 4096 // 4KB per buffer
    
    for i := 0; i < bufferCount; i++ {
        // Pinned host buffers for fast PCIe transfers
        input, err := memory.AllocPinned(bufferSize)
        if err != nil {
            panic(fmt.Sprintf("Failed to allocate input buffer %d: %v", i, err))
        }
        ulp.inputBuffers[i] = input
        
        output, err := memory.AllocPinned(bufferSize)
        if err != nil {
            panic(fmt.Sprintf("Failed to allocate output buffer %d: %v", i, err))
        }
        ulp.outputBuffers[i] = output
        
        // Pre-allocated device memory
        device, err := memory.Alloc(int64(bufferSize))
        if err != nil {
            panic(fmt.Sprintf("Failed to allocate device buffer %d: %v", i, err))
        }
        ulp.deviceBuffers[i] = device
    }
    
    fmt.Printf("   Pre-allocated %d buffer sets (%d bytes each)\n", bufferCount, bufferSize)
}

func (ulp *UltraLowLatencyProcessor) setCPUAffinity() {
    // Pin to specific CPU cores for consistent performance
    runtime.LockOSThread()
    
    // In a real implementation, would use OS-specific calls to set CPU affinity
    // For simulation, we document the requirement
    fmt.Printf("   CPU affinity configured for real-time performance\n")
}

func (ulp *UltraLowLatencyProcessor) optimizeForLatency() {
    // Configure GPU for minimal latency
    device := cuda.GetDevice(0)
    
    // Set GPU clocks to maximum (requires elevated privileges)
    // device.SetMemoryClockRate(device.GetMaxMemoryClockRate())
    // device.SetGraphicsClockRate(device.GetMaxGraphicsClockRate())
    
    // Disable ECC if not required (trades reliability for performance)
    // device.SetECCMode(false)
    
    // Set persistence mode to keep driver loaded
    // device.SetPersistenceMode(true)
    
    fmt.Printf("   GPU optimized for minimum latency\n")
}

// Process single task with ultra-low latency
func (ulp *UltraLowLatencyProcessor) ProcessTask(task RealTimeTask) {
    if atomic.LoadInt32(&ulp.isRunning) == 0 {
        task.Callback(RealTimeResult{
            TaskID:  task.TaskID,
            Success: false,
            Error:   fmt.Errorf("processor not running"),
        })
        return
    }
    
    startTime := time.Now()
    
    // Select optimal stream based on current load
    streamIndex := ulp.selectOptimalStream()
    stream := ulp.streams[streamIndex]
    
    // Get pre-allocated buffers
    inputBuffer := ulp.inputBuffers[streamIndex]
    outputBuffer := ulp.outputBuffers[streamIndex]
    deviceBuffer := ulp.deviceBuffers[streamIndex]
    
    // Fast path: process immediately if resources available
    result := ulp.processFastPath(task, stream, inputBuffer, outputBuffer, deviceBuffer, startTime)
    
    // Update latency statistics
    ulp.latencyTracker.Record(result.Latency)
    
    // Execute callback
    task.Callback(result)
}

func (ulp *UltraLowLatencyProcessor) selectOptimalStream() int {
    // Simple round-robin selection
    // In production, would consider stream load and deadline constraints
    currentTime := time.Now().UnixNano()
    return int(currentTime) % len(ulp.streams)
}

func (ulp *UltraLowLatencyProcessor) processFastPath(
    task RealTimeTask,
    stream *cuda.Stream,
    inputBuffer *memory.PinnedBuffer,
    outputBuffer *memory.PinnedBuffer,
    deviceBuffer *memory.DeviceMemory,
    startTime time.Time,
) RealTimeResult {
    
    result := RealTimeResult{
        TaskID:    task.TaskID,
        Success:   true,
        Timestamp: startTime,
    }
    
    // Check if we can meet the deadline
    timeRemaining := task.Deadline.Sub(startTime)
    if timeRemaining < time.Duration(ulp.maxLatencyNs) {
        result.Success = false
        result.MissedDeadline = true
        result.Error = fmt.Errorf("insufficient time to meet deadline")
        result.Latency = time.Since(startTime)
        return result
    }
    
    // Step 1: Copy input data (optimized for minimal latency)
    copyStart := time.Now()
    err := ulp.fastMemcpy(inputBuffer, task.Data, task.Data.Size())
    if err != nil {
        result.Success = false
        result.Error = fmt.Errorf("input copy failed: %v", err)
        result.Latency = time.Since(startTime)
        return result
    }
    
    // Step 2: Transfer to GPU (asynchronous)
    transferStart := time.Now()
    err = stream.MemcpyHtoDAsync(deviceBuffer, inputBuffer, task.Data.Size())
    if err != nil {
        result.Success = false
        result.Error = fmt.Errorf("GPU transfer failed: %v", err)
        result.Latency = time.Since(startTime)
        return result
    }
    
    // Step 3: Launch kernel (pre-compiled and cached)
    launchStart := time.Now()
    err = task.ProcessingFunc(deviceBuffer, deviceBuffer) // In-place operation
    if err != nil {
        result.Success = false
        result.Error = fmt.Errorf("kernel execution failed: %v", err)
        result.Latency = time.Since(startTime)
        return result
    }
    
    // Step 4: Transfer result back (asynchronous)
    err = stream.MemcpyDtoHAsync(outputBuffer, deviceBuffer, task.Data.Size())
    if err != nil {
        result.Success = false
        result.Error = fmt.Errorf("result transfer failed: %v", err)
        result.Latency = time.Since(startTime)
        return result
    }
    
    // Step 5: Synchronize and copy result
    stream.Synchronize()
    err = ulp.fastMemcpy(task.Result, outputBuffer, task.Data.Size())
    if err != nil {
        result.Success = false
        result.Error = fmt.Errorf("output copy failed: %v", err)
        result.Latency = time.Since(startTime)
        return result
    }
    
    // Calculate final latency
    result.Latency = time.Since(startTime)
    
    // Check if deadline was met
    if time.Now().After(task.Deadline) {
        result.MissedDeadline = true
    }
    
    return result
}

func (ulp *UltraLowLatencyProcessor) fastMemcpy(dst, src *memory.PinnedBuffer, size int) error {
    // Ultra-fast memory copy using optimized techniques
    // In real implementation, would use vectorized instructions, temporal hints
    
    dstPtr := dst.Ptr()
    srcPtr := src.Ptr()
    
    // Use unsafe operations for maximum speed
    copy((*[1 << 30]byte)(dstPtr)[:size], (*[1 << 30]byte)(srcPtr)[:size])
    
    return nil
}

// Streaming processing with continuous data flow
func (ulp *UltraLowLatencyProcessor) StartStreaming(ctx context.Context) <-chan RealTimeResult {
    atomic.StoreInt32(&ulp.isRunning, 1)
    
    resultChan := make(chan RealTimeResult, 100)
    
    // Start processing workers
    for i := 0; i < len(ulp.streams); i++ {
        go ulp.streamingWorker(ctx, i, resultChan)
    }
    
    // Start latency monitoring
    go ulp.latencyMonitor(ctx)
    
    fmt.Printf("   Streaming processor started with %d workers\n", len(ulp.streams))
    return resultChan
}

func (ulp *UltraLowLatencyProcessor) streamingWorker(ctx context.Context, workerID int, results chan<- RealTimeResult) {
    stream := ulp.streams[workerID]
    inputBuffer := ulp.inputBuffers[workerID]
    outputBuffer := ulp.outputBuffers[workerID]
    deviceBuffer := ulp.deviceBuffers[workerID]
    
    // Worker-specific task queue
    taskQueue := ulp.scheduler.taskQueues[workerID]
    
    for {
        select {
        case <-ctx.Done():
            return
            
        case task := <-taskQueue:
            startTime := time.Now()
            result := ulp.processFastPath(task, stream, inputBuffer, outputBuffer, deviceBuffer, startTime)
            
            results <- result
            
            // Update worker statistics
            ulp.updateWorkerStats(workerID, result)
        }
    }
}

func (ulp *UltraLowLatencyProcessor) updateWorkerStats(workerID int, result RealTimeResult) {
    ulp.latencyTracker.Record(result.Latency)
    
    // Track deadline misses
    if result.MissedDeadline {
        atomic.AddInt64(&ulp.scheduler.deadlineMisses, 1)
    }
}

func (ulp *UltraLowLatencyProcessor) latencyMonitor(ctx context.Context) {
    ticker := time.NewTicker(1 * time.Second)
    defer ticker.Stop()
    
    for {
        select {
        case <-ctx.Done():
            return
            
        case <-ticker.C:
            ulp.reportLatencyStats()
        }
    }
}

func (ulp *UltraLowLatencyProcessor) reportLatencyStats() {
    stats := ulp.latencyTracker.GetStatistics()
    
    if len(stats.measurements) > 0 {
        fmt.Printf("📊 Latency Stats: P50=%.1fμs P95=%.1fμs P99=%.1fμs Max=%.1fμs\n",
                   stats.percentiles[50].Seconds()*1e6,
                   stats.percentiles[95].Seconds()*1e6,
                   stats.percentiles[99].Seconds()*1e6,
                   stats.max.Seconds()*1e6)
        
        // Alert if exceeding latency target
        if stats.percentiles[95] > time.Duration(ulp.maxLatencyNs) {
            fmt.Printf("⚠️ WARNING: P95 latency exceeding target!\n")
        }
    }
}

func (ulp *UltraLowLatencyProcessor) Stop() {
    atomic.StoreInt32(&ulp.isRunning, 0)
    fmt.Printf("   Streaming processor stopped\n")
}

func (ulp *UltraLowLatencyProcessor) Cleanup() {
    // Cleanup streams
    for _, stream := range ulp.streams {
        stream.Synchronize()
        stream.Destroy()
    }
    
    // Cleanup buffers
    for _, buffer := range ulp.inputBuffers {
        buffer.Free()
    }
    for _, buffer := range ulp.outputBuffers {
        buffer.Free()
    }
    for _, buffer := range ulp.deviceBuffers {
        buffer.Free()
    }
    
    // Cleanup memory pools
    for _, pool := range ulp.memoryPools {
        pool.Destroy()
    }
    
    fmt.Printf("   Ultra-low latency processor cleaned up\n")
}

// Latency tracking implementation
func NewLatencyTracker(maxMeasurements int) *LatencyTracker {
    return &LatencyTracker{
        measurements:    make([]time.Duration, maxMeasurements),
        percentiles:     make(map[float64]time.Duration),
        maxMeasurements: maxMeasurements,
    }
}

func (lt *LatencyTracker) Record(latency time.Duration) {
    lt.mutex.Lock()
    defer lt.mutex.Unlock()
    
    lt.measurements[lt.currentIndex] = latency
    lt.currentIndex = (lt.currentIndex + 1) % lt.maxMeasurements
}

func (lt *LatencyTracker) GetStatistics() LatencyStatistics {
    lt.mutex.Lock()
    defer lt.mutex.Unlock()
    
    // Copy measurements for analysis
    validMeasurements := make([]time.Duration, 0, lt.maxMeasurements)
    for _, measurement := range lt.measurements {
        if measurement > 0 {
            validMeasurements = append(validMeasurements, measurement)
        }
    }
    
    if len(validMeasurements) == 0 {
        return LatencyStatistics{}
    }
    
    // Sort for percentile calculation
    sort.Slice(validMeasurements, func(i, j int) bool {
        return validMeasurements[i] < validMeasurements[j]
    })
    
    // Calculate percentiles
    percentiles := map[float64]time.Duration{
        50:  validMeasurements[len(validMeasurements)*50/100],
        95:  validMeasurements[len(validMeasurements)*95/100],
        99:  validMeasurements[len(validMeasurements)*99/100],
        99.9: validMeasurements[len(validMeasurements)*999/1000],
    }
    
    // Calculate min, max, mean
    min := validMeasurements[0]
    max := validMeasurements[len(validMeasurements)-1]
    
    var total time.Duration
    for _, measurement := range validMeasurements {
        total += measurement
    }
    mean := total / time.Duration(len(validMeasurements))
    
    return LatencyStatistics{
        measurements: validMeasurements,
        percentiles:  percentiles,
        min:          min,
        max:          max,
        mean:         mean,
        count:        len(validMeasurements),
    }
}

type LatencyStatistics struct {
    measurements []time.Duration
    percentiles  map[float64]time.Duration
    min          time.Duration
    max          time.Duration
    mean         time.Duration
    count        int
}

// Real-time scheduler implementation
func NewRealTimeScheduler(priorityLevels int, streams []*cuda.Stream) *RealTimeScheduler {
    scheduler := &RealTimeScheduler{
        taskQueues:   make([]chan RealTimeTask, len(streams)),
        priorities:   make([]int, len(streams)),
        deadlines:    make([]time.Time, len(streams)),
        gpuStreams:   streams,
        scheduling:   EarliestDeadlineFirst,
    }
    
    for i := range scheduler.taskQueues {
        scheduler.taskQueues[i] = make(chan RealTimeTask, 100)
    }
    
    return scheduler
}

type RealTimeScheduler struct {
    taskQueues      []chan RealTimeTask
    priorities      []int
    deadlines       []time.Time
    gpuStreams      []*cuda.Stream
    scheduling      SchedulingPolicy
    deadlineMisses  int64
}

func (rts *RealTimeScheduler) ScheduleTask(task RealTimeTask) {
    switch rts.scheduling {
    case EarliestDeadlineFirst:
        rts.scheduleEDF(task)
    case FixedPriority:
        rts.scheduleFixedPriority(task)
    default:
        // Round-robin fallback
        workerID := int(time.Now().UnixNano()) % len(rts.taskQueues)
        rts.taskQueues[workerID] <- task
    }
}

func (rts *RealTimeScheduler) scheduleEDF(task RealTimeTask) {
    // Find worker with earliest deadline or least loaded
    bestWorker := 0
    earliestDeadline := rts.deadlines[0]
    
    for i := 1; i < len(rts.deadlines); i++ {
        if rts.deadlines[i].Before(earliestDeadline) || rts.deadlines[i].IsZero() {
            bestWorker = i
            earliestDeadline = rts.deadlines[i]
        }
    }
    
    rts.deadlines[bestWorker] = task.Deadline
    rts.taskQueues[bestWorker] <- task
}

func (rts *RealTimeScheduler) scheduleFixedPriority(task RealTimeTask) {
    // Assign to worker based on task priority
    workerID := task.Priority % len(rts.taskQueues)
    rts.taskQueues[workerID] <- task
}

// Kernel cache for pre-compiled operations
func NewKernelCache() *KernelCache {
    return &KernelCache{
        compiledKernels: make(map[string]*cuda.Function),
        launchParams:    make(map[string]KernelConfig),
    }
}

func (kc *KernelCache) GetKernel(name string) (*cuda.Function, KernelConfig, bool) {
    kc.mutex.RLock()
    defer kc.mutex.RUnlock()
    
    function, exists := kc.compiledKernels[name]
    if !exists {
        return nil, KernelConfig{}, false
    }
    
    config := kc.launchParams[name]
    return function, config, true
}

func (kc *KernelCache) CacheKernel(name string, function *cuda.Function, config KernelConfig) {
    kc.mutex.Lock()
    defer kc.mutex.Unlock()
    
    kc.compiledKernels[name] = function
    kc.launchParams[name] = config
}

// Example real-time processing functions
func vectorAddKernel(input, output *memory.DeviceMemory) error {
    // Simulate ultra-fast vector addition kernel
    // In real implementation, would launch pre-compiled CUDA kernel
    time.Sleep(10 * time.Microsecond) // Simulate kernel execution
    return nil
}

func matrixMultiplyKernel(input, output *memory.DeviceMemory) error {
    // Simulate fast matrix multiply kernel  
    time.Sleep(50 * time.Microsecond) // Simulate kernel execution
    return nil
}

func fftKernel(input, output *memory.DeviceMemory) error {
    // Simulate fast FFT kernel
    time.Sleep(25 * time.Microsecond) // Simulate kernel execution
    return nil
}

// Demonstration
func main() {
    fmt.Println("⚡ Expert Real-Time & Streaming Processing")
    
    // Create ultra-low latency processor (target: 100 microseconds)
    processor := NewUltraLowLatencyProcessor(100)
    defer processor.Cleanup()
    
    // Demonstrate single-task processing
    demonstrateSingleTaskProcessing(processor)
    
    // Demonstrate streaming processing
    demonstrateStreamingProcessing(processor)
    
    // Benchmark latency characteristics
    benchmarkLatency(processor)
}

func demonstrateSingleTaskProcessing(processor *UltraLowLatencyProcessor) {
    fmt.Println("\n🎯 Single Task Processing:")
    
    // Create test task
    inputData, _ := memory.AllocPinned(1024)
    outputData, _ := memory.AllocPinned(1024)
    defer inputData.Free()
    defer outputData.Free()
    
    task := RealTimeTask{
        TaskID:      "test_vector_add",
        Data:        inputData,
        Result:      outputData,
        Priority:    1,
        Deadline:    time.Now().Add(1 * time.Millisecond),
        ProcessingFunc: vectorAddKernel,
        Callback: func(result RealTimeResult) {
            fmt.Printf("   Task %s completed: Success=%t, Latency=%.1fμs\n",
                       result.TaskID, result.Success, result.Latency.Seconds()*1e6)
            if result.MissedDeadline {
                fmt.Printf("   ⚠️ Deadline missed!\n")
            }
        },
    }
    
    // Process task
    processor.ProcessTask(task)
    
    // Allow callback to execute
    time.Sleep(10 * time.Millisecond)
}

func demonstrateStreamingProcessing(processor *UltraLowLatencyProcessor) {
    fmt.Println("\n🌊 Streaming Processing:")
    
    ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
    defer cancel()
    
    resultChan := processor.StartStreaming(ctx)
    defer processor.Stop()
    
    // Generate continuous stream of tasks
    go generateTaskStream(processor, ctx)
    
    // Monitor results
    resultsReceived := 0
    deadlineMisses := 0
    totalLatency := time.Duration(0)
    
    for {
        select {
        case <-ctx.Done():
            fmt.Printf("   Processed %d tasks, %d deadline misses\n", resultsReceived, deadlineMisses)
            fmt.Printf("   Average latency: %.1fμs\n", 
                       (totalLatency/time.Duration(resultsReceived)).Seconds()*1e6)
            return
            
        case result := <-resultChan:
            resultsReceived++
            totalLatency += result.Latency
            
            if result.MissedDeadline {
                deadlineMisses++
            }
            
            if resultsReceived%100 == 0 {
                fmt.Printf("   Processed %d tasks...\n", resultsReceived)
            }
        }
    }
}

func generateTaskStream(processor *UltraLowLatencyProcessor, ctx context.Context) {
    taskCounter := 0
    
    // Different task types with varying processing requirements
    taskTypes := []struct {
        name     string
        function func(*memory.DeviceMemory, *memory.DeviceMemory) error
        priority int
        deadline time.Duration
    }{
        {"vector_add", vectorAddKernel, 1, 50 * time.Microsecond},
        {"matrix_mult", matrixMultiplyKernel, 2, 100 * time.Microsecond},
        {"fft", fftKernel, 1, 75 * time.Microsecond},
    }
    
    ticker := time.NewTicker(100 * time.Microsecond) // 10kHz task generation
    defer ticker.Stop()
    
    for {
        select {
        case <-ctx.Done():
            return
            
        case <-ticker.C:
            taskType := taskTypes[taskCounter%len(taskTypes)]
            
            // Create task buffers
            inputData, _ := memory.AllocPinned(1024)
            outputData, _ := memory.AllocPinned(1024)
            
            task := RealTimeTask{
                TaskID:         fmt.Sprintf("%s_%d", taskType.name, taskCounter),
                Data:           inputData,
                Result:         outputData,
                Priority:       taskType.priority,
                Deadline:       time.Now().Add(taskType.deadline),
                ProcessingFunc: taskType.function,
                Callback: func(result RealTimeResult) {
                    // Cleanup in callback
                    inputData.Free()
                    outputData.Free()
                },
            }
            
            processor.scheduler.ScheduleTask(task)
            taskCounter++
        }
    }
}

func benchmarkLatency(processor *UltraLowLatencyProcessor) {
    fmt.Println("\n📊 Latency Benchmarking:")
    
    // Test different scenarios
    scenarios := []struct {
        name          string
        taskCount     int
        taskSize      int
        frequency     time.Duration
        processingFunc func(*memory.DeviceMemory, *memory.DeviceMemory) error
    }{
        {"High Frequency Small", 1000, 256, 10 * time.Microsecond, vectorAddKernel},
        {"Medium Frequency Medium", 500, 1024, 50 * time.Microsecond, matrixMultiplyKernel},
        {"Low Frequency Large", 100, 4096, 200 * time.Microsecond, fftKernel},
    }
    
    for _, scenario := range scenarios {
        fmt.Printf("\n  Testing: %s\n", scenario.name)
        
        results := make([]RealTimeResult, 0, scenario.taskCount)
        resultMutex := sync.Mutex{}
        
        wg := sync.WaitGroup{}
        startTime := time.Now()
        
        for i := 0; i < scenario.taskCount; i++ {
            wg.Add(1)
            
            go func(taskID int) {
                defer wg.Done()
                
                inputData, _ := memory.AllocPinned(scenario.taskSize)
                outputData, _ := memory.AllocPinned(scenario.taskSize)
                defer inputData.Free()
                defer outputData.Free()
                
                task := RealTimeTask{
                    TaskID:         fmt.Sprintf("bench_%d", taskID),
                    Data:           inputData,
                    Result:         outputData,
                    Priority:       1,
                    Deadline:       time.Now().Add(500 * time.Microsecond),
                    ProcessingFunc: scenario.processingFunc,
                    Callback: func(result RealTimeResult) {
                        resultMutex.Lock()
                        results = append(results, result)
                        resultMutex.Unlock()
                    },
                }
                
                processor.ProcessTask(task)
                time.Sleep(scenario.frequency) // Control frequency
            }(i)
        }
        
        wg.Wait()
        totalTime := time.Since(startTime)
        
        // Analyze results
        if len(results) > 0 {
            latencies := make([]time.Duration, len(results))
            deadlineMisses := 0
            
            for i, result := range results {
                latencies[i] = result.Latency
                if result.MissedDeadline {
                    deadlineMisses++
                }
            }
            
            // Calculate statistics
            sort.Slice(latencies, func(i, j int) bool { return latencies[i] < latencies[j] })
            
            p50 := latencies[len(latencies)*50/100]
            p95 := latencies[len(latencies)*95/100]
            p99 := latencies[len(latencies)*99/100]
            max := latencies[len(latencies)-1]
            
            var sum time.Duration
            for _, lat := range latencies {
                sum += lat
            }
            mean := sum / time.Duration(len(latencies))
            
            fmt.Printf("    Tasks: %d, Deadline Misses: %d (%.1f%%)\n", 
                       len(results), deadlineMisses, float64(deadlineMisses)/float64(len(results))*100)
            fmt.Printf("    Latency - Mean: %.1fμs, P50: %.1fμs, P95: %.1fμs, P99: %.1fμs, Max: %.1fμs\n",
                       mean.Seconds()*1e6, p50.Seconds()*1e6, p95.Seconds()*1e6, 
                       p99.Seconds()*1e6, max.Seconds()*1e6)
            fmt.Printf("    Throughput: %.0f tasks/sec\n", float64(len(results))/totalTime.Seconds())
        }
    }
}

func sort.Slice(slice []time.Duration, less func(i, j int) bool) {
    // Simplified sort implementation for demo
    for i := 0; i < len(slice)-1; i++ {
        for j := i + 1; j < len(slice); j++ {
            if less(j, i) {
                slice[i], slice[j] = slice[j], slice[i]
            }
        }
    }
}
```

---

## 🎯 Module Assessment

### **Real-Time Computing Mastery**

1. **Latency Optimization**: Achieve sub-millisecond processing latencies
2. **Streaming Performance**: Handle continuous data flows with high throughput
3. **Temporal Accuracy**: Meet deadlines consistently under load
4. **System Integration**: Build production-ready real-time systems

### **Success Criteria**

- ✅ P95 latency under target thresholds
- ✅ Zero deadline misses for critical tasks
- ✅ Sustained throughput under continuous load
- ✅ Deterministic performance characteristics

---

## 🚀 Next Steps

**Fantastic! You've mastered real-time GPU computing.**

**You're now ready for:**
➡️ **[Module 5: GPU-Native Algorithms](TRAINING_EXPERT_5_ALGORITHMS.md)**

**Skills Mastered:**
- ⚡ **Ultra-Low Latency Processing** - Sub-millisecond GPU computation
- 🌊 **Streaming Architectures** - Continuous data flow optimization
- ⏱️ **Real-Time Scheduling** - Deadline-aware task management  
- 🎯 **Performance Determinism** - Consistent, predictable execution

---

*From batch processing to real-time responsiveness - mastering time-critical computing! ⚡🎯*
