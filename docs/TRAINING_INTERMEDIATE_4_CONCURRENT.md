# 🔄 Module 4: Concurrent Patterns & Streams

**Goal:** Master CUDA streams, concurrent execution patterns, and advanced parallelization strategies for maximum GPU utilization

---

## 📚 Learning Objectives

By the end of this module, you will:
- 🚀 **Master CUDA streams** for overlapping computation and data transfer
- 🔄 **Implement advanced concurrency** patterns with multiple streams
- ⚡ **Optimize memory bandwidth** through asynchronous operations
- 🎯 **Build producer-consumer** systems with GPU-CPU coordination
- 📊 **Monitor and profile** concurrent execution performance

---

## 🧠 Theoretical Foundation

### CUDA Stream Hierarchy

**Stream Types:**
```
NULL Stream (Default)
├── Synchronous operations
├── Blocks all other streams
└── Legacy behavior

Explicit Streams
├── Non-blocking streams (independent)
├── Priority streams (high/low)
└── Callback streams (CPU synchronization)
```

### Concurrency Patterns

**Overlap Strategies:**
1. **Computation + Memory Transfer**: H2D while computing on previous batch
2. **Multiple Kernels**: Different streams executing different kernels
3. **Pipeline Processing**: Multi-stage data processing pipeline
4. **Producer-Consumer**: CPU generates, GPU processes, CPU consumes

### Memory Transfer Optimization

**Transfer Pattern Analysis:**
```
Sequential:     |--H2D--||--Compute--||--D2H--|
Overlapped:     |--H2D--||--Compute--||--D2H--|
                     |--H2D--||--Compute--||--D2H--|
Pipeline:       |H2D|Compute|D2H|H2D|Compute|D2H|...
```

---

## 🏗️ Chapter 1: Stream Management Architecture

### Advanced Stream Controller

Create `concurrent/stream_manager.go`:

```go
package main

import (
    "fmt"
    "sync"
    "time"
    "context"
    "github.com/stitch1968/gocuda"
    "github.com/stitch1968/gocuda/memory"
)

// Multi-stream execution manager
type StreamManager struct {
    ctx            *cuda.Context
    streams        []*cuda.Stream
    priorityStreams []*cuda.Stream
    eventPool      *EventPool
    profiler       *ConcurrencyProfiler
    mutex          sync.RWMutex
}

type StreamConfig struct {
    NumStreams      int
    UsePriority     bool
    HighPriority    int
    LowPriority     int
    EnableProfiling bool
}

type ConcurrencyProfiler struct {
    kernelTimes     map[string][]time.Duration
    transferTimes   map[string][]time.Duration
    overlapRatios   map[string]float64
    utilizationData []UtilizationSnapshot
    mutex          sync.RWMutex
}

type UtilizationSnapshot struct {
    Timestamp    time.Time
    ActiveStreams int
    PendingOps   int
    MemoryBandwidth float64
    ComputeUtil  float64
}

type EventPool struct {
    events []*cuda.Event
    index  int
    mutex  sync.Mutex
}

func NewStreamManager(config StreamConfig) *StreamManager {
    cuda.Initialize()
    ctx := cuda.GetDefaultContext()
    
    sm := &StreamManager{
        ctx:       ctx,
        streams:   make([]*cuda.Stream, config.NumStreams),
        eventPool: NewEventPool(config.NumStreams * 4),
        profiler: &ConcurrencyProfiler{
            kernelTimes:   make(map[string][]time.Duration),
            transferTimes: make(map[string][]time.Duration),
            overlapRatios: make(map[string]float64),
        },
    }
    
    // Create regular streams
    for i := range sm.streams {
        stream, err := ctx.CreateStream()
        if err != nil {
            panic(fmt.Sprintf("Failed to create stream %d: %v", i, err))
        }
        sm.streams[i] = stream
    }
    
    // Create priority streams if requested
    if config.UsePriority {
        sm.priorityStreams = make([]*cuda.Stream, 2)
        highStream, _ := ctx.CreateStreamWithPriority(config.HighPriority)
        lowStream, _ := ctx.CreateStreamWithPriority(config.LowPriority)
        sm.priorityStreams[0] = highStream
        sm.priorityStreams[1] = lowStream
    }
    
    fmt.Printf("🚀 StreamManager initialized with %d streams\n", config.NumStreams)
    if config.UsePriority {
        fmt.Printf("   Priority streams: High=%d, Low=%d\n", config.HighPriority, config.LowPriority)
    }
    
    return sm
}

func NewEventPool(size int) *EventPool {
    events := make([]*cuda.Event, size)
    ctx := cuda.GetDefaultContext()
    
    for i := range events {
        event, err := ctx.CreateEvent()
        if err != nil {
            panic(fmt.Sprintf("Failed to create event %d: %v", i, err))
        }
        events[i] = event
    }
    
    return &EventPool{events: events}
}

func (ep *EventPool) Get() *cuda.Event {
    ep.mutex.Lock()
    defer ep.mutex.Unlock()
    
    event := ep.events[ep.index]
    ep.index = (ep.index + 1) % len(ep.events)
    return event
}

// Get available stream (round-robin)
func (sm *StreamManager) GetStream() *cuda.Stream {
    sm.mutex.RLock()
    defer sm.mutex.RUnlock()
    
    // Simple round-robin for now
    // In practice, would implement load balancing
    streamIdx := int(time.Now().UnixNano()) % len(sm.streams)
    return sm.streams[streamIdx]
}

// Get high priority stream
func (sm *StreamManager) GetPriorityStream(high bool) *cuda.Stream {
    sm.mutex.RLock()
    defer sm.mutex.RUnlock()
    
    if sm.priorityStreams == nil {
        return sm.GetStream()
    }
    
    if high {
        return sm.priorityStreams[0]
    }
    return sm.priorityStreams[1]
}

// Asynchronous memory copy with profiling
func (sm *StreamManager) CopyAsync(dst, src *memory.DeviceMemory, size int64, direction string, stream *cuda.Stream) error {
    operationID := fmt.Sprintf("%s_%d", direction, size)
    
    startEvent := sm.eventPool.Get()
    endEvent := sm.eventPool.Get()
    
    // Record start
    stream.RecordEvent(startEvent)
    
    var err error
    switch direction {
    case "H2D":
        err = dst.CopyFromHostAsync(src, stream)
    case "D2H":
        err = src.CopyToHostAsync(dst, stream)
    case "D2D":
        err = dst.CopyFromDeviceAsync(src, stream)
    default:
        return fmt.Errorf("unknown copy direction: %s", direction)
    }
    
    if err != nil {
        return err
    }
    
    // Record end and measure time asynchronously
    stream.RecordEvent(endEvent)
    
    go sm.measureTransferTime(operationID, startEvent, endEvent)
    
    return nil
}

func (sm *StreamManager) measureTransferTime(operationID string, startEvent, endEvent *cuda.Event) {
    // Wait for completion
    endEvent.Synchronize()
    
    // Calculate elapsed time
    elapsed := endEvent.ElapsedTime(startEvent)
    
    sm.profiler.mutex.Lock()
    sm.profiler.transferTimes[operationID] = append(sm.profiler.transferTimes[operationID], elapsed)
    sm.profiler.mutex.Unlock()
}

// Execute kernel with timing
func (sm *StreamManager) ExecuteKernelAsync(kernel *cuda.Function, gridDim, blockDim cuda.Dim3, 
                                          sharedMem int, args []interface{}, 
                                          kernelName string, stream *cuda.Stream) error {
    
    startEvent := sm.eventPool.Get()
    endEvent := sm.eventPool.Get()
    
    // Record start
    stream.RecordEvent(startEvent)
    
    // Launch kernel
    err := kernel.LaunchAsync(gridDim, blockDim, sharedMem, stream, args...)
    if err != nil {
        return err
    }
    
    // Record end
    stream.RecordEvent(endEvent)
    
    go sm.measureKernelTime(kernelName, startEvent, endEvent)
    
    return nil
}

func (sm *StreamManager) measureKernelTime(kernelName string, startEvent, endEvent *cuda.Event) {
    endEvent.Synchronize()
    elapsed := endEvent.ElapsedTime(startEvent)
    
    sm.profiler.mutex.Lock()
    sm.profiler.kernelTimes[kernelName] = append(sm.profiler.kernelTimes[kernelName], elapsed)
    sm.profiler.mutex.Unlock()
}

// Synchronize all streams
func (sm *StreamManager) SynchronizeAll() {
    for _, stream := range sm.streams {
        stream.Synchronize()
    }
    
    if sm.priorityStreams != nil {
        for _, stream := range sm.priorityStreams {
            stream.Synchronize()
        }
    }
}

func (sm *StreamManager) Destroy() {
    sm.SynchronizeAll()
    
    for _, stream := range sm.streams {
        stream.Destroy()
    }
    
    if sm.priorityStreams != nil {
        for _, stream := range sm.priorityStreams {
            stream.Destroy()
        }
    }
    
    for _, event := range sm.eventPool.events {
        event.Destroy()
    }
}

func main() {
    fmt.Println("🔄 Advanced Stream Management and Concurrency")
    
    config := StreamConfig{
        NumStreams:      4,
        UsePriority:     true,
        HighPriority:    0,
        LowPriority:     -1,
        EnableProfiling: true,
    }
    
    manager := NewStreamManager(config)
    defer manager.Destroy()
    
    // Demonstrate concurrent patterns
    demonstrateConcurrentPatterns(manager)
    
    // Pipeline processing demonstration
    pipelineDemo(manager)
    
    // Producer-consumer pattern
    producerConsumerDemo(manager)
    
    // Performance analysis
    performanceAnalysis(manager)
}

func demonstrateConcurrentPatterns(manager *StreamManager) {
    fmt.Println("\n🚀 Concurrent Execution Patterns:")
    
    // Pattern 1: Overlapped computation and memory transfer
    overlapDemo(manager)
    
    // Pattern 2: Multiple independent kernels
    multiKernelDemo(manager)
    
    // Pattern 3: Dependency chain with events
    dependencyChainDemo(manager)
}

func overlapDemo(manager *StreamManager) {
    fmt.Println("\n1. Overlapped Computation + Memory Transfer:")
    
    dataSize := int64(1024 * 1024 * 4) // 4MB
    numBatches := 4
    
    // Allocate host-pinned memory for better transfer performance
    hostData := make([][]float32, numBatches)
    deviceData := make([]*memory.DeviceMemory, numBatches)
    
    for i := 0; i < numBatches; i++ {
        hostData[i] = make([]float32, dataSize/4)
        for j := range hostData[i] {
            hostData[i][j] = float32(i*len(hostData[i]) + j)
        }
        
        deviceData[i], _ = memory.Alloc(dataSize)
        defer deviceData[i].Free()
    }
    
    fmt.Printf("  Processing %d batches of %.1fMB each\n", numBatches, float64(dataSize)/1024/1024)
    
    startTotal := time.Now()
    
    // Overlapped execution: transfer batch i+1 while processing batch i
    for i := 0; i < numBatches; i++ {
        stream := manager.GetStream()
        
        // Start data transfer for current batch
        manager.CopyAsync(deviceData[i], nil, dataSize, "H2D", stream)
        
        // If not first batch, we can overlap with computation on previous batch
        if i > 0 {
            computeStream := manager.GetStream()
            // Simulate computation on previous batch
            simulateComputation(manager, deviceData[i-1], 
                              fmt.Sprintf("compute_batch_%d", i-1), computeStream)
        }
        
        // Process current batch after transfer completes
        go func(batchIdx int, s *cuda.Stream) {
            s.Synchronize() // Wait for transfer to complete
            computeStream := manager.GetStream()
            simulateComputation(manager, deviceData[batchIdx], 
                              fmt.Sprintf("compute_batch_%d", batchIdx), computeStream)
        }(i, stream)
    }
    
    // Process final batch
    if numBatches > 0 {
        finalStream := manager.GetStream()
        simulateComputation(manager, deviceData[numBatches-1], 
                          fmt.Sprintf("compute_batch_%d", numBatches-1), finalStream)
    }
    
    manager.SynchronizeAll()
    totalTime := time.Since(startTotal)
    
    fmt.Printf("  ✅ Overlapped execution completed in %v\n", totalTime)
}

func multiKernelDemo(manager *StreamManager) {
    fmt.Println("\n2. Multiple Independent Kernels:")
    
    dataSize := int64(1024 * 1024 * 2)
    
    // Create multiple data sets for different operations
    data1, _ := memory.Alloc(dataSize)
    data2, _ := memory.Alloc(dataSize)  
    data3, _ := memory.Alloc(dataSize)
    defer data1.Free()
    defer data2.Free()
    defer data3.Free()
    
    // Initialize data
    initData := make([]float32, dataSize/4)
    for i := range initData {
        initData[i] = float32(i)
    }
    data1.CopyFromHost(initData)
    data2.CopyFromHost(initData)
    data3.CopyFromHost(initData)
    
    fmt.Printf("  Launching 3 independent operations on %.1fMB each\n", 
               float64(dataSize)/1024/1024)
    
    start := time.Now()
    
    // Launch three different operations on different streams
    stream1 := manager.GetStream()
    stream2 := manager.GetStream()
    stream3 := manager.GetPriorityStream(true) // High priority for stream3
    
    // Simulate different kernel types
    simulateComputation(manager, data1, "vector_add", stream1)
    simulateComputation(manager, data2, "matrix_mult", stream2)
    simulateComputation(manager, data3, "fft_transform", stream3)
    
    manager.SynchronizeAll()
    elapsed := time.Since(start)
    
    fmt.Printf("  ✅ Multi-kernel execution completed in %v\n", elapsed)
}

func dependencyChainDemo(manager *StreamManager) {
    fmt.Println("\n3. Dependency Chain with Events:")
    
    dataSize := int64(1024 * 1024)
    
    stage1Data, _ := memory.Alloc(dataSize)
    stage2Data, _ := memory.Alloc(dataSize)
    stage3Data, _ := memory.Alloc(dataSize)
    defer stage1Data.Free()
    defer stage2Data.Free()
    defer stage3Data.Free()
    
    // Initialize first stage
    initData := make([]float32, dataSize/4)
    for i := range initData {
        initData[i] = float32(i)
    }
    stage1Data.CopyFromHost(initData)
    
    fmt.Printf("  Creating dependency chain: Stage1 → Stage2 → Stage3\n")
    
    start := time.Now()
    
    stream1 := manager.GetStream()
    stream2 := manager.GetStream()
    stream3 := manager.GetStream()
    
    // Create events for synchronization
    event1 := manager.eventPool.Get()
    event2 := manager.eventPool.Get()
    
    // Stage 1: Process initial data
    simulateComputation(manager, stage1Data, "preprocessing", stream1)
    stream1.RecordEvent(event1)
    
    // Stage 2: Wait for stage 1, then process
    stream2.WaitEvent(event1)
    manager.CopyAsync(stage2Data, stage1Data, dataSize, "D2D", stream2)
    simulateComputation(manager, stage2Data, "main_processing", stream2)
    stream2.RecordEvent(event2)
    
    // Stage 3: Wait for stage 2, then finalize
    stream3.WaitEvent(event2)
    manager.CopyAsync(stage3Data, stage2Data, dataSize, "D2D", stream3)
    simulateComputation(manager, stage3Data, "postprocessing", stream3)
    
    manager.SynchronizeAll()
    elapsed := time.Since(start)
    
    fmt.Printf("  ✅ Dependency chain completed in %v\n", elapsed)
}

func simulateComputation(manager *StreamManager, data *memory.DeviceMemory, operation string, stream *cuda.Stream) {
    // Simulate kernel execution time based on operation type
    var duration time.Duration
    switch operation {
    case "vector_add", "preprocessing", "postprocessing":
        duration = 50 * time.Millisecond
    case "matrix_mult", "main_processing":
        duration = 100 * time.Millisecond
    case "fft_transform":
        duration = 200 * time.Millisecond
    default:
        duration = 75 * time.Millisecond
    }
    
    // Record simulation in profiler
    start := time.Now()
    
    // Simulate work (in real implementation, would be actual kernel launch)
    time.Sleep(duration)
    
    elapsed := time.Since(start)
    
    manager.profiler.mutex.Lock()
    manager.profiler.kernelTimes[operation] = append(
        manager.profiler.kernelTimes[operation], elapsed)
    manager.profiler.mutex.Unlock()
}

func pipelineDemo(manager *StreamManager) {
    fmt.Println("\n🔄 Pipeline Processing:")
    
    numStages := 3
    batchSize := int64(512 * 1024) // 512KB per batch
    numBatches := 8
    
    fmt.Printf("  %d-stage pipeline processing %d batches of %.1fKB\n", 
               numStages, numBatches, float64(batchSize)/1024)
    
    // Create pipeline stages
    pipeline := NewPipeline(manager, numStages, batchSize)
    defer pipeline.Cleanup()
    
    start := time.Now()
    
    // Feed batches into pipeline
    for i := 0; i < numBatches; i++ {
        batch := createBatch(i, batchSize)
        pipeline.ProcessBatch(batch, i)
    }
    
    // Wait for pipeline to drain
    pipeline.Flush()
    
    elapsed := time.Since(start)
    fmt.Printf("  ✅ Pipeline processing completed in %v\n", elapsed)
    
    throughput := float64(numBatches*int(batchSize)) / elapsed.Seconds() / 1024 / 1024
    fmt.Printf("  Throughput: %.2f MB/s\n", throughput)
}

type Pipeline struct {
    manager   *StreamManager
    stages    []*PipelineStage
    batchSize int64
}

type PipelineStage struct {
    stageID   int
    stream    *cuda.Stream
    inputData *memory.DeviceMemory
    outputData *memory.DeviceMemory
    event     *cuda.Event
}

func NewPipeline(manager *StreamManager, numStages int, batchSize int64) *Pipeline {
    stages := make([]*PipelineStage, numStages)
    
    for i := 0; i < numStages; i++ {
        inputData, _ := memory.Alloc(batchSize)
        outputData, _ := memory.Alloc(batchSize)
        
        stages[i] = &PipelineStage{
            stageID:    i,
            stream:     manager.GetStream(),
            inputData:  inputData,
            outputData: outputData,
            event:      manager.eventPool.Get(),
        }
    }
    
    return &Pipeline{
        manager:   manager,
        stages:    stages,
        batchSize: batchSize,
    }
}

func (p *Pipeline) ProcessBatch(batch []float32, batchID int) {
    // Stage 0: Input processing
    stage := p.stages[0]
    stage.inputData.CopyFromHost(batch)
    simulateComputation(p.manager, stage.inputData, 
                       fmt.Sprintf("stage0_batch%d", batchID), stage.stream)
    stage.stream.RecordEvent(stage.event)
    
    // Subsequent stages process in pipeline fashion
    for i := 1; i < len(p.stages); i++ {
        currentStage := p.stages[i]
        prevStage := p.stages[i-1]
        
        // Wait for previous stage
        currentStage.stream.WaitEvent(prevStage.event)
        
        // Copy data from previous stage
        p.manager.CopyAsync(currentStage.inputData, prevStage.inputData, 
                          p.batchSize, "D2D", currentStage.stream)
        
        // Process current stage
        simulateComputation(p.manager, currentStage.inputData,
                          fmt.Sprintf("stage%d_batch%d", i, batchID), 
                          currentStage.stream)
        
        currentStage.stream.RecordEvent(currentStage.event)
    }
}

func (p *Pipeline) Flush() {
    // Wait for all stages to complete
    for _, stage := range p.stages {
        stage.stream.Synchronize()
    }
}

func (p *Pipeline) Cleanup() {
    for _, stage := range p.stages {
        stage.inputData.Free()
        stage.outputData.Free()
    }
}

func createBatch(batchID int, size int64) []float32 {
    batch := make([]float32, size/4)
    for i := range batch {
        batch[i] = float32(batchID*len(batch) + i)
    }
    return batch
}

func producerConsumerDemo(manager *StreamManager) {
    fmt.Println("\n🔄 Producer-Consumer Pattern:")
    
    bufferSize := 4
    numItems := 20
    itemSize := int64(256 * 1024) // 256KB per item
    
    fmt.Printf("  Buffer size: %d items, Processing %d items of %.1fKB each\n", 
               bufferSize, numItems, float64(itemSize)/1024)
    
    // Create producer-consumer system
    pc := NewProducerConsumer(manager, bufferSize, itemSize)
    defer pc.Cleanup()
    
    start := time.Now()
    
    // Start consumer in background
    ctx, cancel := context.WithCancel(context.Background())
    defer cancel()
    
    var wg sync.WaitGroup
    wg.Add(1)
    go func() {
        defer wg.Done()
        pc.StartConsumer(ctx)
    }()
    
    // Producer generates items
    for i := 0; i < numItems; i++ {
        item := createBatch(i, itemSize)
        pc.Produce(item, i)
    }
    
    // Signal completion and wait
    pc.SignalComplete()
    wg.Wait()
    
    elapsed := time.Since(start)
    fmt.Printf("  ✅ Producer-consumer completed in %v\n", elapsed)
}

type ProducerConsumer struct {
    manager     *StreamManager
    buffer      chan *BufferItem
    itemSize    int64
    completed   chan bool
}

type BufferItem struct {
    Data   []float32
    ItemID int
    Device *memory.DeviceMemory
}

func NewProducerConsumer(manager *StreamManager, bufferSize int, itemSize int64) *ProducerConsumer {
    return &ProducerConsumer{
        manager:   manager,
        buffer:    make(chan *BufferItem, bufferSize),
        itemSize:  itemSize,
        completed: make(chan bool, 1),
    }
}

func (pc *ProducerConsumer) Produce(data []float32, itemID int) {
    deviceMem, _ := memory.Alloc(pc.itemSize)
    
    item := &BufferItem{
        Data:   data,
        ItemID: itemID,
        Device: deviceMem,
    }
    
    // Copy to device asynchronously
    stream := pc.manager.GetStream()
    deviceMem.CopyFromHost(data)
    
    pc.buffer <- item
}

func (pc *ProducerConsumer) StartConsumer(ctx context.Context) {
    processedCount := 0
    
    for {
        select {
        case item := <-pc.buffer:
            // Process item on GPU
            stream := pc.manager.GetStream()
            simulateComputation(pc.manager, item.Device, 
                              fmt.Sprintf("consumer_item_%d", item.ItemID), stream)
            stream.Synchronize()
            
            processedCount++
            fmt.Printf("    Processed item %d\n", item.ItemID)
            
            // Cleanup
            item.Device.Free()
            
        case <-pc.completed:
            fmt.Printf("  Consumer processed %d items\n", processedCount)
            return
            
        case <-ctx.Done():
            return
        }
    }
}

func (pc *ProducerConsumer) SignalComplete() {
    pc.completed <- true
}

func (pc *ProducerConsumer) Cleanup() {
    close(pc.buffer)
    close(pc.completed)
}

func performanceAnalysis(manager *StreamManager) {
    fmt.Println("\n📊 Concurrency Performance Analysis:")
    
    manager.profiler.mutex.RLock()
    defer manager.profiler.mutex.RUnlock()
    
    fmt.Println("\nKernel Execution Times:")
    for operation, times := range manager.profiler.kernelTimes {
        if len(times) > 0 {
            avg := calculateAverage(times)
            min := findMinDuration(times)
            max := findMaxDuration(times)
            fmt.Printf("  %-20s: avg=%v, min=%v, max=%v (%d calls)\n", 
                      operation, avg, min, max, len(times))
        }
    }
    
    fmt.Println("\nMemory Transfer Times:")
    for operation, times := range manager.profiler.transferTimes {
        if len(times) > 0 {
            avg := calculateAverage(times)
            fmt.Printf("  %-20s: avg=%v (%d transfers)\n", 
                      operation, avg, len(times))
        }
    }
    
    fmt.Println("\n💡 Concurrency Optimization Tips:")
    fmt.Println("  1. Keep GPU busy: overlap computation with memory transfer")
    fmt.Println("  2. Use multiple streams: independent operations in parallel") 
    fmt.Println("  3. Pipeline processing: steady-state throughput")
    fmt.Println("  4. Priority streams: critical path optimization")
    fmt.Println("  5. Event synchronization: minimal blocking")
    fmt.Println("  6. Pinned memory: faster host-device transfers")
    fmt.Println("  7. Async operations: maximize utilization")
    
    fmt.Println("\n🎯 Stream Usage Guidelines:")
    fmt.Println("  • 2-4 streams: good balance for most applications")
    fmt.Println("  • More streams: diminishing returns due to overhead")
    fmt.Println("  • Priority streams: use sparingly for critical operations")
    fmt.Println("  • Event overhead: minimize synchronization points")
    fmt.Println("  • Memory bandwidth: often the bottleneck in concurrent operations")
}

func calculateAverage(durations []time.Duration) time.Duration {
    if len(durations) == 0 {
        return 0
    }
    
    var total time.Duration
    for _, d := range durations {
        total += d
    }
    return total / time.Duration(len(durations))
}

func findMinDuration(durations []time.Duration) time.Duration {
    if len(durations) == 0 {
        return 0
    }
    
    min := durations[0]
    for _, d := range durations[1:] {
        if d < min {
            min = d
        }
    }
    return min
}

func findMaxDuration(durations []time.Duration) time.Duration {
    if len(durations) == 0 {
        return 0
    }
    
    max := durations[0]
    for _, d := range durations[1:] {
        if d > max {
            max = d
        }
    }
    return max
}
```

---

## 🎯 Module Assessment

### **Knowledge Validation**

1. **Stream Management**: Create and manage multiple CUDA streams efficiently
2. **Concurrency Patterns**: Implement overlapped execution and pipeline processing
3. **Synchronization**: Use events for inter-stream coordination
4. **Performance Analysis**: Profile and optimize concurrent execution

### **Practical Challenge**

Implement a real-time data processing system:
- **Video Processing**: Multi-stream pipeline for real-time video filtering
- **Signal Processing**: Concurrent FFT processing with overlapped I/O
- **Scientific Computing**: Multi-phase simulation with producer-consumer pattern
- **Machine Learning**: Concurrent batch processing with asynchronous data loading

### **Success Criteria**

- ✅ Achieve >90% GPU utilization in concurrent scenarios
- ✅ Implement efficient producer-consumer with minimal blocking
- ✅ Demonstrate measurable performance improvement over sequential execution
- ✅ Handle synchronization correctly without race conditions

---

## 🚀 Next Steps

**Excellent! You've mastered concurrent GPU programming.**

**You're now ready for:**
➡️ **[Module 5: Hardware Optimization](TRAINING_INTERMEDIATE_5_HARDWARE.md)**

**Skills Mastered:**
- 🚀 **Stream Management** - Multi-stream coordination and load balancing
- 🔄 **Concurrency Patterns** - Overlapped execution and pipeline processing
- ⚡ **Performance Optimization** - Bandwidth utilization and throughput maximization
- 🎯 **Synchronization** - Event-driven coordination and dependency management

---

*From sequential processing to concurrent mastery - unlocking the full power of parallel execution! 🔄⚡*
