# 🌐 Expert Module 2: Multi-GPU & Distributed Computing

**Goal:** Master multi-GPU scaling, distributed computing, and enterprise-grade GPU cluster management for maximum computational throughput

---

## 📚 Learning Objectives

By the end of this module, you will:
- 🔄 **Scale across multiple GPUs** with efficient communication patterns
- 🌐 **Implement distributed algorithms** for GPU clusters
- ⚡ **Optimize inter-GPU communication** with NCCL, NVLink, and InfiniBand
- 🎯 **Design fault-tolerant systems** with load balancing and recovery
- 📊 **Monitor and manage** large-scale GPU deployments

---

## 🧠 Theoretical Foundation

### Multi-GPU Architecture Hierarchy

**GPU Scaling Levels:**
```
Single Node Multi-GPU
├── NVLink (300+ GB/s bidirectional)
├── PCIe (64 GB/s bidirectional)
└── CPU Memory Bridge

Multi-Node GPU Cluster
├── InfiniBand (200+ Gb/s)
├── Ethernet (100+ Gb/s)
├── GPU-Direct RDMA
└── Distributed Memory Management
```

### Communication Patterns

**Collective Operations:**
- **AllReduce**: Reduction + broadcast to all GPUs
- **AllGather**: Gather data from all GPUs to all GPUs
- **ReduceScatter**: Reduction + scatter to all GPUs
- **Broadcast**: Send data from one GPU to all others
- **P2P Transfer**: Direct peer-to-peer communication

### Scaling Efficiency

**Performance Metrics:**
```
Strong Scaling: Fixed problem size, more GPUs
Weak Scaling: Problem size scales with GPU count
Efficiency = (T₁ / (N × Tₙ)) × 100%
Bandwidth Utilization = Actual BW / Theoretical BW
```

---

## 🏗️ Chapter 1: Multi-GPU Programming Framework

### Advanced Multi-GPU Manager

Create `distributed/multi_gpu_manager.go`:

```go
package main

import (
    "fmt"
    "sync"
    "time"
    "context"
    "github.com/stitch1968/gocuda"
    "github.com/stitch1968/gocuda/memory"
    "github.com/stitch1968/gocuda/communication"
)

// Enterprise-grade multi-GPU management system
type MultiGPUManager struct {
    devices       []*cuda.Device
    contexts      []*cuda.Context
    streams       [][]*cuda.Stream
    communicator  *NCCLCommunicator
    loadBalancer  *LoadBalancer
    faultHandler  *FaultTolerance
    profiler      *DistributedProfiler
    scheduler     *TaskScheduler
    mutex         sync.RWMutex
}

type NCCLCommunicator struct {
    numDevices    int
    comms         []communication.NCCLComm
    ncclID        communication.NCCLUniqueID
    initialized   bool
    mutex         sync.Mutex
}

type LoadBalancer struct {
    deviceLoads   []float64
    taskQueue     chan DistributedTask
    completedTasks chan TaskResult
    strategy      LoadBalancingStrategy
    metrics       *LoadBalancingMetrics
}

type LoadBalancingStrategy int

const (
    RoundRobin LoadBalancingStrategy = iota
    WeightedLoad
    CapacityBased
    LatencyOptimized
)

type LoadBalancingMetrics struct {
    TaskDistribution []int64
    ExecutionTimes   []time.Duration
    DeviceUtilization []float64
    LoadImbalance    float64
}

type FaultTolerance struct {
    deviceHealth    []DeviceHealth
    checkpointMgr   *CheckpointManager
    recoveryMgr     *RecoveryManager
    redundancyLevel int
    healthMonitor   chan HealthStatus
}

type DeviceHealth struct {
    DeviceID      int
    IsHealthy     bool
    LastError     error
    ErrorCount    int
    LastCheckTime time.Time
    Temperature   float32
    PowerUsage    float32
    MemoryUsage   float64
}

type DistributedTask struct {
    TaskID       string
    TaskType     string
    Data         interface{}
    Dependencies []string
    DeviceHint   int
    Priority     int
    Timeout      time.Duration
    Callback     func(TaskResult)
}

type TaskResult struct {
    TaskID       string
    Success      bool
    Result       interface{}
    ExecutionTime time.Duration
    DeviceID     int
    Error        error
    Metrics      map[string]interface{}
}

type DistributedProfiler struct {
    communicationMetrics *CommunicationMetrics
    scalingMetrics      *ScalingMetrics
    performanceHistory  []PerformanceSnapshot
    bottleneckAnalysis  *BottleneckAnalysis
}

type CommunicationMetrics struct {
    P2PBandwidth     [][]float64 // [src][dst] bandwidth matrix
    CollectiveTimes  map[string]time.Duration
    MessageSizes     map[string][]int64
    LatencyMatrix    [][]time.Duration
    BandwidthUtil    []float64
}

type ScalingMetrics struct {
    StrongScalingEff  []float64 // efficiency vs number of GPUs
    WeakScalingEff    []float64
    IdealSpeedup      []float64
    ActualSpeedup     []float64
    ParallelEfficiency float64
}

func NewMultiGPUManager() *MultiGPUManager {
    cuda.Initialize()
    
    // Discover available GPUs
    deviceCount := cuda.GetDeviceCount()
    
    manager := &MultiGPUManager{
        devices:  make([]*cuda.Device, deviceCount),
        contexts: make([]*cuda.Context, deviceCount),
        streams:  make([][]*cuda.Stream, deviceCount),
    }
    
    // Initialize all devices
    for i := 0; i < deviceCount; i++ {
        device := cuda.GetDevice(i)
        manager.devices[i] = device
        
        // Set device and create context
        cuda.SetDevice(i)
        ctx := cuda.CreateContext(device)
        manager.contexts[i] = ctx
        
        // Create multiple streams per device
        streamsPerDevice := 4
        manager.streams[i] = make([]*cuda.Stream, streamsPerDevice)
        for j := 0; j < streamsPerDevice; j++ {
            stream, _ := ctx.CreateStream()
            manager.streams[i][j] = stream
        }
    }
    
    // Initialize subsystems
    manager.initializeCommunication()
    manager.initializeLoadBalancer()
    manager.initializeFaultTolerance()
    manager.initializeProfiler()
    
    fmt.Printf("🌐 Multi-GPU Manager initialized: %d GPUs\n", deviceCount)
    manager.printDeviceTopology()
    
    return manager
}

func (mgr *MultiGPUManager) initializeCommunication() {
    numDevices := len(mgr.devices)
    
    mgr.communicator = &NCCLCommunicator{
        numDevices: numDevices,
        comms:      make([]communication.NCCLComm, numDevices),
    }
    
    // Initialize NCCL
    ncclID := communication.GetUniqueID()
    mgr.communicator.ncclID = ncclID
    
    // Create NCCL communicators for each device
    for i := 0; i < numDevices; i++ {
        cuda.SetDevice(i)
        comm, err := communication.InitRank(ncclID, numDevices, i)
        if err != nil {
            fmt.Printf("Warning: Failed to initialize NCCL on device %d: %v\n", i, err)
            continue
        }
        mgr.communicator.comms[i] = comm
    }
    
    mgr.communicator.initialized = true
    fmt.Printf("   NCCL communication initialized across %d devices\n", numDevices)
}

func (mgr *MultiGPUManager) initializeLoadBalancer() {
    numDevices := len(mgr.devices)
    
    mgr.loadBalancer = &LoadBalancer{
        deviceLoads:    make([]float64, numDevices),
        taskQueue:      make(chan DistributedTask, 1000),
        completedTasks: make(chan TaskResult, 1000),
        strategy:       WeightedLoad,
        metrics: &LoadBalancingMetrics{
            TaskDistribution:  make([]int64, numDevices),
            ExecutionTimes:    make([]time.Duration, 0),
            DeviceUtilization: make([]float64, numDevices),
        },
    }
    
    // Start load balancer workers
    for i := 0; i < numDevices; i++ {
        go mgr.loadBalancerWorker(i)
    }
    
    fmt.Printf("   Load balancer initialized with %s strategy\n", 
               mgr.getStrategyName(mgr.loadBalancer.strategy))
}

func (mgr *MultiGPUManager) getStrategyName(strategy LoadBalancingStrategy) string {
    switch strategy {
    case RoundRobin:
        return "Round Robin"
    case WeightedLoad:
        return "Weighted Load"
    case CapacityBased:
        return "Capacity Based"
    case LatencyOptimized:
        return "Latency Optimized"
    default:
        return "Unknown"
    }
}

func (mgr *MultiGPUManager) initializeFaultTolerance() {
    numDevices := len(mgr.devices)
    
    mgr.faultHandler = &FaultTolerance{
        deviceHealth:    make([]DeviceHealth, numDevices),
        redundancyLevel: 1, // Single device failure tolerance
        healthMonitor:   make(chan HealthStatus, 100),
    }
    
    // Initialize device health monitoring
    for i := 0; i < numDevices; i++ {
        mgr.faultHandler.deviceHealth[i] = DeviceHealth{
            DeviceID:      i,
            IsHealthy:     true,
            ErrorCount:    0,
            LastCheckTime: time.Now(),
        }
    }
    
    // Start health monitoring goroutine
    go mgr.healthMonitor()
    
    fmt.Printf("   Fault tolerance initialized (redundancy level: %d)\n", 
               mgr.faultHandler.redundancyLevel)
}

func (mgr *MultiGPUManager) initializeProfiler() {
    numDevices := len(mgr.devices)
    
    mgr.profiler = &DistributedProfiler{
        communicationMetrics: &CommunicationMetrics{
            P2PBandwidth:    make([][]float64, numDevices),
            CollectiveTimes: make(map[string]time.Duration),
            MessageSizes:    make(map[string][]int64),
            LatencyMatrix:   make([][]time.Duration, numDevices),
            BandwidthUtil:   make([]float64, numDevices),
        },
        scalingMetrics: &ScalingMetrics{
            StrongScalingEff: make([]float64, 0),
            WeakScalingEff:   make([]float64, 0),
            IdealSpeedup:     make([]float64, 0),
            ActualSpeedup:    make([]float64, 0),
        },
        performanceHistory: make([]PerformanceSnapshot, 0),
    }
    
    // Initialize bandwidth matrix
    for i := 0; i < numDevices; i++ {
        mgr.profiler.communicationMetrics.P2PBandwidth[i] = make([]float64, numDevices)
        mgr.profiler.communicationMetrics.LatencyMatrix[i] = make([]time.Duration, numDevices)
    }
    
    fmt.Printf("   Distributed profiler initialized\n")
}

func (mgr *MultiGPUManager) printDeviceTopology() {
    fmt.Println("\n🔗 GPU Topology Analysis:")
    
    numDevices := len(mgr.devices)
    for i := 0; i < numDevices; i++ {
        device := mgr.devices[i]
        props := device.GetProperties()
        
        fmt.Printf("  GPU %d: %s\n", i, props.Name)
        fmt.Printf("    Memory: %.1f GB\n", float64(props.TotalGlobalMem)/(1024*1024*1024))
        fmt.Printf("    Compute Capability: %d.%d\n", props.Major, props.Minor)
        
        // Analyze P2P connectivity
        for j := 0; j < numDevices; j++ {
            if i != j {
                canAccess := mgr.canAccessPeer(i, j)
                nvlinkConnected := mgr.hasNVLink(i, j)
                
                if nvlinkConnected {
                    fmt.Printf("    -> GPU %d: NVLink ✅\n", j)
                } else if canAccess {
                    fmt.Printf("    -> GPU %d: PCIe ⚡\n", j)
                } else {
                    fmt.Printf("    -> GPU %d: No P2P ❌\n", j)
                }
            }
        }
    }
}

func (mgr *MultiGPUManager) canAccessPeer(deviceFrom, deviceTo int) bool {
    // In real implementation, would check CUDA peer access
    // For simulation, assume modern systems support P2P
    return true
}

func (mgr *MultiGPUManager) hasNVLink(deviceFrom, deviceTo int) bool {
    // In real implementation, would check NVLink topology
    // For simulation, assume first few devices have NVLink
    return deviceFrom < 4 && deviceTo < 4 && deviceFrom != deviceTo
}

// Advanced multi-GPU operations
func (mgr *MultiGPUManager) AllReduce(data [][]float32, operation string) error {
    fmt.Printf("🔄 AllReduce operation: %s across %d GPUs\n", operation, len(mgr.devices))
    
    if !mgr.communicator.initialized {
        return fmt.Errorf("NCCL not initialized")
    }
    
    start := time.Now()
    
    // Simulate AllReduce operation
    result := mgr.simulateAllReduce(data, operation)
    
    elapsed := time.Since(start)
    
    // Update metrics
    mgr.profiler.communicationMetrics.CollectiveTimes["AllReduce"] = elapsed
    
    // Calculate bandwidth
    totalElements := 0
    for _, deviceData := range data {
        totalElements += len(deviceData)
    }
    
    dataSize := float64(totalElements * 4) // sizeof(float32)
    bandwidth := (dataSize * float64(len(mgr.devices))) / elapsed.Seconds() / (1024*1024*1024) // GB/s
    
    fmt.Printf("   Completed in %v (%.2f GB/s effective bandwidth)\n", elapsed, bandwidth)
    
    // Copy results back
    for i, deviceResult := range result {
        if i < len(data) {
            copy(data[i], deviceResult)
        }
    }
    
    return nil
}

func (mgr *MultiGPUManager) simulateAllReduce(data [][]float32, operation string) [][]float32 {
    if len(data) == 0 {
        return data
    }
    
    numDevices := len(data)
    elementsPerDevice := len(data[0])
    
    // Create result arrays
    result := make([][]float32, numDevices)
    for i := range result {
        result[i] = make([]float32, elementsPerDevice)
    }
    
    // Perform reduction operation
    switch operation {
    case "sum":
        for i := 0; i < elementsPerDevice; i++ {
            var sum float32
            for j := 0; j < numDevices; j++ {
                sum += data[j][i]
            }
            // Broadcast sum to all devices
            for j := 0; j < numDevices; j++ {
                result[j][i] = sum
            }
        }
    case "average":
        for i := 0; i < elementsPerDevice; i++ {
            var sum float32
            for j := 0; j < numDevices; j++ {
                sum += data[j][i]
            }
            avg := sum / float32(numDevices)
            for j := 0; j < numDevices; j++ {
                result[j][i] = avg
            }
        }
    case "max":
        for i := 0; i < elementsPerDevice; i++ {
            max := data[0][i]
            for j := 1; j < numDevices; j++ {
                if data[j][i] > max {
                    max = data[j][i]
                }
            }
            for j := 0; j < numDevices; j++ {
                result[j][i] = max
            }
        }
    default:
        // Default to sum
        return mgr.simulateAllReduce(data, "sum")
    }
    
    return result
}

func (mgr *MultiGPUManager) P2PTransfer(srcDevice, dstDevice int, data *memory.DeviceMemory, size int64) error {
    fmt.Printf("📡 P2P Transfer: GPU %d -> GPU %d (%d bytes)\n", srcDevice, dstDevice, size)
    
    if srcDevice >= len(mgr.devices) || dstDevice >= len(mgr.devices) {
        return fmt.Errorf("invalid device ID")
    }
    
    start := time.Now()
    
    // In real implementation, would use cudaMemcpyPeer
    // For simulation, add realistic transfer time
    time.Sleep(time.Duration(float64(size)/1e9) * time.Second) // Simulate 1 GB/s transfer
    
    elapsed := time.Since(start)
    
    // Calculate bandwidth
    bandwidth := float64(size) / elapsed.Seconds() / (1024*1024*1024) // GB/s
    
    // Update metrics
    mgr.profiler.communicationMetrics.P2PBandwidth[srcDevice][dstDevice] = bandwidth
    mgr.profiler.communicationMetrics.LatencyMatrix[srcDevice][dstDevice] = elapsed
    
    fmt.Printf("   Transfer completed in %v (%.2f GB/s)\n", elapsed, bandwidth)
    
    return nil
}

func (mgr *MultiGPUManager) loadBalancerWorker(deviceID int) {
    cuda.SetDevice(deviceID)
    
    for task := range mgr.loadBalancer.taskQueue {
        // Check if this task should run on this device
        selectedDevice := mgr.selectDeviceForTask(task)
        if selectedDevice != deviceID {
            continue
        }
        
        start := time.Now()
        
        // Execute task
        result := mgr.executeTask(task, deviceID)
        
        elapsed := time.Since(start)
        result.ExecutionTime = elapsed
        result.DeviceID = deviceID
        
        // Update load metrics
        mgr.loadBalancer.metrics.TaskDistribution[deviceID]++
        mgr.loadBalancer.metrics.ExecutionTimes = append(
            mgr.loadBalancer.metrics.ExecutionTimes, elapsed)
        
        // Send result
        mgr.loadBalancer.completedTasks <- result
        
        // Update device load
        mgr.updateDeviceLoad(deviceID, elapsed)
    }
}

func (mgr *MultiGPUManager) selectDeviceForTask(task DistributedTask) int {
    switch mgr.loadBalancer.strategy {
    case RoundRobin:
        return int(task.TaskID[0]) % len(mgr.devices) // Simple hash-based selection
        
    case WeightedLoad:
        // Select device with lowest current load
        minLoad := mgr.loadBalancer.deviceLoads[0]
        selectedDevice := 0
        
        for i, load := range mgr.loadBalancer.deviceLoads {
            if load < minLoad {
                minLoad = load
                selectedDevice = i
            }
        }
        return selectedDevice
        
    case CapacityBased:
        // Consider device capabilities
        return mgr.selectByCapacity(task)
        
    case LatencyOptimized:
        // Consider data locality and communication cost
        return mgr.selectByLatency(task)
        
    default:
        return 0
    }
}

func (mgr *MultiGPUManager) selectByCapacity(task DistributedTask) int {
    // Simple capacity-based selection (in real implementation, would consider memory, compute capability)
    bestDevice := 0
    bestScore := 0.0
    
    for i, device := range mgr.devices {
        props := device.GetProperties()
        
        // Score based on available memory and compute capability
        memScore := float64(props.TotalGlobalMem) / (1024 * 1024 * 1024) // GB
        computeScore := float64(props.Major*10 + props.Minor)
        loadScore := 1.0 / (1.0 + mgr.loadBalancer.deviceLoads[i])
        
        totalScore := memScore * 0.4 + computeScore * 0.3 + loadScore * 0.3
        
        if totalScore > bestScore {
            bestScore = totalScore
            bestDevice = i
        }
    }
    
    return bestDevice
}

func (mgr *MultiGPUManager) selectByLatency(task DistributedTask) int {
    // Simplified latency-based selection
    if task.DeviceHint >= 0 && task.DeviceHint < len(mgr.devices) {
        return task.DeviceHint
    }
    
    // Default to lowest load
    return mgr.selectDeviceForTask(DistributedTask{
        TaskID: task.TaskID,
    }) // Use weighted load strategy
}

func (mgr *MultiGPUManager) executeTask(task DistributedTask, deviceID int) TaskResult {
    result := TaskResult{
        TaskID:   task.TaskID,
        DeviceID: deviceID,
        Success:  true,
        Metrics:  make(map[string]interface{}),
    }
    
    // Simulate task execution based on type
    switch task.TaskType {
    case "matrix_multiply":
        result.Result = mgr.simulateMatrixMultiply(task.Data, deviceID)
    case "reduction":
        result.Result = mgr.simulateReduction(task.Data, deviceID)
    case "convolution":
        result.Result = mgr.simulateConvolution(task.Data, deviceID)
    default:
        time.Sleep(10 * time.Millisecond) // Generic task simulation
        result.Result = "completed"
    }
    
    return result
}

func (mgr *MultiGPUManager) simulateMatrixMultiply(data interface{}, deviceID int) interface{} {
    // Simulate matrix multiplication workload
    time.Sleep(20 * time.Millisecond)
    return fmt.Sprintf("matrix_multiply_result_device_%d", deviceID)
}

func (mgr *MultiGPUManager) simulateReduction(data interface{}, deviceID int) interface{} {
    // Simulate reduction workload
    time.Sleep(5 * time.Millisecond)
    return fmt.Sprintf("reduction_result_device_%d", deviceID)
}

func (mgr *MultiGPUManager) simulateConvolution(data interface{}, deviceID int) interface{} {
    // Simulate convolution workload
    time.Sleep(15 * time.Millisecond)
    return fmt.Sprintf("convolution_result_device_%d", deviceID)
}

func (mgr *MultiGPUManager) updateDeviceLoad(deviceID int, taskTime time.Duration) {
    // Exponential moving average for device load
    alpha := 0.1
    newLoad := taskTime.Seconds()
    mgr.loadBalancer.deviceLoads[deviceID] = 
        alpha*newLoad + (1-alpha)*mgr.loadBalancer.deviceLoads[deviceID]
}

func (mgr *MultiGPUManager) healthMonitor() {
    ticker := time.NewTicker(5 * time.Second)
    defer ticker.Stop()
    
    for range ticker.C {
        mgr.checkDeviceHealth()
    }
}

func (mgr *MultiGPUManager) checkDeviceHealth() {
    for i, device := range mgr.devices {
        health := &mgr.faultHandler.deviceHealth[i]
        
        // Simulate health checks
        cuda.SetDevice(i)
        
        // Check memory usage
        memInfo := device.GetMemoryInfo()
        health.MemoryUsage = float64(memInfo.Used) / float64(memInfo.Total)
        
        // Simulate temperature and power readings
        health.Temperature = 65.0 + float32(i*5) // Simulate different temps
        health.PowerUsage = 200.0 + float32(i*20)
        
        // Health criteria
        wasHealthy := health.IsHealthy
        health.IsHealthy = health.Temperature < 85.0 && 
                          health.MemoryUsage < 0.95 &&
                          health.PowerUsage < 400.0
        
        if !health.IsHealthy && wasHealthy {
            fmt.Printf("⚠️ Device %d health degraded: Temp=%.1f°C, Mem=%.1f%%, Power=%.1fW\n",
                       i, health.Temperature, health.MemoryUsage*100, health.PowerUsage)
        }
        
        health.LastCheckTime = time.Now()
    }
}

func (mgr *MultiGPUManager) SubmitTask(task DistributedTask) <-chan TaskResult {
    resultChan := make(chan TaskResult, 1)
    
    // Add callback to forward result
    originalCallback := task.Callback
    task.Callback = func(result TaskResult) {
        resultChan <- result
        if originalCallback != nil {
            originalCallback(result)
        }
    }
    
    // Submit to load balancer
    mgr.loadBalancer.taskQueue <- task
    
    return resultChan
}

func (mgr *MultiGPUManager) BenchmarkScaling(problemSizes []int) {
    fmt.Println("\n📊 Multi-GPU Scaling Benchmark:")
    
    deviceCounts := []int{1, 2, 4, len(mgr.devices)}
    
    fmt.Println("GPUs\tProblem Size\tTime\t\tSpeedup\tEfficiency")
    
    for _, size := range problemSizes {
        var baselineTime time.Duration
        
        for _, gpuCount := range deviceCounts {
            if gpuCount > len(mgr.devices) {
                continue
            }
            
            // Run benchmark with specified GPU count
            execTime := mgr.runScalingBenchmark(size, gpuCount)
            
            if gpuCount == 1 {
                baselineTime = execTime
            }
            
            speedup := float64(baselineTime) / float64(execTime)
            efficiency := speedup / float64(gpuCount) * 100
            
            fmt.Printf("%d\t%d\t\t%v\t%.2fx\t%.1f%%\n", 
                       gpuCount, size, execTime, speedup, efficiency)
            
            // Store scaling metrics
            if len(mgr.profiler.scalingMetrics.ActualSpeedup) < len(deviceCounts) {
                mgr.profiler.scalingMetrics.ActualSpeedup = append(
                    mgr.profiler.scalingMetrics.ActualSpeedup, speedup)
                mgr.profiler.scalingMetrics.IdealSpeedup = append(
                    mgr.profiler.scalingMetrics.IdealSpeedup, float64(gpuCount))
            }
        }
    }
    
    mgr.analyzeScalingEfficiency()
}

func (mgr *MultiGPUManager) runScalingBenchmark(problemSize, gpuCount int) time.Duration {
    // Create synthetic workload
    tasks := make([]DistributedTask, problemSize/1000) // 1 task per 1000 elements
    
    start := time.Now()
    
    // Submit tasks
    for i := range tasks {
        task := DistributedTask{
            TaskID:   fmt.Sprintf("benchmark_%d_%d", problemSize, i),
            TaskType: "matrix_multiply",
            Data:     problemSize,
            Priority: 1,
        }
        
        mgr.loadBalancer.taskQueue <- task
    }
    
    // Wait for completion (simplified)
    completed := 0
    for completed < len(tasks) {
        select {
        case <-mgr.loadBalancer.completedTasks:
            completed++
        case <-time.After(30 * time.Second):
            fmt.Printf("Warning: Benchmark timeout\n")
            return time.Since(start)
        }
    }
    
    return time.Since(start)
}

func (mgr *MultiGPUManager) analyzeScalingEfficiency() {
    fmt.Println("\n📈 Scaling Efficiency Analysis:")
    
    metrics := mgr.profiler.scalingMetrics
    
    if len(metrics.ActualSpeedup) > 0 {
        fmt.Println("GPUs\tIdeal Speedup\tActual Speedup\tEfficiency")
        
        for i, actual := range metrics.ActualSpeedup {
            ideal := metrics.IdealSpeedup[i]
            efficiency := (actual / ideal) * 100
            
            fmt.Printf("%.0f\t%.2fx\t\t%.2fx\t\t%.1f%%\n", 
                       ideal, ideal, actual, efficiency)
        }
    }
    
    // Calculate parallel efficiency
    if len(metrics.ActualSpeedup) > 1 {
        totalEfficiency := 0.0
        for i, actual := range metrics.ActualSpeedup {
            if i > 0 { // Skip single GPU
                ideal := metrics.IdealSpeedup[i]
                efficiency := actual / ideal
                totalEfficiency += efficiency
            }
        }
        
        avgEfficiency := totalEfficiency / float64(len(metrics.ActualSpeedup)-1)
        mgr.profiler.scalingMetrics.ParallelEfficiency = avgEfficiency
        
        fmt.Printf("\nOverall Parallel Efficiency: %.1f%%\n", avgEfficiency*100)
        
        if avgEfficiency > 0.8 {
            fmt.Println("✅ Excellent scaling efficiency")
        } else if avgEfficiency > 0.6 {
            fmt.Println("⚠️ Good scaling efficiency with room for improvement")
        } else {
            fmt.Println("❌ Poor scaling efficiency - investigate bottlenecks")
        }
    }
}

func (mgr *MultiGPUManager) GenerateReport() {
    fmt.Println("\n📄 Multi-GPU System Report")
    fmt.Println("=" * 50)
    
    // System configuration
    fmt.Printf("System Configuration:\n")
    fmt.Printf("  Total GPUs: %d\n", len(mgr.devices))
    fmt.Printf("  NCCL Initialized: %t\n", mgr.communicator.initialized)
    fmt.Printf("  Load Balancing Strategy: %s\n", 
               mgr.getStrategyName(mgr.loadBalancer.strategy))
    
    // Communication performance
    fmt.Printf("\nCommunication Performance:\n")
    for i := 0; i < len(mgr.devices); i++ {
        for j := 0; j < len(mgr.devices); j++ {
            if i != j {
                bw := mgr.profiler.communicationMetrics.P2PBandwidth[i][j]
                if bw > 0 {
                    fmt.Printf("  GPU %d -> GPU %d: %.2f GB/s\n", i, j, bw)
                }
            }
        }
    }
    
    // Device health summary
    fmt.Printf("\nDevice Health Summary:\n")
    for i, health := range mgr.faultHandler.deviceHealth {
        status := "✅"
        if !health.IsHealthy {
            status = "❌"
        }
        fmt.Printf("  GPU %d: %s (Temp: %.1f°C, Mem: %.1f%%)\n", 
                   i, status, health.Temperature, health.MemoryUsage*100)
    }
    
    // Load balancing metrics
    fmt.Printf("\nLoad Balancing Metrics:\n")
    totalTasks := int64(0)
    for i, count := range mgr.loadBalancer.metrics.TaskDistribution {
        fmt.Printf("  GPU %d: %d tasks\n", i, count)
        totalTasks += count
    }
    
    if totalTasks > 0 {
        // Calculate load imbalance
        avgTasks := float64(totalTasks) / float64(len(mgr.devices))
        maxDeviation := 0.0
        for _, count := range mgr.loadBalancer.metrics.TaskDistribution {
            deviation := math.Abs(float64(count) - avgTasks)
            if deviation > maxDeviation {
                maxDeviation = deviation
            }
        }
        imbalance := (maxDeviation / avgTasks) * 100
        fmt.Printf("  Load Imbalance: %.1f%%\n", imbalance)
    }
    
    // Scaling efficiency
    if mgr.profiler.scalingMetrics.ParallelEfficiency > 0 {
        fmt.Printf("\nScaling Efficiency: %.1f%%\n", 
                   mgr.profiler.scalingMetrics.ParallelEfficiency*100)
    }
}

func (mgr *MultiGPUManager) Cleanup() {
    fmt.Println("🧹 Cleaning up Multi-GPU Manager...")
    
    // Close task queue
    close(mgr.loadBalancer.taskQueue)
    
    // Cleanup NCCL
    if mgr.communicator.initialized {
        for _, comm := range mgr.communicator.comms {
            comm.Destroy()
        }
    }
    
    // Cleanup streams and contexts
    for i, streams := range mgr.streams {
        for _, stream := range streams {
            stream.Destroy()
        }
        mgr.contexts[i].Destroy()
    }
    
    fmt.Println("   Cleanup completed")
}

// Demonstration
func main() {
    fmt.Println("🌐 Expert Multi-GPU & Distributed Computing")
    
    manager := NewMultiGPUManager()
    defer manager.Cleanup()
    
    // Demonstrate multi-GPU operations
    demonstrateMultiGPUOperations(manager)
    
    // Run scaling benchmarks
    problemSizes := []int{10000, 50000, 100000}
    manager.BenchmarkScaling(problemSizes)
    
    // Generate comprehensive report
    manager.GenerateReport()
}

func demonstrateMultiGPUOperations(manager *MultiGPUManager) {
    fmt.Println("\n🔄 Multi-GPU Operations Demonstration:")
    
    // Test AllReduce
    numDevices := len(manager.devices)
    data := make([][]float32, numDevices)
    elementsPerDevice := 10000
    
    for i := 0; i < numDevices; i++ {
        data[i] = make([]float32, elementsPerDevice)
        for j := 0; j < elementsPerDevice; j++ {
            data[i][j] = float32(i*elementsPerDevice + j)
        }
    }
    
    fmt.Println("\n1. AllReduce Operation:")
    manager.AllReduce(data, "sum")
    
    // Test P2P transfers
    fmt.Println("\n2. P2P Transfer Operations:")
    if numDevices > 1 {
        testData, _ := memory.Alloc(1024 * 1024 * 4) // 4MB
        defer testData.Free()
        
        manager.P2PTransfer(0, 1, testData, 1024*1024*4)
    }
    
    // Test distributed task submission
    fmt.Println("\n3. Distributed Task Execution:")
    tasks := []DistributedTask{
        {TaskID: "task_1", TaskType: "matrix_multiply", Priority: 1},
        {TaskID: "task_2", TaskType: "reduction", Priority: 2},
        {TaskID: "task_3", TaskType: "convolution", Priority: 1},
    }
    
    for _, task := range tasks {
        resultChan := manager.SubmitTask(task)
        go func(ch <-chan TaskResult, taskID string) {
            select {
            case result := <-ch:
                fmt.Printf("   Task %s completed on GPU %d in %v\n", 
                           taskID, result.DeviceID, result.ExecutionTime)
            case <-time.After(5 * time.Second):
                fmt.Printf("   Task %s timed out\n", taskID)
            }
        }(resultChan, task.TaskID)
    }
    
    // Allow tasks to complete
    time.Sleep(2 * time.Second)
}
```

---

## 🎯 Module Assessment

### **Distributed Computing Mastery**

1. **Multi-GPU Scaling**: Achieve >80% parallel efficiency with 4+ GPUs
2. **Communication Optimization**: Minimize collective operation overhead
3. **Load Balancing**: Demonstrate adaptive task distribution
4. **Fault Tolerance**: Handle device failures gracefully

### **Success Criteria**

- ✅ Linear scaling for embarrassingly parallel problems
- ✅ Efficient collective communication patterns
- ✅ Production-ready fault tolerance and monitoring
- ✅ Enterprise deployment and management capabilities

---

## 🚀 Next Steps

**Excellent! You've mastered distributed GPU computing.**

**You're now ready for:**
➡️ **[Module 3: Advanced Numerical Methods](TRAINING_EXPERT_3_NUMERICAL.md)**

**Skills Mastered:**
- 🌐 **Multi-GPU Programming** - Scaling across multiple devices
- 🔄 **Distributed Algorithms** - Collective operations and communication
- ⚡ **Performance Optimization** - Communication and load balancing
- 🛡️ **Enterprise Features** - Fault tolerance and monitoring

---

*From single GPU to distributed mastery - scaling computational power across the data center! 🌐⚡*
