# ⚙️ Expert Module 6: Performance Engineering

**Goal:** Master systematic performance optimization, profiling, bottleneck identification, and achieving peak GPU utilization across diverse workloads

---

## 📚 Learning Objectives

By the end of this module, you will:
- 🔍 **Master GPU profiling tools** - Deep performance analysis and bottleneck identification
- ⚙️ **Optimize compute pipelines** - Systematic performance tuning methodology
- 📊 **Analyze performance metrics** - Understanding GPU utilization patterns
- 🎯 **Achieve peak performance** - Extract maximum throughput from GPU hardware
- 🏗️ **Design performance frameworks** - Build profiling and optimization systems

---

## 🧠 Theoretical Foundation

### GPU Performance Model

**Roofline Model:**
```
Performance = min(Compute Roof, Memory Roof)
where:
  Compute Roof = Peak FLOPS
  Memory Roof = Peak Bandwidth × Arithmetic Intensity
  Arithmetic Intensity = FLOPs / Bytes Transferred
```

**Performance Bottlenecks:**
```
Compute-Bound: GPU cores underutilized
├── Insufficient parallelism
├── Thread divergence
├── Low occupancy
└── Suboptimal instruction mix

Memory-Bound: Memory system bottleneck
├── Uncoalesced accesses
├── Cache misses
├── Bank conflicts
└── Limited bandwidth utilization

Latency-Bound: Synchronization overhead
├── Excessive synchronization
├── Small kernel launches
├── CPU-GPU synchronization
└── Memory allocation overhead
```

### Profiling Methodology

**Performance Analysis Pipeline:**
```
1. Baseline Measurement
   ↓
2. Bottleneck Identification
   ↓  
3. Targeted Optimization
   ↓
4. Performance Validation
   ↓
5. Iterative Refinement
```

---

## ⚙️ Chapter 1: Advanced GPU Profiling Framework

### Comprehensive Performance Profiler

Create `performance/gpu_profiler.go`:

```go
package main

import (
    "fmt"
    "time"
    "sync"
    "sort"
    "math"
    "context"
    "runtime"
    "github.com/stitch1968/gocuda"
    "github.com/stitch1968/gocuda/memory"
    "github.com/stitch1968/gocuda/profiler"
)

// Advanced GPU performance profiling framework
type GPUProfiler struct {
    ctx             *cuda.Context
    device          *cuda.Device
    streams         []*cuda.Stream
    
    // Profiling components
    metricsCollector *MetricsCollector
    kernelProfiler   *KernelProfiler
    memoryProfiler   *MemoryProfiler
    timingProfiler   *TimingProfiler
    
    // Configuration
    profilingEnabled bool
    samplingRate     time.Duration
    maxSamples       int
    
    // Results storage
    profileSessions  []*ProfileSession
    currentSession   *ProfileSession
    mutex           sync.RWMutex
}

type MetricsCollector struct {
    device            *cuda.Device
    samples           []MetricsSample
    collectionActive  bool
    collectionStop    chan bool
    samplingInterval  time.Duration
}

type MetricsSample struct {
    Timestamp         time.Time
    GPUUtilization    float64
    MemoryUtilization float64
    Temperature       float64
    PowerUsage        float64
    ClockRates        ClockRates
    MemoryInfo        MemoryInfo
    SMUtilization     []float64
}

type ClockRates struct {
    GraphicsClock int // MHz
    MemoryClock   int // MHz
    SMClock       int // MHz
}

type MemoryInfo struct {
    TotalMemory     int64
    UsedMemory      int64
    FreeMemory      int64
    BandwidthUtil   float64
}

type KernelProfiler struct {
    activeKernels    map[string]*KernelProfile
    completedKernels []*KernelProfile
    mutex           sync.RWMutex
}

type KernelProfile struct {
    Name                string
    LaunchTime         time.Time
    Duration           time.Duration
    GridSize           dim3
    BlockSize          dim3
    SharedMemorySize   int
    RegistersPerThread int
    
    // Performance metrics
    Occupancy          float64
    Instructions       int64
    MemoryTransactions int64
    BranchDivergence   float64
    WarpEfficiency     float64
    
    // Resource utilization
    ComputeUtilization float64
    MemoryUtilization  float64
    CacheHitRate       CacheStats
}

type dim3 struct {
    X, Y, Z int
}

type CacheStats struct {
    L1HitRate float64
    L2HitRate float64
    TextureHitRate float64
}

type MemoryProfiler struct {
    allocations       []*MemoryAllocation
    transfers         []*MemoryTransfer
    accessPatterns    map[string]*AccessPattern
    bandwidthHistory  []BandwidthSample
    mutex            sync.RWMutex
}

type MemoryAllocation struct {
    Address    uintptr
    Size       int64
    Timestamp  time.Time
    StackTrace []uintptr
    Type       MemoryType
}

type MemoryType int

const (
    DeviceMemory MemoryType = iota
    PinnedMemory
    ManagedMemory
    TextureMemory
)

type MemoryTransfer struct {
    Source      MemoryLocation
    Destination MemoryLocation
    Size        int64
    Duration    time.Duration
    Bandwidth   float64
    Timestamp   time.Time
}

type MemoryLocation struct {
    Type   LocationType
    Device int
}

type LocationType int

const (
    Host LocationType = iota
    Device
    Peer
)

type AccessPattern struct {
    KernelName       string
    AccessType       AccessType
    CoalescingRate   float64
    StridedAccesses  int64
    RandomAccesses   int64
    CacheUtilization CacheStats
}

type AccessType int

const (
    Sequential AccessType = iota
    Strided
    Random
    Broadcast
)

type BandwidthSample struct {
    Timestamp       time.Time
    Direction       TransferDirection
    Bandwidth       float64
    Utilization     float64
}

type TransferDirection int

const (
    HostToDevice TransferDirection = iota
    DeviceToHost
    DeviceToDevice
    PeerToPeer
)

type TimingProfiler struct {
    events       []*TimingEvent
    markers      map[string]time.Time
    ranges       []*TimingRange
    mutex       sync.RWMutex
}

type TimingEvent struct {
    Name      string
    Timestamp time.Time
    Duration  time.Duration
    Category  EventCategory
    Details   map[string]interface{}
}

type EventCategory int

const (
    KernelExecution EventCategory = iota
    MemoryTransfer
    Synchronization
    HostFunction
)

type TimingRange struct {
    Name      string
    Start     time.Time
    End       time.Time
    Children  []*TimingRange
}

type ProfileSession struct {
    ID              string
    StartTime       time.Time
    EndTime         time.Time
    DeviceInfo      DeviceInfo
    Metrics         []MetricsSample
    Kernels         []*KernelProfile
    MemoryOps       []*MemoryTransfer
    TimingData      []*TimingEvent
    Summary         *PerformanceSummary
}

type DeviceInfo struct {
    Name               string
    ComputeCapability  [2]int
    TotalMemory        int64
    MultiprocessorCount int
    MaxThreadsPerBlock  int
    WarpSize           int
    ClockRates         ClockRates
}

type PerformanceSummary struct {
    TotalTime             time.Duration
    KernelTime            time.Duration
    MemoryTransferTime    time.Duration
    OverheadTime          time.Duration
    
    AverageGPUUtil        float64
    PeakGPUUtil          float64
    AverageMemoryUtil     float64
    PeakMemoryUtil       float64
    
    TotalKernelLaunches   int
    TotalMemoryTransfers  int
    TotalBytesTransferred int64
    
    Bottlenecks          []string
    Recommendations      []string
}

func NewGPUProfiler() *GPUProfiler {
    cuda.Initialize()
    
    device := cuda.GetDevice(0)
    ctx := cuda.GetDefaultContext()
    
    profiler := &GPUProfiler{
        ctx:              ctx,
        device:           device,
        streams:          make([]*cuda.Stream, 4),
        profilingEnabled: true,
        samplingRate:     10 * time.Millisecond,
        maxSamples:       10000,
        profileSessions:  make([]*ProfileSession, 0),
    }
    
    // Initialize streams
    for i := range profiler.streams {
        stream, _ := ctx.CreateStream()
        profiler.streams[i] = stream
    }
    
    // Initialize profiling components
    profiler.metricsCollector = NewMetricsCollector(device, profiler.samplingRate)
    profiler.kernelProfiler = NewKernelProfiler()
    profiler.memoryProfiler = NewMemoryProfiler()
    profiler.timingProfiler = NewTimingProfiler()
    
    fmt.Printf("⚙️ GPU Profiler initialized for %s\n", device.GetName())
    return profiler
}

func NewMetricsCollector(device *cuda.Device, interval time.Duration) *MetricsCollector {
    return &MetricsCollector{
        device:           device,
        samples:          make([]MetricsSample, 0),
        collectionStop:   make(chan bool),
        samplingInterval: interval,
    }
}

func (mc *MetricsCollector) StartCollection(ctx context.Context) {
    mc.collectionActive = true
    
    go func() {
        ticker := time.NewTicker(mc.samplingInterval)
        defer ticker.Stop()
        
        for {
            select {
            case <-ctx.Done():
                return
            case <-mc.collectionStop:
                return
            case <-ticker.C:
                sample := mc.collectSample()
                mc.samples = append(mc.samples, sample)
                
                // Limit sample history
                if len(mc.samples) > 10000 {
                    mc.samples = mc.samples[1000:]
                }
            }
        }
    }()
}

func (mc *MetricsCollector) collectSample() MetricsSample {
    // Collect comprehensive GPU metrics
    sample := MetricsSample{
        Timestamp: time.Now(),
    }
    
    // GPU utilization
    sample.GPUUtilization = mc.device.GetUtilization()
    
    // Memory information
    memInfo := mc.device.GetMemoryInfo()
    sample.MemoryInfo = MemoryInfo{
        TotalMemory:   memInfo.Total,
        UsedMemory:    memInfo.Used,
        FreeMemory:    memInfo.Free,
        BandwidthUtil: mc.device.GetMemoryBandwidthUtilization(),
    }
    sample.MemoryUtilization = float64(memInfo.Used) / float64(memInfo.Total) * 100
    
    // Temperature and power
    sample.Temperature = float64(mc.device.GetTemperature())
    sample.PowerUsage = float64(mc.device.GetPowerUsage())
    
    // Clock rates
    sample.ClockRates = ClockRates{
        GraphicsClock: mc.device.GetGraphicsClockRate() / 1000, // Convert to MHz
        MemoryClock:   mc.device.GetMemoryClockRate() / 1000,
        SMClock:       mc.device.GetSMClockRate() / 1000,
    }
    
    // SM utilization (per multiprocessor)
    props := mc.device.GetProperties()
    sample.SMUtilization = make([]float64, props.MultiprocessorCount)
    for i := 0; i < props.MultiprocessorCount; i++ {
        sample.SMUtilization[i] = mc.device.GetSMUtilization(i)
    }
    
    return sample
}

func (mc *MetricsCollector) StopCollection() {
    if mc.collectionActive {
        mc.collectionStop <- true
        mc.collectionActive = false
    }
}

func NewKernelProfiler() *KernelProfiler {
    return &KernelProfiler{
        activeKernels:    make(map[string]*KernelProfile),
        completedKernels: make([]*KernelProfile, 0),
    }
}

func (kp *KernelProfiler) StartKernelProfiling(name string, gridSize, blockSize dim3, sharedMem int) string {
    kp.mutex.Lock()
    defer kp.mutex.Unlock()
    
    kernelID := fmt.Sprintf("%s_%d", name, time.Now().UnixNano())
    
    profile := &KernelProfile{
        Name:             name,
        LaunchTime:       time.Now(),
        GridSize:         gridSize,
        BlockSize:        blockSize,
        SharedMemorySize: sharedMem,
    }
    
    kp.activeKernels[kernelID] = profile
    return kernelID
}

func (kp *KernelProfiler) EndKernelProfiling(kernelID string, metrics KernelMetrics) {
    kp.mutex.Lock()
    defer kp.mutex.Unlock()
    
    if profile, exists := kp.activeKernels[kernelID]; exists {
        profile.Duration = time.Since(profile.LaunchTime)
        
        // Update performance metrics
        profile.Occupancy = metrics.Occupancy
        profile.Instructions = metrics.Instructions
        profile.MemoryTransactions = metrics.MemoryTransactions
        profile.BranchDivergence = metrics.BranchDivergence
        profile.WarpEfficiency = metrics.WarpEfficiency
        profile.ComputeUtilization = metrics.ComputeUtilization
        profile.MemoryUtilization = metrics.MemoryUtilization
        profile.CacheHitRate = metrics.CacheHitRate
        
        kp.completedKernels = append(kp.completedKernels, profile)
        delete(kp.activeKernels, kernelID)
    }
}

type KernelMetrics struct {
    Occupancy          float64
    Instructions       int64
    MemoryTransactions int64
    BranchDivergence   float64
    WarpEfficiency     float64
    ComputeUtilization float64
    MemoryUtilization  float64
    CacheHitRate       CacheStats
}

func NewMemoryProfiler() *MemoryProfiler {
    return &MemoryProfiler{
        allocations:      make([]*MemoryAllocation, 0),
        transfers:        make([]*MemoryTransfer, 0),
        accessPatterns:   make(map[string]*AccessPattern),
        bandwidthHistory: make([]BandwidthSample, 0),
    }
}

func (mp *MemoryProfiler) RecordAllocation(addr uintptr, size int64, memType MemoryType) {
    mp.mutex.Lock()
    defer mp.mutex.Unlock()
    
    allocation := &MemoryAllocation{
        Address:    addr,
        Size:       size,
        Timestamp:  time.Now(),
        StackTrace: make([]uintptr, 10),
        Type:       memType,
    }
    
    // Capture stack trace
    n := runtime.Callers(2, allocation.StackTrace)
    allocation.StackTrace = allocation.StackTrace[:n]
    
    mp.allocations = append(mp.allocations, allocation)
}

func (mp *MemoryProfiler) RecordTransfer(src, dst MemoryLocation, size int64, duration time.Duration) {
    mp.mutex.Lock()
    defer mp.mutex.Unlock()
    
    bandwidth := float64(size) / duration.Seconds() / (1024 * 1024 * 1024) // GB/s
    
    transfer := &MemoryTransfer{
        Source:      src,
        Destination: dst,
        Size:        size,
        Duration:    duration,
        Bandwidth:   bandwidth,
        Timestamp:   time.Now(),
    }
    
    mp.transfers = append(mp.transfers, transfer)
    
    // Record bandwidth sample
    direction := HostToDevice
    if src.Type == Device && dst.Type == Host {
        direction = DeviceToHost
    } else if src.Type == Device && dst.Type == Device {
        if src.Device != dst.Device {
            direction = PeerToPeer
        } else {
            direction = DeviceToDevice
        }
    }
    
    sample := BandwidthSample{
        Timestamp:   time.Now(),
        Direction:   direction,
        Bandwidth:   bandwidth,
        Utilization: mp.calculateBandwidthUtilization(bandwidth, direction),
    }
    
    mp.bandwidthHistory = append(mp.bandwidthHistory, sample)
}

func (mp *MemoryProfiler) calculateBandwidthUtilization(actualBW float64, direction TransferDirection) float64 {
    // Theoretical peak bandwidth (device dependent)
    var theoreticalBW float64
    
    switch direction {
    case HostToDevice, DeviceToHost:
        theoreticalBW = 16.0 // PCIe 4.0 x16 theoretical
    case DeviceToDevice:
        theoreticalBW = 900.0 // High-end GPU memory bandwidth
    case PeerToPeer:
        theoreticalBW = 300.0 // NVLink bandwidth
    }
    
    return (actualBW / theoreticalBW) * 100
}

func NewTimingProfiler() *TimingProfiler {
    return &TimingProfiler{
        events:  make([]*TimingEvent, 0),
        markers: make(map[string]time.Time),
        ranges:  make([]*TimingRange, 0),
    }
}

func (tp *TimingProfiler) RecordEvent(name string, duration time.Duration, category EventCategory, details map[string]interface{}) {
    tp.mutex.Lock()
    defer tp.mutex.Unlock()
    
    event := &TimingEvent{
        Name:      name,
        Timestamp: time.Now(),
        Duration:  duration,
        Category:  category,
        Details:   details,
    }
    
    tp.events = append(tp.events, event)
}

func (tp *TimingProfiler) StartRange(name string) {
    tp.mutex.Lock()
    defer tp.mutex.Unlock()
    
    tp.markers[name] = time.Now()
}

func (tp *TimingProfiler) EndRange(name string) {
    tp.mutex.Lock()
    defer tp.mutex.Unlock()
    
    if startTime, exists := tp.markers[name]; exists {
        rangeData := &TimingRange{
            Name:  name,
            Start: startTime,
            End:   time.Now(),
        }
        
        tp.ranges = append(tp.ranges, rangeData)
        delete(tp.markers, name)
    }
}

// Main profiling session management
func (gp *GPUProfiler) StartProfilingSession(sessionID string) {
    gp.mutex.Lock()
    defer gp.mutex.Unlock()
    
    device := gp.device
    props := device.GetProperties()
    
    session := &ProfileSession{
        ID:        sessionID,
        StartTime: time.Now(),
        DeviceInfo: DeviceInfo{
            Name:               props.Name,
            ComputeCapability:  [2]int{props.Major, props.Minor},
            TotalMemory:        props.TotalGlobalMem,
            MultiprocessorCount: props.MultiprocessorCount,
            MaxThreadsPerBlock:  props.MaxThreadsPerBlock,
            WarpSize:           props.WarpSize,
            ClockRates: ClockRates{
                GraphicsClock: props.ClockRate / 1000,
                MemoryClock:   props.MemoryClockRate / 1000,
            },
        },
        Metrics:    make([]MetricsSample, 0),
        Kernels:    make([]*KernelProfile, 0),
        MemoryOps:  make([]*MemoryTransfer, 0),
        TimingData: make([]*TimingEvent, 0),
    }
    
    gp.currentSession = session
    
    // Start metrics collection
    ctx := context.Background()
    gp.metricsCollector.StartCollection(ctx)
    
    fmt.Printf("🔍 Profiling session '%s' started\n", sessionID)
}

func (gp *GPUProfiler) EndProfilingSession() *ProfileSession {
    gp.mutex.Lock()
    defer gp.mutex.Unlock()
    
    if gp.currentSession == nil {
        return nil
    }
    
    // Stop metrics collection
    gp.metricsCollector.StopCollection()
    
    session := gp.currentSession
    session.EndTime = time.Now()
    
    // Collect final data
    session.Metrics = gp.metricsCollector.samples
    session.Kernels = gp.kernelProfiler.completedKernels
    session.MemoryOps = gp.memoryProfiler.transfers
    session.TimingData = gp.timingProfiler.events
    
    // Analyze performance
    session.Summary = gp.analyzePerformance(session)
    
    gp.profileSessions = append(gp.profileSessions, session)
    gp.currentSession = nil
    
    fmt.Printf("📊 Profiling session completed: %v duration\n", 
               session.EndTime.Sub(session.StartTime))
    
    return session
}

func (gp *GPUProfiler) analyzePerformance(session *ProfileSession) *PerformanceSummary {
    summary := &PerformanceSummary{
        TotalTime: session.EndTime.Sub(session.StartTime),
    }
    
    // Analyze kernel execution
    var totalKernelTime time.Duration
    for _, kernel := range session.Kernels {
        totalKernelTime += kernel.Duration
    }
    summary.KernelTime = totalKernelTime
    summary.TotalKernelLaunches = len(session.Kernels)
    
    // Analyze memory transfers
    var totalTransferTime time.Duration
    var totalBytes int64
    for _, transfer := range session.MemoryOps {
        totalTransferTime += transfer.Duration
        totalBytes += transfer.Size
    }
    summary.MemoryTransferTime = totalTransferTime
    summary.TotalMemoryTransfers = len(session.MemoryOps)
    summary.TotalBytesTransferred = totalBytes
    
    // Calculate overhead
    accountedTime := totalKernelTime + totalTransferTime
    if accountedTime < summary.TotalTime {
        summary.OverheadTime = summary.TotalTime - accountedTime
    }
    
    // Analyze GPU utilization
    if len(session.Metrics) > 0 {
        var totalUtil, peakUtil float64
        var totalMemUtil, peakMemUtil float64
        
        for _, sample := range session.Metrics {
            totalUtil += sample.GPUUtilization
            totalMemUtil += sample.MemoryUtilization
            
            if sample.GPUUtilization > peakUtil {
                peakUtil = sample.GPUUtilization
            }
            if sample.MemoryUtilization > peakMemUtil {
                peakMemUtil = sample.MemoryUtilization
            }
        }
        
        summary.AverageGPUUtil = totalUtil / float64(len(session.Metrics))
        summary.PeakGPUUtil = peakUtil
        summary.AverageMemoryUtil = totalMemUtil / float64(len(session.Metrics))
        summary.PeakMemoryUtil = peakMemUtil
    }
    
    // Identify bottlenecks
    summary.Bottlenecks = gp.identifyBottlenecks(session, summary)
    summary.Recommendations = gp.generateRecommendations(session, summary)
    
    return summary
}

func (gp *GPUProfiler) identifyBottlenecks(session *ProfileSession, summary *PerformanceSummary) []string {
    bottlenecks := make([]string, 0)
    
    // Check GPU utilization
    if summary.AverageGPUUtil < 50 {
        bottlenecks = append(bottlenecks, "Low GPU utilization - insufficient parallelism")
    }
    
    // Check memory utilization
    if summary.AverageMemoryUtil > 90 {
        bottlenecks = append(bottlenecks, "High memory pressure - consider memory optimization")
    }
    
    // Check kernel occupancy
    lowOccupancyKernels := 0
    for _, kernel := range session.Kernels {
        if kernel.Occupancy < 0.5 {
            lowOccupancyKernels++
        }
    }
    if lowOccupancyKernels > len(session.Kernels)/2 {
        bottlenecks = append(bottlenecks, "Low occupancy in many kernels - optimize launch configuration")
    }
    
    // Check memory bandwidth utilization
    for _, transfer := range session.MemoryOps {
        if transfer.Bandwidth < 10.0 { // Less than 10 GB/s suggests issues
            bottlenecks = append(bottlenecks, "Poor memory transfer performance")
            break
        }
    }
    
    // Check for excessive overhead
    if summary.OverheadTime > summary.TotalTime/4 {
        bottlenecks = append(bottlenecks, "High overhead - too many small operations")
    }
    
    return bottlenecks
}

func (gp *GPUProfiler) generateRecommendations(session *ProfileSession, summary *PerformanceSummary) []string {
    recommendations := make([]string, 0)
    
    // Based on bottlenecks, suggest optimizations
    for _, bottleneck := range summary.Bottlenecks {
        switch {
        case contains(bottleneck, "Low GPU utilization"):
            recommendations = append(recommendations, 
                "Increase problem size or reduce number of kernel launches")
                
        case contains(bottleneck, "High memory pressure"):
            recommendations = append(recommendations,
                "Optimize memory usage patterns or increase batch processing")
                
        case contains(bottleneck, "Low occupancy"):
            recommendations = append(recommendations,
                "Adjust block size, reduce register usage, or optimize shared memory")
                
        case contains(bottleneck, "Poor memory transfer"):
            recommendations = append(recommendations,
                "Use pinned memory, larger transfers, or asynchronous operations")
                
        case contains(bottleneck, "High overhead"):
            recommendations = append(recommendations,
                "Batch operations, use streams, or reduce CPU-GPU synchronization")
        }
    }
    
    // Performance-specific recommendations
    if summary.AverageGPUUtil > 80 && summary.AverageMemoryUtil > 80 {
        recommendations = append(recommendations,
            "Good utilization! Consider algorithmic improvements or multi-GPU scaling")
    }
    
    return recommendations
}

func contains(str, substr string) bool {
    return len(str) >= len(substr) && str[:len(substr)] == substr ||
           len(str) > len(substr) && (str[len(str)-len(substr):] == substr ||
           indexOf(str, substr) >= 0)
}

func indexOf(str, substr string) int {
    for i := 0; i <= len(str)-len(substr); i++ {
        if str[i:i+len(substr)] == substr {
            return i
        }
    }
    return -1
}

func (gp *GPUProfiler) GenerateReport(session *ProfileSession) {
    fmt.Printf("\n📊 GPU Performance Report: %s\n", session.ID)
    fmt.Printf("="*60 + "\n")
    
    // Device information
    fmt.Printf("Device: %s (Compute %d.%d)\n", 
               session.DeviceInfo.Name,
               session.DeviceInfo.ComputeCapability[0],
               session.DeviceInfo.ComputeCapability[1])
    fmt.Printf("Session Duration: %v\n", session.Summary.TotalTime)
    
    // Time breakdown
    fmt.Printf("\nTime Breakdown:\n")
    fmt.Printf("  Kernel Execution: %v (%.1f%%)\n", 
               session.Summary.KernelTime,
               float64(session.Summary.KernelTime)/float64(session.Summary.TotalTime)*100)
    fmt.Printf("  Memory Transfers: %v (%.1f%%)\n",
               session.Summary.MemoryTransferTime,
               float64(session.Summary.MemoryTransferTime)/float64(session.Summary.TotalTime)*100)
    fmt.Printf("  Overhead: %v (%.1f%%)\n",
               session.Summary.OverheadTime,
               float64(session.Summary.OverheadTime)/float64(session.Summary.TotalTime)*100)
    
    // Utilization metrics
    fmt.Printf("\nUtilization Metrics:\n")
    fmt.Printf("  GPU Utilization: Avg=%.1f%%, Peak=%.1f%%\n",
               session.Summary.AverageGPUUtil, session.Summary.PeakGPUUtil)
    fmt.Printf("  Memory Utilization: Avg=%.1f%%, Peak=%.1f%%\n",
               session.Summary.AverageMemoryUtil, session.Summary.PeakMemoryUtil)
    
    // Kernel analysis
    if len(session.Kernels) > 0 {
        fmt.Printf("\nKernel Analysis:\n")
        fmt.Printf("  Total Kernels: %d\n", len(session.Kernels))
        
        // Find hotspot kernels
        kernelTimes := make(map[string]time.Duration)
        for _, kernel := range session.Kernels {
            kernelTimes[kernel.Name] += kernel.Duration
        }
        
        // Sort by duration
        type kernelTime struct {
            name     string
            duration time.Duration
        }
        
        sorted := make([]kernelTime, 0, len(kernelTimes))
        for name, duration := range kernelTimes {
            sorted = append(sorted, kernelTime{name, duration})
        }
        
        sort.Slice(sorted, func(i, j int) bool {
            return sorted[i].duration > sorted[j].duration
        })
        
        fmt.Printf("  Top Hotspot Kernels:\n")
        for i, kt := range sorted[:min(5, len(sorted))] {
            percentage := float64(kt.duration) / float64(session.Summary.KernelTime) * 100
            fmt.Printf("    %d. %s: %v (%.1f%%)\n", i+1, kt.name, kt.duration, percentage)
        }
    }
    
    // Memory analysis
    if len(session.MemoryOps) > 0 {
        fmt.Printf("\nMemory Analysis:\n")
        fmt.Printf("  Total Transfers: %d\n", len(session.MemoryOps))
        fmt.Printf("  Total Data: %.2f GB\n", 
                   float64(session.Summary.TotalBytesTransferred)/(1024*1024*1024))
        
        // Average bandwidth by direction
        bandwidthByDirection := make(map[TransferDirection][]float64)
        for _, transfer := range session.MemoryOps {
            direction := gp.getTransferDirection(transfer)
            bandwidthByDirection[direction] = append(bandwidthByDirection[direction], transfer.Bandwidth)
        }
        
        for direction, bandwidths := range bandwidthByDirection {
            if len(bandwidths) > 0 {
                var sum float64
                for _, bw := range bandwidths {
                    sum += bw
                }
                avg := sum / float64(len(bandwidths))
                fmt.Printf("  %s: %.2f GB/s average\n", gp.directionString(direction), avg)
            }
        }
    }
    
    // Bottlenecks and recommendations
    if len(session.Summary.Bottlenecks) > 0 {
        fmt.Printf("\n⚠️ Identified Bottlenecks:\n")
        for i, bottleneck := range session.Summary.Bottlenecks {
            fmt.Printf("  %d. %s\n", i+1, bottleneck)
        }
    }
    
    if len(session.Summary.Recommendations) > 0 {
        fmt.Printf("\n💡 Optimization Recommendations:\n")
        for i, rec := range session.Summary.Recommendations {
            fmt.Printf("  %d. %s\n", i+1, rec)
        }
    }
    
    fmt.Printf("\n" + "="*60 + "\n")
}

func (gp *GPUProfiler) getTransferDirection(transfer *MemoryTransfer) TransferDirection {
    if transfer.Source.Type == Host && transfer.Destination.Type == Device {
        return HostToDevice
    } else if transfer.Source.Type == Device && transfer.Destination.Type == Host {
        return DeviceToHost
    } else if transfer.Source.Type == Device && transfer.Destination.Type == Device {
        if transfer.Source.Device != transfer.Destination.Device {
            return PeerToPeer
        } else {
            return DeviceToDevice
        }
    }
    return HostToDevice // Default
}

func (gp *GPUProfiler) directionString(direction TransferDirection) string {
    switch direction {
    case HostToDevice:
        return "Host→Device"
    case DeviceToHost:
        return "Device→Host"
    case DeviceToDevice:
        return "Device→Device"
    case PeerToPeer:
        return "Peer→Peer"
    default:
        return "Unknown"
    }
}

func (gp *GPUProfiler) Cleanup() {
    for _, stream := range gp.streams {
        stream.Destroy()
    }
    
    if gp.currentSession != nil {
        gp.EndProfilingSession()
    }
    
    fmt.Printf("   GPU Profiler cleaned up\n")
}

func min(a, b int) int {
    if a < b {
        return a
    }
    return b
}

// Demonstration
func main() {
    fmt.Println("⚙️ Expert Performance Engineering: GPU Profiling")
    
    profiler := NewGPUProfiler()
    defer profiler.Cleanup()
    
    // Start profiling session
    profiler.StartProfilingSession("performance_demo")
    
    // Run some example workloads
    runProfilingDemo(profiler)
    
    // End session and generate report
    session := profiler.EndProfilingSession()
    if session != nil {
        profiler.GenerateReport(session)
    }
    
    // Demonstrate optimization workflow
    demonstrateOptimizationWorkflow()
}

func runProfilingDemo(profiler *GPUProfiler) {
    fmt.Println("\n🔬 Running profiling demonstration...")
    
    // Simulate various GPU operations
    demonstrateKernelProfiling(profiler)
    demonstrateMemoryProfiling(profiler)
    demonstrateTimingProfiling(profiler)
}

func demonstrateKernelProfiling(profiler *GPUProfiler) {
    fmt.Println("  🔄 Simulating kernel profiling...")
    
    // Simulate different kernel types
    kernels := []struct {
        name      string
        duration  time.Duration
        occupancy float64
        efficiency float64
    }{
        {"vector_add", 100 * time.Microsecond, 0.8, 0.9},
        {"matrix_mult", 2 * time.Millisecond, 0.6, 0.7},
        {"reduction", 50 * time.Microsecond, 0.9, 0.8},
        {"transpose", 200 * time.Microsecond, 0.4, 0.5},
    }
    
    for _, kernel := range kernels {
        // Start profiling
        kernelID := profiler.kernelProfiler.StartKernelProfiling(
            kernel.name,
            dim3{256, 1, 1}, // Grid size
            dim3{128, 1, 1}, // Block size
            1024,            // Shared memory
        )
        
        // Simulate kernel execution
        time.Sleep(kernel.duration)
        
        // End profiling with metrics
        metrics := KernelMetrics{
            Occupancy:          kernel.occupancy,
            Instructions:       int64(kernel.duration.Nanoseconds() / 10), // Simulated
            MemoryTransactions: int64(kernel.duration.Nanoseconds() / 50), // Simulated
            BranchDivergence:   0.1,
            WarpEfficiency:     kernel.efficiency,
            ComputeUtilization: kernel.occupancy * 100,
            MemoryUtilization:  50.0,
            CacheHitRate: CacheStats{
                L1HitRate:      0.9,
                L2HitRate:      0.7,
                TextureHitRate: 0.8,
            },
        }
        
        profiler.kernelProfiler.EndKernelProfiling(kernelID, metrics)
    }
}

func demonstrateMemoryProfiling(profiler *GPUProfiler) {
    fmt.Println("  💾 Simulating memory profiling...")
    
    // Simulate memory allocations
    allocSizes := []int64{1024 * 1024, 4 * 1024 * 1024, 16 * 1024 * 1024}
    
    for i, size := range allocSizes {
        addr := uintptr(0x1000000 + i*int(size))
        profiler.memoryProfiler.RecordAllocation(addr, size, DeviceMemory)
    }
    
    // Simulate memory transfers
    transfers := []struct {
        size      int64
        duration  time.Duration
        direction TransferDirection
    }{
        {1024 * 1024, 500 * time.Microsecond, HostToDevice},
        {4 * 1024 * 1024, 1800 * time.Microsecond, HostToDevice},
        {2 * 1024 * 1024, 900 * time.Microsecond, DeviceToHost},
        {8 * 1024 * 1024, 100 * time.Microsecond, DeviceToDevice},
    }
    
    for _, transfer := range transfers {
        var src, dst MemoryLocation
        
        switch transfer.direction {
        case HostToDevice:
            src = MemoryLocation{Host, 0}
            dst = MemoryLocation{Device, 0}
        case DeviceToHost:
            src = MemoryLocation{Device, 0}
            dst = MemoryLocation{Host, 0}
        case DeviceToDevice:
            src = MemoryLocation{Device, 0}
            dst = MemoryLocation{Device, 0}
        }
        
        profiler.memoryProfiler.RecordTransfer(src, dst, transfer.size, transfer.duration)
        time.Sleep(transfer.duration) // Simulate actual transfer time
    }
}

func demonstrateTimingProfiling(profiler *GPUProfiler) {
    fmt.Println("  ⏱️ Simulating timing profiling...")
    
    // Simulate various timing events
    events := []struct {
        name     string
        duration time.Duration
        category EventCategory
    }{
        {"data_preparation", 2 * time.Millisecond, HostFunction},
        {"kernel_launch", 20 * time.Microsecond, Synchronization},
        {"result_processing", 1 * time.Millisecond, HostFunction},
    }
    
    for _, event := range events {
        details := map[string]interface{}{
            "simulated": true,
            "category":  event.category,
        }
        
        profiler.timingProfiler.RecordEvent(event.name, event.duration, event.category, details)
        time.Sleep(event.duration) // Simulate actual operation time
    }
    
    // Simulate timing ranges
    profiler.timingProfiler.StartRange("full_computation")
    time.Sleep(5 * time.Millisecond)
    profiler.timingProfiler.EndRange("full_computation")
}

func demonstrateOptimizationWorkflow() {
    fmt.Println("\n🎯 Optimization Workflow Demonstration:")
    
    steps := []string{
        "1. Baseline Performance Measurement",
        "2. Profiling and Bottleneck Identification", 
        "3. Targeted Optimization Implementation",
        "4. Performance Validation and Comparison",
        "5. Iterative Refinement",
    }
    
    for _, step := range steps {
        fmt.Printf("   %s\n", step)
        time.Sleep(200 * time.Millisecond)
    }
    
    fmt.Println("\n💡 Key Optimization Strategies:")
    strategies := []string{
        "• Maximize GPU occupancy through optimal launch configurations",
        "• Optimize memory access patterns for coalescing", 
        "• Minimize CPU-GPU synchronization overhead",
        "• Use asynchronous operations and multi-stream execution",
        "• Profile-guided optimization with hardware counters",
        "• Algorithm-specific optimizations for compute vs memory bound workloads",
    }
    
    for _, strategy := range strategies {
        fmt.Printf("   %s\n", strategy)
    }
}
```

---

## 🎯 Module Assessment

### **Performance Engineering Mastery**

1. **Profiling Expertise**: Master GPU profiling tools and methodologies
2. **Bottleneck Identification**: Systematically identify and classify performance issues
3. **Optimization Implementation**: Apply targeted optimizations based on profiling data
4. **Performance Validation**: Demonstrate measurable improvements

### **Success Criteria**

- ✅ Comprehensive profiling framework with multiple metrics
- ✅ Accurate bottleneck identification and root cause analysis
- ✅ Systematic optimization workflow with measurable results
- ✅ Production-ready performance monitoring and reporting

---

## 🚀 Next Steps

**Exceptional! You've mastered systematic performance engineering.**

**You're now ready for:**
➡️ **[Module 7: Integration & Architecture](TRAINING_EXPERT_7_INTEGRATION.md)**

**Skills Mastered:**
- ⚙️ **Advanced GPU Profiling** - Comprehensive performance analysis
- 🔍 **Bottleneck Identification** - Systematic root cause analysis  
- 📊 **Performance Optimization** - Data-driven improvement strategies
- 🎯 **Engineering Methodology** - Reproducible optimization workflows

---

*From guesswork to precision - engineering peak GPU performance through systematic analysis! ⚙️📊*
