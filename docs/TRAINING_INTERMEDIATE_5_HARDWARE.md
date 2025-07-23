# ⚡ Module 5: Hardware Optimization & Architecture

**Goal:** Master GPU architecture optimization, warp-level programming, and hardware-specific performance tuning for maximum efficiency

---

## 📚 Learning Objectives

By the end of this module, you will:
- 🔧 **Master GPU architecture** - SM, warps, memory hierarchy, and compute capability
- ⚡ **Optimize for hardware** - warp divergence, bank conflicts, and occupancy
- 🎯 **Implement warp primitives** - shuffle, vote, cooperative groups
- 📊 **Profile hardware metrics** - instruction throughput, memory bandwidth, occupancy
- 🚀 **Architecture-specific tuning** - Ampere, Turing, Pascal optimization strategies

---

## 🧠 Theoretical Foundation

### GPU Architecture Hierarchy

**CUDA Architecture Stack:**
```
GPU Device
├── Streaming Multiprocessors (SMs)
│   ├── CUDA Cores (32-128 per SM)
│   ├── Warp Schedulers (2-4 per SM)
│   ├── Shared Memory (48-164KB)
│   ├── L1 Cache (32-128KB)
│   └── Register File (256KB-1MB)
├── L2 Cache (1-40MB)
├── Global Memory (GDDR/HBM)
└── Host Interface (PCIe)
```

### Warp Execution Model

**Warp Characteristics:**
- **Size**: 32 threads (SIMT execution)
- **Scheduling**: Hardware thread scheduler
- **Divergence**: Performance impact of branch divergence
- **Occupancy**: Active warps per SM

**Memory Hierarchy Performance:**
```
Register:     ~1 cycle
Shared Mem:   ~1-32 cycles  
L1 Cache:     ~32 cycles
L2 Cache:     ~200 cycles
Global Mem:   ~400-800 cycles
```

### Architecture Evolution

**Compute Capabilities:**
- **Pascal (6.x)**: Unified memory, NVLink
- **Volta (7.0)**: Tensor cores, independent thread scheduling
- **Turing (7.5)**: RT cores, variable precision
- **Ampere (8.x)**: 3rd gen Tensor cores, structural sparsity

---

## 🏗️ Chapter 1: Hardware-Aware Programming

### GPU Architecture Analyzer

Create `hardware/architecture_optimizer.go`:

```go
package main

import (
    "fmt"
    "math"
    "time"
    "github.com/stitch1968/gocuda"
    "github.com/stitch1968/gocuda/memory"
    "github.com/stitch1968/gocuda/profiler"
)

// Hardware architecture analysis and optimization
type ArchitectureOptimizer struct {
    ctx           *cuda.Context
    device        *cuda.Device
    deviceProps   *cuda.DeviceProperties
    profiler      *HardwareProfiler
    metrics       *ArchitectureMetrics
}

type HardwareProfiler struct {
    occupancyData    map[string]*OccupancyInfo
    memoryMetrics    map[string]*MemoryMetrics
    warpMetrics      map[string]*WarpMetrics
    throughputData   map[string]*ThroughputInfo
}

type ArchitectureMetrics struct {
    ComputeCapability  string
    SMCount           int
    CoresPerSM        int
    MaxThreadsPerSM   int
    MaxWarpsPerSM     int
    SharedMemPerSM    int64
    RegistersPerSM    int
    L2CacheSize       int64
    MemoryBandwidth   float64
    BaseClockRate     int
    MemoryClockRate   int
}

type OccupancyInfo struct {
    TheoreticalOccupancy float64
    AchievedOccupancy    float64
    LimitingFactor       string
    RegistersPer Thread  int
    SharedMemPerBlock    int64
    BlockSize            int
    GridSize             int
}

type MemoryMetrics struct {
    GlobalMemThroughput  float64
    SharedMemBandwidth   float64
    BankConflicts        int
    CacheHitRate         float64
    CoalescingEfficiency float64
}

type WarpMetrics struct {
    DivergenceRatio     float64
    ActiveWarps         int
    Instructions        int64
    Stalls             int64
    MemoryStalls       int64
    BranchEfficiency   float64
}

type ThroughputInfo struct {
    Instructions     int64
    ExecutionTime    time.Duration
    GFLOPS          float64
    MemoryBandwidth float64
    Efficiency      float64
}

func NewArchitectureOptimizer() *ArchitectureOptimizer {
    cuda.Initialize()
    ctx := cuda.GetDefaultContext()
    device, _ := cuda.GetDevice(0)
    props := device.GetProperties()
    
    optimizer := &ArchitectureOptimizer{
        ctx:         ctx,
        device:      device,
        deviceProps: props,
        profiler: &HardwareProfiler{
            occupancyData:  make(map[string]*OccupancyInfo),
            memoryMetrics:  make(map[string]*MemoryMetrics),
            warpMetrics:    make(map[string]*WarpMetrics),
            throughputData: make(map[string]*ThroughputInfo),
        },
    }
    
    // Analyze architecture
    optimizer.analyzeArchitecture()
    
    return optimizer
}

func (ao *ArchitectureOptimizer) analyzeArchitecture() {
    props := ao.deviceProps
    
    ao.metrics = &ArchitectureMetrics{
        ComputeCapability: fmt.Sprintf("%d.%d", props.Major, props.Minor),
        SMCount:          props.MultiProcessorCount,
        MaxThreadsPerSM:  props.MaxThreadsPerMultiProcessor,
        SharedMemPerSM:   props.SharedMemPerMultiProcessor,
        RegistersPerSM:   props.RegsPerMultiProcessor,
        L2CacheSize:      props.L2CacheSize,
        MemoryBandwidth:  calculateTheoreticalBandwidth(props),
        BaseClockRate:    props.ClockRate,
        MemoryClockRate:  props.MemoryClockRate,
    }
    
    // Calculate cores per SM based on compute capability
    ao.metrics.CoresPerSM = ao.calculateCoresPerSM()
    ao.metrics.MaxWarpsPerSM = ao.metrics.MaxThreadsPerSM / 32
    
    fmt.Printf("🔧 GPU Architecture Analysis:\n")
    fmt.Printf("  Device: %s\n", props.Name)
    fmt.Printf("  Compute Capability: %s\n", ao.metrics.ComputeCapability)
    fmt.Printf("  Streaming Multiprocessors: %d\n", ao.metrics.SMCount)
    fmt.Printf("  CUDA Cores per SM: %d (Total: %d)\n", 
               ao.metrics.CoresPerSM, ao.metrics.SMCount * ao.metrics.CoresPerSM)
    fmt.Printf("  Max Threads per SM: %d\n", ao.metrics.MaxThreadsPerSM)
    fmt.Printf("  Max Warps per SM: %d\n", ao.metrics.MaxWarpsPerSM)
    fmt.Printf("  Shared Memory per SM: %d KB\n", ao.metrics.SharedMemPerSM/1024)
    fmt.Printf("  L2 Cache Size: %d MB\n", ao.metrics.L2CacheSize/1024/1024)
    fmt.Printf("  Memory Bandwidth: %.1f GB/s\n", ao.metrics.MemoryBandwidth)
}

func (ao *ArchitectureOptimizer) calculateCoresPerSM() int {
    major := ao.deviceProps.Major
    minor := ao.deviceProps.Minor
    
    // CUDA cores per SM by architecture
    switch {
    case major == 2: // Fermi
        if minor == 1 {
            return 48
        }
        return 32
    case major == 3: // Kepler
        return 192
    case major == 5: // Maxwell
        return 128
    case major == 6: // Pascal
        return 64
    case major == 7: // Volta/Turing
        return 64
    case major == 8: // Ampere
        return 64
    case major >= 9: // Hopper and beyond
        return 128
    default:
        return 64 // Default estimate
    }
}

func calculateTheoreticalBandwidth(props *cuda.DeviceProperties) float64 {
    // Theoretical bandwidth calculation
    // Memory bus width * memory clock rate * 2 (DDR) / 8 (bits to bytes)
    busWidth := float64(props.MemoryBusWidth)
    clockRate := float64(props.MemoryClockRate) * 1000 // Convert to Hz
    
    return (busWidth * clockRate * 2) / (8 * 1e9) // GB/s
}

// Occupancy analysis and optimization
func (ao *ArchitectureOptimizer) AnalyzeOccupancy(kernelName string, blockSize int, 
                                                 registersPerThread int, 
                                                 sharedMemPerBlock int64) *OccupancyInfo {
    fmt.Printf("🔍 Occupancy Analysis for %s:\n", kernelName)
    
    info := &OccupancyInfo{
        RegistersPer Thread: registersPerThread,
        SharedMemPerBlock:   sharedMemPerBlock,
        BlockSize:          blockSize,
    }
    
    // Calculate theoretical occupancy
    info.TheoreticalOccupancy = ao.calculateTheoreticalOccupancy(
        blockSize, registersPerThread, sharedMemPerBlock)
    
    // Determine limiting factor
    info.LimitingFactor = ao.findLimitingFactor(
        blockSize, registersPerThread, sharedMemPerBlock)
    
    ao.profiler.occupancyData[kernelName] = info
    
    fmt.Printf("  Block Size: %d threads\n", blockSize)
    fmt.Printf("  Registers per Thread: %d\n", registersPerThread)
    fmt.Printf("  Shared Memory per Block: %d KB\n", sharedMemPerBlock/1024)
    fmt.Printf("  Theoretical Occupancy: %.2f%%\n", info.TheoreticalOccupancy*100)
    fmt.Printf("  Limiting Factor: %s\n", info.LimitingFactor)
    
    return info
}

func (ao *ArchitectureOptimizer) calculateTheoreticalOccupancy(blockSize int, 
                                                              registersPerThread int, 
                                                              sharedMemPerBlock int64) float64 {
    maxThreadsPerSM := float64(ao.metrics.MaxThreadsPerSM)
    maxWarpsPerSM := float64(ao.metrics.MaxWarpsPerSM)
    maxSharedMem := float64(ao.metrics.SharedMemPerSM)
    maxRegisters := float64(ao.metrics.RegistersPerSM)
    
    // Calculate blocks per SM based on different constraints
    
    // Thread limit
    blocksByThreads := math.Floor(maxThreadsPerSM / float64(blockSize))
    
    // Warp limit
    warpsPerBlock := math.Ceil(float64(blockSize) / 32.0)
    blocksByWarps := math.Floor(maxWarpsPerSM / warpsPerBlock)
    
    // Shared memory limit
    blocksBySharedMem := math.Floor(maxSharedMem / float64(sharedMemPerBlock))
    
    // Register limit
    registersPerBlock := float64(blockSize * registersPerThread)
    blocksByRegisters := math.Floor(maxRegisters / registersPerBlock)
    
    // The minimum determines the actual number of blocks
    actualBlocks := math.Min(math.Min(blocksByThreads, blocksByWarps),
                           math.Min(blocksBySharedMem, blocksByRegisters))
    
    // Occupancy is the ratio of active warps to maximum warps
    activeWarps := actualBlocks * warpsPerBlock
    occupancy := activeWarps / maxWarpsPerSM
    
    return math.Min(occupancy, 1.0)
}

func (ao *ArchitectureOptimizer) findLimitingFactor(blockSize int, 
                                                   registersPerThread int, 
                                                   sharedMemPerBlock int64) string {
    maxThreadsPerSM := float64(ao.metrics.MaxThreadsPerSM)
    maxWarpsPerSM := float64(ao.metrics.MaxWarpsPerSM)
    maxSharedMem := float64(ao.metrics.SharedMemPerSM)
    maxRegisters := float64(ao.metrics.RegistersPerSM)
    
    blocksByThreads := math.Floor(maxThreadsPerSM / float64(blockSize))
    warpsPerBlock := math.Ceil(float64(blockSize) / 32.0)
    blocksByWarps := math.Floor(maxWarpsPerSM / warpsPerBlock)
    blocksBySharedMem := math.Floor(maxSharedMem / float64(sharedMemPerBlock))
    registersPerBlock := float64(blockSize * registersPerThread)
    blocksByRegisters := math.Floor(maxRegisters / registersPerBlock)
    
    minBlocks := math.Min(math.Min(blocksByThreads, blocksByWarps),
                         math.Min(blocksBySharedMem, blocksByRegisters))
    
    switch minBlocks {
    case blocksByThreads:
        return "Thread Limit"
    case blocksByWarps:
        return "Warp Limit"
    case blocksBySharedMem:
        return "Shared Memory"
    case blocksByRegisters:
        return "Register Usage"
    default:
        return "Unknown"
    }
}

// Memory access pattern optimization
func (ao *ArchitectureOptimizer) AnalyzeMemoryPattern(pattern string, 
                                                     dataSize int64, 
                                                     accessPattern string) *MemoryMetrics {
    fmt.Printf("🧠 Memory Pattern Analysis (%s):\n", pattern)
    
    metrics := &MemoryMetrics{}
    
    // Simulate memory analysis based on access pattern
    switch accessPattern {
    case "coalesced":
        metrics.CoalescingEfficiency = 1.0
        metrics.GlobalMemThroughput = ao.metrics.MemoryBandwidth * 0.9
        metrics.CacheHitRate = 0.95
        metrics.BankConflicts = 0
        
    case "strided":
        stride := 2 // Example stride
        metrics.CoalescingEfficiency = 1.0 / float64(stride)
        metrics.GlobalMemThroughput = ao.metrics.MemoryBandwidth * 0.5
        metrics.CacheHitRate = 0.7
        metrics.BankConflicts = stride - 1
        
    case "random":
        metrics.CoalescingEfficiency = 0.1
        metrics.GlobalMemThroughput = ao.metrics.MemoryBandwidth * 0.1
        metrics.CacheHitRate = 0.2
        metrics.BankConflicts = 16 // High conflicts
        
    default:
        metrics.CoalescingEfficiency = 0.5
        metrics.GlobalMemThroughput = ao.metrics.MemoryBandwidth * 0.5
        metrics.CacheHitRate = 0.6
        metrics.BankConflicts = 4
    }
    
    // Shared memory bandwidth (theoretical)
    metrics.SharedMemBandwidth = float64(ao.metrics.SMCount) * 1000.0 // GB/s estimate
    
    ao.profiler.memoryMetrics[pattern] = metrics
    
    fmt.Printf("  Access Pattern: %s\n", accessPattern)
    fmt.Printf("  Coalescing Efficiency: %.2f%%\n", metrics.CoalescingEfficiency*100)
    fmt.Printf("  Global Memory Throughput: %.1f GB/s\n", metrics.GlobalMemThroughput)
    fmt.Printf("  Cache Hit Rate: %.2f%%\n", metrics.CacheHitRate*100)
    fmt.Printf("  Bank Conflicts: %d\n", metrics.BankConflicts)
    
    return metrics
}

// Warp-level optimization analysis
func (ao *ArchitectureOptimizer) AnalyzeWarpExecution(kernelName string, 
                                                     hasBranching bool, 
                                                     branchDivergence float64) *WarpMetrics {
    fmt.Printf("🔄 Warp Execution Analysis (%s):\n", kernelName)
    
    metrics := &WarpMetrics{
        ActiveWarps:      ao.metrics.MaxWarpsPerSM,
        Instructions:     1000000, // Example instruction count
        DivergenceRatio:  branchDivergence,
        BranchEfficiency: 1.0 - branchDivergence,
    }
    
    // Calculate stalls based on divergence and memory access
    if hasBranching {
        metrics.MemoryStalls = int64(float64(metrics.Instructions) * 0.3 * branchDivergence)
        metrics.Stalls = int64(float64(metrics.Instructions) * 0.1 * branchDivergence)
    } else {
        metrics.MemoryStalls = int64(float64(metrics.Instructions) * 0.1)
        metrics.Stalls = int64(float64(metrics.Instructions) * 0.05)
    }
    
    ao.profiler.warpMetrics[kernelName] = metrics
    
    fmt.Printf("  Active Warps: %d\n", metrics.ActiveWarps)
    fmt.Printf("  Branch Divergence: %.2f%%\n", metrics.DivergenceRatio*100)
    fmt.Printf("  Branch Efficiency: %.2f%%\n", metrics.BranchEfficiency*100)
    fmt.Printf("  Memory Stalls: %d\n", metrics.MemoryStalls)
    fmt.Printf("  Total Stalls: %d\n", metrics.Stalls)
    
    return metrics
}

// Architecture-specific optimizations
func (ao *ArchitectureOptimizer) GetArchitectureOptimizations() []string {
    major := ao.deviceProps.Major
    minor := ao.deviceProps.Minor
    
    var optimizations []string
    
    switch {
    case major == 6: // Pascal
        optimizations = []string{
            "Use unified memory for easier development",
            "Optimize for 64 cores per SM",
            "Take advantage of improved atomics",
            "Use FP16 operations where applicable",
        }
        
    case major == 7 && minor == 0: // Volta
        optimizations = []string{
            "Leverage Tensor Cores for mixed-precision training",
            "Use independent thread scheduling carefully",
            "Optimize for larger shared memory (96KB)",
            "Take advantage of improved L1 cache",
            "Use cooperative groups for better synchronization",
        }
        
    case major == 7 && minor == 5: // Turing
        optimizations = []string{
            "Use Tensor Cores with INT8/INT4 precision",
            "Leverage RT cores for ray tracing workloads",
            "Optimize for variable precision operations",
            "Use mesh shaders for graphics workloads",
        }
        
    case major == 8: // Ampere
        optimizations = []string{
            "Leverage 3rd generation Tensor Cores",
            "Use structural sparsity (2:4 sparse patterns)",
            "Optimize for larger shared memory (164KB)",
            "Take advantage of asynchronous data movement",
            "Use fine-grained memory management",
            "Leverage multi-instance GPU (MIG) if available",
        }
        
    case major >= 9: // Hopper and beyond
        optimizations = []string{
            "Use 4th generation Tensor Cores",
            "Leverage transformer engine optimizations",
            "Optimize for NVLink and NVSwitch",
            "Use advanced memory hierarchy features",
            "Take advantage of improved concurrent execution",
        }
        
    default:
        optimizations = []string{
            "General optimization strategies",
            "Optimize memory coalescing",
            "Minimize branch divergence",
            "Use shared memory effectively",
            "Maximize occupancy",
        }
    }
    
    return optimizations
}

func main() {
    fmt.Println("⚡ Hardware Architecture Optimization")
    
    optimizer := NewArchitectureOptimizer()
    
    // Demonstrate architecture analysis
    demonstrateArchitectureAnalysis(optimizer)
    
    // Occupancy optimization
    occupancyOptimization(optimizer)
    
    // Memory pattern analysis
    memoryPatternAnalysis(optimizer)
    
    // Warp execution analysis
    warpExecutionAnalysis(optimizer)
    
    // Architecture-specific recommendations
    architectureRecommendations(optimizer)
}

func demonstrateArchitectureAnalysis(optimizer *ArchitectureOptimizer) {
    fmt.Println("\n🔧 Architecture Analysis:")
    
    // Create test scenarios with different configurations
    testConfigurations := []struct {
        name             string
        blockSize        int
        registersPerThread int
        sharedMemPerBlock int64
    }{
        {"Optimal Config", 256, 32, 12288},      // 12KB shared memory
        {"Register Limited", 512, 64, 8192},     // High register usage
        {"Memory Limited", 1024, 16, 32768},     // 32KB shared memory
        {"Small Block", 64, 24, 4096},           // Small block size
    }
    
    fmt.Println("\nOccupancy Analysis for Different Configurations:")
    for _, config := range testConfigurations {
        optimizer.AnalyzeOccupancy(config.name, config.blockSize, 
                                  config.registersPerThread, config.sharedMemPerBlock)
        fmt.Println()
    }
}

func occupancyOptimization(optimizer *ArchitectureOptimizer) {
    fmt.Println("\n🎯 Occupancy Optimization:")
    
    // Find optimal block size for a given kernel
    fmt.Println("Finding optimal block size (32 registers, 8KB shared memory):")
    
    registersPerThread := 32
    sharedMemPerBlock := int64(8192)
    
    bestOccupancy := 0.0
    bestBlockSize := 0
    
    blockSizes := []int{64, 96, 128, 160, 192, 224, 256, 320, 384, 448, 512, 640, 768, 896, 1024}
    
    for _, blockSize := range blockSizes {
        info := optimizer.AnalyzeOccupancy(fmt.Sprintf("test_block_%d", blockSize),
                                          blockSize, registersPerThread, sharedMemPerBlock)
        
        if info.TheoreticalOccupancy > bestOccupancy {
            bestOccupancy = info.TheoreticalOccupancy
            bestBlockSize = blockSize
        }
    }
    
    fmt.Printf("🏆 Optimal Block Size: %d threads (%.2f%% occupancy)\n", 
               bestBlockSize, bestOccupancy*100)
}

func memoryPatternAnalysis(optimizer *ArchitectureOptimizer) {
    fmt.Println("\n🧠 Memory Access Pattern Analysis:")
    
    patterns := []struct {
        name    string
        pattern string
    }{
        {"Sequential Access", "coalesced"},
        {"Strided Access", "strided"}, 
        {"Random Access", "random"},
        {"Mixed Pattern", "mixed"},
    }
    
    for _, p := range patterns {
        optimizer.AnalyzeMemoryPattern(p.name, 1024*1024*4, p.pattern) // 4MB
        fmt.Println()
    }
}

func warpExecutionAnalysis(optimizer *ArchitectureOptimizer) {
    fmt.Println("\n🔄 Warp Execution Analysis:")
    
    kernels := []struct {
        name           string
        hasBranching   bool
        divergence     float64
    }{
        {"No Branching", false, 0.0},
        {"Uniform Branching", true, 0.0},
        {"Moderate Divergence", true, 0.3},
        {"High Divergence", true, 0.8},
    }
    
    for _, kernel := range kernels {
        optimizer.AnalyzeWarpExecution(kernel.name, kernel.hasBranching, kernel.divergence)
        fmt.Println()
    }
}

func architectureRecommendations(optimizer *ArchitectureOptimizer) {
    fmt.Println("\n💡 Architecture-Specific Optimizations:")
    
    optimizations := optimizer.GetArchitectureOptimizations()
    
    for i, opt := range optimizations {
        fmt.Printf("  %d. %s\n", i+1, opt)
    }
    
    fmt.Println("\n🎯 Performance Tuning Checklist:")
    fmt.Println("  ✅ Maximize occupancy (>75% theoretical)")
    fmt.Println("  ✅ Optimize memory coalescing (>90% efficiency)")
    fmt.Println("  ✅ Minimize bank conflicts (<2 per warp)")
    fmt.Println("  ✅ Reduce branch divergence (<10%)")
    fmt.Println("  ✅ Balance register usage vs. occupancy")
    fmt.Println("  ✅ Use appropriate shared memory size")
    fmt.Println("  ✅ Leverage architecture-specific features")
    
    fmt.Println("\n🔬 Advanced Hardware Features:")
    
    major := optimizer.deviceProps.Major
    if major >= 7 {
        fmt.Println("  • Tensor Cores: Available for mixed-precision operations")
    }
    if major >= 8 {
        fmt.Println("  • Structural Sparsity: 2:4 sparse pattern optimization")
    }
    if major >= 6 {
        fmt.Println("  • Unified Memory: Simplified memory management")
    }
    
    fmt.Println("  • Cooperative Groups: Advanced thread synchronization")
    fmt.Println("  • Warp Primitives: Shuffle, vote, and collective operations")
    fmt.Println("  • Multi-stream Execution: Concurrent kernel execution")
    
    fmt.Println("\n📊 Performance Monitoring Tools:")
    fmt.Println("  • NVIDIA Nsight Compute: Kernel-level profiling")
    fmt.Println("  • NVIDIA Nsight Systems: System-wide timeline analysis") 
    fmt.Println("  • nvprof: Command-line profiler (legacy)")
    fmt.Println("  • CUPTI: CUDA Profiling Tools Interface")
    fmt.Println("  • GPU Performance Counters: Hardware-level metrics")
}
```

---

## 🎯 Module Assessment

### **Knowledge Validation**

1. **Architecture Mastery**: Understand GPU hierarchy and execution model
2. **Occupancy Optimization**: Maximize hardware utilization through configuration tuning
3. **Memory Optimization**: Achieve high coalescing efficiency and minimize bank conflicts
4. **Warp Programming**: Minimize divergence and leverage warp-level primitives

### **Practical Challenge**

Implement architecture-optimized kernels for:
- **High-Performance Computing**: Optimized GEMM for your specific GPU architecture  
- **Machine Learning**: Tensor operation kernels with architecture-specific optimizations
- **Scientific Computing**: Memory-intensive simulation with optimal access patterns
- **Graphics Processing**: Compute shader optimization for rendering workloads

### **Success Criteria**

- ✅ Achieve >75% theoretical occupancy on target kernels
- ✅ Demonstrate >90% memory coalescing efficiency
- ✅ Minimize branch divergence to <10% where branching is necessary
- ✅ Show measurable performance improvements from architecture-specific tuning

---

## 🚀 Next Steps

**Outstanding! You've mastered hardware-level optimization.**

**You're now ready for:**
➡️ **[Module 6: Performance Tuning](TRAINING_INTERMEDIATE_6_PERFORMANCE.md)**

**Skills Mastered:**
- 🔧 **Architecture Analysis** - Deep understanding of GPU hardware characteristics
- ⚡ **Occupancy Optimization** - Maximizing SM utilization through configuration tuning
- 🧠 **Memory Optimization** - Coalescing, caching, and bandwidth optimization
- 🔄 **Warp Programming** - Minimizing divergence and leveraging warp primitives

---

*From general programming to hardware-specific mastery - squeezing every ounce of performance from the silicon! ⚡🔧*
