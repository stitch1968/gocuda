// Package multigpu provides multi-GPU support for scaling computations
// across multiple GPUs with peer-to-peer communication and load balancing
package performance

import (
	"fmt"
	"sync"
	"time"

	"github.com/stitch1968/gocuda/memory"
)

// MultiGPU manages computations across multiple GPU devices
type MultiGPU struct {
	devices    []DeviceInfo               // Available GPU devices
	streams    map[int][]*StreamInfo      // Streams per device
	memory     map[int]*DistributedMemory // Memory pools per device
	p2pEnabled map[string]bool            // Peer-to-peer connectivity matrix
	scheduler  *LoadBalancer              // Work distribution scheduler
	metrics    *MultiGPUMetrics           // Performance tracking
	mutex      sync.RWMutex
	active     bool
}

// DeviceInfo contains information about a GPU device
type DeviceInfo struct {
	ID                int     `json:"id"`
	Name              string  `json:"name"`
	TotalMemory       int64   `json:"total_memory"`
	FreeMemory        int64   `json:"free_memory"`
	ComputeCapability string  `json:"compute_capability"`
	MultiProcessors   int     `json:"multiprocessors"`
	ClockRate         int     `json:"clock_rate"`
	MemoryBandwidth   float64 `json:"memory_bandwidth"`
	Active            bool    `json:"active"`
}

// DistributedMemory manages memory across multiple devices
type DistributedMemory struct {
	DeviceID   int
	LocalPool  *memory.Memory
	RemoteRefs map[int]*memory.Memory // References to memory on other devices
	totalSize  int64
	mutex      sync.RWMutex
}

// LoadBalancer distributes work across available GPUs
type LoadBalancer struct {
	devices        []DeviceInfo
	workloads      map[int]float64 // Current workload per device (0-1.0)
	strategy       BalancingStrategy
	lastAssignment map[int]time.Time // Last work assignment per device
	mutex          sync.RWMutex
}

// BalancingStrategy defines how work is distributed across GPUs
type BalancingStrategy int

const (
	RoundRobin BalancingStrategy = iota
	LoadAware
	CapabilityWeighted
	MemoryAware
	CustomBalancing
)

// MultiGPUMetrics tracks multi-GPU performance
type MultiGPUMetrics struct {
	TotalDevices      int             `json:"total_devices"`
	ActiveDevices     int             `json:"active_devices"`
	P2PConnections    int             `json:"p2p_connections"`
	ScalingEfficiency float64         `json:"scaling_efficiency"`
	LoadBalance       float64         `json:"load_balance"`
	Throughput        map[int]float64 `json:"throughput"`
	MemoryUtilization map[int]float64 `json:"memory_utilization"`
	LastUpdated       time.Time       `json:"last_updated"`
	mutex             sync.RWMutex
}

// DistributionStrategy defines how data is distributed across GPUs
type DistributionStrategy int

const (
	EvenSplit DistributionStrategy = iota
	WeightedByCapability
	WeightedByMemory
	CustomDistribution
)

// Kernel represents a GPU kernel that can be executed on multiple devices
type Kernel interface {
	Execute(deviceID int, input *memory.Memory, params map[string]interface{}) (*memory.Memory, error)
	GetName() string
	GetRequiredMemory(inputSize int64) int64
}

// MultiGPUKernel is a kernel that can be executed across multiple GPUs
type MultiGPUKernel struct {
	Name           string
	Implementation func(int, *memory.Memory, map[string]interface{}) (*memory.Memory, error)
	MemoryRequired func(int64) int64
}

// Execute implements the Kernel interface
func (mk *MultiGPUKernel) Execute(deviceID int, input *memory.Memory, params map[string]interface{}) (*memory.Memory, error) {
	return mk.Implementation(deviceID, input, params)
}

// GetName implements the Kernel interface
func (mk *MultiGPUKernel) GetName() string {
	return mk.Name
}

// GetRequiredMemory implements the Kernel interface
func (mk *MultiGPUKernel) GetRequiredMemory(inputSize int64) int64 {
	if mk.MemoryRequired != nil {
		return mk.MemoryRequired(inputSize)
	}
	return inputSize * 2 // Default: assume 2x input size needed
}

// NewMultiGPU creates a new multi-GPU manager
func NewMultiGPU() (*MultiGPU, error) {
	mg := &MultiGPU{
		devices:    make([]DeviceInfo, 0),
		streams:    make(map[int][]*StreamInfo),
		memory:     make(map[int]*DistributedMemory),
		p2pEnabled: make(map[string]bool),
		metrics: &MultiGPUMetrics{
			Throughput:        make(map[int]float64),
			MemoryUtilization: make(map[int]float64),
			LastUpdated:       time.Now(),
		},
		active: false,
	}

	// Discover available GPUs
	err := mg.discoverDevices()
	if err != nil {
		return nil, fmt.Errorf("failed to discover GPU devices: %v", err)
	}

	// Initialize load balancer
	mg.scheduler = &LoadBalancer{
		devices:        mg.devices,
		workloads:      make(map[int]float64),
		strategy:       LoadAware,
		lastAssignment: make(map[int]time.Time),
	}

	return mg, nil
}

// discoverDevices finds all available GPU devices
func (mg *MultiGPU) discoverDevices() error {
	// In a real implementation, this would use CUDA APIs to enumerate devices
	// For simulation, create mock devices
	mockDevices := []DeviceInfo{
		{
			ID:                0,
			Name:              "NVIDIA RTX 4090",
			TotalMemory:       24 * 1024 * 1024 * 1024, // 24GB
			FreeMemory:        20 * 1024 * 1024 * 1024, // 20GB free
			ComputeCapability: "8.9",
			MultiProcessors:   128,
			ClockRate:         2230,
			MemoryBandwidth:   1008.0, // GB/s
			Active:            true,
		},
		{
			ID:                1,
			Name:              "NVIDIA RTX 4080",
			TotalMemory:       16 * 1024 * 1024 * 1024, // 16GB
			FreeMemory:        14 * 1024 * 1024 * 1024, // 14GB free
			ComputeCapability: "8.9",
			MultiProcessors:   76,
			ClockRate:         2205,
			MemoryBandwidth:   717.0, // GB/s
			Active:            true,
		},
	}

	mg.devices = mockDevices
	mg.metrics.TotalDevices = len(mockDevices)
	mg.metrics.ActiveDevices = len(mockDevices)

	// Initialize streams for each device
	for _, device := range mg.devices {
		mg.streams[device.ID] = make([]*StreamInfo, 4) // 4 streams per device
		for i := 0; i < 4; i++ {
			mg.streams[device.ID][i] = &StreamInfo{
				ID:       i,
				Name:     fmt.Sprintf("Device%d_Stream%d", device.ID, i),
				Active:   false,
				LastUsed: time.Now(),
			}
		}

		// Initialize distributed memory
		mg.memory[device.ID] = &DistributedMemory{
			DeviceID:   device.ID,
			RemoteRefs: make(map[int]*memory.Memory),
			totalSize:  0,
		}
	}

	return nil
}

// EnableP2P enables peer-to-peer communication between GPUs
func (mg *MultiGPU) EnableP2P() error {
	mg.mutex.Lock()
	defer mg.mutex.Unlock()

	if len(mg.devices) < 2 {
		return fmt.Errorf("need at least 2 GPUs for P2P communication")
	}

	connections := 0
	// Test P2P connectivity between all device pairs
	for i := 0; i < len(mg.devices); i++ {
		for j := i + 1; j < len(mg.devices); j++ {
			device1 := mg.devices[i]
			device2 := mg.devices[j]

			p2pKey := fmt.Sprintf("%d-%d", device1.ID, device2.ID)

			// In real implementation, would check CUDA P2P capability
			canAccess := mg.checkP2PCapability(device1.ID, device2.ID)

			if canAccess {
				mg.p2pEnabled[p2pKey] = true
				connections++
			} else {
				mg.p2pEnabled[p2pKey] = false
			}
		}
	}

	mg.metrics.mutex.Lock()
	mg.metrics.P2PConnections = connections
	mg.metrics.mutex.Unlock()

	return nil
}

// checkP2PCapability checks if P2P is possible between two devices
func (mg *MultiGPU) checkP2PCapability(device1, device2 int) bool {
	// Simplified P2P capability check
	// Real implementation would use cuDeviceCanAccessPeer
	return true // Assume P2P is available for simulation
}

// DistributeData distributes data across multiple GPUs
func (mg *MultiGPU) DistributeData(data []float32, strategy DistributionStrategy) error {
	if len(data) == 0 {
		return fmt.Errorf("data cannot be empty")
	}

	mg.mutex.Lock()
	defer mg.mutex.Unlock()

	activeDevices := mg.getActiveDevices()
	if len(activeDevices) == 0 {
		return fmt.Errorf("no active devices available")
	}

	// Calculate data distribution based on strategy
	distribution := mg.calculateDistribution(len(data), activeDevices, strategy)

	// Distribute data to devices
	offset := 0
	for i, deviceID := range activeDevices {
		chunkSize := distribution[i]
		if chunkSize == 0 {
			continue
		}

		// Allocate memory on device
		deviceMem, err := memory.Alloc(int64(chunkSize * 4)) // float32 = 4 bytes
		if err != nil {
			return fmt.Errorf("failed to allocate memory on device %d: %v", deviceID, err)
		}

		// Copy data chunk to device (simulated)
		// In real implementation, would use cudaMemcpy
		dataChunk := data[offset : offset+chunkSize]
		err = mg.copyToDevice(deviceMem, dataChunk, deviceID)
		if err != nil {
			deviceMem.Free()
			return fmt.Errorf("failed to copy data to device %d: %v", deviceID, err)
		}

		// Store in distributed memory
		mg.memory[deviceID].LocalPool = deviceMem
		mg.memory[deviceID].totalSize = int64(chunkSize * 4)

		offset += chunkSize
	}

	return nil
}

// calculateDistribution determines how to split data across devices
func (mg *MultiGPU) calculateDistribution(dataSize int, devices []int, strategy DistributionStrategy) []int {
	distribution := make([]int, len(devices))

	switch strategy {
	case EvenSplit:
		chunkSize := dataSize / len(devices)
		remainder := dataSize % len(devices)

		for i := range distribution {
			distribution[i] = chunkSize
			if i < remainder {
				distribution[i]++
			}
		}

	case WeightedByCapability:
		totalCapability := 0.0
		capabilities := make([]float64, len(devices))

		// Calculate relative compute capabilities
		for i, deviceID := range devices {
			device := mg.getDeviceByID(deviceID)
			capability := float64(device.MultiProcessors) * float64(device.ClockRate)
			capabilities[i] = capability
			totalCapability += capability
		}

		// Distribute based on capability weights
		assigned := 0
		for i, capability := range capabilities {
			weight := capability / totalCapability
			distribution[i] = int(float64(dataSize) * weight)
			assigned += distribution[i]
		}

		// Handle rounding errors
		if assigned < dataSize {
			distribution[0] += dataSize - assigned
		}

	case WeightedByMemory:
		totalMemory := int64(0)
		memories := make([]int64, len(devices))

		// Calculate relative memory sizes
		for i, deviceID := range devices {
			device := mg.getDeviceByID(deviceID)
			memories[i] = device.FreeMemory
			totalMemory += device.FreeMemory
		}

		// Distribute based on memory weights
		assigned := 0
		for i, memSize := range memories {
			weight := float64(memSize) / float64(totalMemory)
			distribution[i] = int(float64(dataSize) * weight)
			assigned += distribution[i]
		}

		// Handle rounding errors
		if assigned < dataSize {
			distribution[0] += dataSize - assigned
		}

	default:
		// Default to even split
		return mg.calculateDistribution(dataSize, devices, EvenSplit)
	}

	return distribution
}

// ParallelExecute executes a kernel on all active GPUs
func (mg *MultiGPU) ParallelExecute(kernel Kernel, params map[string]interface{}) (map[int]*memory.Memory, error) {
	mg.mutex.RLock()
	activeDevices := mg.getActiveDevices()
	mg.mutex.RUnlock()

	if len(activeDevices) == 0 {
		return nil, fmt.Errorf("no active devices available")
	}

	results := make(map[int]*memory.Memory)
	var mu sync.Mutex
	var wg sync.WaitGroup
	var execError error

	// Execute kernel on each device concurrently
	for _, deviceID := range activeDevices {
		wg.Add(1)
		go func(devID int) {
			defer wg.Done()

			// Get input data for this device
			deviceMem := mg.memory[devID].LocalPool
			if deviceMem == nil {
				mu.Lock()
				if execError == nil {
					execError = fmt.Errorf("no data allocated on device %d", devID)
				}
				mu.Unlock()
				return
			}

			// Execute kernel
			start := time.Now()
			result, err := kernel.Execute(devID, deviceMem, params)
			duration := time.Since(start)

			if err != nil {
				mu.Lock()
				if execError == nil {
					execError = fmt.Errorf("kernel execution failed on device %d: %v", devID, err)
				}
				mu.Unlock()
				return
			}

			// Store result
			mu.Lock()
			results[devID] = result

			// Update metrics
			mg.updateDeviceMetrics(devID, duration, result.Size())
			mu.Unlock()
		}(deviceID)
	}

	wg.Wait()

	if execError != nil {
		// Clean up any successful results
		for _, result := range results {
			result.Free()
		}
		return nil, execError
	}

	// Update scaling efficiency
	mg.calculateScalingEfficiency(len(activeDevices))

	return results, nil
}

// GatherResults collects results from all devices back to the host
func (mg *MultiGPU) GatherResults(deviceResults map[int]*memory.Memory) ([]float32, error) {
	if len(deviceResults) == 0 {
		return nil, fmt.Errorf("no results to gather")
	}

	// Calculate total result size
	totalSize := int64(0)
	for _, result := range deviceResults {
		totalSize += result.Size()
	}

	// Allocate host memory
	hostData := make([]float32, totalSize/4) // float32 = 4 bytes
	offset := 0

	// Copy data from each device
	for deviceID, result := range deviceResults {
		deviceSize := int(result.Size() / 4)

		// Simulate copying from device to host
		err := mg.copyFromDevice(result, hostData[offset:offset+deviceSize], deviceID)
		if err != nil {
			return nil, fmt.Errorf("failed to copy from device %d: %v", deviceID, err)
		}

		offset += deviceSize
	}

	return hostData, nil
}

// P2PCopy performs direct GPU-to-GPU memory transfer
func (mg *MultiGPU) P2PCopy(srcDevice, dstDevice int, src, dst *memory.Memory, size int64) error {
	p2pKey := fmt.Sprintf("%d-%d", srcDevice, dstDevice)

	mg.mutex.RLock()
	enabled, exists := mg.p2pEnabled[p2pKey]
	mg.mutex.RUnlock()

	if !exists || !enabled {
		return fmt.Errorf("P2P not enabled between devices %d and %d", srcDevice, dstDevice)
	}

	// In real implementation, would use cudaMemcpyPeer
	// For simulation, just record the operation
	time.Sleep(time.Duration(size/1024/1024) * time.Millisecond) // Simulate transfer time

	return nil
}

// Helper functions

// getActiveDevices returns list of currently active device IDs
func (mg *MultiGPU) getActiveDevices() []int {
	activeDevices := make([]int, 0)
	for _, device := range mg.devices {
		if device.Active {
			activeDevices = append(activeDevices, device.ID)
		}
	}
	return activeDevices
}

// getDeviceByID finds device info by ID
func (mg *MultiGPU) getDeviceByID(deviceID int) *DeviceInfo {
	for i := range mg.devices {
		if mg.devices[i].ID == deviceID {
			return &mg.devices[i]
		}
	}
	return nil
}

// copyToDevice simulates copying data to GPU device
func (mg *MultiGPU) copyToDevice(deviceMem *memory.Memory, data []float32, deviceID int) error {
	// Simulate GPU memory copy
	copyTime := time.Duration(len(data)) * time.Nanosecond
	time.Sleep(copyTime)
	return nil
}

// copyFromDevice simulates copying data from GPU device
func (mg *MultiGPU) copyFromDevice(deviceMem *memory.Memory, hostData []float32, deviceID int) error {
	// Simulate GPU memory copy
	copyTime := time.Duration(len(hostData)) * time.Nanosecond
	time.Sleep(copyTime)
	return nil
}

// updateDeviceMetrics updates performance metrics for a device
func (mg *MultiGPU) updateDeviceMetrics(deviceID int, duration time.Duration, dataSize int64) {
	mg.metrics.mutex.Lock()
	defer mg.metrics.mutex.Unlock()

	// Calculate throughput (MB/s)
	throughput := float64(dataSize) / duration.Seconds() / 1024 / 1024
	mg.metrics.Throughput[deviceID] = throughput

	// Estimate memory utilization
	device := mg.getDeviceByID(deviceID)
	if device != nil {
		utilization := float64(device.TotalMemory-device.FreeMemory) / float64(device.TotalMemory) * 100
		mg.metrics.MemoryUtilization[deviceID] = utilization
	}

	mg.metrics.LastUpdated = time.Now()
}

// calculateScalingEfficiency calculates multi-GPU scaling efficiency
func (mg *MultiGPU) calculateScalingEfficiency(numDevices int) {
	mg.metrics.mutex.Lock()
	defer mg.metrics.mutex.Unlock()

	// Simplified scaling efficiency calculation
	// Real implementation would compare single-GPU vs multi-GPU performance
	idealSpeedup := float64(numDevices)

	// Account for overheads (communication, synchronization, etc.)
	overhead := 1.0 - (0.1 * float64(numDevices-1)) // 10% overhead per additional GPU
	actualSpeedup := idealSpeedup * overhead

	mg.metrics.ScalingEfficiency = (actualSpeedup / idealSpeedup) * 100
}

// GetMetrics returns current multi-GPU metrics
func (mg *MultiGPU) GetMetrics() *MultiGPUMetrics {
	mg.metrics.mutex.RLock()
	defer mg.metrics.mutex.RUnlock()

	// Return a copy to avoid race conditions
	metrics := &MultiGPUMetrics{
		TotalDevices:      mg.metrics.TotalDevices,
		ActiveDevices:     mg.metrics.ActiveDevices,
		P2PConnections:    mg.metrics.P2PConnections,
		ScalingEfficiency: mg.metrics.ScalingEfficiency,
		LoadBalance:       mg.metrics.LoadBalance,
		Throughput:        make(map[int]float64),
		MemoryUtilization: make(map[int]float64),
		LastUpdated:       mg.metrics.LastUpdated,
	}

	// Copy maps
	for k, v := range mg.metrics.Throughput {
		metrics.Throughput[k] = v
	}
	for k, v := range mg.metrics.MemoryUtilization {
		metrics.MemoryUtilization[k] = v
	}

	return metrics
}

// GetDeviceInfo returns information about all devices
func (mg *MultiGPU) GetDeviceInfo() []DeviceInfo {
	mg.mutex.RLock()
	defer mg.mutex.RUnlock()

	// Return a copy
	devices := make([]DeviceInfo, len(mg.devices))
	copy(devices, mg.devices)
	return devices
}

// String returns a formatted string representation of multi-GPU metrics
func (mgm *MultiGPUMetrics) String() string {
	mgm.mutex.RLock()
	defer mgm.mutex.RUnlock()

	return fmt.Sprintf(`Multi-GPU Performance Metrics
===============================
Total Devices: %d
Active Devices: %d  
P2P Connections: %d
Scaling Efficiency: %.1f%%
Load Balance: %.1f%%
Last Updated: %s

Benefits:
- Linear scaling across multiple GPUs
- Direct P2P memory transfers
- Load-balanced work distribution
- Automatic device discovery and management`,
		mgm.TotalDevices,
		mgm.ActiveDevices,
		mgm.P2PConnections,
		mgm.ScalingEfficiency,
		mgm.LoadBalance,
		mgm.LastUpdated.Format("2006-01-02 15:04:05"))
}
