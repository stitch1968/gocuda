// Package profiler provides GPU performance profiling and monitoring for GoCUDA.
// This package includes timing, memory usage tracking, kernel performance analysis,
// and optimization suggestions for GPU code.
package profiler

import (
	"fmt"
	"sort"
	"sync"
	"time"
)

// Event represents a profiling event
type Event struct {
	Name      string
	Type      EventType
	StartTime time.Time
	EndTime   time.Time
	Duration  time.Duration
	StreamID  int
	DeviceID  int
	Metadata  map[string]interface{}
}

// EventType defines the type of profiling event
type EventType int

const (
	EventKernel EventType = iota
	EventMemoryAlloc
	EventMemoryFree
	EventMemoryCopy
	EventStream
	EventOther
)

// String returns the string representation of EventType
func (et EventType) String() string {
	switch et {
	case EventKernel:
		return "Kernel"
	case EventMemoryAlloc:
		return "MemoryAlloc"
	case EventMemoryFree:
		return "MemoryFree"
	case EventMemoryCopy:
		return "MemoryCopy"
	case EventStream:
		return "Stream"
	case EventOther:
		return "Other"
	default:
		return "Unknown"
	}
}

// Profiler manages performance profiling
type Profiler struct {
	enabled    bool
	events     []Event
	startTimes map[string]time.Time
	mu         sync.RWMutex
	memTracker *MemoryTracker
}

// MemoryTracker tracks GPU memory usage
type MemoryTracker struct {
	allocations map[uintptr]*AllocationInfo
	totalBytes  int64
	peakBytes   int64
	mu          sync.RWMutex
}

// AllocationInfo stores information about a memory allocation
type AllocationInfo struct {
	Size       int64
	Timestamp  time.Time
	StackTrace string
}

// Statistics contains profiling statistics
type Statistics struct {
	TotalEvents     int
	TotalDuration   time.Duration
	AverageDuration time.Duration
	EventsByType    map[EventType]int
	MemoryPeak      int64
	MemoryCurrent   int64
	Recommendations []string
}

// KernelProfile contains kernel-specific profiling information
type KernelProfile struct {
	Name           string
	CallCount      int
	TotalTime      time.Duration
	AverageTime    time.Duration
	MinTime        time.Duration
	MaxTime        time.Duration
	ThroughputGBps float64
	Occupancy      float64
}

var (
	globalProfiler *Profiler
	profilerOnce   sync.Once
)

// GetProfiler returns the global profiler instance
func GetProfiler() *Profiler {
	profilerOnce.Do(func() {
		globalProfiler = &Profiler{
			enabled:    false,
			events:     make([]Event, 0),
			startTimes: make(map[string]time.Time),
			memTracker: &MemoryTracker{
				allocations: make(map[uintptr]*AllocationInfo),
			},
		}
	})
	return globalProfiler
}

// Enable enables profiling
func (p *Profiler) Enable() {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.enabled = true
}

// Disable disables profiling
func (p *Profiler) Disable() {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.enabled = false
}

// IsEnabled returns whether profiling is enabled
func (p *Profiler) IsEnabled() bool {
	p.mu.RLock()
	defer p.mu.RUnlock()
	return p.enabled
}

// StartEvent starts timing an event
func (p *Profiler) StartEvent(name string) {
	if !p.IsEnabled() {
		return
	}

	p.mu.Lock()
	defer p.mu.Unlock()
	p.startTimes[name] = time.Now()
}

// EndEvent ends timing an event and records it
func (p *Profiler) EndEvent(name string, eventType EventType) {
	if !p.IsEnabled() {
		return
	}

	endTime := time.Now()
	p.mu.Lock()
	defer p.mu.Unlock()

	if startTime, exists := p.startTimes[name]; exists {
		duration := endTime.Sub(startTime)
		event := Event{
			Name:      name,
			Type:      eventType,
			StartTime: startTime,
			EndTime:   endTime,
			Duration:  duration,
			StreamID:  0, // Default stream
			DeviceID:  0, // Default device
			Metadata:  make(map[string]interface{}),
		}
		p.events = append(p.events, event)
		delete(p.startTimes, name)
	}
}

// RecordEvent records a complete event
func (p *Profiler) RecordEvent(event Event) {
	if !p.IsEnabled() {
		return
	}

	p.mu.Lock()
	defer p.mu.Unlock()
	p.events = append(p.events, event)
}

// Clear clears all profiling data
func (p *Profiler) Clear() {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.events = p.events[:0]
	p.startTimes = make(map[string]time.Time)
	p.memTracker.allocations = make(map[uintptr]*AllocationInfo)
	p.memTracker.totalBytes = 0
	p.memTracker.peakBytes = 0
}

// GetEvents returns all recorded events
func (p *Profiler) GetEvents() []Event {
	p.mu.RLock()
	defer p.mu.RUnlock()

	events := make([]Event, len(p.events))
	copy(events, p.events)
	return events
}

// GetStatistics returns profiling statistics
func (p *Profiler) GetStatistics() Statistics {
	p.mu.RLock()
	defer p.mu.RUnlock()

	stats := Statistics{
		TotalEvents:   len(p.events),
		EventsByType:  make(map[EventType]int),
		MemoryCurrent: p.memTracker.totalBytes,
		MemoryPeak:    p.memTracker.peakBytes,
	}

	if len(p.events) == 0 {
		return stats
	}

	var totalDuration time.Duration
	for _, event := range p.events {
		totalDuration += event.Duration
		stats.EventsByType[event.Type]++
	}

	stats.TotalDuration = totalDuration
	stats.AverageDuration = totalDuration / time.Duration(len(p.events))
	stats.Recommendations = p.generateRecommendations()

	return stats
}

// GetKernelProfiles returns profiling information for kernels
func (p *Profiler) GetKernelProfiles() []KernelProfile {
	p.mu.RLock()
	defer p.mu.RUnlock()

	kernelData := make(map[string][]time.Duration)

	for _, event := range p.events {
		if event.Type == EventKernel {
			kernelData[event.Name] = append(kernelData[event.Name], event.Duration)
		}
	}

	profiles := make([]KernelProfile, 0, len(kernelData))
	for name, durations := range kernelData {
		if len(durations) == 0 {
			continue
		}

		profile := KernelProfile{
			Name:      name,
			CallCount: len(durations),
		}

		// Calculate statistics
		var total time.Duration
		minTime := durations[0]
		maxTime := durations[0]

		for _, d := range durations {
			total += d
			if d < minTime {
				minTime = d
			}
			if d > maxTime {
				maxTime = d
			}
		}

		profile.TotalTime = total
		profile.AverageTime = total / time.Duration(len(durations))
		profile.MinTime = minTime
		profile.MaxTime = maxTime
		profile.ThroughputGBps = 0.0 // Would need actual data transfer info
		profile.Occupancy = 1.0      // Simulated full occupancy

		profiles = append(profiles, profile)
	}

	// Sort by total time (descending)
	sort.Slice(profiles, func(i, j int) bool {
		return profiles[i].TotalTime > profiles[j].TotalTime
	})

	return profiles
}

// generateRecommendations generates optimization recommendations
func (p *Profiler) generateRecommendations() []string {
	recommendations := make([]string, 0)

	// Analyze memory usage
	if p.memTracker.peakBytes > 1024*1024*1024 { // > 1GB
		recommendations = append(recommendations, "High memory usage detected. Consider optimizing memory allocation patterns.")
	}

	// Analyze kernel performance
	kernelEvents := 0
	memoryEvents := 0
	var totalKernelTime, totalMemoryTime time.Duration

	for _, event := range p.events {
		switch event.Type {
		case EventKernel:
			kernelEvents++
			totalKernelTime += event.Duration
		case EventMemoryCopy:
			memoryEvents++
			totalMemoryTime += event.Duration
		}
	}

	if memoryEvents > 0 && kernelEvents > 0 {
		memoryRatio := float64(totalMemoryTime) / float64(totalKernelTime+totalMemoryTime)
		if memoryRatio > 0.5 {
			recommendations = append(recommendations, "High memory transfer overhead. Consider using unified memory or reducing host-device transfers.")
		}
	}

	if len(p.events) > 1000 {
		recommendations = append(recommendations, "Large number of events detected. Consider batching operations for better performance.")
	}

	return recommendations
}

// Memory tracking methods

// TrackAllocation tracks a memory allocation
func (p *Profiler) TrackAllocation(ptr uintptr, size int64) {
	if !p.IsEnabled() {
		return
	}

	p.memTracker.mu.Lock()
	defer p.memTracker.mu.Unlock()

	p.memTracker.allocations[ptr] = &AllocationInfo{
		Size:      size,
		Timestamp: time.Now(),
	}
	p.memTracker.totalBytes += size
	if p.memTracker.totalBytes > p.memTracker.peakBytes {
		p.memTracker.peakBytes = p.memTracker.totalBytes
	}

	// Record allocation event
	go func() {
		p.RecordEvent(Event{
			Name:      "MemoryAllocation",
			Type:      EventMemoryAlloc,
			StartTime: time.Now(),
			EndTime:   time.Now(),
			Duration:  0,
			Metadata: map[string]interface{}{
				"size": size,
				"ptr":  ptr,
			},
		})
	}()
}

// TrackDeallocation tracks a memory deallocation
func (p *Profiler) TrackDeallocation(ptr uintptr) {
	if !p.IsEnabled() {
		return
	}

	p.memTracker.mu.Lock()
	defer p.memTracker.mu.Unlock()

	if info, exists := p.memTracker.allocations[ptr]; exists {
		p.memTracker.totalBytes -= info.Size
		delete(p.memTracker.allocations, ptr)

		// Record deallocation event
		go func() {
			p.RecordEvent(Event{
				Name:      "MemoryDeallocation",
				Type:      EventMemoryFree,
				StartTime: time.Now(),
				EndTime:   time.Now(),
				Duration:  0,
				Metadata: map[string]interface{}{
					"size": info.Size,
					"ptr":  ptr,
				},
			})
		}()
	}
}

// GetMemoryUsage returns current memory usage information
func (p *Profiler) GetMemoryUsage() (current, peak int64) {
	p.memTracker.mu.RLock()
	defer p.memTracker.mu.RUnlock()
	return p.memTracker.totalBytes, p.memTracker.peakBytes
}

// Convenience functions

// Enable enables global profiling
func Enable() {
	GetProfiler().Enable()
}

// Disable disables global profiling
func Disable() {
	GetProfiler().Disable()
}

// StartEvent starts timing a global event
func StartEvent(name string) {
	GetProfiler().StartEvent(name)
}

// EndEvent ends timing a global event
func EndEvent(name string, eventType EventType) {
	GetProfiler().EndEvent(name, eventType)
}

// ProfileKernel profiles a kernel execution
func ProfileKernel(name string, kernelFunc func()) error {
	profiler := GetProfiler()
	if !profiler.IsEnabled() {
		kernelFunc()
		return nil
	}

	profiler.StartEvent(name)
	kernelFunc()
	profiler.EndEvent(name, EventKernel)
	return nil
}

// ProfileMemoryTransfer profiles a memory transfer operation
func ProfileMemoryTransfer(name string, transferFunc func()) error {
	profiler := GetProfiler()
	if !profiler.IsEnabled() {
		transferFunc()
		return nil
	}

	profiler.StartEvent(name)
	transferFunc()
	profiler.EndEvent(name, EventMemoryCopy)
	return nil
}

// GetStatistics returns global profiling statistics
func GetStatistics() Statistics {
	return GetProfiler().GetStatistics()
}

// GetKernelProfiles returns global kernel profiles
func GetKernelProfiles() []KernelProfile {
	return GetProfiler().GetKernelProfiles()
}

// Clear clears global profiling data
func Clear() {
	GetProfiler().Clear()
}

// Report generation

// GenerateReport generates a comprehensive profiling report
func GenerateReport() string {
	profiler := GetProfiler()
	stats := profiler.GetStatistics()
	kernelProfiles := profiler.GetKernelProfiles()

	report := fmt.Sprintf("GoCUDA Profiling Report\n")
	report += fmt.Sprintf("======================\n\n")

	report += fmt.Sprintf("Summary:\n")
	report += fmt.Sprintf("  Total Events: %d\n", stats.TotalEvents)
	report += fmt.Sprintf("  Total Duration: %v\n", stats.TotalDuration)
	report += fmt.Sprintf("  Average Duration: %v\n", stats.AverageDuration)
	report += fmt.Sprintf("  Memory Peak: %d bytes (%.2f MB)\n", stats.MemoryPeak, float64(stats.MemoryPeak)/(1024*1024))
	report += fmt.Sprintf("  Memory Current: %d bytes (%.2f MB)\n\n", stats.MemoryCurrent, float64(stats.MemoryCurrent)/(1024*1024))

	report += fmt.Sprintf("Events by Type:\n")
	for eventType, count := range stats.EventsByType {
		report += fmt.Sprintf("  %s: %d\n", eventType.String(), count)
	}
	report += fmt.Sprintf("\n")

	if len(kernelProfiles) > 0 {
		report += fmt.Sprintf("Kernel Profiles:\n")
		for i, profile := range kernelProfiles {
			if i >= 10 { // Limit to top 10
				break
			}
			report += fmt.Sprintf("  %s:\n", profile.Name)
			report += fmt.Sprintf("    Calls: %d\n", profile.CallCount)
			report += fmt.Sprintf("    Total Time: %v\n", profile.TotalTime)
			report += fmt.Sprintf("    Average Time: %v\n", profile.AverageTime)
			report += fmt.Sprintf("    Min/Max Time: %v / %v\n", profile.MinTime, profile.MaxTime)
			report += fmt.Sprintf("\n")
		}
	}

	if len(stats.Recommendations) > 0 {
		report += fmt.Sprintf("Recommendations:\n")
		for _, rec := range stats.Recommendations {
			report += fmt.Sprintf("  - %s\n", rec)
		}
	}

	return report
}

// WriteReport writes a profiling report to a file
func WriteReport(filename string) error {
	// For now, just return the report string (file writing would require os package)
	_ = GenerateReport()
	return fmt.Errorf("file writing not implemented in simulation mode")
}
