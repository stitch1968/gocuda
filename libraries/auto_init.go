package libraries

import (
	"fmt"
	"sync"

	cuda "github.com/stitch1968/gocuda"
)

var (
	libraryInitOnce sync.Once
	libraryInitErr  error
)

func ensureCudaReady() error {
	libraryInitOnce.Do(func() {
		if err := cuda.Initialize(); err != nil {
			libraryInitErr = err
			return
		}

		if !cuda.ShouldUseCuda() {
			return
		}

		devices, err := cuda.GetDevices()
		if err != nil {
			libraryInitErr = err
			return
		}
		if len(devices) == 0 {
			return
		}

		preferredID := selectPreferredDevice(devices)
		if err := cuda.SetCudaDevice(preferredID); err != nil {
			libraryInitErr = err
			return
		}

		fmt.Printf("GoCUDA: selected CUDA device %d (%s)\n", preferredID, devices[preferredID].Name)
	})

	return libraryInitErr
}

func selectPreferredDevice(devices []*cuda.Device) int {
	bestIdx := 0
	best := devices[0]
	for i := 1; i < len(devices); i++ {
		candidate := devices[i]
		if devicePreferred(candidate, best) {
			best = candidate
			bestIdx = i
		}
	}
	return bestIdx
}

func devicePreferred(candidate, current *cuda.Device) bool {
	if candidate.Properties.Major != current.Properties.Major {
		return candidate.Properties.Major > current.Properties.Major
	}
	if candidate.Properties.Minor != current.Properties.Minor {
		return candidate.Properties.Minor > current.Properties.Minor
	}
	if candidate.Properties.TotalGlobalMem != current.Properties.TotalGlobalMem {
		return candidate.Properties.TotalGlobalMem > current.Properties.TotalGlobalMem
	}
	if candidate.Properties.MultiProcessorCount != current.Properties.MultiProcessorCount {
		return candidate.Properties.MultiProcessorCount > current.Properties.MultiProcessorCount
	}
	if candidate.Properties.ClockRate != current.Properties.ClockRate {
		return candidate.Properties.ClockRate > current.Properties.ClockRate
	}
	if candidate.Properties.MemoryClockRate != current.Properties.MemoryClockRate {
		return candidate.Properties.MemoryClockRate > current.Properties.MemoryClockRate
	}
	if candidate.Properties.MemoryBusWidth != current.Properties.MemoryBusWidth {
		return candidate.Properties.MemoryBusWidth > current.Properties.MemoryBusWidth
	}
	return false
}
