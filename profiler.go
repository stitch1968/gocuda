package cuda

// Re-exports from profiler package for backward compatibility

import (
	"github.com/stitch1968/gocuda/profiler"
)

// Profiler functions and types
var (
	GetProfiler = profiler.GetProfiler
)

// Profiler event types re-exported
type EventType = profiler.EventType

// Event type constants
const (
	EventKernel      = profiler.EventKernel
	EventMemoryAlloc = profiler.EventMemoryAlloc
	EventMemoryFree  = profiler.EventMemoryFree
)
