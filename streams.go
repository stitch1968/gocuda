package cuda

// Re-exports from streams package for backward compatibility

import (
	"github.com/stitch1968/gocuda/streams"
)

// Stream management functions and types
var (
	CreateStream             = streams.CreateStream
	CreateStreamNonBlocking  = streams.CreateStreamNonBlocking
	CreateHighPriorityStream = streams.CreateHighPriorityStream
	CreateLowPriorityStream  = streams.CreateLowPriorityStream
	GetManager               = streams.GetManager
	SynchronizeDevice        = streams.SynchronizeDevice
)

// Types re-exported
type StreamFlags = streams.StreamFlags
type Priority = streams.Priority

// Stream flags constants
const (
	StreamDefault     = streams.StreamDefault
	StreamNonBlocking = streams.StreamNonBlocking
)

// Priority constants
const (
	PriorityLow    = streams.PriorityLow
	PriorityNormal = streams.PriorityNormal
	PriorityHigh   = streams.PriorityHigh
)
