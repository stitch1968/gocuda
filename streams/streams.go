// Package streams provides CUDA stream management and execution for GoCUDA.
// This package handles asynchronous execution, stream synchronization,
// and performance optimization through parallel execution.
package streams

import (
	"fmt"
	"sync"
	"time"

	"github.com/stitch1968/gocuda/internal"
)

// Priority defines stream execution priority
type Priority int

const (
	PriorityLow Priority = iota
	PriorityNormal
	PriorityHigh
)

// StreamFlags defines stream creation flags
type StreamFlags int

const (
	StreamDefault StreamFlags = iota
	StreamNonBlocking
)

// Event represents a CUDA event for stream synchronization
type Event struct {
	id        int
	timestamp time.Time
	completed bool
	mu        sync.Mutex
}

// Record records the event in the stream
func (e *Event) Record(stream *internal.Stream) error {
	e.mu.Lock()
	defer e.mu.Unlock()

	e.timestamp = time.Now()
	e.completed = false

	// Schedule event completion after stream operations
	stream.Execute(func() {
		e.mu.Lock()
		defer e.mu.Unlock()
		e.completed = true
	})

	return nil
}

// Query checks if the event has completed
func (e *Event) Query() bool {
	e.mu.Lock()
	defer e.mu.Unlock()
	return e.completed
}

// Synchronize waits for the event to complete
func (e *Event) Synchronize() error {
	for !e.Query() {
		time.Sleep(time.Microsecond * 10)
	}
	return nil
}

// ElapsedTime returns the elapsed time between two events
func (e *Event) ElapsedTime(end *Event) (time.Duration, error) {
	if !e.completed || !end.completed {
		return 0, fmt.Errorf("events must be completed to measure elapsed time")
	}

	if end.timestamp.Before(e.timestamp) {
		return 0, fmt.Errorf("end event timestamp is before start event")
	}

	return end.timestamp.Sub(e.timestamp), nil
}

// Stream represents a CUDA stream
type Stream struct {
	*internal.Stream
	priority Priority
	flags    StreamFlags
	events   []*Event
	mu       sync.Mutex
}

// Manager manages multiple CUDA streams
type Manager struct {
	streams       map[int]*Stream
	defaultStream *Stream
	nextID        int
	mu            sync.RWMutex
}

var (
	globalManager *Manager
	managerOnce   sync.Once
)

// GetManager returns the global stream manager
func GetManager() *Manager {
	managerOnce.Do(func() {
		globalManager = &Manager{
			streams: make(map[int]*Stream),
			nextID:  1,
		}

		// Create default stream
		defaultInternalStream := internal.GetDefaultStream()
		globalManager.defaultStream = &Stream{
			Stream:   defaultInternalStream,
			priority: PriorityNormal,
			flags:    StreamDefault,
			events:   make([]*Event, 0),
		}
		globalManager.streams[0] = globalManager.defaultStream
	})
	return globalManager
}

// CreateStream creates a new CUDA stream
func (m *Manager) CreateStream(flags StreamFlags) (*Stream, error) {
	m.mu.Lock()
	defer m.mu.Unlock()

	internalStream := internal.NewStream()
	stream := &Stream{
		Stream:   internalStream,
		priority: PriorityNormal,
		flags:    flags,
		events:   make([]*Event, 0),
	}

	id := m.nextID
	m.nextID++
	m.streams[id] = stream

	return stream, nil
}

// CreateStreamWithPriority creates a new CUDA stream with priority
func (m *Manager) CreateStreamWithPriority(flags StreamFlags, priority Priority) (*Stream, error) {
	stream, err := m.CreateStream(flags)
	if err != nil {
		return nil, err
	}

	stream.priority = priority
	return stream, nil
}

// GetDefaultStream returns the default stream
func (m *Manager) GetDefaultStream() *Stream {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return m.defaultStream
}

// DestroyStream destroys a stream
func (m *Manager) DestroyStream(stream *Stream) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	// Find and remove stream
	for id, s := range m.streams {
		if s == stream {
			delete(m.streams, id)
			break
		}
	}

	// Wait for stream to complete
	return stream.Synchronize()
}

// SynchronizeAll waits for all streams to complete
func (m *Manager) SynchronizeAll() error {
	m.mu.RLock()
	streams := make([]*Stream, 0, len(m.streams))
	for _, stream := range m.streams {
		streams = append(streams, stream)
	}
	m.mu.RUnlock()

	for _, stream := range streams {
		if err := stream.Synchronize(); err != nil {
			return err
		}
	}

	return nil
}

// GetActiveStreams returns the number of active streams
func (m *Manager) GetActiveStreams() int {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return len(m.streams)
}

// Stream methods

// SetPriority sets the stream priority
func (s *Stream) SetPriority(priority Priority) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.priority = priority
}

// GetPriority returns the stream priority
func (s *Stream) GetPriority() Priority {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.priority
}

// WaitEvent makes the stream wait for an event
func (s *Stream) WaitEvent(event *Event) error {
	s.Execute(func() {
		event.Synchronize()
	})
	return nil
}

// AddCallback adds a callback to be executed when the stream completes
func (s *Stream) AddCallback(callback func()) error {
	s.Execute(callback)
	return nil
}

// Convenience functions

// CreateStream creates a new stream with default settings
func CreateStream() (*Stream, error) {
	return GetManager().CreateStream(StreamDefault)
}

// CreateStreamNonBlocking creates a new non-blocking stream
func CreateStreamNonBlocking() (*Stream, error) {
	return GetManager().CreateStream(StreamNonBlocking)
}

// CreateHighPriorityStream creates a high-priority stream
func CreateHighPriorityStream() (*Stream, error) {
	return GetManager().CreateStreamWithPriority(StreamDefault, PriorityHigh)
}

// CreateLowPriorityStream creates a low-priority stream
func CreateLowPriorityStream() (*Stream, error) {
	return GetManager().CreateStreamWithPriority(StreamDefault, PriorityLow)
}

// GetDefaultStream returns the default stream
func GetDefaultStream() *Stream {
	return GetManager().GetDefaultStream()
}

// SynchronizeDevice waits for all streams to complete
func SynchronizeDevice() error {
	return GetManager().SynchronizeAll()
}

// Event creation and management

var (
	eventID  int
	eventsMu sync.Mutex
)

// CreateEvent creates a new CUDA event
func CreateEvent() *Event {
	eventsMu.Lock()
	defer eventsMu.Unlock()

	id := eventID
	eventID++

	return &Event{
		id:        id,
		completed: false,
	}
}

// CreateEventWithFlags creates a new event with flags for CUDA compatibility
func CreateEventWithFlags(flags int) *Event {
	// In simulation mode, we create a standard event and note the flags
	event := CreateEvent()
	// Flags like cudaEventDisableTiming, cudaEventBlockingSync are noted but not enforced in simulation
	fmt.Printf("Event created with flags: 0x%x (simulation mode)\n", flags)
	return event
}

// Performance measurement utilities

// MeasureKernelTime measures the execution time of a kernel
func MeasureKernelTime(stream *Stream, kernelFunc func()) (time.Duration, error) {
	start := CreateEvent()
	end := CreateEvent()

	// Record start event
	start.Record(stream.Stream)

	// Execute kernel
	stream.Execute(kernelFunc)

	// Record end event
	end.Record(stream.Stream)

	// Wait for completion
	if err := end.Synchronize(); err != nil {
		return 0, err
	}

	return start.ElapsedTime(end)
}

// Profile provides profiling information for streams
type Profile struct {
	StreamID       int
	OperationCount int
	TotalTime      time.Duration
	AverageTime    time.Duration
}

// GetStreamProfile returns profiling information for a stream
func GetStreamProfile(stream *Stream) *Profile {
	// Basic profile information (can be extended)
	return &Profile{
		StreamID:       0, // Would need to track stream IDs
		OperationCount: 0, // Would need to track operations
		TotalTime:      0,
		AverageTime:    0,
	}
}

// Multi-stream utilities

// ExecuteParallel executes multiple functions in parallel on different streams
func ExecuteParallel(functions ...func()) error {
	if len(functions) == 0 {
		return nil
	}

	streams := make([]*Stream, len(functions))
	var err error

	// Create streams
	for i := range functions {
		streams[i], err = CreateStream()
		if err != nil {
			return err
		}
	}

	// Execute functions in parallel
	for i, fn := range functions {
		streams[i].Execute(fn)
	}

	// Synchronize all streams
	for _, stream := range streams {
		if err := stream.Synchronize(); err != nil {
			return err
		}
	}

	// Clean up streams
	manager := GetManager()
	for _, stream := range streams {
		manager.DestroyStream(stream)
	}

	return nil
}

// ExecuteOnStream executes a function on a specific stream
func ExecuteOnStream(stream *Stream, fn func()) error {
	if stream == nil {
		stream = GetDefaultStream()
	}
	stream.Execute(fn)
	return nil
}
