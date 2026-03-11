// Package internal contains shared internal utilities for GoCUDA packages.
// This package is not intended for external use.
package internal

import (
	"fmt"
	"sync"
	"time"
	"unsafe"
)

// Stream represents a CUDA stream for asynchronous operations
type Stream struct {
	ptr        unsafe.Pointer
	deviceID   int
	priority   int
	flags      uint32
	tasks      chan func()
	isBlocking bool
	name       string
	mu         sync.Mutex
	cond       *sync.Cond
	pending    int
	closed     bool
}

var (
	defaultStream *Stream
	streamOnce    sync.Once
)

// GetDefaultStream returns the default CUDA stream
func GetDefaultStream() *Stream {
	streamOnce.Do(func() {
		defaultStream = newStream("DefaultStream")
	})
	return defaultStream
}

func newStream(name string) *Stream {
	stream := &Stream{
		deviceID:   0,
		priority:   0,
		flags:      0,
		tasks:      make(chan func(), 100),
		isBlocking: false,
		name:       name,
	}
	stream.cond = sync.NewCond(&stream.mu)
	go stream.processor()
	return stream
}

// processor handles stream task processing
func (s *Stream) processor() {
	for task := range s.tasks {
		func() {
			defer s.completeTask()
			task()
		}()
	}
}

// Execute executes a task on the stream
func (s *Stream) Execute(task func()) {
	s.mu.Lock()
	if s.closed {
		s.mu.Unlock()
		return
	}
	s.pending++
	s.mu.Unlock()

	s.tasks <- task
}

func (s *Stream) completeTask() {
	s.mu.Lock()
	s.pending--
	if s.pending == 0 {
		s.cond.Broadcast()
	}
	s.mu.Unlock()
}

// Synchronize waits for all stream operations to complete
func (s *Stream) Synchronize() error {
	s.mu.Lock()
	for s.pending > 0 {
		s.cond.Wait()
	}
	s.mu.Unlock()
	return nil
}

// Close closes the stream
func (s *Stream) Close() error {
	if err := s.Synchronize(); err != nil {
		return err
	}

	s.mu.Lock()
	if s.closed {
		s.mu.Unlock()
		return nil
	}
	s.closed = true
	close(s.tasks)
	s.mu.Unlock()
	return nil
}

// Error handling utilities

// CudaError represents a CUDA error
type CudaError struct {
	Code    int
	Message string
}

func (e *CudaError) Error() string {
	return fmt.Sprintf("CUDA error %d: %s", e.Code, e.Message)
}

// GetLastError returns the last CUDA error (simulation/wrapper)
func GetLastError() error {
	// For now this is a stub or needs split impl if real CUDA state tracking is needed
	// In the common case, we just return nil as checking last error is complex with Go/CGO
	return nil
}

// Profiling utilities

// Event represents a CUDA event for timing
type Event struct {
	time time.Time
}

// CreateEvent creates a new CUDA event
func CreateEvent() *Event {
	return &Event{time: time.Now()}
}

// Record records the event time
func (e *Event) Record() {
	e.time = time.Now()
}

// ElapsedTime returns the elapsed time between two events in milliseconds
func ElapsedTime(start, end *Event) float32 {
	return float32(end.time.Sub(start.time).Nanoseconds()) / 1000000.0
}

// NewStream creates a new CUDA stream
func NewStream() *Stream {
	return newStream("UserStream")
}
