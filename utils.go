package cuda

import (
	"fmt"
	"time"
)

// Profiler provides timing and performance metrics for CUDA operations
type Profiler struct {
	events map[string]time.Duration
	starts map[string]time.Time
}

// NewProfiler creates a new profiler
func NewProfiler() *Profiler {
	return &Profiler{
		events: make(map[string]time.Duration),
		starts: make(map[string]time.Time),
	}
}

// Start begins timing an event
func (p *Profiler) Start(name string) {
	p.starts[name] = time.Now()
}

// End stops timing an event
func (p *Profiler) End(name string) {
	if start, exists := p.starts[name]; exists {
		p.events[name] = time.Since(start)
		delete(p.starts, name)
	}
}

// GetDuration returns the duration of an event
func (p *Profiler) GetDuration(name string) time.Duration {
	return p.events[name]
}

// PrintSummary prints all recorded events
func (p *Profiler) PrintSummary() {
	fmt.Println("=== CUDA Profiler Summary ===")
	for name, duration := range p.events {
		fmt.Printf("%-20s: %v\n", name, duration)
	}
}

// Event represents a CUDA event for timing
type Event struct {
	recorded bool
	time     time.Time
}

// NewEvent creates a new CUDA event
func NewEvent() *Event {
	return &Event{}
}

// Record records the event
func (e *Event) Record() error {
	e.time = time.Now()
	e.recorded = true
	return nil
}

// ElapsedTime returns the time elapsed between two events
func ElapsedTime(start, end *Event) time.Duration {
	if !start.recorded || !end.recorded {
		return 0
	}
	return end.time.Sub(start.time)
}

// Timer provides a convenient way to time CUDA operations
type Timer struct {
	start *Event
	end   *Event
}

// NewTimer creates a new timer
func NewTimer() *Timer {
	return &Timer{
		start: NewEvent(),
		end:   NewEvent(),
	}
}

// Start starts the timer
func (t *Timer) Start() error {
	return t.start.Record()
}

// Stop stops the timer
func (t *Timer) Stop() error {
	return t.end.Record()
}

// Elapsed returns the elapsed time
func (t *Timer) Elapsed() time.Duration {
	return ElapsedTime(t.start, t.end)
}

// Benchmark runs a function multiple times and returns statistics
func Benchmark(name string, iterations int, fn func() error) {
	fmt.Printf("Benchmarking %s (%d iterations)...\n", name, iterations)

	var total time.Duration
	var min, max time.Duration
	first := true

	for i := 0; i < iterations; i++ {
		timer := NewTimer()
		timer.Start()

		err := fn()
		if err != nil {
			fmt.Printf("Error in iteration %d: %v\n", i, err)
			continue
		}

		timer.Stop()
		duration := timer.Elapsed()
		total += duration

		if first {
			min = duration
			max = duration
			first = false
		} else {
			if duration < min {
				min = duration
			}
			if duration > max {
				max = duration
			}
		}
	}

	avg := total / time.Duration(iterations)

	fmt.Printf("Results for %s:\n", name)
	fmt.Printf("  Average: %v\n", avg)
	fmt.Printf("  Min:     %v\n", min)
	fmt.Printf("  Max:     %v\n", max)
	fmt.Printf("  Total:   %v\n", total)
}

// MemoryPool manages a pool of reusable memory allocations
type MemoryPool struct {
	available map[int64][]*Memory
	used      map[*Memory]int64
}

// NewMemoryPool creates a new memory pool
func NewMemoryPool() *MemoryPool {
	return &MemoryPool{
		available: make(map[int64][]*Memory),
		used:      make(map[*Memory]int64),
	}
}

// Get gets a memory allocation from the pool
func (mp *MemoryPool) Get(size int64) (*Memory, error) {
	if mems, exists := mp.available[size]; exists && len(mems) > 0 {
		// Reuse existing allocation
		mem := mems[len(mems)-1]
		mp.available[size] = mems[:len(mems)-1]
		mp.used[mem] = size
		return mem, nil
	}

	// Allocate new memory
	mem, err := Malloc(size)
	if err != nil {
		return nil, err
	}

	mp.used[mem] = size
	return mem, nil
}

// Put returns a memory allocation to the pool
func (mp *MemoryPool) Put(mem *Memory) {
	if size, exists := mp.used[mem]; exists {
		delete(mp.used, mem)
		mp.available[size] = append(mp.available[size], mem)
	}
}

// Clear frees all memory in the pool
func (mp *MemoryPool) Clear() {
	for _, mems := range mp.available {
		for _, mem := range mems {
			mem.Free()
		}
	}

	for mem := range mp.used {
		mem.Free()
	}

	mp.available = make(map[int64][]*Memory)
	mp.used = make(map[*Memory]int64)
}

// WithContext executes a function with a CUDA context
func WithContext(deviceID int, fn func(*Context) error) error {
	ctx, err := NewContext(deviceID)
	if err != nil {
		return err
	}
	// Note: Context cleanup is handled by the context itself

	return fn(ctx)
}

// WithStream executes a function with a CUDA stream
func WithStream(fn func(*Stream) error) error {
	ctx := GetDefaultContext()
	stream, err := ctx.NewStream()
	if err != nil {
		return err
	}
	// Note: Stream cleanup is handled automatically

	return fn(stream)
}

// Launch provides a higher-level interface for launching kernels
func Launch(name string, gridDim, blockDim Dim3, kernel Kernel, args ...interface{}) error {
	timer := NewTimer()
	timer.Start()

	// Execute kernel using the kernel's Execute method
	fmt.Printf("Launching kernel '%s' with grid %+v, block %+v\n", name, gridDim, blockDim)

	// Use default stream for execution
	err := kernel.Execute(gridDim, blockDim, 0, GetDefaultStream(), args...)
	if err != nil {
		return fmt.Errorf("kernel execution failed: %v", err)
	}

	err = SynchronizeDevice()
	if err != nil {
		return err
	}

	timer.Stop()
	fmt.Printf("Kernel '%s' executed in %v\n", name, timer.Elapsed())
	return nil
}

// LaunchAsync launches a kernel asynchronously
func LaunchAsync(stream *Stream, gridDim, blockDim Dim3, kernel Kernel, args ...interface{}) error {
	// Execute kernel asynchronously using the provided stream
	fmt.Printf("Launching async kernel with grid %+v, block %+v\n", gridDim, blockDim)

	return kernel.Execute(gridDim, blockDim, 0, stream, args...)
}

// BatchExecute executes multiple operations in batches
func BatchExecute(operations []func() error, batchSize int) error {
	for i := 0; i < len(operations); i += batchSize {
		end := i + batchSize
		if end > len(operations) {
			end = len(operations)
		}

		// Execute batch
		for j := i; j < end; j++ {
			err := operations[j]()
			if err != nil {
				return fmt.Errorf("operation %d failed: %v", j, err)
			}
		}

		// Synchronize after each batch
		err := SynchronizeDevice()
		if err != nil {
			return fmt.Errorf("synchronization failed after batch %d: %v", i/batchSize, err)
		}
	}

	return nil
}

// Pipeline executes a series of operations in a pipeline
func Pipeline(stages []func(*Stream) error) error {
	ctx := GetDefaultContext()
	streams := make([]*Stream, len(stages))

	// Create streams for each stage
	for i := range streams {
		stream, err := ctx.NewStream()
		if err != nil {
			return err
		}
		// Note: Stream cleanup is handled automatically
		streams[i] = stream
	}

	// Execute stages
	for i, stage := range stages {
		err := stage(streams[i])
		if err != nil {
			return fmt.Errorf("stage %d failed: %v", i, err)
		}
	}

	// Synchronize all streams
	for _, stream := range streams {
		err := stream.Synchronize()
		if err != nil {
			return err
		}
	}

	return nil
}

// AsyncExecutor manages asynchronous execution of CUDA operations
type AsyncExecutor struct {
	tasks chan func() error
	wg    CudaWaitGroup
}

// NewAsyncExecutor creates a new async executor
func NewAsyncExecutor(workers int) *AsyncExecutor {
	ae := &AsyncExecutor{
		tasks: make(chan func() error, workers*2),
	}

	// Start worker goroutines
	for i := 0; i < workers; i++ {
		go ae.worker()
	}

	return ae
}

// Submit submits a task for asynchronous execution
func (ae *AsyncExecutor) Submit(task func() error) {
	ae.wg.Add(1)
	ae.tasks <- task
}

// Wait waits for all tasks to complete
func (ae *AsyncExecutor) Wait() {
	ae.wg.Wait()
}

// Close closes the executor
func (ae *AsyncExecutor) Close() {
	close(ae.tasks)
	ae.wg.Wait()
}

func (ae *AsyncExecutor) worker() {
	for task := range ae.tasks {
		err := task()
		if err != nil {
			fmt.Printf("Async task failed: %v\n", err)
		}
		ae.wg.Done()
	}
}
