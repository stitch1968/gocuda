package cuda

import (
	"context"
	"fmt"
	"sync"

	"github.com/stitch1968/gocuda/kernels"
)

// GoFunc represents a function that can be executed on GPU
type GoFunc func(ctx context.Context, args ...interface{}) error

// CudaWaitGroup is similar to sync.WaitGroup but for CUDA operations
type CudaWaitGroup struct {
	wg sync.WaitGroup
}

// Add adds delta to the WaitGroup counter
func (cwg *CudaWaitGroup) Add(delta int) {
	cwg.wg.Add(delta)
}

// Done decrements the WaitGroup counter by one
func (cwg *CudaWaitGroup) Done() {
	cwg.wg.Done()
}

// Wait blocks until the WaitGroup counter is zero
func (cwg *CudaWaitGroup) Wait() {
	cwg.wg.Wait()
}

// CudaChannel provides a channel-like interface for CUDA operations
type CudaChannel struct {
	ch     chan interface{}
	stream *Stream
}

// NewCudaChannel creates a new CUDA channel
func NewCudaChannel(buffer int) *CudaChannel {
	return &CudaChannel{
		ch:     make(chan interface{}, buffer),
		stream: GetDefaultStream(),
	}
}

// Send sends a value to the channel
func (cc *CudaChannel) Send(value interface{}) {
	cc.ch <- value
}

// Receive receives a value from the channel
func (cc *CudaChannel) Receive() interface{} {
	return <-cc.ch
}

// Close closes the channel
func (cc *CudaChannel) Close() {
	close(cc.ch)
}

// Go executes a function on the GPU similar to go routine
// Usage: cuda.Go(func(ctx context.Context, args ...interface{}) error { ... }, arg1, arg2, ...)
func Go(fn GoFunc, args ...interface{}) error {
	return GoWithStream(GetDefaultStream(), fn, args...)
}

// GoWithStream executes a function on the GPU with a specific stream
func GoWithStream(stream *Stream, fn GoFunc, args ...interface{}) error {
	kernel := &SimpleKernel{
		Name: "UserDefinedKernel",
		Func: func(kernelArgs ...interface{}) error {
			ctx := context.Background()
			return fn(ctx, kernelArgs...)
		},
	}

	// Execute kernel on stream using the proper stream interface
	stream.Execute(func() {
		// Simulate kernel execution
		kernel.Func(args...)
	})

	return nil
}

// GoWithDimensions executes a function on the GPU with custom grid and block dimensions
func GoWithDimensions(gridDim, blockDim kernels.Dim3, fn GoFunc, args ...interface{}) error {
	return GoWithDimensionsAndStream(GetDefaultStream(), gridDim, blockDim, fn, args...)
}

// GoWithDimensionsAndStream executes a function on the GPU with custom dimensions and stream
func GoWithDimensionsAndStream(stream *Stream, gridDim, blockDim kernels.Dim3, fn GoFunc, args ...interface{}) error {
	kernel := &SimpleKernel{
		Name: "UserDefinedKernel",
		Func: func(kernelArgs ...interface{}) error {
			ctx := context.Background()
			return fn(ctx, kernelArgs...)
		},
	}

	// Execute kernel on stream with dimensions
	stream.Execute(func() {
		// Simulate kernel execution with dimensions
		fmt.Printf("Executing kernel with grid %+v, block %+v\n", gridDim, blockDim)
		kernel.Func(args...)
	})

	return nil
}

// Synchronize waits for all CUDA operations to complete (similar to runtime.Gosched())
func Synchronize() error {
	// Create a new stream for synchronization to avoid WaitGroup reuse issues
	stream := GetDefaultStream()
	// Use a simple approach that doesn't cause WaitGroup conflicts
	stream.Execute(func() {
		// This is a no-op that forces synchronization
	})
	return nil
}

// SynchronizeStream waits for all operations in a specific stream to complete
func SynchronizeStream(stream *Stream) error {
	// Use a simple approach that doesn't cause WaitGroup conflicts
	stream.Execute(func() {
		// This is a no-op that forces synchronization
	})
	return nil
}

// ParallelFor executes a function in parallel across multiple GPU threads
func ParallelFor(start, end int, fn func(i int) error) error {
	numThreads := end - start
	if numThreads <= 0 {
		return nil
	}

	// For small ranges, execute sequentially to avoid overhead
	if numThreads <= 10 {
		for i := start; i < end; i++ {
			if err := fn(i); err != nil {
				return err
			}
		}
		return nil
	}

	// Calculate optimal grid and block dimensions
	blockSize := 256
	gridSize := (numThreads + blockSize - 1) / blockSize

	gridDim := kernels.Dim3{X: gridSize, Y: 1, Z: 1}
	blockDim := kernels.Dim3{X: blockSize, Y: 1, Z: 1}

	return GoWithDimensions(gridDim, blockDim, func(ctx context.Context, args ...interface{}) error {
		// Execute in parallel using goroutines for simulation
		var wg sync.WaitGroup
		errorChan := make(chan error, numThreads)

		for i := start; i < end; i++ {
			wg.Add(1)
			go func(index int) {
				defer wg.Done()
				if err := fn(index); err != nil {
					select {
					case errorChan <- err:
					default:
					}
				}
			}(i)
		}

		wg.Wait()
		close(errorChan)

		// Return the first error if any
		select {
		case err := <-errorChan:
			return err
		default:
			return nil
		}
	})
}

// Reduce performs a reduction operation across GPU threads
func Reduce(data []float64, operation func(a, b float64) float64) (float64, error) {
	if len(data) == 0 {
		return 0, fmt.Errorf("empty data slice")
	}

	if len(data) == 1 {
		return data[0], nil
	}

	// Use a sequential approach for correctness in simulation mode
	result := data[0]
	for i := 1; i < len(data); i++ {
		result = operation(result, data[i])
	}

	return result, nil
}

// Map applies a function to each element of a slice in parallel
func Map(input []float64, fn func(float64) float64) ([]float64, error) {
	if len(input) == 0 {
		return []float64{}, nil
	}

	output := make([]float64, len(input))
	var mu sync.Mutex

	err := ParallelFor(0, len(input), func(i int) error {
		result := fn(input[i])
		mu.Lock()
		output[i] = result
		mu.Unlock()
		return nil
	})

	return output, err
}

// Select provides a select-like operation for CUDA channels
func Select(cases ...SelectCase) (int, interface{}, bool) {
	// Simplified select implementation
	for i, c := range cases {
		select {
		case val := <-c.Channel.ch:
			if c.Handler != nil {
				c.Handler(val)
			}
			return i, val, true
		default:
			continue
		}
	}
	return -1, nil, false
}

// SelectCase represents a case in a select operation
type SelectCase struct {
	Channel *CudaChannel
	Handler func(interface{})
}
