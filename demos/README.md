# Demo Examples

This folder contains demonstration programs showcasing different aspects of the GoCUDA library.

## Files

### Basic Demos
- **`basic_demo.go`** - Core GoCUDA features demonstration including:
  - Device information and initialization
  - `cuda.Go()` usage (Go routine-like GPU execution)
  - Vector addition on GPU
  - Parallel processing patterns
  - CUDA channels for GPU-host communication
  - Multiple streams for concurrent execution
  - Memory management

- **`memory_demo.go`** - CUDA-compatible memory management features:
  - Different memory types (Device, Host, Pinned, Unified)
  - 2D memory allocation with pitch alignment
  - Explicit memory copy operations with direction
  - 2D memory copy operations
  - Asynchronous memory operations
  - Unified memory management and prefetching
  - Memory alignment and optimization

### Advanced Examples
- **`examples/main.go`** - Comprehensive examples including:
  - Vector addition
  - Matrix multiplication
  - Parallel for loops
  - CUDA channels
  - Memory management
  - Multiple streams
  - CUDA wait groups
  - Map-reduce patterns

## Running the Demos

To run any demo, use:

```bash
# For build-only demos (with //go:build ignore)
go run -tags ignore basic_demo.go

# For regular Go programs
go run examples/main.go
```

## Features Demonstrated

### Core Features
- **Go routine-like API**: Use `cuda.Go()` to execute functions on GPU
- **Channels**: `cuda.CudaChannel` for GPU-host communication
- **Parallel patterns**: `cuda.ParallelFor()`, `cuda.Map()`, `cuda.Reduce()`
- **Synchronization**: `cuda.CudaWaitGroup` similar to `sync.WaitGroup`
- **Multiple streams**: Concurrent GPU execution
- **Memory management**: Automatic cleanup with finalizers

### CUDA Compatibility Features
- **Memory types**: Device, Host, Pinned, and Unified memory
- **2D memory**: Pitch-aligned allocations for optimal performance
- **Memory operations**: Explicit copy directions, async operations
- **Advanced memory**: Unified memory with advice and prefetching
- **Kernels**: Built-in vector, matrix, and convolution operations

### Performance Features
- **Stream management**: Multiple concurrent execution streams
- **Memory optimization**: 256-byte alignment for optimal performance
- **Asynchronous operations**: Non-blocking memory and compute operations
- **Profiling**: Built-in timing and performance measurement

## API Comparison

| Go | GoCUDA |
|----|--------|
| `go func() { ... }()` | `cuda.Go(func(ctx context.Context, args ...interface{}) error { ... }, args...)` |
| `make(chan T, size)` | `cuda.NewCudaChannel(size)` |
| `var wg sync.WaitGroup` | `var wg cuda.CudaWaitGroup` |
| `for i := range items { ... }` | `cuda.ParallelFor(0, len(items), func(i int) error { ... })` |

The GoCUDA library provides a familiar Go-like interface for CUDA programming while maintaining full compatibility with CUDA specifications and optimal GPU performance.
