# CUDA Detection and Dual-Mode Operation Guide

## Overview

GoCUDA now features **automatic CUDA detection** with intelligent fallback to CPU simulation. This allows your code to run seamlessly on both GPU-equipped systems and systems without CUDA, using the same API.

## How It Works

### Automatic Detection Process

1. **Build Time**: Uses Go build tags to conditionally compile CUDA support
2. **Runtime**: Automatically detects CUDA availability and device count
3. **Fallback**: Gracefully falls back to CPU simulation when CUDA is unavailable
4. **Unified API**: Same function calls work in both modes

### Build Modes

#### CPU Simulation Mode (Default)
```bash
# No CUDA required - works on any system
go build ./...
```

#### Real CUDA Mode
```bash
# Requires NVIDIA CUDA Toolkit
go build -tags cuda ./...
```

## Using the Detection API

### Basic Detection

```go
package main

import (
    "fmt"
    "github.com/stitch1968/gocuda"
)

func main() {
    // Initialize runtime (detects CUDA automatically)
    cuda.Initialize()
    
    // Check CUDA availability
    if cuda.IsCudaAvailable() {
        fmt.Println("🚀 Real CUDA GPU detected!")
    } else {
        fmt.Println("💻 Using CPU simulation")
    }
    
    // Check execution mode
    if cuda.ShouldUseCuda() {
        fmt.Println("Using GPU execution")
    } else {
        fmt.Println("Using CPU simulation")
    }
    
    // Print detailed system information
    cuda.PrintCudaInfo()
}
```

### Runtime Information

```go
// Get CUDA runtime instance
runtime := cuda.GetCudaRuntime()

// Check properties
fmt.Printf("CUDA Available: %t\n", runtime.Available)
fmt.Printf("Device Count: %d\n", runtime.DeviceCount)
fmt.Printf("Runtime Version: %d\n", runtime.RuntimeVersion)
fmt.Printf("Driver Version: %d\n", runtime.DriverVersion)

// Get device information
for i := 0; i < cuda.GetCudaDeviceCount(); i++ {
    device, err := cuda.GetCudaDevice(i)
    if err == nil {
        fmt.Printf("Device %d: %s\n", i, device.Name)
        fmt.Printf("  Compute: %d.%d\n", 
            device.Properties.Major, device.Properties.Minor)
        fmt.Printf("  Memory: %d MB\n", 
            device.Properties.TotalGlobalMem/(1024*1024))
    }
}
```

### Memory Management

Memory operations automatically use the appropriate backend:

```go
// This works in both CUDA and simulation modes
mem, err := cuda.Malloc(1024 * 1024) // 1MB
if err != nil {
    log.Fatal(err)
}
defer mem.Free()

// Memory types work consistently
deviceMem, _ := cuda.MallocWithTypeAndStream(stream, size, cuda.MemoryTypeDevice)
hostMem, _ := cuda.MallocWithTypeAndStream(stream, size, cuda.MemoryTypeHost)
unifiedMem, _ := cuda.MallocManaged(size)
```

### Kernel Execution

Kernels execute on GPU when available, CPU when not:

```go
// Vector addition example
func vectorAdd(a, b, c *cuda.Memory, size int) error {
    kernel := &cuda.SimpleKernel{
        Name: "VectorAdd",
        Func: func(args ...interface{}) error {
            // Implementation works in both modes
            if cuda.ShouldUseCuda() {
                // Real GPU kernel would be PTX/SASS
                // For demo, we simulate the operation
            }
            
            // CPU simulation implementation
            if a.Data() != nil && b.Data() != nil && c.Data() != nil {
                for i := 0; i < size; i++ {
                    c.Data()[i] = a.Data()[i] + b.Data()[i]
                }
            }
            return nil
        },
    }
    
    gridDim := cuda.Dim3{X: (size + 255) / 256, Y: 1, Z: 1}
    blockDim := cuda.Dim3{X: 256, Y: 1, Z: 1}
    
    stream := cuda.GetDefaultStream()
    return stream.Execute(kernel, gridDim, blockDim, 0)
}
```

## Build System

### Using Build Scripts

**Windows:**
```cmd
# CPU simulation only
.\build.bat nocuda

# With CUDA support (requires CUDA toolkit)
.\build.bat cuda
```

**Linux/macOS:**
```bash
# CPU simulation only
./build.sh nocuda

# With CUDA support (requires CUDA toolkit)
./build.sh cuda
```

### Manual Building

**For CPU simulation:**
```bash
go build ./...
go test ./...
```

**For real CUDA:**
```bash
# Set CGO_ENABLED and build with cuda tag
CGO_ENABLED=1 go build -tags cuda ./...
CGO_ENABLED=1 go test -tags cuda ./...
```

## Environment Requirements

### CPU Simulation Mode
- ✅ Go 1.21 or later
- ✅ Any operating system
- ✅ No GPU required
- ✅ No additional dependencies

### Real CUDA Mode
- ✅ Go 1.21 or later
- ✅ NVIDIA GPU with CUDA support
- ✅ NVIDIA CUDA Toolkit (11.0 or later)
- ✅ Appropriate GPU drivers
- ✅ CGO enabled

## Force Fallback Mode

You can force CPU simulation even when CUDA is available:

```go
// Force simulation mode (useful for testing)
cuda.ForceFallbackToSimulation(true)

// Check mode
if cuda.ShouldUseCuda() {
    fmt.Println("Using GPU")
} else {
    fmt.Println("Using CPU simulation") // Will print this
}
```

## Performance Considerations

### CPU Simulation Mode
- ✅ Great for development and testing
- ✅ Consistent behavior across platforms
- ✅ No GPU hardware required
- ⚠️ Limited by CPU performance
- ⚠️ No true parallel GPU execution

### Real CUDA Mode
- 🚀 Maximum performance on compatible hardware
- 🚀 True GPU parallel execution
- 🚀 Access to GPU memory hierarchy
- ⚠️ Requires specific hardware/software setup
- ⚠️ Platform-dependent

## Error Handling

The library gracefully handles missing CUDA:

```go
func main() {
    // Initialize always succeeds
    err := cuda.Initialize()
    if err != nil {
        log.Fatal("Should never happen") // Initialize doesn't fail
    }
    
    // Check what mode we're in
    runtime := cuda.GetCudaRuntime()
    if !runtime.Available {
        fmt.Println("CUDA not available, using simulation")
        // Your code continues to work normally
    }
    
    // Memory allocation automatically adapts
    mem, err := cuda.Malloc(1024)
    if err != nil {
        log.Printf("Memory allocation failed: %v", err)
        // Handle as appropriate for your application
    }
}
```

## Migration Guide

If you have existing GoCUDA code, no changes are required! The new detection system is fully backward compatible:

```go
// Old code - still works exactly the same
cuda.Initialize()
devices, _ := cuda.GetDevices()
mem, _ := cuda.Malloc(1024)
// etc.
```

The only difference is that it now automatically detects and uses real GPU hardware when available.

## Development Workflow

1. **Develop** using CPU simulation mode (fast, no special hardware)
2. **Test** on systems with different CUDA configurations
3. **Deploy** with CUDA support for production performance

```bash
# Development cycle
go build ./...           # Fast simulation build
go test ./...           # Test in simulation

# Production build
go build -tags cuda ./... # Real CUDA support
```

This allows seamless development on laptops/desktops without GPUs while deploying with full GPU acceleration in production.
