# GoCUDA Copilot Instructions

## 🧠 Project Architecture & Mental Model

### Core Concept: Dual-Mode Runtime
This project implements a transparent dual-mode architecture using Go build tags. **Understanding this is critical for all code generation.**

1.  **Hardware Mode (`//go:build cuda`)**: 
    -   Interacts with real NVIDIA hardware via CGO.
    -   Links against `cudart` and `cuda` libraries.
    -   Key file: `cuda_runtime.go`
2.  **Simulation Mode (`//go:build !cuda`)**: 
    -   **Default behavior** (no GPU required).
    -   Simulates memory allocation, kernel delays, and device properties on CPU.
    -   Key files: `cuda_runtime_nocuda.go`, `libraries/*.go` (simulation logic).

### Component Boundaries
-   `cuda/`: Core driver/runtime API (Device, Context, Stream, Event). The entry point.
-   `libraries/`: High-level wrappers for CUDA ecosystem (cuFFT, cuDNN, etc.). **Note**: Current implementations often primarily provide simulation/mocking logic.
-   `memory/`: Memory management strategies (Unified, Pinned, Device).
-   `kernels/`: Kernel loading and execution helpers.
-   `demos/`: Runnable examples demonstrating API usage.

## 🛠 Developer Workflow

### Build Commands
*   **Simulation Build (Default)**: `go build ./...`
*   **Hardware Build**: `CGO_ENABLED=1 go build -tags cuda ./...`
*   **Build Scripts**: Prefer using `./build.sh [cuda|nocuda]` over raw `go` commands to ensure correct flags.

### Testing
*   **Run Tests (Simulated)**: `go test ./...`
*   **Run Tests (Hardware)**: `CGO_ENABLED=1 go test -tags cuda -v ./...`
*   **Verify Environment**: `verify_build.sh` runs a comprehensive suite of tests in implementation modes.

## 📝 Coding Standards & Patterns

### 1. Build Tag Discipline
When adding new core runtime functionality, **ALWAYS** implement both sides:
-   `file.go` (Interface/Types)
-   `file_impl.go` (Real CGO impl `//go:build cuda`)
-   `file_nocuda.go` (Simulation impl `//go:build !cuda`)

### 2. Error Handling
-   Wrappers should return idiomatic Go errors.
-   Check `cuda.Initialize()` at the start of `main()`.

### 3. Simulation Logic
-   When implementing "missing" features in `libraries/` or `nocuda` files, use `simulateKernelExecution` (or similar logic) to mimic computational cost (`time.Sleep`).
-   Do **not** leave empty functions; provide a "happy path" simulation.

### 4. Memory Management
-   Follow the pattern: `ptr, err := memory.Alloc(size)` -> `defer memory.Free(ptr)` (if available).
-   Demos often skip `Free` for brevity, but library code should be leak-free.

## 🔍 Key Files to Read
-   `build.sh`: The source of truth for build flags.
-   `cuda.go`: Public API definitions.
-   `cuda_runtime.go` vs `cuda_runtime_nocuda.go`: The reference for how the dual-mode split is implemented.
-   `libraries/cufft.go`: Example of the library wrapping pattern.
-   `demos/basic_demo.go`: Example of consumer API usage.

## ⚠️ Important Context
-   External dependencies: `github.com/stitch1968/gocuda/internal` is strict internal usage.
-   The `libraries/` packages currently function heavily as **simulators**. Do not assume they call CGO unless you verify the CGO bindings exist in those specific files.
