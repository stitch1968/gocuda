# GoCUDA Support Matrix

This document defines the currently supported environments and the parts of the surface that are suitable for production use.

## Runtime Modes

| Mode | Status | Notes |
| --- | --- | --- |
| Simulation mode (`go build ./...`) | Supported for development and testing | No GPU required. Intended for API development, unit tests, and functional fallback coverage. |
| CUDA hardware mode (`go build -tags cuda ./...`) | Supported on validated configurations below | Requires NVIDIA CUDA Toolkit, CGO, and compatible runtime libraries. |

## Platform Support

| Platform | Simulation Mode | CUDA Hardware Mode | Notes |
| --- | --- | --- | --- |
| Windows x64 | Supported | Supported with documented native-library caveats | Requires CUDA Toolkit, MSVC-compatible build environment, and repo import libs for `cuda`/`cudart`. Native cuDNN wiring is present, but successful Windows CUDA builds depend on generating `lib_mingw\libcudnn.a` from the installed cuDNN DLL and validating on hardware. |
| Linux x86_64 | Supported | Supported | Requires CUDA Toolkit with `lib64` runtime libraries available. |
| macOS | Supported for simulation only | Not supported | NVIDIA CUDA hardware mode is not supported by this repository on macOS. |

## Toolchain Support

| Component | Supported Range | Notes |
| --- | --- | --- |
| Go | 1.26+ | Module currently targets Go 1.26 in `go.mod`. |
| CUDA Toolkit | 12.x to 13.x target | Build paths and verification logic assume standard CUDA Toolkit layouts. |
| GPU Architecture | NVIDIA CUDA-capable GPUs supported by installed toolkit | Real coverage depends on vendor library availability and toolkit compatibility. |

## Production-Ready Library Surface

| Surface | Status | Notes |
| --- | --- | --- |
| Core runtime, memory, streams, checked views | Production-focused | Primary runtime path for both simulation and CUDA-tagged builds. |
| cuFFT | Production-ready in CUDA mode | Native cuFFT backend in CUDA builds with simulation fallback when CUDA runtime mode is unavailable. |
| cuRAND | Production-ready in CUDA mode | Native cuRAND backend in CUDA builds with simulation fallback when CUDA runtime mode is unavailable. |
| cuSOLVER | Production-ready in CUDA mode | Native cuSOLVER backend in CUDA builds with simulation fallback when CUDA runtime mode is unavailable. |
| cuSPARSE | Production-ready in CUDA mode | Native cuSPARSE context and execution paths in CUDA builds with simulation fallback when CUDA runtime mode is unavailable. |
| cuDNN | Production-ready in simulation and CUDA modes | Deterministic descriptor-driven tensor, convolution, pooling, activation, and batch-normalization execution with explicit device-transfer support. Native cuDNN bindings are wired for CUDA-tagged builds; Windows requires generated `lib_mingw\libcudnn.a` and hardware validation. |
| AmgX | Production-ready in simulation and CUDA modes | Deterministic CSR-based setup, solve, multi-RHS solve, and matrix-update execution. |
| CUDA Math API | Production-ready in simulation and CUDA modes | Validated Go execution with explicit device-transfer support. |
| Thrust | Production-ready in simulation and CUDA modes | Validated Go algorithm execution with explicit device-transfer support. |
| nvJPEG | Production-ready in simulation and CUDA modes | Deterministic JPEG encode/decode with explicit device-transfer support. Native nvJPEG bindings are wired for CUDA-tagged builds; Windows requires generated `lib_mingw\libnvjpeg.a` and hardware validation. |
| nvJPEG2000 | Production-ready in simulation and CUDA modes | Real JPEG 2000 encode/decode and metadata extraction backed by `ffmpeg`/`ffprobe`. |
| cuTENSOR | Production-ready in simulation and CUDA modes | Validated descriptor-driven tensor execution with explicit device-transfer support. |
| CUTLASS | Production-ready in simulation and CUDA modes | Validated host/device-safe linear algebra execution. |
| cuDSS | Production-ready in simulation and CUDA modes | Validated host/device-safe sparse direct solving. |

## Remaining Native Gaps

The production verification command now passes, but some wrappers still rely on validated Go or external-tool execution rather than native CUDA vendor bindings:

- `cuTENSOR` native CUDA bindings
- `CUTLASS` native CUDA kernels
- `cuDNN` hardware validation on a Windows CUDA runner after import-library generation
- `cuDSS` native CUDA bindings
- `AmgX` native CUDA bindings
- `nvJPEG` hardware validation on a Windows CUDA runner after import-library generation
- `nvJPEG2000` native CUDA bindings (current implementation uses `ffmpeg`/`ffprobe`)

## Verification Commands

Use these commands before treating a build as production-capable:

```bash
go test ./...
go run ./cmd/gocuda-verify-toolkit
go run ./cmd/gocuda-verify-production
```

For CUDA hardware validation on a supported machine:

```bash
go test -tags cuda ./...
```
