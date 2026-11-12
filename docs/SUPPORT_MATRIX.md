# GoCUDA Support Matrix

GoCUDA is experimental. A passing test suite does not establish production
readiness or complete CUDA API coverage.

## Runtime modes

| Mode | Purpose | Limitations |
| --- | --- | --- |
| Default simulation build | Development and testing without a GPU | Host execution; not all CUDA semantics are modeled. |
| CUDA-tagged build | Selected NVIDIA runtime and library integrations | Requires CGO, CUDA and vendor libraries; some operations still run on the host or use external tools. |

CUDA-tagged builds are intended for supported Windows and Linux CUDA
configurations. macOS supports simulation only. The module targets Go 1.26 or
later; CUDA toolkit and driver compatibility depends on the installed libraries.

## Implementation boundaries

| Surface | Current behavior |
| --- | --- |
| Memory | Legacy `TypeDevice` and unified allocations use managed memory in CUDA mode. `TypeDeviceOnly` uses device-only memory; pinned allocations use page-locked host memory. Simulation uses Go memory while enforcing the device-only view restriction. Call `Free` after outstanding work completes. |
| Streams and events | Existing stream/event APIs model host queues and timing. Separate `NativeStream` and `NativeEvent` APIs provide CUDA streams, GPU events and async copies for pinned/device buffers in CUDA builds. |
| Kernels and graphs | PTX modules and `NativeOps` launch selected GPU kernels; explicit native graphs support kernel, copy and memset nodes with dependencies and replay. Existing built-in kernel and host graph APIs remain host-side. |
| cuFFT, cuRAND, cuSOLVER, cuSPARSE | Selected native paths are available in CUDA builds. Simulation and individual operations have separate limitations. |
| Other library wrappers | `NativeBLAS` explicitly uses cuBLAS for float32/float64 GEMM, including transpose and row-stride options. Other wrappers select native paths, Go fallbacks, or external tools depending on operation and configuration. CUTLASS-named native GEMM paths also use cuBLAS. |

The programmatic readiness matrix reports library-wide production readiness as
unverified until operation-level coverage and validation are established.
Consult [library runtime boundaries](LIBRARY_RUNTIME_BOUNDARIES.md) before
assuming a call executes on a GPU.

## Verification

```bash
go test ./...
go test -race ./...
go run ./cmd/gocuda-verify-toolkit
go run ./cmd/gocuda-verify-production
```

The production verifier currently reports the outstanding readiness blockers.
For CUDA hardware, run `go test -tags cuda ./...` with the required vendor DLLs
or shared libraries available. On Windows,
`.\run_hardware_mode.ps1 -Command go test -tags cuda ./...` configures runtime
DLL paths before testing.

The GitHub Actions workflow runs simulation tests on Linux and Windows and a
Linux race job. Its CUDA job is manually enabled and requires a self-hosted
Linux runner labelled `cuda` with the toolkit, GPU, and vendor libraries already
installed. Its required nvcc integration test needs compute capability 7.5 or
newer. The native smoke command fails if CUDA falls back to simulation;
individual optional-library tests may still skip unavailable capabilities.
An independent opt-in `multi-gpu` job requires a self-hosted runner labelled
`multi-gpu` with at least two peer-capable GPUs; missing devices fail that job
instead of silently skipping the cross-device tests.
The CUDA job also runs Compute Sanitizer memcheck and initcheck on the native
regression tests. API-error reporting is disabled for those checks because the tests and
kernel metadata discovery intentionally probe CUDA error paths; the ordinary
test suite still checks returned API errors.
