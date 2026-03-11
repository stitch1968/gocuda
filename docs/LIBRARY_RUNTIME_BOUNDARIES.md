# Library Runtime Boundaries

GoCUDA exposes a broad surface area across CUDA ecosystem packages, but not every package is currently a direct binding to the vendor runtime.

## Current State

- `cuda`, `memory`, `streams`, and the tested kernel helpers are the primary runtime path for both simulation and CUDA-tagged builds.
- High-level allocations in CUDA mode use managed memory so host-side helper implementations can inspect and mutate buffers safely.
- Some production-ready packages in `libraries/` are implemented as validated deterministic Go execution or explicitly documented external-tool-backed flows rather than thin wrappers over the native CUDA libraries.

## Production-Ready Surface

- `libraries/curand.go`: CUDA-tagged builds execute through native cuRAND bindings; non-CUDA runtime mode falls back to the Go helper implementation.
- `libraries/cusparse.go`: CUDA-tagged builds execute through native cuSPARSE-backed entry points and CUDA-mode sparse helpers; non-CUDA runtime mode falls back to the Go compatibility implementation.
- `libraries/cusolver.go`: CUDA-tagged builds execute through native cuSOLVER bindings; non-CUDA runtime mode falls back to the Go helper implementation.
- `libraries/cufft.go`: CUDA-tagged builds execute through native cuFFT bindings; non-CUDA runtime mode falls back to the Go helper implementation.
- `libraries/cudamath.go`: math API behavior is implemented by deterministic Go execution with explicit host/device transfers and is production-ready across simulation and CUDA runtime modes.
- `libraries/amgx.go`: sparse-system setup, solve, multi-RHS solve, and matrix-update behavior are implemented by deterministic CSR-based execution with explicit host/device transfers and are production-ready across simulation and CUDA runtime modes.
- `libraries/cudnn.go`: convolution, pooling, activation, and batch normalization behavior is implemented by deterministic descriptor-driven execution with explicit host/device transfers and is production-ready across simulation and CUDA runtime modes.
- `libraries/thrust.go`: Thrust-style algorithms are implemented by deterministic Go execution with explicit host/device transfers and are production-ready across simulation and CUDA runtime modes.
- `libraries/nvjpeg.go`: JPEG encode/decode behavior is production-ready across simulation and CUDA runtime modes via deterministic Go execution, with native nvJPEG dispatch now wired for CUDA-tagged builds and deterministic fallback preserved elsewhere.
- `libraries/nvjpeg2000.go`: JPEG 2000 encode/decode and metadata extraction are implemented by `ffmpeg`/`ffprobe`-backed execution with explicit host/device transfers and are production-ready when those tools are installed. Native nvJPEG2000 dispatch is now wired for CUDA-tagged builds for a bounded 8-bit interleaved encode/decode/probe slice, with deterministic fallback preserved elsewhere.
- `libraries/cutensor.go`: tensor contraction and transformation behavior is implemented by deterministic descriptor-driven execution with explicit host/device transfers and is production-ready across simulation and CUDA runtime modes.
- `libraries/cutlass.go`: GEMM, convolution, sparse-dense multiplication, and triangular-update behavior are implemented by deterministic host/device-safe execution and are production-ready across simulation and CUDA runtime modes. Native cuBLAS-backed CUTLASS dispatch is now wired for CUDA-tagged builds for bounded float32/float64 row-major linear-combination GEMM, float32 `Rank2k` with `beta=0`, and float32 `Trmm`, with deterministic fallback preserved elsewhere.
- `libraries/cudss.go`: sparse direct-solver analysis, factorization, and solve behavior are implemented by deterministic host/device-safe execution and are production-ready across simulation and CUDA runtime modes.

## Remaining Native Parity Gaps

- `libraries/cudnn.go`: native CUDA-tagged cuDNN bindings now exist for CUDA-tagged builds; Windows requires generated `lib_mingw\libcudnn.a` and hardware validation before the native path can be considered validated.
- `libraries/thrust.go`: native CUDA-tagged Thrust-backed execution is still a roadmap item.
- `libraries/nvjpeg.go`: native CUDA-tagged nvJPEG bindings now exist for CUDA-tagged builds; Windows requires generated `lib_mingw\libnvjpeg.a` and hardware validation before the native path can be considered validated.
- `libraries/nvjpeg2000.go`: native CUDA-tagged nvJPEG2000 bindings now exist for CUDA-tagged builds for a bounded encode/decode/probe slice; Windows requires generated `lib_mingw\libnvjpeg2k.a` and hardware validation before the native path can be considered validated.
- `libraries/cudss.go`: native CUDA-tagged cuDSS bindings now exist for CUDA-tagged builds; Windows requires generated `lib_mingw\libcudss.a` and hardware validation before the native path can be considered validated.
- `libraries/amgx.go`: native CUDA-tagged AmgX bindings now exist for CUDA-tagged builds; Windows requires generated `lib_mingw\libamgxsh.a` and hardware validation before the native path can be considered validated.
- `libraries/cutensor.go`: native CUDA-tagged cuTENSOR bindings now exist for CUDA-tagged builds; Windows requires generated `lib_mingw\libcutensor.a` and hardware validation before the native path can be considered validated.
- `libraries/cutlass.go`: native CUDA-tagged CUTLASS parity now exists for a bounded cuBLAS-backed GEMM/`Rank2k`/`Trmm` slice; Windows requires generated `lib_mingw\libcublas.a` and hardware validation before the native path can be considered validated.

## Practical Guidance

- Use the production-ready surface today for supported workloads, tests, demos, and integration scenarios that match the documented execution model.
- Do not assume validated-go or external-tool-backed implementations reflect native CUDA library throughput unless the package is explicitly backed by native CUDA bindings.
- The current production-ready matrix no longer includes helper-backed high-level libraries.
- If a package needs native GPU-library execution, add a real CUDA-tagged implementation alongside the validated production path and keep the simulation path for non-CUDA builds.
