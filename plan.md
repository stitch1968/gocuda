# Production Readiness Plan

This file is the execution tracker for moving GoCUDA from a dual-mode development/runtime experiment into something that can be deployed with clear production guarantees.

## Foundation

- [x] Stabilize the core CUDA/runtime, stream, memory, and checked-view paths
- [x] Align Windows CUDA build and linker configuration
- [x] Remove panic-only public access patterns where error-returning APIs exist
- [x] Document helper-backed library boundaries honestly in the README and runtime-boundary docs
- [x] Fail fast in CUDA mode when a high-level library is still helper-backed
- [x] Add tests for the helper-backed runtime policy
- [x] Add a machine-readable library readiness matrix in code

## Release Gating

- [x] Add a release-mode verification command that fails when any non-production library is enabled in a production build
- [x] Add CI coverage for `go test ./...` in simulation mode and `go test -tags cuda ./...` in hardware mode
- [x] Add build verification for required CUDA toolkit components per supported platform
- [x] Add a documented support matrix for OS, Go version, CUDA toolkit version, and GPU architecture

## Library Productionization

- [x] cuFFT: replace helper-backed FFT execution with native CUDA-tagged cuFFT bindings
- [x] cuDNN: replace helper-backed tensor/convolution/pooling/activation paths with deterministic production-ready descriptor-driven execution and explicit device-transfer support
- [x] cuRAND: replace helper-backed RNG with native CUDA-tagged cuRAND bindings
- [x] cuSPARSE: replace helper-backed sparse operations with native CUDA-tagged cuSPARSE bindings
- [x] cuSOLVER: replace helper-backed solver paths with native CUDA-tagged cuSOLVER bindings
- [x] AmgX: replace helper-backed AMG solver paths with deterministic production-ready CSR solving and explicit device-transfer support
- [x] CUDA Math API: replace helper-backed math wrappers with deterministic production-ready execution and explicit device-transfer support
- [x] Thrust: replace helper-backed algorithm wrappers with deterministic production-ready execution and explicit device-transfer support
- [x] nvJPEG: replace helper-backed JPEG wrappers with deterministic production-ready encode/decode and explicit device-transfer support
- [x] cuTENSOR: replace helper-backed tensor wrappers with deterministic production-ready descriptor-driven execution and explicit device-transfer support
- [x] CUTLASS: replace helper-backed template-kernel wrappers with deterministic production-ready linear algebra execution and explicit device-transfer support
- [x] cuDSS: replace helper-backed sparse direct solver wrappers with deterministic production-ready direct solving and explicit device-transfer support
- [x] nvJPEG2000: replace CPU helper path with production-ready `ffmpeg`/`ffprobe`-backed JPEG 2000 encode/decode and metadata extraction

## Native Parity Backlog

- [ ] cuDNN: add native CUDA-tagged cuDNN bindings to complement the current deterministic production-ready implementation
- [ ] Thrust: add native CUDA-tagged implementations to complement the current deterministic production-ready implementation
- [ ] nvJPEG: add native CUDA-tagged nvJPEG bindings to complement the current deterministic production-ready implementation
- [ ] nvJPEG2000: add native CUDA-tagged nvJPEG2000 bindings to complement the current `ffmpeg`/`ffprobe`-backed production-ready implementation
- [ ] CUTLASS: add native CUDA kernels to complement the current deterministic production-ready implementation
- [ ] cuDSS: add native CUDA-tagged bindings to complement the current deterministic production-ready implementation
- [ ] AmgX: add native CUDA-tagged bindings to complement the current deterministic production-ready implementation
- [ ] cuTENSOR: add native CUDA-tagged bindings to complement the current deterministic production-ready implementation

## Operational Hardening

- [x] Add concurrency and lifecycle tests for contexts, streams, and memory ownership under load
- [x] Add negative tests for unsupported devices, missing drivers, and partial toolkit installs
- [x] Add benchmark baselines for native-library paths to detect performance regressions
- [x] Add production diagnostics for runtime mode, selected device, toolkit version, and linked-library availability
- [x] Add smoke-test demos that validate the native path for each productionized library
- [x] Ensure user-created stream destruction closes worker goroutines while preserving the default stream
- [x] Expose explicit initialization-state APIs so callers can detect successful runtime setup without relying on panics or implicit side effects
- [x] Make memory freeing idempotent and race-safe against finalizer-triggered cleanup paths
- [x] Add device-aware internal CUDA wrapper execution and a context-bound execution helper for locked-thread device selection
- [x] Guard zero-length transfer operations and reject implicit cross-device copies until explicit peer-copy support exists

## Runtime Safety Backlog

- [ ] Add explicit peer-copy support for cross-device transfers and multi-GPU memory ownership
- [ ] Expand context-bound execution across more high-level APIs so native CUDA mode consistently inherits device/thread affinity
- [ ] Add CUDA-tagged validation covering the locked-thread device wrapper paths under real hardware mode

## Exit Criteria

- [x] Every library advertised as production-ready has a validated production path, whether native CUDA, deterministic validated-go, or external-tool-backed where explicitly documented
- [x] Simulation mode remains available for development without changing production behavior claims
- [x] The README, docs, and readiness matrix all agree on what is and is not production-ready
- [x] Supported production configurations pass automated build, test, and smoke-test gates
