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

- [x] cuDNN: add native CUDA-tagged cuDNN bindings to complement the current deterministic production-ready implementation (done: native dispatch, Windows+non-Windows build wiring, CUDA-tagged native tests, and import-lib setup path are in place; the full Windows CUDA-tagged `go test -tags cuda ./...` sweep now passes on the primary validation machine)
- [x] Thrust: add native CUDA-tagged implementations to complement the current deterministic production-ready implementation (done: native context creation; CUDA-tagged runtime-backed dispatch for the full public Thrust surface including copy/fill/generate, sort, sort-by-key, reduce, unary/binary transform, inclusive/exclusive scan, find, count, unique, partition, copy-if, merge, set operations, and min/max element helpers; CUDA-tagged native tests; and broader Windows CUDA-tagged library validation are in place, with `go test -tags cuda ./libraries -run Thrust` and `go test -tags cuda ./libraries` passing on the primary validation machine)
- [x] nvJPEG: add native CUDA-tagged nvJPEG bindings to complement the current deterministic production-ready implementation (done: native dispatch, Windows+non-Windows build wiring, CUDA-tagged native tests, import-lib setup path, Windows GPU-backend encoder creation, and explicit 4:4:4 sampling-factor configuration are in place; the full Windows CUDA-tagged `go test -tags cuda ./...` sweep now passes on the primary validation machine)
- [x] nvJPEG2000: add native CUDA-tagged nvJPEG2000 bindings to complement the current `ffmpeg`/`ffprobe`-backed production-ready implementation (done: native handle/state lifecycle, native metadata probing, native 8-bit interleaved encode/decode dispatch, CUDA-tagged native tests, Windows import-lib setup path, and C-owned encode component metadata are in place; the full Windows CUDA-tagged `go test -tags cuda ./...` sweep now passes on the primary validation machine)
- [x] CUTLASS: add native CUDA kernels to complement the current deterministic production-ready implementation (done: native cuBLAS-backed float32/float64 row-major GEMM dispatch, batched GEMM via native handle reuse, float32 Rank2k with beta=0, float32 Trmm, float32 SpMM, and convolution forward/dgrad/wgrad coverage are in place alongside corrected row-major Rank2k mapping, CUDA-tagged native tests, and Windows import-lib setup path; `go test -tags cuda ./libraries -run CUTLASS` and `go test -tags cuda ./libraries` pass on the primary validation machine)
- [x] cuDSS: add native CUDA-tagged bindings to complement the current deterministic production-ready implementation (done: native dispatch, CUDA-tagged solve backend, CUDA-tagged native tests, Windows float32-to-float64 native-buffer marshaling, and CSR descriptor wiring aligned with NVIDIA cuDSS examples are in place; the full Windows CUDA-tagged `go test -tags cuda ./...` sweep now passes on the primary validation machine)
- [x] AmgX: add native CUDA-tagged bindings to complement the current deterministic production-ready implementation (done: native dispatch, CUDA-tagged setup/solve backend, CUDA-tagged native tests, Windows import-lib setup path, and existing-matrix coefficient-update reuse are in place; the full Windows CUDA-tagged `go test -tags cuda ./...` sweep now passes on the primary validation machine)
- [ ] cuTENSOR: add native CUDA-tagged bindings to complement the current deterministic production-ready implementation (in progress: native handle creation, native contraction/elementwise/reduction/permutation dispatch, cached native contraction plans, CUDA-tagged native tests, Windows import-lib setup path, deterministic fallback for unsupported native descriptor layouts, and mixed native/fallback plan execution fixes added; `go test -tags cuda ./libraries -run CuTensor` and `go test -tags cuda ./libraries` pass on the primary validation machine with the conservative packed-column-major native gate restored, but broader native layout coverage remains blocked because widening row-major packed descriptor support caused a native-path crash)

- [x] Add Windows-native verification for optional CUDA backlog headers, DLLs, and import libs so hardware-mode validation failures surface as actionable environment gaps instead of first-error compile failures
- [x] Add a machine-specific Windows SDK installation checklist keyed to the exact missing native backlog files reported by the verifier
- [x] Install the missing optional Windows vendor SDK headers, DLLs, and import libs for cuDNN, nvJPEG2000, cuTENSOR, and cuDSS on the primary validation machine
- [x] Regenerate MinGW import libs from the locally discovered DLL set and rerun CUDA-tagged validation after the missing vendor SDKs are installed (done: verifier and auto-helper now auto-discover the versioned vendor install layout and regenerate `libcudnn.a`, `libnvjpeg2k.a`, `libcudss.a`, `libcutensor.a`, and `libamgxsh.a`; current CUDA-tagged library validation now compiles on Windows and has moved on to runtime DLL-path and native-behavior failures in specific libraries)
- [x] Wire Windows CUDA-tagged cgo include discovery or a build-wrapper env path for non-toolkit vendor headers so cuDNN, nvJPEG2000, cuDSS, and cuTENSOR native builds do not require manual compiler include path injection
- [x] Install or build optional AmgX Windows headers, DLLs, and import libs if native AmgX validation is required on the primary validation machine (done: built upstream NVIDIA/AMGX locally on Windows, copied `include/amgx_c.h` and related headers into `C:\amgx\include`, copied `amgxsh.dll` into `C:\amgx\bin`, and generated `libamgxsh.a` for MinGW-linked validation)
- [x] Add a Windows CUDA runtime-path wrapper or equivalent documented test harness so CUDA-tagged validation picks up CUDA/vendor DLL directories without requiring an ad hoc shell `PATH` override
- [x] Clear the current Windows native-runtime blocker set in `go test -tags cuda ./libraries` (done: cuDSS float64 staging/CSR wiring, AmgX matrix-update reuse, cuTENSOR mixed plan fallback fixes, CUTLASS Rank2k row-major mapping correction, nvJPEG GPU-backend encoder setup, and nvJPEG2000 C-owned encode metadata fixes now allow the CUDA-tagged `./libraries` suite to pass on the primary Windows validation machine)
- [x] Validate the broader Windows CUDA-tagged repository sweep with `go test -tags cuda ./...` (done: repaired a malformed test block in `tests/cuda_test.go` and gated peer-copy tests on actual device count so the full CUDA-tagged repo sweep now passes on the primary single-GPU Windows validation machine)
- [x] Broaden the bounded native Thrust slice beyond context/copy/fill/generate once the Windows validation environment can exercise additional native algorithm coverage (done: the CUDA-tagged backend now also covers sort, sort-by-key, reduce, unary/binary transform, inclusive/exclusive scan, partition, copy-if, and set operations; validated with `go test -tags cuda ./libraries -run Thrust` and the broader `go test -tags cuda ./libraries` sweep on the primary Windows validation machine)

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
- [x] Guard zero-length transfer operations and keep same-device copies strict while introducing explicit peer-copy APIs for cross-device transfers

## Runtime Safety Backlog

- [x] Add explicit peer-copy support for cross-device transfers and multi-GPU memory ownership primitives
- [x] Expand context-bound execution across more high-level APIs so native CUDA mode consistently inherits device/thread affinity
- [x] Add CUDA-tagged validation covering the locked-thread device wrapper paths under real hardware mode

## Exit Criteria

- [x] Every library advertised as production-ready has a validated production path, whether native CUDA, deterministic validated-go, or external-tool-backed where explicitly documented
- [x] Simulation mode remains available for development without changing production behavior claims
- [x] The README, docs, and readiness matrix all agree on what is and is not production-ready
- [x] Supported production configurations pass automated build, test, and smoke-test gates
