# Library Runtime Boundaries

GoCUDA exposes a broad surface area across CUDA ecosystem packages, but not every package is currently a direct binding to the vendor runtime.

## Current State

- `cuda`, `memory`, `streams`, and the tested kernel helpers are the primary runtime path for both simulation and CUDA-tagged builds.
- High-level allocations in CUDA mode use managed memory so host-side helper implementations can inspect and mutate buffers safely.
- Some packages in `libraries/` are still functional compatibility layers implemented in Go rather than thin wrappers over the native CUDA libraries.

## Known Helper-Backed Packages

- `libraries/cufft.go`: FFT behavior is implemented with Go-side DFT helpers operating over managed or host-visible buffers.
- `libraries/cudnn.go`: convolution, pooling, activation, and batch normalization paths are simulated with Go-side tensor transforms.

## Practical Guidance

- Use these packages today for API experimentation, tests, and demos.
- Do not assume the current `libraries/` implementations reflect production CUDA library throughput or exact numerical behavior.
- If a package needs production-grade GPU execution, add a real CUDA-tagged implementation alongside the existing helper-backed path and keep the simulation path for non-CUDA builds.