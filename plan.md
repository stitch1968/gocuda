# Plan

- [x] Verify and install Windows CUDA build prerequisites
- [x] Fix default stream synchronization in simulation mode
- [x] Align Windows CUDA cgo include and linker configuration
- [x] Make CUDA-mode allocations host-accessible for CPU-simulated algorithms and tests
- [x] Add bounds-checked typed memory views in core algorithm and kernel paths
- [x] Fix advanced BFS to use 4-byte graph/distance storage consistently
- [x] Update top-level docs to reflect the current dual-mode runtime and Windows CUDA workflow
- [x] Validate simulation and CUDA-tagged test paths

## Remaining Work

- [x] Replace remaining direct `Ptr()` slicing in demos, tests, and helper APIs with checked memory views or explicit copy helpers
- [x] Migrate public examples and helper code toward `DefaultContext()` and `DefaultStream()` instead of the panic-based getters
- [x] Resolve remaining `errcheck` and unused-symbol warnings in production packages (`streams`, `libraries`, `performance`, `memory`, `cuda`)
- [x] Review library wrappers for places where CUDA mode still depends on CPU-side helper implementations and document or refactor those boundaries
- [x] Add focused tests for bounds-check failures and invalid memory-view requests
