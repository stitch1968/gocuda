# Library Runtime Boundaries

GoCUDA's CUDA-tagged build combines native NVIDIA library calls with host-side
Go and external-tool implementations. A CUDA build or a library name alone does
not guarantee that a particular operation runs on the GPU.

## Core runtime

- Legacy `TypeDevice` and unified allocations use CUDA managed memory;
  `TypeDeviceOnly` uses `cudaMalloc`, and pinned host allocations use
  `cudaMallocHost`. Simulation uses Go memory, but device-only allocations still
  require copy helpers and reject host views. Allocations require explicit
  `Free` after outstanding work completes, and a view cannot outlive its allocation.
- `Context.Run` binds a callback to its device on a locked OS thread and
  restores the prior selection. `SetDevice` and `SetCudaDevice` affect the
  calling OS thread only: callers must use `runtime.LockOSThread` until their
  CUDA calls finish. Library initialization does not silently select a device.
  Allocations using an explicit host stream use that stream's device; other
  allocations use the current OS-thread device.
- `DeviceSynchronize` waits for the default host queue and the current CUDA
  device. `DeviceReset` returns an unsupported error in hardware mode rather
  than invalidating managed handles. Device-wide synchronization alone does
  not release buffer retentions held by native streams; synchronize or close
  those streams separately.
- `ForceFallbackToSimulation` changes high-level runtime dispatch, not every
  lower-level CUDA allocation or native API. Device discovery is cached; do not
  switch fallback mode after initialization. For an entirely CPU-only process,
  use the default build without the `cuda` tag.
- Legacy `Copy*WithStream` helpers are blocking copies and do not order work
  on the supplied host queue. Use `NativeStream.CopyAsync` with pinned buffers
  for actual CUDA stream ordering.
- Memory pools accept only unretained legacy `TypeDevice` allocations on their
  selected device. `Put` transfers ownership and returns an error on duplicate,
  retained, freed or incompatible buffers; the caller must not use a buffer
  after `Put`. `Close` frees cached allocations but not checked-out ones.
- `Go` and `GoWithStream` schedule host callbacks. Callback errors are returned
  by the next stream synchronization. Grid and block dimensions on these APIs
  do not launch GPU threads.
- `kernels.LoadPTX` loads a CUDA driver module and launches named kernels on a
  `NativeStream`. It accepts device/managed buffer pointers and 32- or 64-bit
  scalar arguments. Parameter metadata must be available from the CUDA driver;
  unsupported parameter sizes return errors. Launch validation checks argument
  counts and sizes, not PTX types or kernel memory accesses. Launch buffers
  remain retained until the stream is synchronized; closing a module
  synchronizes streams used by its direct launches.
  See the [native kernel test](../tests/native_kernel_cuda_test.go) for a
  complete PTX launch and pinned-buffer transfer example.
- `kernels.CompileCUDA` invokes a local `nvcc` to compile source to PTX with
  explicit virtual architecture, include paths and definitions. It supports
  cancellation and returns compiler diagnostics; it does not use NVRTC.
  Windows compilation requires an MSVC developer environment. See the
  [compiler integration test](../tests/compiler_cuda_test.go).
- `kernels.NativeOps` provides native float32 vector addition, SAXPY, sum,
  2D cross-correlation and a baseline row-major matrix multiply on compute
  capability 7.0 or newer. Sum uses global atomics, so accumulation order and
  floating-point rounding are nondeterministic. The matrix and reduction
  kernels are not optimized library replacements. These operations require
  native streams and device or managed buffers; they do not fall back to the
  CPU. See the [native operations test](../tests/native_ops_cuda_test.go) and
  [reduction and convolution test](../tests/native_ops_extra_cuda_test.go).
- `libraries.NativeBLAS` provides explicit cuBLAS-backed float32 and float64
  row-major GEMM on a native stream, with alpha/beta scaling, transpose and
  row-stride options. It does not fall back to Go. Closing the handle synchronizes
  its stream.
- Built-in kernels execute host-side algorithms. Existing root-level graphs
  schedule host work and do not track asynchronous work nested inside a node.
  `kernels.NativeGraph` separately supports CUDA driver kernel, memcpy and
  memset nodes, dependencies and replay. It retains referenced modules and
  buffers until   closed; `Free` returns an error while a graph retains a buffer. Closing a
  graph synchronizes the streams used for its launches before releasing its
  resources. Native graph host nodes and updates are not yet exposed.
  See the [native graph test](../tests/native_graph_cuda_test.go) and
  [graph memory-node test](../tests/native_graph_memory_cuda_test.go).
- Existing `Stream` and `Event` APIs model host queues and timing. The separate
  `streams.NativeStream` and `streams.NativeEvent` APIs use CUDA streams and
  GPU events; they return an explicit error when CUDA is unavailable.
- Native async copies accept CUDA device/managed and pinned host allocations,
  not pageable Go memory. Copies, native kernel launches, native-stream cuFFT
  and cuBLAS GEMM retain their buffers until the stream is synchronized or
  closed successfully; event completion alone does not release those retentions.
  Do not read or write buffers before the associated operation completes.
  The [native stream test](../tests/native_stream_cuda_test.go) demonstrates
  cross-stream event ordering and cuFFT execution.

## Library wrappers

- cuFFT, cuRAND, cuSOLVER and cuSPARSE have selected native CUDA paths. The
  cuFFT complex and real single-precision execution APIs support out-of-place
  transforms; real transforms use packed Hermitian layout and inverse transforms
  are unnormalized. Real in-place FFTs are not currently supported. Native
  cuFFT plans can attach to `streams.NativeStream`; their input and output
  buffers and the stream must belong to the plan's device. Changing or
  destroying a plan synchronizes its prior native stream. Without an attached
  native stream, cuFFT calls synchronize the CUDA default stream before returning.
- `CreateNativeFFTContextOnDevice`, `CreateNativeRandomGeneratorOnDevice` and
  `CreateNativeSolverContextOnDevice` require native CUDA and bind their
  respective objects to a chosen device.
  Their `Backend` methods distinguish native CUDA from Go simulation; other
  wrappers can still use fallback paths, so these checks are not a global
  strict-native policy.
- cuSOLVER simulation computes column-major float32 LU, solve, upper Cholesky,
  QR, SVD, symmetric eigenvalues and pseudoinverse results. SVD and eigenvalue
  decompositions use Gonum and compute internally in float64. Eigenvalues use
  the upper triangle, matching the native symmetric solver; eigenvalues are
  returned as complex64 with zero imaginary part and eigenvectors as
  column-major float32. Wide native SVD and pseudoinverse transpose buffers
  on the host before/after the cuSOLVER decomposition; they are not fully
  GPU-resident. Native solver context calls are serialized and bound to the
  context's device.
- `QRInfo.Tau`, `LUInfo.Pivots` and the `SVDInfo` output accessors expose
  allocations owned by their result object. Call `Destroy` on the result; do
  not free these allocations separately.
- cuSPARSE CSR and COO SpMV and sparse-to-dense conversion, CSR triangular
  solve, CSR sparse matrix multiplication, and dense-to-CSR conversion use
  cuSPARSE in CUDA mode and synchronize the default stream before returning.
  Sparse inputs undergo host-side index validation; COO data and duplicate
  CSR entries are canonicalized on the host before execution because
  cuSPARSE requires COO row ordering and does not guarantee duplicate-index
  results. CSR multiplication canonicalizes both inputs on the host.
  Dense-to-CSR accepts device or managed input on the context device and
  limits the dense element count to the 32-bit sparse index capacity.
  Mixed-format sparse multiplication, sparse LU, COO triangular solve, SpMM
  and dense-to-COO conversion currently execute on the host in both modes.
  Sparse LU does not pivot and returns an error when a nonzero pivot is
  unavailable. `CreateNativeSparseContextOnDevice` creates a strict
  native-only context: the native operations listed above are supported and
  host-only operations return `ErrNativeSparseOperationUnsupported`.
  `SpMVBackend`, `SpSVBackend`, `SpGEMMBackend`, `SparseToDenseBackend` and
  `DenseToSparseBackend` report operation backends. Native
  operations reject output buffers that alias the input vector or sparse
  matrix buffers. Triangular solve requires explicit finite, nonzero diagonal
  entries and infers lower or upper fill from the matrix.
- cuRAND simulation generates uniform, normal, log-normal and Poisson values
  using Go's pseudorandom generator, not the vendor's exact sequences. It
  rejects quasi-random generators and Poisson lambda above 1024. `SetSeed`
  returns an error and resets the pseudorandom sequence offset. Normal and
  log-normal generation requires an even sample count.
- Native cuRAND generators can attach to `streams.NativeStream` with
  `SetStream`. Stream execution retains output buffers until synchronization;
  default-stream generation completes before returning. Changing the stream,
  reseeding or destroying the generator synchronizes its outstanding work.
- cuDNN, cuDSS, AmgX, cuTENSOR, nvJPEG and nvJPEG2000 include selected native
  paths but can also use host or external-tool fallbacks.
- CUTLASS-named native linear algebra paths use cuBLAS, not CUTLASS kernels.
  Other parameter combinations may fall back to Go.
- The CUDA Math API and portions of Thrust-style operations use Go execution.

The readiness matrix is library-wide and does not provide per-operation backend
selection. Apart from explicitly documented native-only APIs, production
validation and a library-wide strict-native policy remain outstanding; inspect
the operation implementation before relying on GPU execution or measuring its
performance.
