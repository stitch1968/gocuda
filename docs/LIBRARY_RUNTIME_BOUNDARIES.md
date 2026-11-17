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
  row-stride options. `SGEMMStridedBatched` and `DGEMMStridedBatched` enqueue
  uniformly shaped batches with positive element strides; overlapping input
  batches are supported, but zero-stride broadcasting is not. Output batch
  strides must cover the complete output matrix storage span. These calls
  retain their buffers until stream synchronization. They do not fall back to
  Go. Closing the handle synchronizes its stream.
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
  cuFFT complex and real single- and double-precision execution APIs support
  out-of-place transforms; complex transforms also support in-place execution.
  Real transforms use packed Hermitian layout and inverse transforms are
  unnormalized. Native inverse real transforms may overwrite their input.
  Real in-place FFTs are not currently supported. `CreatePlanMany` supports
  contiguous batches of one- to three-dimensional transforms, but not
  advanced embedded or strided layouts. Native
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
- cuSPARSE CSR and COO SpMV, sparse-dense multiplication, triangular solve, dense/sparse conversion and
  sparse matrix multiplication use
  cuSPARSE in CUDA mode and synchronize before returning.
  `SparseContext.SetStream` attaches a native stream on the same device, or
  selects the default stream with nil. Operations wait for earlier work on
  the attached stream before host-side validation, then execute and synchronize
  on that stream. These calls are blocking, not asynchronous. `Synchronize`,
  stream changes and context destruction synchronize prior work and retry
  pending cleanup; `Synchronize` keeps the current stream attached.
  A closed attached stream causes operations to fail until detached; it may
  be detached or the context destroyed after the stream has closed.
  NativeStream-owned retentions from other queued operations remain held
  until the stream or context is explicitly synchronized, the stream is
  closed, or the sparse context changes streams or is destroyed. Do not call an attached sparse
  context from a callback holding that stream's native handle.
  Sparse inputs undergo host-side index validation; COO data and duplicate
  CSR entries are canonicalized on the host before execution because
  cuSPARSE requires COO row ordering and does not guarantee duplicate-index
  results. Sparse multiplication canonicalizes both inputs to CSR on the
  host, including mixed CSR/COO inputs, before the native cuSPARSE product.
  `SpMVEx` supports both non-transposed and transposed CSR/COO multiplication;
  for transpose, x has `A.rows` elements and y has `A.cols` elements. The
  simulation does not read the previous y values when beta is zero. Native
  transpose execution may be nondeterministic.
  `SpSVEx` solves with either the original triangular matrix or its transpose;
  the triangular fill is inferred from the stored matrix before transposition.
  Dense-to-sparse accepts device or managed input on the context device and
  limits the dense element count to the 32-bit sparse index capacity. Native
  dense-to-sparse output indices are not guaranteed to be sorted.
  `SpMMDense` multiplies a float32 CSR/COO matrix by a contiguous row-major
  dense matrix and writes a contiguous row-major result. `SpMMDenseEx` also
  supports transposed inputs and row strides; each dense buffer must contain
  all padded rows, including the last. Output padding is preserved. Both
  reject output aliasing, require device or managed buffers in CUDA mode and
  synchronize before returning. `SpMMDenseBatched` applies one shared sparse matrix to
  1 to 65535 separate dense B and C matrices. Batch strides are measured in
  elements, must include complete padded matrices and may not overlap; gaps
  and output padding are preserved. Broadcasting a dense B or batching
  different sparse A matrices is not supported. Native algorithms may be
  nondeterministic.
  Sparse-sparse `SpMM` computes the product with cuSPARSE SpGEMM and adds
  the initial output using the legacy `csrgeam2` API. It canonicalizes inputs,
  prunes zero results and copies the final CSR or COO output on the host; it
  is not fully GPU-resident. The destination may alias an input and must
  have enough capacity after pruning. Native addition rejects cases where
  the sum of input nonzero counts exceeds the 32-bit index limit.
  Sparse LU currently executes on the host in both modes, does not pivot,
  and returns an error when a nonzero pivot is
  unavailable. `CreateNativeSparseContextOnDevice` creates a strict
  native-only context: the native operations listed above are supported and
  host-only operations return `ErrNativeSparseOperationUnsupported`.
  `SpMVBackend` (including transposed calls), `SpMMDenseBackend` (including batched calls), `SpMMBackend`, `SpSVBackend` (including transposed calls), `SpGEMMBackend`, `SparseToDenseBackend` and
  `DenseToSparseBackend` report operation backends. Native
  vector and sparse-dense operations reject output buffers that alias input
  or sparse matrix buffers. Native triangular solves require device or managed input
  and output buffers on the context device, including for COO matrices.
  They require explicit finite, nonzero diagonal entries and infer lower or
  upper fill from the matrix.
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
- cuDSS CSR analysis, factorization and refactorization validate host-read
  row offsets and column indices, rejecting out-of-range and duplicate entries.
  Native handles and CSR descriptors are bound to their CUDA device; native
  matrices and solve buffers must use device or managed memory on that device.
  `CreateNativeDSSHandleOnDevice` rejects host fallbacks and unsupported
  non-default configuration; it accepts CSR LU/general and Cholesky/SPD.
  Its zero-valued ordering and pivot fields use cuDSS defaults, not guarantees
  of no ordering or pivoting. `Backend` reports the handle backend, not every
  operation of a non-strict handle. Native execution still performs host-side
  validation, conversion and diagnostics. Operations and destruction on the
  same handle are serialized.
  Factorization, refactorization and solve require the same CSR offsets,
  column order and symmetry as the last successful analysis. Reanalysis
  invalidates prior factors. Update matrix values only before factorization
  or refactorization; native float32 value copies are refreshed at those phases.
  Destroying a matrix does not free its caller-owned value buffer.
  Callers must not concurrently use a matrix from multiple handles, destroy
  its descriptor, or modify or free its buffers while cuDSS uses them;
  asynchronous native resource lifetime protection remains incomplete.
  Factorization also constructs a
  dense host copy, including in native mode. `Solve` reports the maximum
  infinity-norm residual, normwise backward error and componentwise backward
  error calculated from the returned (possibly float32-rounded) solution.
  `SolveMultiple` reports these separately for each right-hand side.
  Unmeasured `Error`, `PivotGrowth` and `ConditionNumber` values are NaN;
  `Iterations` is -1 and `InertiaAvailable` is false when unavailable.
  `GetInertia` returns `ErrDSSInertiaUnavailable` rather than fabricated
  inertia, and the refinement configuration is not applied by this wrapper.
  These paths are not yet suitable for scalable or production-verified sparse
  solves.
- Native nvJPEG decoding binds each decoder to its creation device and
  serializes its operations. Supported native decodes use pinned bitstream
  storage and synchronize the default stream before returning; output is
  allocated on the decoder's device. Failed synchronization retains internal
  buffers for retry by `JpegDecoderState.Synchronize` or `Destroy`.
  The constructor selects the native decoder backend; the per-call
  `JpegDecodeParams.Backend` field is not currently applied.
  Crop, scale and unsupported output formats may still use the host fallback;
  encoder and nvJPEG2000 lifetime behavior is not covered by this guarantee.
  On the tested Windows CUDA 13.2 setup, Compute Sanitizer initcheck reports
  uninitialized reads in nvJPEG's internal `rgba2rgb` kernel and grayscale
  copy; their cause is not established, and nvJPEG initcheck validation
  remains outstanding.
- cuDNN 4D and N-dimensional tensor and filter descriptor dimensions must
  be positive and fit 32-bit values; tensor strides must fit signed 32-bit
  values. Failed native setters do not update Go-side descriptor metadata.
  The float32 activation
  simulation honors positive input and output strides and preserves padding;
  unsupported activation modes, including Swish, return an error in simulation.
  The float32 convolution simulation supports packed NCHW/NHWC tensors and
  packed NCHW filters, and distinguishes convolution from cross-correlation.
  Invalid convolution configuration or output dimensions return errors.
  Float32 pooling simulation supports positive 4D strides, propagates NaNs,
  excludes padding from maxima, and requires padding smaller than the window
  and matching output dimensions. Float32 batch-normalization inference
  simulation supports positive 4D input, output and parameter strides for
  spatial and per-activation modes and validates their dimensions. Other
  cuDNN host operations should not be assumed to support arbitrary strided
  layouts.
  `CreateNativeDNNHandleOnDevice` requires native cuDNN and
  binds the handle to a selected device; `Backend` reports its mode. Native
  execution accepts device or managed buffers on that device, validates
  storage extents and requires native descriptors with positive strides.
  Vectorized descriptor layouts without explicit strides are not supported
  by these execution paths. In-place activation requires matching dimensions
  and strides. Native activation is tested with float32, float64, float16
  and bfloat16; bfloat16 requires supported hardware and cuDNN. Mixed
  double/non-double scaling types are rejected, and the public alpha and beta
  arguments are float32 even for double-precision tensors.
  `DNNHandle.SetStream` attaches a native stream or selects the default with
  nil. Operations synchronize before returning; `Synchronize`, stream changes
  and destruction also retry pending cleanup. Buffers, workspaces and
  descriptors remain retained if synchronization fails, and descriptor
  mutation or destruction returns `ErrDNNDescriptorInUse` while retained.
  NativeStream-owned retentions remain held until the stream or handle is
  explicitly synchronized, the stream is closed, or the handle changes
  streams or is destroyed. A closed attached stream rejects new operations
  until detached. Do not call a handle from a callback holding its stream.
- CUTLASS-named native linear algebra paths use cuBLAS, not CUTLASS kernels.
  Other parameter combinations may fall back to Go.
- The CUDA Math API and portions of Thrust-style operations use Go execution.

The readiness matrix is library-wide and does not provide per-operation backend
selection. Apart from explicitly documented native-only APIs, production
validation and a library-wide strict-native policy remain outstanding; inspect
the operation implementation before relying on GPU execution or measuring its
performance.
