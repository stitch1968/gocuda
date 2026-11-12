# Public API Functions

Generated from exported functions in non-internal packages. Excludes demos, cmd, tests, and docs.

This inventory includes separate build-tag variants and does not indicate
whether an operation runs on the GPU. See [runtime boundaries](LIBRARY_RUNTIME_BOUNDARIES.md).

| Package | Function | Description |
| --- | --- | --- |
| github.com/stitch1968/gocuda | Alloc | Alloc creates a new memory allocation builder (Simple API - Layer 1) |
| github.com/stitch1968/gocuda | AsyncExecutor.Close | Close closes the executor |
| github.com/stitch1968/gocuda | AsyncExecutor.Submit | Submit submits a task for asynchronous execution |
| github.com/stitch1968/gocuda | AsyncExecutor.Wait | Wait waits for all tasks to complete |
| github.com/stitch1968/gocuda | BatchExecute | BatchExecute executes multiple operations in batches |
| github.com/stitch1968/gocuda | Benchmark | Benchmark runs a function multiple times and returns statistics |
| github.com/stitch1968/gocuda | BitsToFloat | No doc comment available. |
| github.com/stitch1968/gocuda | CalculateOccupancy | CalculateOccupancy calculates theoretical occupancy for a kernel |
| github.com/stitch1968/gocuda | CheckKernelParameters | CheckKernelParameters validates kernel launch parameters |
| github.com/stitch1968/gocuda | CheckMemoryAllocation | CheckMemoryAllocation validates memory allocation parameters |
| github.com/stitch1968/gocuda | Compute | Compute provides auto-managed memory computation (Simplest API) |
| github.com/stitch1968/gocuda | Context.NewStream | NewStream creates a new CUDA stream in this context - delegates to streams package |
| github.com/stitch1968/gocuda | Context.Run | Run executes fn while binding the calling OS thread to this context's device. |
| github.com/stitch1968/gocuda | CreateArray | CreateArray creates a CUDA array (like cudaMallocArray) |
| github.com/stitch1968/gocuda | CreateEvent | CreateEvent creates a CUDA event (like cudaEventCreate) |
| github.com/stitch1968/gocuda | CreateGraph | CreateGraph creates a new CUDA graph |
| github.com/stitch1968/gocuda | CreateSurfaceObject | CreateSurfaceObject creates a surface object from an array |
| github.com/stitch1968/gocuda | CreateTextureObject | CreateTextureObject creates a texture object from an array |
| github.com/stitch1968/gocuda | CudaChannel.Close | Close closes the channel |
| github.com/stitch1968/gocuda | CudaChannel.Receive | Receive receives a value from the channel |
| github.com/stitch1968/gocuda | CudaChannel.Send | Send sends a value to the channel |
| github.com/stitch1968/gocuda | CudaError.Error | No doc comment available. |
| github.com/stitch1968/gocuda | CudaEvent.Record | Record records an event in a stream (like cudaEventRecord) |
| github.com/stitch1968/gocuda | CudaEvent.Synchronize | Synchronize waits for the event to complete (like cudaEventSynchronize) |
| github.com/stitch1968/gocuda | CudaFree | CudaFree frees GPU memory using real CUDA |
| github.com/stitch1968/gocuda | CudaFree | CudaFree is a no-op in simulation mode |
| github.com/stitch1968/gocuda | CudaGraph.AddKernelNode | AddKernelNode adds a kernel node to the graph |
| github.com/stitch1968/gocuda | CudaGraph.Launch | Launch launches the graph (like cudaGraphLaunch) |
| github.com/stitch1968/gocuda | CudaMalloc | CudaMalloc allocates memory on the GPU using real CUDA |
| github.com/stitch1968/gocuda | CudaMalloc | CudaMalloc returns an error in simulation mode |
| github.com/stitch1968/gocuda | CudaMemcpy | CudaMemcpy returns an error in simulation mode |
| github.com/stitch1968/gocuda | CudaMemcpy | CudaMemcpy copies memory using real CUDA |
| github.com/stitch1968/gocuda | CudaWaitGroup.Add | Add adds delta to the WaitGroup counter |
| github.com/stitch1968/gocuda | CudaWaitGroup.Done | Done decrements the WaitGroup counter by one |
| github.com/stitch1968/gocuda | CudaWaitGroup.Wait | Wait blocks until the WaitGroup counter is zero |
| github.com/stitch1968/gocuda | CurrentCudaDeviceID | CurrentCudaDeviceID returns the simulated current device ID. |
| github.com/stitch1968/gocuda | CurrentCudaDeviceID | CurrentCudaDeviceID returns the currently selected CUDA device ID. |
| github.com/stitch1968/gocuda | DefaultContext | DefaultContext returns the default CUDA context without panicking on init failure. |
| github.com/stitch1968/gocuda | DefaultStream | DefaultStream returns the default CUDA stream without panicking on init failure. |
| github.com/stitch1968/gocuda | DeviceReset | DeviceReset reports unsupported in hardware mode to avoid invalidating live resources. |
| github.com/stitch1968/gocuda | DeviceSynchronize | DeviceSynchronize waits for the default host queue and current CUDA device. |
| github.com/stitch1968/gocuda | ElapsedTime | ElapsedTime returns the time elapsed between two events |
| github.com/stitch1968/gocuda | EnhancedError.Error | No doc comment available. |
| github.com/stitch1968/gocuda | EnhancedError.GetDetails | GetDetails returns additional error context |
| github.com/stitch1968/gocuda | EnhancedError.IsRecoverable | IsRecoverable returns true if the error can be handled gracefully |
| github.com/stitch1968/gocuda | Event.Record | Record records the event |
| github.com/stitch1968/gocuda | EventElapsedTime | ElapsedTime calculates time between two events (like cudaEventElapsedTime) |
| github.com/stitch1968/gocuda | FallbackToCPU | FallbackToCPU provides a fallback mechanism for recoverable errors |
| github.com/stitch1968/gocuda | FloatToBits | Helper functions for float conversion |
| github.com/stitch1968/gocuda | ForceFallbackToSimulation | ForceFallbackToSimulation forces the library to use CPU simulation even if CUDA is available |
| github.com/stitch1968/gocuda | ForceFallbackToSimulation | ForceFallbackToSimulation is a no-op in simulation mode |
| github.com/stitch1968/gocuda | FormatMemorySize | FormatMemorySize formats memory size in human-readable format |
| github.com/stitch1968/gocuda | GetCudaDevice | GetCudaDevice returns a CUDA device by ID |
| github.com/stitch1968/gocuda | GetCudaDevice | GetCudaDevice returns an error in simulation mode |
| github.com/stitch1968/gocuda | GetCudaDeviceCount | GetCudaDeviceCount returns 0 in simulation mode |
| github.com/stitch1968/gocuda | GetCudaDeviceCount | GetCudaDeviceCount returns the number of CUDA devices |
| github.com/stitch1968/gocuda | GetCudaDeviceProperties | GetCudaDeviceProperties returns an error in simulation mode |
| github.com/stitch1968/gocuda | GetCudaDeviceProperties | GetCudaDeviceProperties returns properties for a CUDA device |
| github.com/stitch1968/gocuda | GetCudaRuntime | GetCudaRuntime returns the global CUDA runtime instance |
| github.com/stitch1968/gocuda | GetCudaRuntime | GetCudaRuntime returns the global CUDA runtime instance |
| github.com/stitch1968/gocuda | GetCurrentDevice | GetCurrentDevice returns the current device ID (like cudaGetDevice) |
| github.com/stitch1968/gocuda | GetDefaultContext | GetDefaultContext returns the default CUDA context |
| github.com/stitch1968/gocuda | GetDefaultStream | GetDefaultStream returns the default CUDA stream |
| github.com/stitch1968/gocuda | GetDevice | GetDevice returns device information by ID |
| github.com/stitch1968/gocuda | GetDevices | GetDevices returns all available CUDA devices (real or simulated) |
| github.com/stitch1968/gocuda | GetErrorSuggestion | GetErrorSuggestion provides context-aware error suggestions |
| github.com/stitch1968/gocuda | Go | Go schedules a host callback on the default stream. |
| github.com/stitch1968/gocuda | GoWithDimensions | GoWithDimensions schedules a host callback; dimensions do not launch GPU threads. |
| github.com/stitch1968/gocuda | GoWithDimensionsAndStream | GoWithDimensionsAndStream schedules a host callback; dimensions are informational. |
| github.com/stitch1968/gocuda | GoWithStream | GoWithStream schedules a host callback; synchronization returns callback errors. |
| github.com/stitch1968/gocuda | InitializationError | InitializationError returns the initialization error recorded by Initialize, or nil if initialization has not failed. |
| github.com/stitch1968/gocuda | Initialize | Initialize initializes the CUDA runtime (real or simulated) |
| github.com/stitch1968/gocuda | InitializeCudaRuntime | InitializeCudaRuntime initializes the simulated CUDA runtime |
| github.com/stitch1968/gocuda | InitializeCudaRuntime | InitializeCudaRuntime initializes the CUDA runtime and detects available devices |
| github.com/stitch1968/gocuda | IsCudaAvailable | IsCudaAvailable returns false in simulation mode |
| github.com/stitch1968/gocuda | IsCudaAvailable | IsCudaAvailable returns true if CUDA is available and devices are found |
| github.com/stitch1968/gocuda | IsInitialized | IsInitialized reports whether the CUDA runtime has been initialized successfully. |
| github.com/stitch1968/gocuda | Launch | Launch provides a higher-level interface for launching kernels |
| github.com/stitch1968/gocuda | LaunchAsync | LaunchAsync launches a kernel asynchronously |
| github.com/stitch1968/gocuda | Map | Map applies a function to each element of a slice in parallel |
| github.com/stitch1968/gocuda | Matrix.Cols | Cols returns the number of columns |
| github.com/stitch1968/gocuda | Matrix.Free | Free releases the matrix memory |
| github.com/stitch1968/gocuda | Matrix.Multiply | Multiply performs matrix multiplication (Simple API) |
| github.com/stitch1968/gocuda | Matrix.Rows | Rows returns the number of rows |
| github.com/stitch1968/gocuda | Matrix.Size | Size returns the matrix size in bytes |
| github.com/stitch1968/gocuda | MemoryPool.Clear | Clear frees all memory in the pool |
| github.com/stitch1968/gocuda | MemoryPool.Get | Get gets a memory allocation from the pool |
| github.com/stitch1968/gocuda | MemoryPool.Put | Put returns a memory allocation to the pool |
| github.com/stitch1968/gocuda | NewAsyncExecutor | NewAsyncExecutor creates a new async executor |
| github.com/stitch1968/gocuda | NewContext | NewContext creates a new CUDA context for the specified device |
| github.com/stitch1968/gocuda | NewCudaChannel | NewCudaChannel creates a new CUDA channel |
| github.com/stitch1968/gocuda | NewDeviceError | NewDeviceError creates a device-related error with helpful suggestions |
| github.com/stitch1968/gocuda | NewEvent | NewEvent creates a new CUDA event |
| github.com/stitch1968/gocuda | NewKernelError | NewKernelError creates a kernel execution error with helpful suggestions |
| github.com/stitch1968/gocuda | NewMatrix | NewMatrix creates a matrix (Simple API) |
| github.com/stitch1968/gocuda | NewMemoryError | NewMemoryError creates a memory-related error with helpful suggestions |
| github.com/stitch1968/gocuda | NewMemoryPool | NewMemoryPool creates a new memory pool |
| github.com/stitch1968/gocuda | NewProfiler | NewProfiler creates a new profiler |
| github.com/stitch1968/gocuda | NewSimpleContext | NewSimpleContext creates a new simplified context (addresses complex initialization) |
| github.com/stitch1968/gocuda | NewTimer | NewTimer creates a new timer |
| github.com/stitch1968/gocuda | NewVector | NewVector creates a vector (Simple API - Layer 1) |
| github.com/stitch1968/gocuda | Parallel | Parallel executes a parallel loop (Simple API - consistent naming) |
| github.com/stitch1968/gocuda | ParallelFor | ParallelFor executes a function in parallel across multiple GPU threads |
| github.com/stitch1968/gocuda | Pipeline | Pipeline executes a series of operations in a pipeline |
| github.com/stitch1968/gocuda | PrintCudaInfo | PrintCudaInfo prints information about the CUDA runtime and devices |
| github.com/stitch1968/gocuda | PrintCudaInfo | PrintCudaInfo prints simulation information |
| github.com/stitch1968/gocuda | Profiler.End | End stops timing an event |
| github.com/stitch1968/gocuda | Profiler.GetDuration | GetDuration returns the duration of an event |
| github.com/stitch1968/gocuda | Profiler.PrintSummary | PrintSummary prints all recorded events |
| github.com/stitch1968/gocuda | Profiler.Start | Start begins timing an event |
| github.com/stitch1968/gocuda | Reduce | Reduce performs a reduction operation across GPU threads |
| github.com/stitch1968/gocuda | Run | Run executes a function on GPU (Simple API - addresses naming consistency) |
| github.com/stitch1968/gocuda | Select | Select provides a select-like operation for CUDA channels |
| github.com/stitch1968/gocuda | SetCudaDevice | SetCudaDevice is a no-op in simulation mode |
| github.com/stitch1968/gocuda | SetCudaDevice | SetCudaDevice sets the current CUDA device |
| github.com/stitch1968/gocuda | SetDevice | SetDevice sets the current device (like cudaSetDevice) |
| github.com/stitch1968/gocuda | ShouldUseCuda | ShouldUseCuda returns true if CUDA should be used instead of simulation |
| github.com/stitch1968/gocuda | ShouldUseCuda | ShouldUseCuda returns false in simulation mode |
| github.com/stitch1968/gocuda | SimpleContext.Run | Run executes a function in the GPU context (Simple API) |
| github.com/stitch1968/gocuda | SimpleKernel.Execute | Execute implements the Kernel interface for SimpleKernel |
| github.com/stitch1968/gocuda | SimpleMatrixMultiply | SimpleMatrixMultiply provides an easy-to-use matrix multiplication function |
| github.com/stitch1968/gocuda | SimpleMemory.Allocate | Allocate performs the actual memory allocation |
| github.com/stitch1968/gocuda | SimpleMemory.DeviceOnly | DeviceOnly specifies device-only memory, which cannot be viewed on the host. |
| github.com/stitch1968/gocuda | SimpleMemory.OnDevice | OnDevice specifies device memory allocation |
| github.com/stitch1968/gocuda | SimpleMemory.OnHost | OnHost specifies host memory allocation |
| github.com/stitch1968/gocuda | SimpleMemory.Pinned | Pinned specifies pinned host memory allocation |
| github.com/stitch1968/gocuda | SimpleMemory.Unified | Unified specifies unified memory allocation |
| github.com/stitch1968/gocuda | SimpleMemory.WithStream | WithStream specifies a custom stream |
| github.com/stitch1968/gocuda | SimpleVectorAdd | SimpleVectorAdd provides an easy-to-use vector addition function |
| github.com/stitch1968/gocuda | Synchronize | Synchronize waits for all CUDA operations to complete (similar to runtime.Gosched()) |
| github.com/stitch1968/gocuda | SynchronizeStream | SynchronizeStream waits for all operations in a specific stream to complete |
| github.com/stitch1968/gocuda | Timer.Elapsed | Elapsed returns the elapsed time |
| github.com/stitch1968/gocuda | Timer.Start | Start starts the timer |
| github.com/stitch1968/gocuda | Timer.Stop | Stop stops the timer |
| github.com/stitch1968/gocuda | Vector.Add | Add performs vector addition (Simple API) |
| github.com/stitch1968/gocuda | Vector.Free | Free releases the vector memory |
| github.com/stitch1968/gocuda | Vector.Length | Length returns the vector length |
| github.com/stitch1968/gocuda | Vector.Multiply | Multiply performs scalar multiplication (Simple API) |
| github.com/stitch1968/gocuda | Vector.Size | Size returns the vector size in bytes |
| github.com/stitch1968/gocuda | Vector.ToSlice | ToSlice returns the vector as a Go slice (Simple API) |
| github.com/stitch1968/gocuda | View | View returns a typed slice over an allocation after validating bounds. |
| github.com/stitch1968/gocuda | WithContext | WithContext executes a function with a CUDA context |
| github.com/stitch1968/gocuda | WithStream | WithStream executes a function with a CUDA stream |
| github.com/stitch1968/gocuda | WrapError | WrapError wraps an existing error with enhanced context |
| github.com/stitch1968/gocuda/advanced | AsyncFFT | AsyncFFT performs asynchronous FFT |
| github.com/stitch1968/gocuda/advanced | AsyncSort | AsyncSort performs asynchronous sorting |
| github.com/stitch1968/gocuda/advanced | BFS | BFS performs breadth-first search on GPU |
| github.com/stitch1968/gocuda/advanced | Batch.Add | Add adds an operation to the batch |
| github.com/stitch1968/gocuda/advanced | Batch.Clear | Clear clears the batch |
| github.com/stitch1968/gocuda/advanced | Batch.Execute | Execute executes all operations in the batch |
| github.com/stitch1968/gocuda/advanced | BitonicSort | BitonicSort performs bitonic sort (good for GPU parallelization) |
| github.com/stitch1968/gocuda/advanced | FFT | FFT performs Fast Fourier Transform on complex data |
| github.com/stitch1968/gocuda/advanced | GEMM | GEMM performs General Matrix Multiply (C = alpha*A*B + beta*C) |
| github.com/stitch1968/gocuda/advanced | LUDecomposition | LUDecomposition performs LU decomposition |
| github.com/stitch1968/gocuda/advanced | NewBatch | NewBatch creates a new batch processor |
| github.com/stitch1968/gocuda/advanced | PageRank | PageRank performs PageRank algorithm on GPU |
| github.com/stitch1968/gocuda/advanced | RadixSort | RadixSort performs parallel radix sort on GPU |
| github.com/stitch1968/gocuda/advanced | Transpose | Transpose performs matrix transpose |
| github.com/stitch1968/gocuda/hardware | CoalescedGroup.IsValid | No doc comment available. |
| github.com/stitch1968/gocuda/hardware | CoalescedGroup.Size | No doc comment available. |
| github.com/stitch1968/gocuda/hardware | CoalescedGroup.Sync | No doc comment available. |
| github.com/stitch1968/gocuda/hardware | CoalescedGroup.ThreadRank | No doc comment available. |
| github.com/stitch1968/gocuda/hardware | GetTensorCoreInfo | GetTensorCoreInfo returns information about Tensor Core support |
| github.com/stitch1968/gocuda/hardware | GetWarpInfo | GetWarpInfo returns information about the current warp context |
| github.com/stitch1968/gocuda/hardware | GroupBarrier | GroupBarrier synchronizes all threads in the group |
| github.com/stitch1968/gocuda/hardware | GroupReduce | GroupReduce performs reduction within a cooperative group |
| github.com/stitch1968/gocuda/hardware | NewCoalescedGroup | NewCoalescedGroup creates a coalesced group from active threads |
| github.com/stitch1968/gocuda/hardware | NewThreadBlock | NewThreadBlock creates a new thread block cooperative group |
| github.com/stitch1968/gocuda/hardware | NewWarp | NewWarp creates a new warp cooperative group |
| github.com/stitch1968/gocuda/hardware | NewWarpReduce | NewWarpReduce creates a new warp reduction context |
| github.com/stitch1968/gocuda/hardware | NewWarpShuffle | NewWarpShuffle creates a new warp shuffle context |
| github.com/stitch1968/gocuda/hardware | NewWarpVote | NewWarpVote creates a new warp vote context |
| github.com/stitch1968/gocuda/hardware | TensorCoreMMA | TensorCoreMMA performs mixed-precision matrix multiply-accumulate |
| github.com/stitch1968/gocuda/hardware | ThreadBlock.IsValid | No doc comment available. |
| github.com/stitch1968/gocuda/hardware | ThreadBlock.Size | No doc comment available. |
| github.com/stitch1968/gocuda/hardware | ThreadBlock.Sync | No doc comment available. |
| github.com/stitch1968/gocuda/hardware | ThreadBlock.ThreadRank | No doc comment available. |
| github.com/stitch1968/gocuda/hardware | Warp.IsValid | No doc comment available. |
| github.com/stitch1968/gocuda/hardware | Warp.Size | No doc comment available. |
| github.com/stitch1968/gocuda/hardware | Warp.Sync | No doc comment available. |
| github.com/stitch1968/gocuda/hardware | Warp.ThreadRank | No doc comment available. |
| github.com/stitch1968/gocuda/hardware | WarpReduce.ReduceMax | ReduceMax performs warp-level max reduction |
| github.com/stitch1968/gocuda/hardware | WarpReduce.ReduceMin | ReduceMin performs warp-level min reduction |
| github.com/stitch1968/gocuda/hardware | WarpReduce.ReduceSum | ReduceSum performs warp-level sum reduction |
| github.com/stitch1968/gocuda/hardware | WarpShuffle.ShuffleDown | ShuffleDown shifts value down by delta lanes |
| github.com/stitch1968/gocuda/hardware | WarpShuffle.ShuffleIndex | ShuffleIndex shuffles from specific lane |
| github.com/stitch1968/gocuda/hardware | WarpShuffle.ShuffleUp | ShuffleUp shifts value up by delta lanes |
| github.com/stitch1968/gocuda/hardware | WarpShuffle.ShuffleXor | ShuffleXor performs XOR-based shuffle |
| github.com/stitch1968/gocuda/hardware | WarpVote.All | All returns true if all active threads have predicate true |
| github.com/stitch1968/gocuda/hardware | WarpVote.Any | Any returns true if any active thread has predicate true |
| github.com/stitch1968/gocuda/hardware | WarpVote.Ballot | Ballot returns a bitmask of threads where predicate is true |
| github.com/stitch1968/gocuda/kernels | Convolution2D.Execute | No doc comment available. |
| github.com/stitch1968/gocuda/kernels | BufferArg | BufferArg passes a device or managed buffer with a byte offset. |
| github.com/stitch1968/gocuda/kernels | CompileCUDA | CompileCUDA compiles CUDA source to PTX using a local nvcc installation. |
| github.com/stitch1968/gocuda/kernels | Float32Arg | Float32Arg passes a 32-bit floating-point value. |
| github.com/stitch1968/gocuda/kernels | Float64Arg | Float64Arg passes a 64-bit floating-point value. |
| github.com/stitch1968/gocuda/kernels | Int32Arg | Int32Arg passes a 32-bit signed integer. |
| github.com/stitch1968/gocuda/kernels | Int64Arg | Int64Arg passes a 64-bit signed integer. |
| github.com/stitch1968/gocuda/kernels | Uint32Arg | Uint32Arg passes a 32-bit unsigned integer. |
| github.com/stitch1968/gocuda/kernels | Uint64Arg | Uint64Arg passes a 64-bit unsigned integer. |
| github.com/stitch1968/gocuda/kernels | LoadPTX | LoadPTX loads a CUDA driver module from PTX. |
| github.com/stitch1968/gocuda/kernels | NativeFunction.Launch | Launch enqueues a GPU kernel on a native stream. |
| github.com/stitch1968/gocuda/kernels | NewNativeGraph | NewNativeGraph creates a CUDA driver graph on a device. |
| github.com/stitch1968/gocuda/kernels | NativeGraph.AddKernelNode | AddKernelNode adds a native CUDA kernel node and dependencies. |
| github.com/stitch1968/gocuda/kernels | NativeGraph.AddMemcpyNode | AddMemcpyNode adds a one-dimensional CUDA memory copy node. |
| github.com/stitch1968/gocuda/kernels | NativeGraph.AddMemsetNode | AddMemsetNode adds a CUDA memory initialization node. |
| github.com/stitch1968/gocuda/kernels | NativeGraph.Instantiate | Instantiate creates an executable CUDA graph. |
| github.com/stitch1968/gocuda/kernels | NativeGraph.Launch | Launch enqueues the instantiated graph on a native stream. |
| github.com/stitch1968/gocuda/kernels | NativeGraph.Close | Close destroys the native graph and executable. |
| github.com/stitch1968/gocuda/kernels | NativeGraphNode.DependsOn | DependsOn checks a direct CUDA graph dependency. |
| github.com/stitch1968/gocuda/kernels | NewNativeOps | NewNativeOps loads the built-in GPU kernels on a device. |
| github.com/stitch1968/gocuda/kernels | NativeOps.VectorAdd | VectorAdd enqueues float32 vector addition on the GPU. |
| github.com/stitch1968/gocuda/kernels | NativeOps.SAXPY | SAXPY enqueues float32 scaled vector addition on the GPU. |
| github.com/stitch1968/gocuda/kernels | NativeOps.MatrixMultiply | MatrixMultiply enqueues baseline row-major float32 matrix multiplication. |
| github.com/stitch1968/gocuda/kernels | NativeOps.Sum | Sum enqueues a float32 reduction using atomic accumulation. |
| github.com/stitch1968/gocuda/kernels | NativeOps.Convolution2D | Convolution2D enqueues float32 2D cross-correlation. |
| github.com/stitch1968/gocuda/kernels | NativeOps.Close | Close unloads the GPU kernels. |
| github.com/stitch1968/gocuda/kernels | NativeModule.Close | Close unloads a CUDA driver module. |
| github.com/stitch1968/gocuda/kernels | NativeModule.Function | Function resolves a named GPU kernel in the module. |
| github.com/stitch1968/gocuda/kernels | CreateMatrix | CreateMatrix creates a matrix memory allocation |
| github.com/stitch1968/gocuda/kernels | CreateVector | CreateVector creates a vector memory allocation with optional initial values |
| github.com/stitch1968/gocuda/kernels | CustomKernel.Execute | No doc comment available. |
| github.com/stitch1968/gocuda/kernels | ElementwiseAdd | ElementwiseAdd performs element-wise addition |
| github.com/stitch1968/gocuda/kernels | ElementwiseMultiply | ElementwiseMultiply performs element-wise multiplication |
| github.com/stitch1968/gocuda/kernels | ExecuteMatrixMultiply | ExecuteMatrixMultiply performs matrix multiplication: c = a * b |
| github.com/stitch1968/gocuda/kernels | ExecuteMatrixMultiplyAsync | ExecuteMatrixMultiplyAsync performs asynchronous matrix multiplication |
| github.com/stitch1968/gocuda/kernels | ExecuteVectorAdd | ExecuteVectorAdd performs vector addition: c = a + b |
| github.com/stitch1968/gocuda/kernels | ExecuteVectorAddAsync | ExecuteVectorAddAsync performs asynchronous vector addition |
| github.com/stitch1968/gocuda/kernels | FillVector | FillVector fills a vector with a constant value |
| github.com/stitch1968/gocuda/kernels | GetVectorValue | GetVectorValue gets a value from a vector at the specified index |
| github.com/stitch1968/gocuda/kernels | MatrixMultiply.Execute | No doc comment available. |
| github.com/stitch1968/gocuda/kernels | NewCustomKernel | NewCustomKernel creates a new custom kernel |
| github.com/stitch1968/gocuda/kernels | Reduction.Execute | No doc comment available. |
| github.com/stitch1968/gocuda/kernels | SAXPY.Execute | No doc comment available. |
| github.com/stitch1968/gocuda/kernels | ScalarMultiply | ScalarMultiply performs scalar multiplication: c = a * scalar |
| github.com/stitch1968/gocuda/kernels | SetVectorValue | SetVectorValue sets a value in a vector at the specified index |
| github.com/stitch1968/gocuda/kernels | VectorAdd.Execute | No doc comment available. |
| github.com/stitch1968/gocuda/libraries | ActivationDescriptor.DestroyActivationDescriptor | DestroyActivationDescriptor destroys an activation descriptor |
| github.com/stitch1968/gocuda/libraries | ActivationDescriptor.SetActivationDescriptor | SetActivationDescriptor sets the activation descriptor |
| github.com/stitch1968/gocuda/libraries | AmgXHandle.Destroy | Destroy cleans up AmgX handle resources |
| github.com/stitch1968/gocuda/libraries | AmgXHandle.GetGridComplexity | GetGridComplexity returns the grid complexity of the AMG hierarchy |
| github.com/stitch1968/gocuda/libraries | AmgXHandle.GetOperatorComplexity | GetOperatorComplexity returns the operator complexity of the AMG hierarchy |
| github.com/stitch1968/gocuda/libraries | AmgXHandle.PrintInfo | PrintInfo prints information about the AMG hierarchy |
| github.com/stitch1968/gocuda/libraries | AmgXHandle.Setup | Setup performs the AMG setup phase (coarsening, interpolation, etc.) |
| github.com/stitch1968/gocuda/libraries | AmgXHandle.Solve | Solve solves the linear system using AMG |
| github.com/stitch1968/gocuda/libraries | AmgXHandle.SolveMultiple | SolveMultiple solves multiple systems with the same matrix |
| github.com/stitch1968/gocuda/libraries | AmgXHandle.UpdateMatrix | UpdateMatrix updates the matrix values (keeping the same sparsity pattern) |
| github.com/stitch1968/gocuda/libraries | AmgXMatrix.Destroy | Destroy cleans up AmgX matrix resources |
| github.com/stitch1968/gocuda/libraries | AmgXVector.Destroy | Destroy cleans up AmgX vector resources |
| github.com/stitch1968/gocuda/libraries | ApplyActivation | ApplyActivation provides simplified activation function application |
| github.com/stitch1968/gocuda/libraries | BatchNormDescriptor.DestroyBatchNormDescriptor | DestroyBatchNormDescriptor destroys a batch normalization descriptor |
| github.com/stitch1968/gocuda/libraries | ComputeElementwise | ComputeElementwise performs elementwise operations on vectors |
| github.com/stitch1968/gocuda/libraries | ComputeUnary | ComputeUnary performs unary operations on vectors |
| github.com/stitch1968/gocuda/libraries | ConvolutionDescriptor.DestroyConvolutionDescriptor | DestroyConvolutionDescriptor destroys a convolution descriptor |
| github.com/stitch1968/gocuda/libraries | ConvolutionDescriptor.GetConvolution2dForwardOutputDim | GetConvolution2dForwardOutputDim calculates output dimensions for convolution |
| github.com/stitch1968/gocuda/libraries | ConvolutionDescriptor.SetConvolution2dDescriptor | SetConvolution2dDescriptor sets the convolution descriptor for 2D convolutions |
| github.com/stitch1968/gocuda/libraries | ConvolutionForward | ConvolutionForward provides simplified convolution operation |
| github.com/stitch1968/gocuda/libraries | CreateActivationDescriptor | CreateActivationDescriptor creates an activation descriptor |
| github.com/stitch1968/gocuda/libraries | CreateAmgXHandle | CreateAmgXHandle creates a new AmgX solver handle |
| github.com/stitch1968/gocuda/libraries | CreateAmgXMatrix | CreateAmgXMatrix creates an AmgX matrix |
| github.com/stitch1968/gocuda/libraries | CreateAmgXVector | CreateAmgXVector creates an AmgX vector |
| github.com/stitch1968/gocuda/libraries | CreateBatchNormDescriptor | CreateBatchNormDescriptor creates a batch normalization descriptor |
| github.com/stitch1968/gocuda/libraries | CreateConvolutionDescriptor | CreateConvolutionDescriptor creates a convolution descriptor |
| github.com/stitch1968/gocuda/libraries | CreateCuTensorDescriptor | CreateCuTensorDescriptor creates a tensor descriptor |
| github.com/stitch1968/gocuda/libraries | CreateCuTensorHandle | CreateCuTensorHandle creates a new cuTENSOR handle |
| github.com/stitch1968/gocuda/libraries | CreateCutlassConv | CreateCutlassConv creates a CUTLASS convolution operation handle |
| github.com/stitch1968/gocuda/libraries | CreateCutlassGemm | CreateCutlassGemm creates a CUTLASS GEMM operation handle |
| github.com/stitch1968/gocuda/libraries | CreateDNNHandle | CreateDNNHandle creates a cuDNN library handle |
| github.com/stitch1968/gocuda/libraries | CreateDSSHandle | CreateDSSHandle creates a new cuDSS solver handle |
| github.com/stitch1968/gocuda/libraries | CreateDSSMatrix | CreateDSSMatrix creates a DSS matrix from sparse data |
| github.com/stitch1968/gocuda/libraries | CreateFFTContext | CreateFFTContext creates a new cuFFT context |
| github.com/stitch1968/gocuda/libraries | CreateNativeFFTContextOnDevice | CreateNativeFFTContextOnDevice creates a device-bound native-only cuFFT context. |
| github.com/stitch1968/gocuda/libraries | FFTContext.Backend | Backend reports whether the context uses native CUDA or Go simulation. |
| github.com/stitch1968/gocuda/libraries | CreateNativeRandomGeneratorOnDevice | CreateNativeRandomGeneratorOnDevice creates a device-bound native-only cuRAND generator. |
| github.com/stitch1968/gocuda/libraries | CreateNativeSolverContextOnDevice | CreateNativeSolverContextOnDevice creates a device-bound native-only cuSOLVER context. |
| github.com/stitch1968/gocuda/libraries | RandomGenerator.Backend | Backend reports whether the generator uses native CUDA or Go simulation. |
| github.com/stitch1968/gocuda/libraries | RandomGenerator.SetStream | SetStream attaches or detaches a native CUDA stream. |
| github.com/stitch1968/gocuda/libraries | SolverContext.Backend | Backend reports whether the solver context uses native CUDA or Go simulation. |
| github.com/stitch1968/gocuda/libraries | QRInfo.Tau | Tau returns the QR result's Householder scalars. |
| github.com/stitch1968/gocuda/libraries | LUInfo.Pivots | Pivots returns the LU result's one-based pivot indices. |
| github.com/stitch1968/gocuda/libraries | SVDInfo.SingularValues | SingularValues returns the SVD result's singular values. |
| github.com/stitch1968/gocuda/libraries | SVDInfo.U | U returns the SVD result's left singular vectors. |
| github.com/stitch1968/gocuda/libraries | SVDInfo.VT | VT returns the SVD result's transposed right singular vectors. |
| github.com/stitch1968/gocuda/libraries | CreateFilterDescriptor | CreateFilterDescriptor creates a filter descriptor |
| github.com/stitch1968/gocuda/libraries | CreateJpeg2000Decoder | CreateJpeg2000Decoder creates a new JPEG2000 decoder |
| github.com/stitch1968/gocuda/libraries | CreateJpeg2000Encoder | CreateJpeg2000Encoder creates a new JPEG2000 encoder |
| github.com/stitch1968/gocuda/libraries | CreateJpegDecoder | CreateJpegDecoder creates a new JPEG decoder |
| github.com/stitch1968/gocuda/libraries | CreateJpegEncoder | CreateJpegEncoder creates a new JPEG encoder |
| github.com/stitch1968/gocuda/libraries | CreateMathContext | CreateMathContext creates a new CUDA Math API context |
| github.com/stitch1968/gocuda/libraries | CreatePoolingDescriptor | CreatePoolingDescriptor creates a pooling descriptor |
| github.com/stitch1968/gocuda/libraries | CreateRandomGenerator | CreateRandomGenerator creates a new random number generator |
| github.com/stitch1968/gocuda/libraries | CreateSolverContext | CreateSolverContext creates a new cuSOLVER context |
| github.com/stitch1968/gocuda/libraries | CreateSparseContext | CreateSparseContext creates a new cuSPARSE context |
| github.com/stitch1968/gocuda/libraries | CreateNativeSparseContextOnDevice | CreateNativeSparseContextOnDevice creates a device-bound native-only sparse context. |
| github.com/stitch1968/gocuda/libraries | SparseContext.CreateCSRMatrix | CreateCSRMatrix creates a validated CSR matrix. |
| github.com/stitch1968/gocuda/libraries | SparseContext.CreateCOOMatrix | CreateCOOMatrix creates a validated COO matrix. |
| github.com/stitch1968/gocuda/libraries | SparseContext.SpMVBackend | SpMVBackend reports the backend for a sparse matrix format. |
| github.com/stitch1968/gocuda/libraries | SparseContext.SpSVBackend | SpSVBackend reports the backend for a triangular sparse matrix format. |
| github.com/stitch1968/gocuda/libraries | SparseContext.SpGEMMBackend | SpGEMMBackend reports the backend for sparse matrix multiplication. |
| github.com/stitch1968/gocuda/libraries | SparseContext.DenseToSparseBackend | DenseToSparseBackend reports the backend for a dense-to-sparse format. |
| github.com/stitch1968/gocuda/libraries | SparseContext.SparseToDenseBackend | SparseToDenseBackend reports the backend for a sparse-to-dense format. |
| github.com/stitch1968/gocuda/libraries | SparseMatrix.Values | Values returns the matrix-owned value buffer. |
| github.com/stitch1968/gocuda/libraries | SparseMatrix.RowData | RowData returns matrix-owned row indices or offsets. |
| github.com/stitch1968/gocuda/libraries | SparseMatrix.ColumnIndices | ColumnIndices returns the matrix-owned column indices. |
| github.com/stitch1968/gocuda/libraries | CreateTensorDescriptor | CreateTensorDescriptor creates a tensor descriptor |
| github.com/stitch1968/gocuda/libraries | CreateThrustContext | CreateThrustContext creates a new Thrust context |
| github.com/stitch1968/gocuda/libraries | CuTensorHandle.BatchedTensorContraction | BatchedTensorContraction performs batched tensor contractions |
| github.com/stitch1968/gocuda/libraries | CuTensorHandle.CreateContractionPlan | CreateContractionPlan creates an optimized execution plan for tensor contraction |
| github.com/stitch1968/gocuda/libraries | CuTensorHandle.Destroy | Destroy cleans up cuTENSOR handle resources |
| github.com/stitch1968/gocuda/libraries | CuTensorHandle.ExecuteContractionPlan | ExecuteContractionPlan executes a pre-compiled contraction plan |
| github.com/stitch1968/gocuda/libraries | CuTensorHandle.TensorContraction | TensorContraction performs general tensor contraction |
| github.com/stitch1968/gocuda/libraries | CuTensorHandle.TensorCopy | TensorCopy performs tensor copy with potential layout conversion |
| github.com/stitch1968/gocuda/libraries | CuTensorHandle.TensorElementwiseAdd | TensorElementwiseAdd performs element-wise addition |
| github.com/stitch1968/gocuda/libraries | CuTensorHandle.TensorElementwiseMul | TensorElementwiseMul performs element-wise multiplication |
| github.com/stitch1968/gocuda/libraries | CuTensorHandle.TensorPermute | TensorPermute performs tensor permutation (transpose generalization) |
| github.com/stitch1968/gocuda/libraries | CuTensorHandle.TensorReduce | TensorReduce performs tensor reduction along specified modes |
| github.com/stitch1968/gocuda/libraries | CutlassConvHandle.CutlassConv | CutlassConv performs convolution using CUTLASS |
| github.com/stitch1968/gocuda/libraries | CutlassConvHandle.Destroy | Destroy cleans up convolution handle resources |
| github.com/stitch1968/gocuda/libraries | CutlassGemmHandle.CutlassGemm | CutlassGemm performs General Matrix Multiplication using CUTLASS |
| github.com/stitch1968/gocuda/libraries | CutlassGemmHandle.CutlassGemmBatched | CutlassGemmBatched performs batched GEMM operations |
| github.com/stitch1968/gocuda/libraries | CutlassGemmHandle.Destroy | Destroy cleans up GEMM handle resources |
| github.com/stitch1968/gocuda/libraries | CutlassRank2k | CutlassRank2k performs rank-2k update: C = alpha*A*B^T + alpha*B*A^T + beta*C |
| github.com/stitch1968/gocuda/libraries | CutlassSpmm | CutlassSpmm performs Sparse Matrix-Dense Matrix Multiplication |
| github.com/stitch1968/gocuda/libraries | CutlassTrmm | CutlassTrmm performs Triangular Matrix Multiplication |
| github.com/stitch1968/gocuda/libraries | DNNHandle.ActivationForward | ActivationForward performs forward activation |
| github.com/stitch1968/gocuda/libraries | DNNHandle.BatchNormalizationForwardInference | BatchNormalizationForwardInference performs batch normalization inference |
| github.com/stitch1968/gocuda/libraries | DNNHandle.ConvolutionForward | ConvolutionForward performs forward convolution |
| github.com/stitch1968/gocuda/libraries | DNNHandle.DestroyHandle | DestroyHandle destroys the cuDNN handle |
| github.com/stitch1968/gocuda/libraries | DNNHandle.PoolingForward | PoolingForward performs forward pooling |
| github.com/stitch1968/gocuda/libraries | DSSHandle.Analyze | Analyze performs symbolic analysis of the sparse matrix |
| github.com/stitch1968/gocuda/libraries | DSSHandle.Destroy | Destroy cleans up DSS handle resources |
| github.com/stitch1968/gocuda/libraries | DSSHandle.Factor | Factor performs numerical factorization |
| github.com/stitch1968/gocuda/libraries | DSSHandle.GetDeterminant | GetDeterminant computes the determinant of the factored matrix |
| github.com/stitch1968/gocuda/libraries | DSSHandle.GetInertia | GetInertia computes the inertia of the factored matrix |
| github.com/stitch1968/gocuda/libraries | DSSHandle.Refactor | Refactor updates the numerical factorization with new values |
| github.com/stitch1968/gocuda/libraries | DSSHandle.Solve | Solve solves the linear system Ax = b |
| github.com/stitch1968/gocuda/libraries | DSSHandle.SolveMultiple | SolveMultiple solves multiple linear systems with the same matrix |
| github.com/stitch1968/gocuda/libraries | DSSMatrix.Destroy | Destroy cleans up DSS matrix resources |
| github.com/stitch1968/gocuda/libraries | DecodeJpeg2000Image | DecodeJpeg2000Image provides a simple interface for JPEG2000 decoding |
| github.com/stitch1968/gocuda/libraries | DecodeJpeg2000Quick | DecodeJpeg2000Quick provides a simple interface for JPEG2000 decoding |
| github.com/stitch1968/gocuda/libraries | DecodeJpegImage | DecodeJpegImage provides a simple interface for JPEG decoding |
| github.com/stitch1968/gocuda/libraries | DecodeJpegQuick | DecodeJpegQuick provides a simple interface for JPEG decoding |
| github.com/stitch1968/gocuda/libraries | EncodeJpeg2000Image | EncodeJpeg2000Image provides a simple interface for JPEG2000 encoding |
| github.com/stitch1968/gocuda/libraries | EncodeJpeg2000Lossless | EncodeJpeg2000Lossless provides lossless JPEG2000 encoding |
| github.com/stitch1968/gocuda/libraries | EncodeJpeg2000Quick | EncodeJpeg2000Quick provides a simple interface for JPEG2000 encoding |
| github.com/stitch1968/gocuda/libraries | EncodeJpegImage | EncodeJpegImage provides a simple interface for JPEG encoding |
| github.com/stitch1968/gocuda/libraries | EncodeJpegQuick | EncodeJpegQuick provides a simple interface for JPEG encoding |
| github.com/stitch1968/gocuda/libraries | ExperimentalHelpersEnabled | ExperimentalHelpersEnabled reports whether helper-backed library execution is explicitly allowed in CUDA mode. |
| github.com/stitch1968/gocuda/libraries | FFT1D | FFT1D provides simplified 1D FFT computation |
| github.com/stitch1968/gocuda/libraries | FFTContext.CreatePlan1D | CreatePlan1D creates a 1D FFT plan |
| github.com/stitch1968/gocuda/libraries | FFTContext.CreatePlan2D | CreatePlan2D creates a 2D FFT plan |
| github.com/stitch1968/gocuda/libraries | FFTContext.CreatePlan3D | CreatePlan3D creates a 3D FFT plan |
| github.com/stitch1968/gocuda/libraries | FFTContext.DestroyContext | DestroyContext destroys the cuFFT context |
| github.com/stitch1968/gocuda/libraries | FFTContext.EstimateMemory | EstimateMemory estimates memory requirements for FFT operations |
| github.com/stitch1968/gocuda/libraries | FFTContext.ExecC2C | ExecC2C executes a complex-to-complex FFT |
| github.com/stitch1968/gocuda/libraries | FFTContext.ExecC2R | ExecC2R executes a complex-to-real FFT |
| github.com/stitch1968/gocuda/libraries | FFTContext.ExecR2C | ExecR2C executes a real-to-complex FFT |
| github.com/stitch1968/gocuda/libraries | FFTPlan.DestroyPlan | DestroyPlan destroys an FFT plan |
| github.com/stitch1968/gocuda/libraries | FFTPlan.SetStream | SetStream sets the CUDA stream for the plan |
| github.com/stitch1968/gocuda/libraries | FastMathOperations | FastMathOperations provides optimized math operations with reduced precision |
| github.com/stitch1968/gocuda/libraries | FilterDescriptor.DestroyFilterDescriptor | DestroyFilterDescriptor destroys a filter descriptor |
| github.com/stitch1968/gocuda/libraries | FilterDescriptor.SetFilter4dDescriptor | SetFilter4dDescriptor sets the filter descriptor for 4D filters |
| github.com/stitch1968/gocuda/libraries | FilterDescriptor.SetFilterNdDescriptor | SetFilterNdDescriptor sets the filter descriptor for N-dimensional filters |
| github.com/stitch1968/gocuda/libraries | GemmOperation | GemmOperation performs a simple GEMM operation using CUTLASS |
| github.com/stitch1968/gocuda/libraries | GetCutlassVersion | GetCutlassVersion returns the simulated CUTLASS version |
| github.com/stitch1968/gocuda/libraries | GetFFTSize | GetFFTSize returns the optimal FFT size for a given input size |
| github.com/stitch1968/gocuda/libraries | GetJpeg2000ImageInfo | GetJpeg2000ImageInfo extracts detailed information from JPEG2000 data |
| github.com/stitch1968/gocuda/libraries | GetJpegImageInfo | GetJpegImageInfo extracts basic information from JPEG data without full decoding |
| github.com/stitch1968/gocuda/libraries | GetOptimalGemmAlgorithm | GetOptimalGemmAlgorithm suggests optimal GEMM algorithm based on problem size |
| github.com/stitch1968/gocuda/libraries | HelperBackedLibraries | HelperBackedLibraries returns the high-level wrappers that currently rely on helper-backed or simulated implementations in this repository. |
| github.com/stitch1968/gocuda/libraries | HighPrecisionMath | HighPrecisionMath provides IEEE 754 compliant high-precision operations |
| github.com/stitch1968/gocuda/libraries | Jpeg2000DecoderState.DecodeJpeg2000 | DecodeJpeg2000 decodes a JPEG2000 image from byte data |
| github.com/stitch1968/gocuda/libraries | Jpeg2000DecoderState.DecodeJpeg2000Batch | DecodeJpeg2000Batch decodes multiple JPEG2000 images in batch |
| github.com/stitch1968/gocuda/libraries | Jpeg2000DecoderState.Destroy | Destroy cleans up decoder resources |
| github.com/stitch1968/gocuda/libraries | Jpeg2000EncoderState.Destroy | Destroy cleans up encoder resources |
| github.com/stitch1968/gocuda/libraries | Jpeg2000EncoderState.EncodeJpeg2000 | EncodeJpeg2000 encodes image data to JPEG2000 format |
| github.com/stitch1968/gocuda/libraries | Jpeg2000EncoderState.SetCompressionRatio | SetCompressionRatio sets the JPEG2000 compression ratio |
| github.com/stitch1968/gocuda/libraries | Jpeg2000EncoderState.SetNumLayers | SetNumLayers sets the number of quality layers |
| github.com/stitch1968/gocuda/libraries | Jpeg2000EncoderState.SetNumLevels | SetNumLevels sets the number of wavelet decomposition levels |
| github.com/stitch1968/gocuda/libraries | JpegDecoderState.DecodeJpeg | DecodeJpeg decodes a JPEG image from byte data |
| github.com/stitch1968/gocuda/libraries | JpegDecoderState.DecodeJpegBatch | DecodeJpegBatch decodes multiple JPEG images in batch |
| github.com/stitch1968/gocuda/libraries | JpegDecoderState.Destroy | Destroy cleans up decoder resources |
| github.com/stitch1968/gocuda/libraries | JpegEncoderState.Destroy | Destroy cleans up encoder resources |
| github.com/stitch1968/gocuda/libraries | JpegEncoderState.EncodeJpeg | EncodeJpeg encodes image data to JPEG format |
| github.com/stitch1968/gocuda/libraries | JpegEncoderState.SetQuality | SetQuality sets the JPEG encoding quality (0-100) |
| github.com/stitch1968/gocuda/libraries | LUInfo.Destroy | No doc comment available. |
| github.com/stitch1968/gocuda/libraries | LibraryReadinessMatrix | LibraryReadinessMatrix returns the current readiness state for high-level CUDA ecosystem wrappers exposed by this repository. |
| github.com/stitch1968/gocuda/libraries | NewNativeBLAS | NewNativeBLAS creates a cuBLAS handle on a native stream. |
| github.com/stitch1968/gocuda/libraries | NativeBLAS.SGEMM | SGEMM enqueues row-major float32 matrix multiplication using cuBLAS. |
| github.com/stitch1968/gocuda/libraries | NativeBLAS.SGEMMEx | SGEMMEx enqueues float32 GEMM with transpose and row-stride options. |
| github.com/stitch1968/gocuda/libraries | NativeBLAS.DGEMM | DGEMM enqueues row-major float64 matrix multiplication using cuBLAS. |
| github.com/stitch1968/gocuda/libraries | NativeBLAS.DGEMMEx | DGEMMEx enqueues float64 GEMM with transpose and row-stride options. |
| github.com/stitch1968/gocuda/libraries | NativeBLAS.Close | Close destroys the cuBLAS handle. |
| github.com/stitch1968/gocuda/libraries | MathContext.BatchVectorOps | BatchVectorOps performs multiple vector operations in parallel |
| github.com/stitch1968/gocuda/libraries | MathContext.Destroy | Destroy cleans up math context resources |
| github.com/stitch1968/gocuda/libraries | MathContext.VectorAdd | VectorAdd performs element-wise addition: output[i] = a[i] + b[i] |
| github.com/stitch1968/gocuda/libraries | MathContext.VectorBesselJ0 | VectorBessel performs element-wise Bessel function J0: output[i] = j0(a[i]) |
| github.com/stitch1968/gocuda/libraries | MathContext.VectorComplexAbs | VectorComplexAbs performs element-wise complex absolute value: output[i] = \|a[i]\| |
| github.com/stitch1968/gocuda/libraries | MathContext.VectorComplexArg | VectorComplexArg performs element-wise complex argument: output[i] = arg(a[i]) |
| github.com/stitch1968/gocuda/libraries | MathContext.VectorCos | VectorCos performs element-wise cosine: output[i] = cos(a[i]) |
| github.com/stitch1968/gocuda/libraries | MathContext.VectorErf | VectorErf performs element-wise error function: output[i] = erf(a[i]) |
| github.com/stitch1968/gocuda/libraries | MathContext.VectorExp | VectorExp performs element-wise exponential: output[i] = exp(a[i]) |
| github.com/stitch1968/gocuda/libraries | MathContext.VectorFMA | VectorFMA performs fused multiply-add: output[i] = a[i] * b[i] + c[i] |
| github.com/stitch1968/gocuda/libraries | MathContext.VectorGamma | VectorGamma performs element-wise gamma function: output[i] = gamma(a[i]) |
| github.com/stitch1968/gocuda/libraries | MathContext.VectorLog | VectorLog performs element-wise natural logarithm: output[i] = log(a[i]) |
| github.com/stitch1968/gocuda/libraries | MathContext.VectorMax | VectorMax finds the maximum element in a vector |
| github.com/stitch1968/gocuda/libraries | MathContext.VectorMul | VectorMul performs element-wise multiplication: output[i] = a[i] * b[i] |
| github.com/stitch1968/gocuda/libraries | MathContext.VectorNorm | VectorNorm computes the L2 norm of a vector |
| github.com/stitch1968/gocuda/libraries | MathContext.VectorPow | VectorPow performs element-wise power: output[i] = pow(a[i], b[i]) |
| github.com/stitch1968/gocuda/libraries | MathContext.VectorRsqrt | VectorRsqrt performs element-wise reciprocal square root: output[i] = 1/sqrt(a[i]) |
| github.com/stitch1968/gocuda/libraries | MathContext.VectorSin | VectorSin performs element-wise sine: output[i] = sin(a[i]) |
| github.com/stitch1968/gocuda/libraries | MathContext.VectorSinCos | VectorSinCos computes both sine and cosine: sin_out[i] = sin(a[i]), cos_out[i] = cos(a[i]) |
| github.com/stitch1968/gocuda/libraries | MathContext.VectorSqrt | VectorSqrt performs element-wise square root: output[i] = sqrt(a[i]) |
| github.com/stitch1968/gocuda/libraries | MathContext.VectorSum | VectorSum computes the sum of all elements in a vector |
| github.com/stitch1968/gocuda/libraries | MathContext.VectorTan | VectorTan performs element-wise tangent: output[i] = tan(a[i]) |
| github.com/stitch1968/gocuda/libraries | MatrixMultiply | MatrixMultiply performs matrix multiplication using tensor contraction |
| github.com/stitch1968/gocuda/libraries | PoolingDescriptor.DestroyPoolingDescriptor | DestroyPoolingDescriptor destroys a pooling descriptor |
| github.com/stitch1968/gocuda/libraries | PoolingDescriptor.SetPooling2dDescriptor | SetPooling2dDescriptor sets the pooling descriptor for 2D pooling |
| github.com/stitch1968/gocuda/libraries | ProductionReadinessViolations | ProductionReadinessViolations returns the set of readiness issues that block the current high-level library surface from being considered production-ready. |
| github.com/stitch1968/gocuda/libraries | ProductionReadyLibraries | ProductionReadyLibraries returns the set of high-level wrappers currently considered production-ready for CUDA mode. |
| github.com/stitch1968/gocuda/libraries | QRInfo.Destroy | No doc comment available. |
| github.com/stitch1968/gocuda/libraries | RandomGenerator.Destroy | Destroy cleans up the generator |
| github.com/stitch1968/gocuda/libraries | RandomGenerator.GenerateLogNormal | GenerateLogNormal generates log-normal random numbers |
| github.com/stitch1968/gocuda/libraries | RandomGenerator.GenerateNormal | GenerateNormal generates normal random numbers (mean=0, stddev=1) |
| github.com/stitch1968/gocuda/libraries | RandomGenerator.GeneratePoisson | GeneratePoisson generates Poisson-distributed random numbers |
| github.com/stitch1968/gocuda/libraries | RandomGenerator.GenerateUniform | GenerateUniform generates uniform random numbers in [0, 1) |
| github.com/stitch1968/gocuda/libraries | RandomGenerator.SetSeed | SetSeed sets a pseudorandom generator's seed and resets its sequence offset. |
| github.com/stitch1968/gocuda/libraries | RandomNumbers | RandomNumbers provides a simplified interface for random number generation |
| github.com/stitch1968/gocuda/libraries | ReduceArray | ReduceArray provides simplified parallel reduction |
| github.com/stitch1968/gocuda/libraries | SVDInfo.Destroy | No doc comment available. |
| github.com/stitch1968/gocuda/libraries | SimpleContraction | SimpleContraction performs a simple tensor contraction with default settings |
| github.com/stitch1968/gocuda/libraries | SolveAMG | SolveAMG provides a simple interface for AMG solving |
| github.com/stitch1968/gocuda/libraries | SolveAmgX | SolveAmgX demonstrates AmgX usage for algebraic multigrid solving |
| github.com/stitch1968/gocuda/libraries | SolveDirect | SolveDirect provides a simple interface for direct sparse solving |
| github.com/stitch1968/gocuda/libraries | SolvePCG | SolvePCG solves using Preconditioned Conjugate Gradient with AMG preconditioning |
| github.com/stitch1968/gocuda/libraries | SolveSparseSystem | SolveSparseSystem demonstrates cuDSS usage for sparse linear systems |
| github.com/stitch1968/gocuda/libraries | SolveSymmetric | SolveSymmetric solves a symmetric positive definite system using Cholesky |
| github.com/stitch1968/gocuda/libraries | SolveSystem | SolveSystem provides simplified linear system solving |
| github.com/stitch1968/gocuda/libraries | SolverContext.CholeskyFactorization | CholeskyFactorization performs Cholesky decomposition for positive definite matrices |
| github.com/stitch1968/gocuda/libraries | SolverContext.DestroyContext | DestroyContext cleans up the solver context |
| github.com/stitch1968/gocuda/libraries | SolverContext.Eigenvalues | Eigenvalues computes eigenvalues and optionally eigenvectors |
| github.com/stitch1968/gocuda/libraries | SolverContext.LUFactorization | LUFactorization performs LU decomposition with partial pivoting |
| github.com/stitch1968/gocuda/libraries | SolverContext.PseudoInverse | PseudoInverse computes the Moore-Penrose pseudoinverse using SVD |
| github.com/stitch1968/gocuda/libraries | SolverContext.QRFactorization | QRFactorization performs QR decomposition of matrix A |
| github.com/stitch1968/gocuda/libraries | SolverContext.SVDDecomposition | SVDDecomposition performs Singular Value Decomposition |
| github.com/stitch1968/gocuda/libraries | SolverContext.SolveLinearSystem | SolveLinearSystem solves Ax = b using LU factorization |
| github.com/stitch1968/gocuda/libraries | SortArray | SortArray provides simplified array sorting using Thrust |
| github.com/stitch1968/gocuda/libraries | SparseContext.CreateSparseMatrix | CreateSparseMatrix creates a sparse matrix |
| github.com/stitch1968/gocuda/libraries | SparseContext.DenseToSparse | DenseToSparse converts a dense matrix to sparse format |
| github.com/stitch1968/gocuda/libraries | SparseContext.DestroyContext | DestroyContext destroys the cuSPARSE context |
| github.com/stitch1968/gocuda/libraries | SparseContext.SpGEMM | SpGEMM performs general sparse matrix-matrix multiplication |
| github.com/stitch1968/gocuda/libraries | SparseContext.SpLU | SpLU performs sparse LU factorization |
| github.com/stitch1968/gocuda/libraries | SparseContext.SpMM | SpMM performs sparse matrix-matrix multiplication: C = alpha*A*B + beta*C |
| github.com/stitch1968/gocuda/libraries | SparseContext.SpMV | SpMV performs sparse matrix-vector multiplication: y = alpha*A*x + beta*y |
| github.com/stitch1968/gocuda/libraries | SparseContext.SpSV | SpSV solves sparse triangular system: A*x = b |
| github.com/stitch1968/gocuda/libraries | SparseContext.SparseToDense | SparseToDense converts a sparse matrix to dense format |
| github.com/stitch1968/gocuda/libraries | SparseMatrix.Destroy | Destroy cleans up the sparse matrix |
| github.com/stitch1968/gocuda/libraries | SparseMatrix.GetMatrixInfo | GetMatrixInfo returns information about the sparse matrix |
| github.com/stitch1968/gocuda/libraries | SparseMatrixMultiply | SparseMatrixMultiply provides simplified sparse matrix multiplication |
| github.com/stitch1968/gocuda/libraries | TensorContract | TensorContract performs tensor contraction operations |
| github.com/stitch1968/gocuda/libraries | TensorDescriptor.DestroyTensorDescriptor | DestroyTensorDescriptor destroys a tensor descriptor |
| github.com/stitch1968/gocuda/libraries | TensorDescriptor.SetTensor4dDescriptor | SetTensor4dDescriptor sets the tensor descriptor for 4D tensors |
| github.com/stitch1968/gocuda/libraries | TensorDescriptor.SetTensorNdDescriptor | SetTensorNdDescriptor sets the tensor descriptor for N-dimensional tensors |
| github.com/stitch1968/gocuda/libraries | TensorElementwiseOp | TensorElementwiseOp performs element-wise operations on tensors |
| github.com/stitch1968/gocuda/libraries | TensorMatMul | TensorMatMul performs matrix multiplication using tensor operations |
| github.com/stitch1968/gocuda/libraries | TensorPlan.Destroy | Destroy destroys a tensor plan and frees its resources |
| github.com/stitch1968/gocuda/libraries | ThrustContext.Copy | Copy copies elements from source to destination |
| github.com/stitch1968/gocuda/libraries | ThrustContext.CopyIf | CopyIf copies elements that satisfy predicate |
| github.com/stitch1968/gocuda/libraries | ThrustContext.Count | Count counts occurrences of value |
| github.com/stitch1968/gocuda/libraries | ThrustContext.DestroyContext | DestroyContext cleans up Thrust context |
| github.com/stitch1968/gocuda/libraries | ThrustContext.ExclusiveScan | ExclusiveScan performs exclusive prefix sum |
| github.com/stitch1968/gocuda/libraries | ThrustContext.Fill | Fill fills memory with specified value |
| github.com/stitch1968/gocuda/libraries | ThrustContext.Find | Find locates first occurrence of value |
| github.com/stitch1968/gocuda/libraries | ThrustContext.Generate | Generate fills memory using generator function |
| github.com/stitch1968/gocuda/libraries | ThrustContext.MaxElement | MaxElement finds maximum element |
| github.com/stitch1968/gocuda/libraries | ThrustContext.Merge | Merge merges two sorted sequences |
| github.com/stitch1968/gocuda/libraries | ThrustContext.MinElement | MinElement finds minimum element |
| github.com/stitch1968/gocuda/libraries | ThrustContext.Partition | Partition partitions elements based on predicate |
| github.com/stitch1968/gocuda/libraries | ThrustContext.Reduce | Reduce performs parallel reduction |
| github.com/stitch1968/gocuda/libraries | ThrustContext.Scan | Scan performs inclusive prefix sum |
| github.com/stitch1968/gocuda/libraries | ThrustContext.SetIntersection | SetIntersection computes intersection of two sorted sequences |
| github.com/stitch1968/gocuda/libraries | ThrustContext.SetUnion | SetUnion computes union of two sorted sequences |
| github.com/stitch1968/gocuda/libraries | ThrustContext.Sort | Sort sorts elements in ascending order |
| github.com/stitch1968/gocuda/libraries | ThrustContext.SortByKey | SortByKey sorts key-value pairs by keys |
| github.com/stitch1968/gocuda/libraries | ThrustContext.Transform | Transform applies unary operation to each element |
| github.com/stitch1968/gocuda/libraries | ThrustContext.TransformBinary | TransformBinary applies binary operation to pairs of elements |
| github.com/stitch1968/gocuda/libraries | ThrustContext.Unique | Unique removes consecutive duplicate elements |
| github.com/stitch1968/gocuda/libraries | VectorMath | VectorMath performs element-wise mathematical operations with default settings |
| github.com/stitch1968/gocuda/libraries | VerifyProductionReadiness | VerifyProductionReadiness reports whether the currently exposed high-level library surface meets the repository's production-readiness gate. |
| github.com/stitch1968/gocuda/memory | Alloc | Alloc allocates memory. Call Free when the allocation is no longer needed. |
| github.com/stitch1968/gocuda/memory | AllocDeviceOnly | AllocDeviceOnly allocates device-only memory. Use copy helpers to access it. |
| github.com/stitch1968/gocuda/memory | AllocDeviceOnlyOnDevice | AllocDeviceOnlyOnDevice allocates device-only memory on the selected device. |
| github.com/stitch1968/gocuda/memory | AllocOnDevice | AllocOnDevice allocates device memory associated with a specific device ID. |
| github.com/stitch1968/gocuda/memory | AllocWithStream | AllocWithStream allocates memory on the GPU with a specific stream |
| github.com/stitch1968/gocuda/memory | AllocWithTypeAndStream | AllocWithTypeAndStream allocates memory with specific type and stream |
| github.com/stitch1968/gocuda/memory | CopyDeviceToDevice | CopyDeviceToDevice copies data between device memories |
| github.com/stitch1968/gocuda/memory | CopyDeviceToDevicePeer | CopyDeviceToDevicePeer performs an explicit device-to-device copy across devices using peer access when available or a host bounce buffer in simulation. |
| github.com/stitch1968/gocuda/memory | CopyDeviceToDevicePeerWithStream | CopyDeviceToDevicePeerWithStream performs an explicit cross-device copy. |
| github.com/stitch1968/gocuda/memory | CopyDeviceToDeviceWithStream | CopyDeviceToDeviceWithStream copies data between devices with a specific stream |
| github.com/stitch1968/gocuda/memory | CopyDeviceToHost | CopyDeviceToHost copies data from device to host memory |
| github.com/stitch1968/gocuda/memory | CopyDeviceToHostWithStream | CopyDeviceToHostWithStream copies data from device to host with a specific stream |
| github.com/stitch1968/gocuda/memory | CopyHostToDevice | CopyHostToDevice copies data from host to device memory |
| github.com/stitch1968/gocuda/memory | CopyHostToDeviceWithStream | CopyHostToDeviceWithStream copies data from host to device with a specific stream |
| github.com/stitch1968/gocuda/memory | GetInfo | GetInfo returns the total and free memory information |
| github.com/stitch1968/gocuda/memory | GetManager | GetManager returns the global memory manager |
| github.com/stitch1968/gocuda/memory | Manager.GetAllocationCount | GetAllocationCount returns the number of active allocations |
| github.com/stitch1968/gocuda/memory | Manager.GetTotalAllocated | GetTotalAllocated returns the total bytes allocated |
| github.com/stitch1968/gocuda/memory | Memory.Data | Data returns the simulation data slice |
| github.com/stitch1968/gocuda/memory | Memory.Free | Free releases memory unless an active retention or pool ownership prevents it. |
| github.com/stitch1968/gocuda/memory | Memory.Retain | Retain prevents Free while a consumer holds a retention. |
| github.com/stitch1968/gocuda/memory | Retention.Release | Release relinquishes a retention. |
| github.com/stitch1968/gocuda/memory | Memory.GetAlignment | GetAlignment returns the memory alignment |
| github.com/stitch1968/gocuda/memory | Memory.GetDeviceID | GetDeviceID returns the device ID where memory is allocated |
| github.com/stitch1968/gocuda/memory | Memory.GetPitch | GetPitch returns the memory pitch for 2D allocations |
| github.com/stitch1968/gocuda/memory | Memory.GetType | GetType returns the memory type |
| github.com/stitch1968/gocuda/memory | Memory.Ptr | Ptr returns the memory pointer |
| github.com/stitch1968/gocuda/memory | Memory.Size | Size returns the memory size |
| github.com/stitch1968/gocuda/memory | NewPool | NewPool creates a new memory pool |
| github.com/stitch1968/gocuda/memory | Pool.Get | Get gets memory from the pool or allocates new |
| github.com/stitch1968/gocuda/memory | Pool.Put | Put transfers compatible, unretained memory to the pool. |
| github.com/stitch1968/gocuda/memory | Pool.Close | Close frees cached allocations; checked-out allocations remain caller-owned. |
| github.com/stitch1968/gocuda/memory | View | View returns a typed slice over the allocation after validating size bounds. |
| github.com/stitch1968/gocuda/performance | AlgorithmMetrics.String | String returns formatted string representation of algorithm metrics |
| github.com/stitch1968/gocuda/performance | AsyncPipeline.AddStage | AddStage adds a computation stage to the pipeline |
| github.com/stitch1968/gocuda/performance | AsyncPipeline.Execute | Execute runs the async pipeline with overlapped compute and transfer |
| github.com/stitch1968/gocuda/performance | AsyncPipeline.GetMetrics | GetMetrics returns current pipeline performance metrics |
| github.com/stitch1968/gocuda/performance | AsyncPipeline.GetPipelineInfo | GetPipelineInfo returns information about the pipeline configuration |
| github.com/stitch1968/gocuda/performance | AsyncPipeline.OptimizePipeline | OptimizePipeline analyzes and optimizes the pipeline configuration |
| github.com/stitch1968/gocuda/performance | AsyncPipeline.Stop | Stop gracefully stops the async pipeline |
| github.com/stitch1968/gocuda/performance | BellmanFordGPU | BellmanFordGPU performs single-source shortest path with negative edge weights |
| github.com/stitch1968/gocuda/performance | BenchmarkAlgorithm | BenchmarkAlgorithm benchmarks an algorithm and returns performance metrics |
| github.com/stitch1968/gocuda/performance | BinarySearchBatched | BinarySearchBatched performs batched binary search on sorted arrays |
| github.com/stitch1968/gocuda/performance | Convolve1D | Convolve1D performs 1D convolution: output[i] = sum(signal[i+j] * kernel[j]) |
| github.com/stitch1968/gocuda/performance | Convolve2D | Convolve2D performs 2D convolution (commonly used in image processing) |
| github.com/stitch1968/gocuda/performance | DoubleBuffer.GetBackBuffer | GetBackBuffer returns the inactive buffer for background operations |
| github.com/stitch1968/gocuda/performance | DoubleBuffer.GetCurrentBuffer | GetCurrentBuffer returns the currently active buffer |
| github.com/stitch1968/gocuda/performance | DoubleBuffer.SwapBuffers | SwapBuffers swaps the primary and secondary buffers |
| github.com/stitch1968/gocuda/performance | FFTPlanner.Execute1D | Execute1D performs 1D FFT |
| github.com/stitch1968/gocuda/performance | FFTPlanner.Execute2D | Execute2D performs 2D FFT |
| github.com/stitch1968/gocuda/performance | FFTPlanner.ExecuteBatched | ExecuteBatched performs batched 1D FFT on multiple signals |
| github.com/stitch1968/gocuda/performance | FloydWarshallGPU | FloydWarshallGPU performs all-pairs shortest path |
| github.com/stitch1968/gocuda/performance | FusionReport.String | String returns a formatted string representation of the fusion report |
| github.com/stitch1968/gocuda/performance | GPUProfiler.GenerateReport | GenerateReport generates a comprehensive profiling report |
| github.com/stitch1968/gocuda/performance | GPUProfiler.RecordKernelExecution | RecordKernelExecution records a kernel execution for profiling |
| github.com/stitch1968/gocuda/performance | GPUProfiler.RecordMemoryAllocation | RecordMemoryAllocation records a memory allocation for tracking |
| github.com/stitch1968/gocuda/performance | GPUProfiler.RecordMemoryFree | RecordMemoryFree records a memory deallocation |
| github.com/stitch1968/gocuda/performance | GPUProfiler.StartProfiling | StartProfiling begins performance profiling |
| github.com/stitch1968/gocuda/performance | GPUProfiler.StopProfiling | StopProfiling stops performance profiling |
| github.com/stitch1968/gocuda/performance | JacobiEigenvalues | Eigenvalue computation (simplified Jacobi method) |
| github.com/stitch1968/gocuda/performance | KNearestNeighbors | KNearestNeighbors finds K nearest neighbors for query points |
| github.com/stitch1968/gocuda/performance | KernelFusion.CreateCustomFusedKernel | CreateCustomFusedKernel allows creation of custom fused kernels |
| github.com/stitch1968/gocuda/performance | KernelFusion.FusedElementwiseOperations | FusedElementwiseOperations combines multiple element-wise operations |
| github.com/stitch1968/gocuda/performance | KernelFusion.FusedMatrixMultiplyAdd | FusedMatrixMultiplyAdd performs fused matrix operations: D = A * B + C |
| github.com/stitch1968/gocuda/performance | KernelFusion.FusedVectorSaxpyAdd | FusedVectorSaxpyAdd performs fused scalar-vector operations: result = alpha * x + y + z |
| github.com/stitch1968/gocuda/performance | KernelFusion.GetPerformanceReport | GetPerformanceReport generates a report of all fused kernels performance |
| github.com/stitch1968/gocuda/performance | KernelFusion.OptimizeFusionStrategy | OptimizeFusionStrategy analyzes operation patterns and suggests optimal fusion strategies |
| github.com/stitch1968/gocuda/performance | LUDecomposition | LUDecomposition performs LU decomposition with partial pivoting |
| github.com/stitch1968/gocuda/performance | MatrixTranspose | MatrixTranspose performs in-place or out-of-place matrix transpose |
| github.com/stitch1968/gocuda/performance | MemoryBandwidthOptimizer.AnalyzeAccessPattern | AnalyzeAccessPattern performs comprehensive memory access pattern analysis |
| github.com/stitch1968/gocuda/performance | MemoryBandwidthOptimizer.BenchmarkMemoryPattern | BenchmarkMemoryPattern measures memory access performance |
| github.com/stitch1968/gocuda/performance | MemoryBandwidthOptimizer.GetOptimizationReport | GetOptimizationReport generates a comprehensive optimization report |
| github.com/stitch1968/gocuda/performance | MemoryBandwidthOptimizer.OptimalLayout | OptimalLayout converts data layout for optimal GPU memory access |
| github.com/stitch1968/gocuda/performance | MergeSortParallel | MergeSortParallel performs parallel merge sort |
| github.com/stitch1968/gocuda/performance | MultiGPU.DistributeData | DistributeData distributes data across multiple GPUs |
| github.com/stitch1968/gocuda/performance | MultiGPU.EnableP2P | EnableP2P enables peer-to-peer communication between GPUs |
| github.com/stitch1968/gocuda/performance | MultiGPU.GatherResults | GatherResults collects results from all devices back to the host |
| github.com/stitch1968/gocuda/performance | MultiGPU.GetDeviceInfo | GetDeviceInfo returns information about all devices |
| github.com/stitch1968/gocuda/performance | MultiGPU.GetMetrics | GetMetrics returns current multi-GPU metrics |
| github.com/stitch1968/gocuda/performance | MultiGPU.P2PCopy | P2PCopy performs direct GPU-to-GPU memory transfer |
| github.com/stitch1968/gocuda/performance | MultiGPU.ParallelExecute | ParallelExecute executes a kernel on all active GPUs |
| github.com/stitch1968/gocuda/performance | MultiGPUKernel.Execute | Execute implements the Kernel interface |
| github.com/stitch1968/gocuda/performance | MultiGPUKernel.GetName | GetName implements the Kernel interface |
| github.com/stitch1968/gocuda/performance | MultiGPUKernel.GetRequiredMemory | GetRequiredMemory implements the Kernel interface |
| github.com/stitch1968/gocuda/performance | MultiGPUMetrics.String | String returns a formatted string representation of multi-GPU metrics |
| github.com/stitch1968/gocuda/performance | NewAsyncPipeline | NewAsyncPipeline creates a new async execution pipeline |
| github.com/stitch1968/gocuda/performance | NewFFTPlanner | NewFFTPlanner creates an optimized FFT planner |
| github.com/stitch1968/gocuda/performance | NewGPUProfiler | NewGPUProfiler creates a new GPU profiler |
| github.com/stitch1968/gocuda/performance | NewKernelFusion | NewKernelFusion creates a new kernel fusion manager |
| github.com/stitch1968/gocuda/performance | NewMemoryBandwidthOptimizer | NewMemoryBandwidthOptimizer creates a new memory bandwidth optimizer |
| github.com/stitch1968/gocuda/performance | NewMultiGPU | NewMultiGPU creates a new multi-GPU manager |
| github.com/stitch1968/gocuda/performance | OptimizationReport.String | String returns a formatted string representation of the optimization report |
| github.com/stitch1968/gocuda/performance | PipelineMetrics.String | String returns a formatted string representation of pipeline metrics |
| github.com/stitch1968/gocuda/performance | ProfilingReport.String | String returns formatted profiling report |
| github.com/stitch1968/gocuda/performance | QRDecomposition | QRDecomposition performs QR decomposition using Householder reflections |
| github.com/stitch1968/gocuda/performance | QuickSortGPU | QuickSortGPU performs GPU-optimized quicksort |
| github.com/stitch1968/gocuda/performance | RadixSortOptimized | RadixSortOptimized performs GPU-optimized radix sort |
| github.com/stitch1968/gocuda/performance | SAXPY | SAXPY performs y = alpha*x + y (Single precision Alpha X Plus Y) |
| github.com/stitch1968/gocuda/performance | SDOT | SDOT computes the dot product x dot y |
| github.com/stitch1968/gocuda/performance | SGEMM | SGEMM performs matrix-matrix multiplication: C = alpha*A*B + beta*C |
| github.com/stitch1968/gocuda/performance | SGEMV | SGEMV performs matrix-vector multiplication: y = A*x + y |
| github.com/stitch1968/gocuda/performance | SGER | SGER performs rank-1 update: A = alpha*x*y^T + A |
| github.com/stitch1968/gocuda/performance | SNRM2 | SNRM2 computes the Euclidean norm \|\|x\|\| |
| github.com/stitch1968/gocuda/profiler | Clear | Clear clears global profiling data |
| github.com/stitch1968/gocuda/profiler | Disable | Disable disables global profiling |
| github.com/stitch1968/gocuda/profiler | Enable | Enable enables global profiling |
| github.com/stitch1968/gocuda/profiler | EndEvent | EndEvent ends timing a global event |
| github.com/stitch1968/gocuda/profiler | EventType.String | String returns the string representation of EventType |
| github.com/stitch1968/gocuda/profiler | GenerateReport | GenerateReport generates a comprehensive profiling report |
| github.com/stitch1968/gocuda/profiler | GetKernelProfiles | GetKernelProfiles returns global kernel profiles |
| github.com/stitch1968/gocuda/profiler | GetProfiler | GetProfiler returns the global profiler instance |
| github.com/stitch1968/gocuda/profiler | GetStatistics | GetStatistics returns global profiling statistics |
| github.com/stitch1968/gocuda/profiler | ProfileKernel | ProfileKernel profiles a kernel execution |
| github.com/stitch1968/gocuda/profiler | ProfileMemoryTransfer | ProfileMemoryTransfer profiles a memory transfer operation |
| github.com/stitch1968/gocuda/profiler | Profiler.Clear | Clear clears all profiling data |
| github.com/stitch1968/gocuda/profiler | Profiler.Disable | Disable disables profiling |
| github.com/stitch1968/gocuda/profiler | Profiler.Enable | Enable enables profiling |
| github.com/stitch1968/gocuda/profiler | Profiler.EndEvent | EndEvent ends timing an event and records it |
| github.com/stitch1968/gocuda/profiler | Profiler.GetEvents | GetEvents returns all recorded events |
| github.com/stitch1968/gocuda/profiler | Profiler.GetKernelProfiles | GetKernelProfiles returns profiling information for kernels |
| github.com/stitch1968/gocuda/profiler | Profiler.GetMemoryUsage | GetMemoryUsage returns current memory usage information |
| github.com/stitch1968/gocuda/profiler | Profiler.GetStatistics | GetStatistics returns profiling statistics |
| github.com/stitch1968/gocuda/profiler | Profiler.IsEnabled | IsEnabled returns whether profiling is enabled |
| github.com/stitch1968/gocuda/profiler | Profiler.RecordEvent | RecordEvent records a complete event |
| github.com/stitch1968/gocuda/profiler | Profiler.StartEvent | StartEvent starts timing an event |
| github.com/stitch1968/gocuda/profiler | Profiler.TrackAllocation | TrackAllocation tracks a memory allocation |
| github.com/stitch1968/gocuda/profiler | Profiler.TrackDeallocation | TrackDeallocation tracks a memory deallocation |
| github.com/stitch1968/gocuda/profiler | StartEvent | StartEvent starts timing a global event |
| github.com/stitch1968/gocuda/profiler | WriteReport | WriteReport writes a profiling report to a file |
| github.com/stitch1968/gocuda/streams | CreateEvent | CreateEvent creates a new CUDA event |
| github.com/stitch1968/gocuda/streams | CreateEventWithFlags | CreateEventWithFlags creates a new event with flags for CUDA compatibility |
| github.com/stitch1968/gocuda/streams | CreateHighPriorityStream | CreateHighPriorityStream creates a high-priority stream |
| github.com/stitch1968/gocuda/streams | CreateLowPriorityStream | CreateLowPriorityStream creates a low-priority stream |
| github.com/stitch1968/gocuda/streams | CreateStream | CreateStream creates a new stream with default settings |
| github.com/stitch1968/gocuda/streams | CreateStreamNonBlocking | CreateStreamNonBlocking creates a new non-blocking stream |
| github.com/stitch1968/gocuda/streams | CreateStreamOnDevice | CreateStreamOnDevice creates a new stream on a specific device with default settings. |
| github.com/stitch1968/gocuda/streams | Event.ElapsedTime | ElapsedTime returns the elapsed time between two events |
| github.com/stitch1968/gocuda/streams | Event.Query | Query checks if the event has completed |
| github.com/stitch1968/gocuda/streams | Event.Record | Record records the event in the stream |
| github.com/stitch1968/gocuda/streams | Event.Synchronize | Synchronize waits for the event to complete |
| github.com/stitch1968/gocuda/streams | ExecuteOnStream | ExecuteOnStream executes a function on a specific stream |
| github.com/stitch1968/gocuda/streams | ExecuteParallel | ExecuteParallel executes multiple functions in parallel on different streams |
| github.com/stitch1968/gocuda/streams | GetDefaultStream | GetDefaultStream returns the default stream |
| github.com/stitch1968/gocuda/streams | GetManager | GetManager returns the global stream manager |
| github.com/stitch1968/gocuda/streams | GetStreamProfile | GetStreamProfile returns profiling information for a stream |
| github.com/stitch1968/gocuda/streams | Manager.CreateStream | CreateStream creates a new CUDA stream |
| github.com/stitch1968/gocuda/streams | Manager.CreateStreamOnDevice | CreateStreamOnDevice creates a new CUDA stream bound to a specific device. |
| github.com/stitch1968/gocuda/streams | Manager.CreateStreamWithPriority | CreateStreamWithPriority creates a new CUDA stream with priority |
| github.com/stitch1968/gocuda/streams | Manager.CreateStreamWithPriorityOnDevice | CreateStreamWithPriorityOnDevice creates a new CUDA stream with priority on a specific device. |
| github.com/stitch1968/gocuda/streams | Manager.DestroyStream | DestroyStream destroys a stream |
| github.com/stitch1968/gocuda/streams | Manager.GetActiveStreams | GetActiveStreams returns the number of active streams |
| github.com/stitch1968/gocuda/streams | Manager.GetDefaultStream | GetDefaultStream returns the default stream |
| github.com/stitch1968/gocuda/streams | Manager.SynchronizeAll | SynchronizeAll waits for all streams to complete |
| github.com/stitch1968/gocuda/streams | MeasureKernelTime | MeasureKernelTime measures the execution time of a kernel |
| github.com/stitch1968/gocuda/streams | NewNativeEvent | NewNativeEvent creates a CUDA event on a device. |
| github.com/stitch1968/gocuda/streams | NewNativeStream | NewNativeStream creates a CUDA stream on a device. |
| github.com/stitch1968/gocuda/streams | NativeEvent.Close | Close destroys the CUDA event. |
| github.com/stitch1968/gocuda/streams | NativeEvent.ElapsedTime | ElapsedTime returns CUDA device time between completed events. |
| github.com/stitch1968/gocuda/streams | NativeEvent.Query | Query reports whether a recorded event has completed. |
| github.com/stitch1968/gocuda/streams | NativeEvent.Record | Record enqueues an event on a native stream. |
| github.com/stitch1968/gocuda/streams | NativeEvent.Synchronize | Synchronize waits for the recorded event. |
| github.com/stitch1968/gocuda/streams | NativeStream.Close | Close waits for device work and destroys the stream. |
| github.com/stitch1968/gocuda/streams | NativeStream.CopyAsync | CopyAsync queues a copy between device, managed, or pinned buffers. |
| github.com/stitch1968/gocuda/streams | NativeStream.DeviceID | DeviceID returns the stream device. |
| github.com/stitch1968/gocuda/streams | NativeStream.EnqueueWithBuffers | EnqueueWithBuffers retains buffers during a native stream submission. |
| github.com/stitch1968/gocuda/streams | NativeStream.Synchronize | Synchronize waits for queued device operations. |
| github.com/stitch1968/gocuda/streams | NativeStream.WaitEvent | WaitEvent makes the stream wait for a recorded CUDA event. |
| github.com/stitch1968/gocuda/streams | NativeStream.WithNativeHandle | WithNativeHandle runs a callback with a protected CUDA stream handle. |
| github.com/stitch1968/gocuda/streams | Stream.AddCallback | AddCallback adds a callback to be executed when the stream completes |
| github.com/stitch1968/gocuda/streams | Stream.GetPriority | GetPriority returns the stream priority |
| github.com/stitch1968/gocuda/streams | Stream.SetPriority | SetPriority sets the stream priority |
