//go:build ignore
// +build ignore

// Package main demonstrates the missing CUDA features that have been implemented
// This shows cuRAND, cuSPARSE, cuSOLVER, Thrust, and hardware-specific features
package main

import (
	"encoding/binary"
	"fmt"
	"log"
	"math"
	"time"

	"github.com/stitch1968/gocuda/hardware"
	"github.com/stitch1968/gocuda/libraries"
	"github.com/stitch1968/gocuda/memory"
)

func main() {
	fmt.Println("🚀 GoCUDA Missing Features Demo")
	fmt.Println("=====================================")

	// Demo cuRAND - Random Number Generation
	demoCuRAND()
	fmt.Println()

	// Demo cuSPARSE - Sparse Matrix Operations
	demoCuSPARSE()
	fmt.Println()

	// Demo cuSOLVER - Linear Algebra Solvers
	demoCuSOLVER()
	fmt.Println()

	// Demo Thrust - Parallel Algorithms
	demoThrust()
	fmt.Println()

	// Demo cuFFT - Fast Fourier Transform
	demoCuFFT()
	fmt.Println()

	// Demo cuDNN - Deep Neural Networks
	demoCuDNN()
	fmt.Println()

	// Demo Hardware-Specific Features
	demoHardwareFeatures()
	fmt.Println()

	fmt.Println("Demo complete; review any logged errors and backend distinctions.")
}

func demoCuRAND() {
	fmt.Println("📊 cuRAND - Random Number Generation")

	// Create random number generator
	rng, err := libraries.CreateRandomGenerator(libraries.RngTypeXorwow)
	if err != nil {
		log.Printf("Failed to create RNG: %v", err)
		return
	}
	defer rng.Destroy()

	// Set seed
	if err := rng.SetSeed(12345); err != nil {
		log.Printf("Failed to seed RNG: %v", err)
		return
	}

	// Generate uniform random numbers
	size := 10000
	output, err := memory.Alloc(int64(size * 4))
	if err != nil {
		log.Printf("Memory allocation failed: %v", err)
		return
	}
	defer output.Free()

	start := time.Now()
	err = rng.GenerateUniform(output, size)
	if err != nil {
		log.Printf("Uniform generation failed: %v", err)
		return
	}
	elapsed := time.Since(start)
	fmt.Printf("   ✅ Generated %d uniform random numbers in %v\n", size, elapsed)

	// Generate normal random numbers
	start = time.Now()
	err = rng.GenerateNormal(output, size, 0.0, 1.0)
	if err != nil {
		log.Printf("Normal generation failed: %v", err)
		return
	}
	elapsed = time.Since(start)
	fmt.Printf("   ✅ Generated %d normal random numbers (μ=0, σ=1) in %v\n", size, elapsed)

	// Generate Poisson random numbers
	start = time.Now()
	err = rng.GeneratePoisson(output, size, 5.0)
	if err != nil {
		log.Printf("Poisson generation failed: %v", err)
		return
	}
	elapsed = time.Since(start)
	fmt.Printf("   ✅ Generated %d Poisson random numbers (λ=5) in %v\n", size, elapsed)
}

func demoCuSPARSE() {
	fmt.Println("🕸️  cuSPARSE - Sparse Matrix Operations")

	ctx, err := libraries.CreateSparseContext()
	if err != nil {
		log.Printf("Failed to create sparse context: %v", err)
		return
	}
	defer ctx.DestroyContext()

	A, err := ctx.CreateCSRMatrix(3, 3,
		[]int32{0, 3, 6, 9},
		[]int32{0, 1, 2, 0, 1, 2, 0, 1, 2},
		[]float32{4, 2, 2, 2, 5, 1, 2, 1, 6})
	if err != nil {
		log.Printf("Failed to create sparse matrix A: %v", err)
		return
	}
	defer A.Destroy()

	B, err := ctx.CreateCSRMatrix(3, 2, []int32{0, 1, 2, 3}, []int32{0, 1, 0}, []float32{1, 2, 3})
	if err != nil {
		log.Printf("Failed to create sparse matrix B: %v", err)
		return
	}
	defer B.Destroy()

	x, err := memory.Alloc(3 * 4)
	if err != nil {
		log.Printf("Failed to allocate vector x: %v", err)
		return
	}
	defer x.Free()

	y, err := memory.Alloc(3 * 4)
	if err != nil {
		log.Printf("Failed to allocate vector y: %v", err)
		return
	}
	defer y.Free()

	if err := memory.CopyHostToDevice(x, demoFloat32Bytes([]float32{1, 2, -1})); err != nil {
		log.Printf("Failed to initialize vector x: %v", err)
		return
	}
	if err := memory.CopyHostToDevice(y, demoFloat32Bytes([]float32{0, 0, 0})); err != nil {
		log.Printf("Failed to initialize vector y: %v", err)
		return
	}
	backend, err := ctx.SpMVBackend(libraries.MatrixFormatCSR)
	if err != nil {
		log.Printf("Failed to determine SpMV backend: %v", err)
		return
	}
	start := time.Now()
	err = ctx.SpMV(1.0, A, x, 0.0, y)
	if err != nil {
		log.Printf("SpMV failed: %v", err)
		return
	}
	elapsed := time.Since(start)
	fmt.Printf("   ✅ CSR SpMV (%s) completed in %v\n", backend, elapsed)

	start = time.Now()
	productBackend, err := ctx.SpGEMMBackend(libraries.MatrixFormatCSR, libraries.MatrixFormatCSR)
	if err != nil {
		log.Printf("Failed to determine SpGEMM backend: %v", err)
		return
	}
	C, err := ctx.SpGEMM(A, B)
	if err != nil {
		log.Printf("SpGEMM failed: %v", err)
		return
	}
	defer C.Destroy()
	elapsed = time.Since(start)
	fmt.Printf("   ✅ Sparse matrix-matrix multiply (%s) completed in %v\n", productBackend, elapsed)

	start = time.Now()
	L, U, err := ctx.SpLU(A)
	if err != nil {
		log.Printf("SpLU failed: %v", err)
		return
	}
	defer L.Destroy()
	defer U.Destroy()
	elapsed = time.Since(start)
	fmt.Printf("   ✅ Sparse LU factorization without pivoting (host) completed in %v\n", elapsed)
}

func demoFloat32Bytes(values []float32) []byte {
	data := make([]byte, len(values)*4)
	for i, value := range values {
		binary.NativeEndian.PutUint32(data[i*4:], math.Float32bits(value))
	}
	return data
}

func demoCuSOLVER() {
	fmt.Println("🔧 cuSOLVER - Linear Algebra Solvers")

	// Create solver context
	ctx, err := libraries.CreateSolverContext()
	if err != nil {
		log.Printf("Failed to create solver context: %v", err)
		return
	}
	defer ctx.DestroyContext()

	const n = 3
	matrixValues := []float32{4, 2, 2, 2, 5, 1, 2, 1, 6}
	rightHandSide := []float32{6, 11, -2}
	matrixBytes := demoFloat32Bytes(matrixValues)
	fmt.Printf("   Backend: %s\n", ctx.Backend())

	// Create matrices
	A, err := memory.Alloc(int64(n * n * 4)) // n x n matrix
	if err != nil {
		log.Printf("Failed to allocate matrix A: %v", err)
		return
	}
	defer A.Free()

	b, err := memory.Alloc(int64(n * 4)) // Vector b
	if err != nil {
		log.Printf("Failed to allocate vector b: %v", err)
		return
	}
	defer b.Free()

	if err := memory.CopyHostToDevice(b, demoFloat32Bytes(rightHandSide)); err != nil {
		log.Printf("Failed to initialize vector b: %v", err)
		return
	}
	resetMatrix := func() bool {
		if err := memory.CopyHostToDevice(A, matrixBytes); err != nil {
			log.Printf("Failed to initialize matrix A: %v", err)
			return false
		}
		return true
	}

	// Perform QR factorization
	if !resetMatrix() {
		return
	}
	start := time.Now()
	qrInfo, err := ctx.QRFactorization(A, n, n)
	if err != nil {
		log.Printf("QR factorization failed: %v", err)
		return
	}
	defer qrInfo.Destroy()
	elapsed := time.Since(start)
	fmt.Printf("   ✅ QR factorization (%dx%d) completed in %v\n", n, n, elapsed)

	// Perform SVD decomposition
	if !resetMatrix() {
		return
	}
	start = time.Now()
	svdInfo, err := ctx.SVDDecomposition(A, n, n, true)
	if err != nil {
		log.Printf("SVD failed: %v", err)
		return
	}
	defer svdInfo.Destroy()
	elapsed = time.Since(start)
	fmt.Printf("   ✅ SVD decomposition (%dx%d) completed in %v\n", n, n, elapsed)

	// Solve linear system
	if !resetMatrix() {
		return
	}
	start = time.Now()
	x, err := ctx.SolveLinearSystem(A, b, n)
	if err != nil {
		log.Printf("Linear system solve failed: %v", err)
		return
	}
	defer x.Free()
	solutionBytes := make([]byte, n*4)
	if err := memory.CopyDeviceToHost(solutionBytes, x); err != nil {
		log.Printf("Failed to read solution: %v", err)
		return
	}
	solution := make([]float32, n)
	for i := range solution {
		solution[i] = math.Float32frombits(binary.NativeEndian.Uint32(solutionBytes[i*4:]))
	}
	elapsed = time.Since(start)
	fmt.Printf("   ✅ Linear system Ax=b solved in %v: x=%v\n", elapsed, solution)

	// Compute eigenvalues
	if !resetMatrix() {
		return
	}
	start = time.Now()
	eigenvals, eigenvecs, err := ctx.Eigenvalues(A, n, true)
	if err != nil {
		log.Printf("Eigenvalue computation failed: %v", err)
		return
	}
	defer eigenvals.Free()
	if eigenvecs != nil {
		defer eigenvecs.Free()
	}
	elapsed = time.Since(start)
	fmt.Printf("   ✅ Eigenvalue decomposition completed in %v\n", elapsed)

	// Cholesky factorization (for positive definite matrices)
	if !resetMatrix() {
		return
	}
	start = time.Now()
	err = ctx.CholeskyFactorization(A, n)
	if err != nil {
		log.Printf("Cholesky factorization failed: %v", err)
	} else {
		elapsed = time.Since(start)
		fmt.Printf("   ✅ Cholesky factorization completed in %v\n", elapsed)
	}
}

func demoThrust() {
	fmt.Println("⚡ Thrust - Parallel Algorithms")

	// Create Thrust context
	ctx, err := libraries.CreateThrustContext()
	if err != nil {
		log.Printf("Failed to create Thrust context: %v", err)
		return
	}
	defer ctx.DestroyContext()

	size := 100000

	// Create test data
	data, err := memory.Alloc(int64(size * 4))
	if err != nil {
		log.Printf("Failed to allocate data: %v", err)
		return
	}
	defer data.Free()

	output, err := memory.Alloc(int64(size * 4))
	if err != nil {
		log.Printf("Failed to allocate output: %v", err)
		return
	}
	defer output.Free()

	// Fill with generated data
	start := time.Now()
	err = ctx.Generate(data, size, "sequence", libraries.PolicyDevice)
	if err != nil {
		log.Printf("Generate failed: %v", err)
		return
	}
	elapsed := time.Since(start)
	fmt.Printf("   ✅ Generated %d elements in %v\n", size, elapsed)

	// Sort the data
	start = time.Now()
	err = ctx.Sort(data, size, libraries.PolicyDevice)
	if err != nil {
		log.Printf("Sort failed: %v", err)
		return
	}
	elapsed = time.Since(start)
	fmt.Printf("   ✅ Sorted %d elements in %v\n", size, elapsed)

	// Perform reduction
	start = time.Now()
	result, err := ctx.Reduce(data, size, 0.0, libraries.PolicyDevice)
	if err != nil {
		log.Printf("Reduce failed: %v", err)
		return
	}
	elapsed = time.Since(start)
	fmt.Printf("   ✅ Reduced %d elements (result: %.2f) in %v\n", size, result, elapsed)

	// Perform transform
	start = time.Now()
	err = ctx.Transform(data, output, size, "square", libraries.PolicyDevice)
	if err != nil {
		log.Printf("Transform failed: %v", err)
		return
	}
	elapsed = time.Since(start)
	fmt.Printf("   ✅ Transformed %d elements (square operation) in %v\n", size, elapsed)

	// Perform scan (prefix sum)
	start = time.Now()
	err = ctx.Scan(data, output, size, libraries.PolicyDevice)
	if err != nil {
		log.Printf("Scan failed: %v", err)
		return
	}
	elapsed = time.Since(start)
	fmt.Printf("   ✅ Prefix sum of %d elements in %v\n", size, elapsed)

	// Find min/max elements
	start = time.Now()
	minVal, minIdx, err := ctx.MinElement(data, size, libraries.PolicyDevice)
	if err != nil {
		log.Printf("MinElement failed: %v", err)
		return
	}
	maxVal, maxIdx, err := ctx.MaxElement(data, size, libraries.PolicyDevice)
	if err != nil {
		log.Printf("MaxElement failed: %v", err)
		return
	}
	elapsed = time.Since(start)
	fmt.Printf("   ✅ Found min/max: %.2f@%d, %.2f@%d in %v\n", minVal, minIdx, maxVal, maxIdx, elapsed)
}

func demoHardwareFeatures() {
	fmt.Println("🔧 Hardware-Specific Features")

	// Warp-level primitives
	fmt.Println("   🌊 Warp-Level Primitives:")

	warpInfo := hardware.GetWarpInfo()
	fmt.Printf("      • Warp ID: %d, Lane ID: %d, Warp Size: %d\n",
		warpInfo.WarpID, warpInfo.LaneID, warpInfo.WarpSize)

	// Warp shuffle operations
	shuffle := hardware.NewWarpShuffle(0xFFFFFFFF)
	testVal := float32(42.0)

	start := time.Now()
	result, err := shuffle.ShuffleDown(testVal, 1)
	if err != nil {
		log.Printf("Shuffle down failed: %v", err)
	} else {
		elapsed := time.Since(start)
		fmt.Printf("      • Shuffle down: %.2f -> %.2f in %v\n", testVal, result, elapsed)
	}

	// Warp reduction
	reduce := hardware.NewWarpReduce(0xFFFFFFFF)
	start = time.Now()
	sumResult, err := reduce.ReduceSum(testVal)
	if err != nil {
		log.Printf("Warp reduce failed: %v", err)
	} else {
		elapsed := time.Since(start)
		fmt.Printf("      • Warp sum reduction: %.2f -> %.2f in %v\n", testVal, sumResult, elapsed)
	}

	// Warp voting
	vote := hardware.NewWarpVote(0xFFFFFFFF)
	start = time.Now()
	allTrue := vote.All(true)
	anyTrue := vote.Any(true)
	ballot := vote.Ballot(true)
	elapsed := time.Since(start)
	fmt.Printf("      • Warp vote - All: %t, Any: %t, Ballot: 0x%08X in %v\n",
		allTrue, anyTrue, ballot, elapsed)

	// Cooperative Groups
	fmt.Println("   🤝 Cooperative Groups:")

	blockDim := [3]int{16, 16, 1}
	threadIdx := [3]int{0, 0, 0}
	threadBlock := hardware.NewThreadBlock(blockDim, threadIdx)

	fmt.Printf("      • Thread Block - Size: %d, Rank: %d, Valid: %t\n",
		threadBlock.Size(), threadBlock.ThreadRank(), threadBlock.IsValid())

	warp := hardware.NewWarp(0, 0)
	fmt.Printf("      • Warp Group - Size: %d, Rank: %d, Valid: %t\n",
		warp.Size(), warp.ThreadRank(), warp.IsValid())

	coalescedGroup := hardware.NewCoalescedGroup(0xFFFFFFFF, 0)
	fmt.Printf("      • Coalesced Group - Size: %d, Rank: %d, Valid: %t\n",
		coalescedGroup.Size(), coalescedGroup.ThreadRank(), coalescedGroup.IsValid())

	// Tensor Core support
	fmt.Println("   🧮 Tensor Core Support:")

	tensorInfo := hardware.GetTensorCoreInfo()
	fmt.Printf("      • Compute Capability: %d.%d\n",
		tensorInfo.ComputeCapability[0], tensorInfo.ComputeCapability[1])
	fmt.Printf("      • Precision Support - FP16: %t, BF16: %t, INT8: %t, INT4: %t\n",
		tensorInfo.SupportsFP16, tensorInfo.SupportsBF16,
		tensorInfo.SupportsINT8, tensorInfo.SupportsINT4)

	// Simulate Tensor Core GEMM
	m, n, k := 128, 128, 128
	A, _ := memory.Alloc(int64(m * k * 2)) // FP16
	B, _ := memory.Alloc(int64(k * n * 2))
	C, _ := memory.Alloc(int64(m * n * 4)) // FP32 accumulate
	D, _ := memory.Alloc(int64(m * n * 4))
	defer A.Free()
	defer B.Free()
	defer C.Free()
	defer D.Free()

	start = time.Now()
	err = hardware.TensorCoreMMA(A, B, C, D, m, n, k, "fp16")
	if err != nil {
		log.Printf("Tensor Core MMA failed: %v", err)
	} else {
		elapsed = time.Since(start)
		fmt.Printf("      • Tensor Core GEMM (FP16, %dx%dx%d) in %v\n", m, n, k, elapsed)
	}
}

func demoCuFFT() {
	fmt.Println("🌊 cuFFT - Fast Fourier Transform")

	// Create FFT context
	ctx, err := libraries.CreateFFTContext()
	if err != nil {
		log.Printf("Failed to create FFT context: %v", err)
		return
	}
	defer ctx.DestroyContext()

	// Demo 1D FFT
	size := 1024
	complexSize := int64(size * 8) // Complex64 = 2 * float32

	input, err := memory.Alloc(complexSize)
	if err != nil {
		log.Printf("Failed to allocate FFT input: %v", err)
		return
	}
	defer input.Free()

	output, err := memory.Alloc(complexSize)
	if err != nil {
		log.Printf("Failed to allocate FFT output: %v", err)
		return
	}
	defer output.Free()

	// Initialize input data
	inputData, err := memory.View[libraries.Complex64](input, size)
	if err != nil {
		log.Printf("Failed to create FFT input view: %v", err)
		return
	}
	for i := 0; i < size; i++ {
		inputData[i].Real = float32(i%100) / 100.0
		inputData[i].Imag = 0.0
	}

	// Create and execute 1D FFT plan
	plan, err := ctx.CreatePlan1D(size, libraries.FFTTypeC2C, 1)
	if err != nil {
		log.Printf("Failed to create FFT plan: %v", err)
		return
	}
	defer plan.DestroyPlan()

	start := time.Now()
	err = ctx.ExecC2C(plan, input, output, libraries.FFTForward)
	elapsed := time.Since(start)
	if err != nil {
		log.Printf("FFT execution error: %v", err)
	}

	fmt.Printf("      • 1D Complex FFT (%d points) completed in %v\n", size, elapsed)

	// Demo simplified API
	start = time.Now()
	err = libraries.FFT1D(input, output, size, true)
	elapsed = time.Since(start)
	if err != nil {
		log.Printf("Simple FFT error: %v", err)
	}

	fmt.Printf("      • Simplified FFT API (%d points) completed in %v\n", size, elapsed)

	// Demo Real-to-Complex FFT
	realInput, _ := memory.Alloc(int64(size * 4)) // float32
	defer realInput.Free()

	realData, err := memory.View[float32](realInput, size)
	if err != nil {
		log.Printf("Failed to create R2C input view: %v", err)
		return
	}
	for i := 0; i < size; i++ {
		realData[i] = float32(i%50) / 25.0
	}

	r2cPlan, err := ctx.CreatePlan1D(size, libraries.FFTTypeR2C, 1)
	if err != nil {
		log.Printf("Failed to create R2C plan: %v", err)
		return
	}
	defer r2cPlan.DestroyPlan()

	start = time.Now()
	err = ctx.ExecR2C(r2cPlan, realInput, output)
	elapsed = time.Since(start)
	if err != nil {
		log.Printf("R2C FFT error: %v", err)
	}

	fmt.Printf("      • Real-to-Complex FFT (%d points) completed in %v\n", size, elapsed)
}

func demoCuDNN() {
	fmt.Println("🧠 cuDNN - Deep Neural Networks")

	// Create cuDNN handle
	handle, err := libraries.CreateDNNHandle()
	if err != nil {
		log.Printf("Failed to create cuDNN handle: %v", err)
		return
	}
	defer handle.DestroyHandle()

	// Demo convolution
	batchSize, channels, height, width := 1, 16, 28, 28
	filterCount := 32
	filterSize := 5

	inputSize := int64(batchSize * channels * height * width * 4)
	filterMemSize := int64(filterCount * channels * filterSize * filterSize * 4)
	outputSize := int64(batchSize * filterCount * height * width * 4)

	input, _ := memory.Alloc(inputSize)
	defer input.Free()
	filter, _ := memory.Alloc(filterMemSize)
	defer filter.Free()
	output, _ := memory.Alloc(outputSize)
	defer output.Free()

	start := time.Now()
	err = libraries.ConvolutionForward(
		input, filter, output,
		[]int{batchSize, channels, height, width},
		[]int{filterCount, channels, filterSize, filterSize},
		[]int{batchSize, filterCount, height, width},
		2, 2, 1, 1, // padding and stride
	)
	elapsed := time.Since(start)
	if err != nil {
		log.Printf("Convolution error: %v", err)
	}

	fmt.Printf("      • Convolution (%dx%dx%dx%d * %dx%dx%dx%d) completed in %v\n",
		batchSize, channels, height, width, filterCount, channels, filterSize, filterSize, elapsed)

	// Demo activation functions
	activations := []struct {
		name string
		mode libraries.DNNActivationMode
	}{
		{"ReLU", libraries.DNNActivationRelu},
		{"Sigmoid", libraries.DNNActivationSigmoid},
		{"Tanh", libraries.DNNActivationTanh},
	}

	for _, activ := range activations {
		start = time.Now()
		err = libraries.ApplyActivation(
			input, output,
			[]int{batchSize, channels, height, width},
			activ.mode,
		)
		elapsed = time.Since(start)
		if err != nil {
			log.Printf("%s activation error: %v", activ.name, err)
		}

		fmt.Printf("      • %s activation (%dx%dx%dx%d) completed in %v\n",
			activ.name, batchSize, channels, height, width, elapsed)
	}

	// Demo pooling
	poolInput, _ := memory.Alloc(int64(batchSize * 64 * 32 * 32 * 4))
	defer poolInput.Free()
	poolOutput, _ := memory.Alloc(int64(batchSize * 64 * 16 * 16 * 4))
	defer poolOutput.Free()

	// Create descriptors for pooling
	inputDesc, _ := libraries.CreateTensorDescriptor()
	defer inputDesc.DestroyTensorDescriptor()
	outputDesc, _ := libraries.CreateTensorDescriptor()
	defer outputDesc.DestroyTensorDescriptor()
	poolDesc, _ := libraries.CreatePoolingDescriptor()
	defer poolDesc.DestroyPoolingDescriptor()

	inputDesc.SetTensor4dDescriptor(libraries.DNNTensorNCHW, libraries.DNNDataFloat, batchSize, 64, 32, 32)
	outputDesc.SetTensor4dDescriptor(libraries.DNNTensorNCHW, libraries.DNNDataFloat, batchSize, 64, 16, 16)
	poolDesc.SetPooling2dDescriptor(libraries.DNNPoolingMax, 2, 2, 0, 0, 2, 2)

	start = time.Now()
	err = handle.PoolingForward(poolDesc, 1.0, inputDesc, poolInput, 0.0, outputDesc, poolOutput)
	elapsed = time.Since(start)
	if err != nil {
		log.Printf("Pooling error: %v", err)
	}

	fmt.Printf("      • Max Pooling (1x64x32x32 -> 1x64x16x16) completed in %v\n", elapsed)
}
