// Package libraries provides missing CUDA library functionality
// This creates the main export interface for all CUDA runtime libraries
package libraries

import (
	"github.com/stitch1968/gocuda/memory"
)

// Library initialization functions
var (
	// cuRAND
	CreateRNG = CreateRandomGenerator

	// cuSPARSE
	CreateSparseCtx = CreateSparseContext

	// cuSOLVER
	CreateSolverCtx = CreateSolverContext

	// Thrust
	CreateThrustCtx = CreateThrustContext

	// cuFFT
	CreateFFTCtx = CreateFFTContext

	// cuDNN
	CreateDNNCtx = CreateDNNHandle

	// nvJPEG
	CreateJpegDec = CreateJpegDecoder
	CreateJpegEnc = CreateJpegEncoder

	// nvJPEG2000
	CreateJpeg2000Dec = CreateJpeg2000Decoder
	CreateJpeg2000Enc = CreateJpeg2000Encoder

	// CUTLASS
	CreateCutlassGemmCtx = CreateCutlassGemm
	CreateCutlassConvCtx = CreateCutlassConv
)

// Quick access patterns for common operations

// RandomNumbers provides a simplified interface for random number generation
func RandomNumbers(size int, rngType RngType) ([]float32, error) {
	rng, err := CreateRandomGenerator(rngType)
	if err != nil {
		return nil, err
	}
	defer rng.Destroy()

	output, err := memory.Alloc(int64(size * 4)) // float32
	if err != nil {
		return nil, err
	}
	defer output.Free()

	err = rng.GenerateUniform(output, size)
	if err != nil {
		return nil, err
	}

	// Generate realistic random data for simulation
	result := make([]float32, size)
	for i := range result {
		result[i] = rng.rng.Float32() // Use the actual random number generator
	}

	return result, nil
}

// SparseMatrixMultiply provides simplified sparse matrix multiplication
func SparseMatrixMultiply(A, B *SparseMatrix) (*SparseMatrix, error) {
	ctx, err := CreateSparseContext()
	if err != nil {
		return nil, err
	}
	defer ctx.DestroyContext()

	return ctx.SpGEMM(A, B)
}

// SolveSystem provides simplified linear system solving
func SolveSystem(A, b *memory.Memory, n int) (*memory.Memory, error) {
	ctx, err := CreateSolverContext()
	if err != nil {
		return nil, err
	}
	defer ctx.DestroyContext()

	return ctx.SolveLinearSystem(A, b, n)
}

// SortArray provides simplified array sorting using Thrust
func SortArray(data *memory.Memory, n int) error {
	ctx, err := CreateThrustContext()
	if err != nil {
		return err
	}
	defer ctx.DestroyContext()

	return ctx.Sort(data, n, PolicyDevice)
}

// ReduceArray provides simplified parallel reduction
func ReduceArray(data *memory.Memory, n int) (float32, error) {
	ctx, err := CreateThrustContext()
	if err != nil {
		return 0, err
	}
	defer ctx.DestroyContext()

	return ctx.Reduce(data, n, 0.0, PolicyDevice)
}

// FFT1D provides simplified 1D FFT computation
func FFT1D(input, output *memory.Memory, size int, forward bool) error {
	ctx, err := CreateFFTContext()
	if err != nil {
		return err
	}
	defer ctx.DestroyContext()

	plan, err := ctx.CreatePlan1D(size, FFTTypeC2C, 1)
	if err != nil {
		return err
	}
	defer plan.DestroyPlan()

	direction := FFTForward
	if !forward {
		direction = FFTInverse
	}

	return ctx.ExecC2C(plan, input, output, direction)
}

// ConvolutionForward provides simplified convolution operation
func ConvolutionForward(input, filter, output *memory.Memory, inputDims, filterDims, outputDims []int,
	padH, padW, strideH, strideW int) error {

	handle, err := CreateDNNHandle()
	if err != nil {
		return err
	}
	defer handle.DestroyHandle()

	// Create descriptors
	inputDesc, err := CreateTensorDescriptor()
	if err != nil {
		return err
	}
	defer inputDesc.DestroyTensorDescriptor()

	filterDesc, err := CreateFilterDescriptor()
	if err != nil {
		return err
	}
	defer filterDesc.DestroyFilterDescriptor()

	outputDesc, err := CreateTensorDescriptor()
	if err != nil {
		return err
	}
	defer outputDesc.DestroyTensorDescriptor()

	convDesc, err := CreateConvolutionDescriptor()
	if err != nil {
		return err
	}
	defer convDesc.DestroyConvolutionDescriptor()

	// Set descriptor parameters
	if len(inputDims) == 4 {
		inputDesc.SetTensor4dDescriptor(DNNTensorNCHW, DNNDataFloat, inputDims[0], inputDims[1], inputDims[2], inputDims[3])
	}
	if len(filterDims) == 4 {
		filterDesc.SetFilter4dDescriptor(DNNDataFloat, DNNTensorNCHW, filterDims[0], filterDims[1], filterDims[2], filterDims[3])
	}
	if len(outputDims) == 4 {
		outputDesc.SetTensor4dDescriptor(DNNTensorNCHW, DNNDataFloat, outputDims[0], outputDims[1], outputDims[2], outputDims[3])
	}

	convDesc.SetConvolution2dDescriptor(padH, padW, strideH, strideW, 1, 1, DNNConvolution, DNNDataFloat)

	// Perform convolution
	return handle.ConvolutionForward(1.0, inputDesc, input, filterDesc, filter, convDesc, 0.0, outputDesc, output)
}

// ApplyActivation provides simplified activation function application
func ApplyActivation(input, output *memory.Memory, dims []int, activationType DNNActivationMode) error {
	handle, err := CreateDNNHandle()
	if err != nil {
		return err
	}
	defer handle.DestroyHandle()

	// Create descriptors
	inputDesc, err := CreateTensorDescriptor()
	if err != nil {
		return err
	}
	defer inputDesc.DestroyTensorDescriptor()

	outputDesc, err := CreateTensorDescriptor()
	if err != nil {
		return err
	}
	defer outputDesc.DestroyTensorDescriptor()

	activDesc, err := CreateActivationDescriptor()
	if err != nil {
		return err
	}
	defer activDesc.DestroyActivationDescriptor()

	// Set descriptors
	if len(dims) == 4 {
		inputDesc.SetTensor4dDescriptor(DNNTensorNCHW, DNNDataFloat, dims[0], dims[1], dims[2], dims[3])
		outputDesc.SetTensor4dDescriptor(DNNTensorNCHW, DNNDataFloat, dims[0], dims[1], dims[2], dims[3])
	}
	activDesc.SetActivationDescriptor(activationType, DNNNotPropagateNaN, 0.0)

	// Apply activation
	return handle.ActivationForward(activDesc, 1.0, inputDesc, input, 0.0, outputDesc, output)
}

// JPEG convenience functions

// DecodeJpegImage provides a simple interface for JPEG decoding
func DecodeJpegImage(jpegData []byte) (*memory.Memory, int, int, error) {
	return DecodeJpegQuick(jpegData, JpegFormatRGB)
}

// EncodeJpegImage provides a simple interface for JPEG encoding
func EncodeJpegImage(imageData *memory.Memory, width, height int, quality int) ([]byte, error) {
	return EncodeJpegQuick(imageData, width, height, JpegFormatRGB, quality)
}

// JPEG2000 convenience functions

// DecodeJpeg2000Image provides a simple interface for JPEG2000 decoding
func DecodeJpeg2000Image(j2kData []byte) (*memory.Memory, int, int, error) {
	return DecodeJpeg2000Quick(j2kData, Jpeg2000FormatRGB)
}

// EncodeJpeg2000Image provides a simple interface for JPEG2000 encoding
func EncodeJpeg2000Image(imageData *memory.Memory, width, height int, compressionRatio float32) ([]byte, error) {
	return EncodeJpeg2000Quick(imageData, width, height, Jpeg2000FormatRGB, compressionRatio)
}

// CUTLASS convenience functions

// GemmOperation performs a simple GEMM operation using CUTLASS
func GemmOperation(A, B, C *memory.Memory, M, N, K int, alpha, beta float32) error {
	desc := CutlassGemmDesc{
		M:            M,
		N:            N,
		K:            K,
		DataType:     CutlassFloat32,
		LayoutA:      CutlassRowMajor,
		LayoutB:      CutlassRowMajor,
		LayoutC:      CutlassRowMajor,
		OpA:          CutlassOpN,
		OpB:          CutlassOpN,
		Algorithm:    GetOptimalGemmAlgorithm(M, N, K, CutlassFloat32),
		EpilogueOp:   CutlassEpilogueLinearCombination,
		Alpha:        alpha,
		Beta:         beta,
		SplitKSlices: 1,
	}

	handle, err := CreateCutlassGemm(desc)
	if err != nil {
		return err
	}
	defer handle.Destroy()

	return handle.CutlassGemm(A, B, C)
}
