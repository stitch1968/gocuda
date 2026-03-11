// Package libraries provides CUDA runtime library bindings for GoCUDA
// This implements CUTLASS functionality for high-performance CUDA C++ template library
package libraries

import (
	"errors"
	"fmt"
	"math"

	cuda "github.com/stitch1968/gocuda"
	"github.com/stitch1968/gocuda/memory"
)

// CUTLASS - CUDA Templates for Linear Algebra Subroutines and Solvers

// CUTLASS data types
type CutlassDataType int

const (
	CutlassFloat16 CutlassDataType = iota
	CutlassFloat32
	CutlassFloat64
	CutlassBFloat16
	CutlassInt8
	CutlassInt16
	CutlassInt32
	CutlassTensorFloat32
	CutlassComplexFloat32
	CutlassComplexFloat64
)

// CUTLASS matrix layouts
type CutlassLayout int

const (
	CutlassRowMajor CutlassLayout = iota
	CutlassColumnMajor
)

// CUTLASS operation types
type CutlassOperation int

const (
	CutlassOpN CutlassOperation = iota // No transpose
	CutlassOpT                         // Transpose
	CutlassOpC                         // Conjugate transpose
)

// CUTLASS GEMM algorithms
type CutlassGemmAlgorithm int

const (
	CutlassGemmDefault CutlassGemmAlgorithm = iota
	CutlassGemmSiMt
	CutlassGemmAnalytic
	CutlassGemmPlanar
	CutlassGemmWmma
	CutlassGemmTensorOp
	CutlassGemmSparseTensorOp
)

// CUTLASS epilogue operations
type CutlassEpilogueOp int

const (
	CutlassEpilogueLinearCombination CutlassEpilogueOp = iota
	CutlassEpilogueLinearCombinationClamp
	CutlassEpilogueBias
	CutlassEpilogueRelu
	CutlassEpilogueGelu
	CutlassEpilogueSigmoid
)

// CUTLASS GEMM descriptor
type CutlassGemmDesc struct {
	M, N, K      int
	DataType     CutlassDataType
	LayoutA      CutlassLayout
	LayoutB      CutlassLayout
	LayoutC      CutlassLayout
	OpA          CutlassOperation
	OpB          CutlassOperation
	Algorithm    CutlassGemmAlgorithm
	EpilogueOp   CutlassEpilogueOp
	Alpha        float32
	Beta         float32
	SplitKSlices int
}

// CUTLASS convolution modes
type CutlassConvMode int

const (
	CutlassConvForward CutlassConvMode = iota
	CutlassConvDgrad
	CutlassConvWgrad
)

// CUTLASS convolution descriptor
type CutlassConvDesc struct {
	N, H, W, C           int // Input dimensions
	K                    int // Output channels
	R, S                 int // Filter dimensions
	PadH, PadW           int // Padding
	StrideH, StrideW     int // Stride
	DilationH, DilationW int // Dilation
	Mode                 CutlassConvMode
	DataType             CutlassDataType
	Algorithm            CutlassGemmAlgorithm
}

// CUTLASS GEMM handle
type CutlassGemmHandle struct {
	handle       *memory.Memory
	descriptor   CutlassGemmDesc
	workspace    *memory.Memory
	nativeHandle uintptr
	native       bool
}

// CUTLASS convolution handle
type CutlassConvHandle struct {
	handle     *memory.Memory
	descriptor CutlassConvDesc
	workspace  *memory.Memory
}

var errCUTLASSUnsupported = errors.New("cutlass native path unsupported for requested parameters")

// CreateCutlassGemm creates a CUTLASS GEMM operation handle
func CreateCutlassGemm(desc CutlassGemmDesc) (*CutlassGemmHandle, error) {
	if err := ensureCudaReady(); err != nil {
		return nil, err
	}
	if cuda.ShouldUseCuda() && cutlassNativeAvailable() {
		handle, err := createNativeCutlassGemm(desc)
		if err == nil {
			return handle, nil
		}
	}
	return createDeterministicCutlassGemm(desc)
}

func createDeterministicCutlassGemm(desc CutlassGemmDesc) (*CutlassGemmHandle, error) {
	handle := &CutlassGemmHandle{
		descriptor: desc,
	}

	// Allocate handle memory
	var err error
	handle.handle, err = memory.Alloc(4096) // CUTLASS kernel state
	if err != nil {
		return nil, fmt.Errorf("failed to allocate CUTLASS GEMM handle: %v", err)
	}

	// Calculate and allocate workspace size
	workspaceSize := calculateGemmWorkspaceSize(desc)
	if workspaceSize > 0 {
		handle.workspace, err = memory.Alloc(int64(workspaceSize))
		if err != nil {
			handle.handle.Free()
			return nil, fmt.Errorf("failed to allocate GEMM workspace: %v", err)
		}
	}

	return handle, nil
}

// CreateCutlassConv creates a CUTLASS convolution operation handle
func CreateCutlassConv(desc CutlassConvDesc) (*CutlassConvHandle, error) {
	if err := ensureCudaReady(); err != nil {
		return nil, err
	}

	handle := &CutlassConvHandle{
		descriptor: desc,
	}

	// Allocate handle memory
	var err error
	handle.handle, err = memory.Alloc(8192) // Convolution needs more state memory
	if err != nil {
		return nil, fmt.Errorf("failed to allocate CUTLASS convolution handle: %v", err)
	}

	// Calculate and allocate workspace size
	workspaceSize := calculateConvWorkspaceSize(desc)
	if workspaceSize > 0 {
		handle.workspace, err = memory.Alloc(int64(workspaceSize))
		if err != nil {
			handle.handle.Free()
			return nil, fmt.Errorf("failed to allocate convolution workspace: %v", err)
		}
	}

	return handle, nil
}

// CutlassGemm performs General Matrix Multiplication using CUTLASS
func (handle *CutlassGemmHandle) CutlassGemm(A, B, C *memory.Memory) error {
	if A == nil || B == nil || C == nil {
		return fmt.Errorf("input matrices cannot be nil")
	}

	desc := handle.descriptor

	// Validate matrix dimensions
	if desc.M <= 0 || desc.N <= 0 || desc.K <= 0 {
		return fmt.Errorf("invalid matrix dimensions: M=%d, N=%d, K=%d", desc.M, desc.N, desc.K)
	}
	if handle != nil && handle.native {
		err := executeNativeCutlassGemm(handle, A, B, C)
		if err == nil {
			return nil
		}
		if !errors.Is(err, errCUTLASSUnsupported) {
			return err
		}
	}

	return executeCutlassGemm(desc, A, B, C)
}

// CutlassGemmBatched performs batched GEMM operations
func (handle *CutlassGemmHandle) CutlassGemmBatched(A, B, C []*memory.Memory, batchCount int) error {
	if len(A) != batchCount || len(B) != batchCount || len(C) != batchCount {
		return fmt.Errorf("batch arrays must have length %d", batchCount)
	}

	// Validate all matrices
	for i := range batchCount {
		if A[i] == nil || B[i] == nil || C[i] == nil {
			return fmt.Errorf("batch matrix %d cannot be nil", i)
		}
	}

	for index := range batchCount {
		if err := handle.CutlassGemm(A[index], B[index], C[index]); err != nil {
			return fmt.Errorf("CUTLASS batched GEMM execution failed at %d: %v", index, err)
		}
	}

	return nil
}

// CutlassConv performs convolution using CUTLASS
func (handle *CutlassConvHandle) CutlassConv(input, filter, output *memory.Memory) error {
	if input == nil || filter == nil || output == nil {
		return fmt.Errorf("input tensors cannot be nil")
	}

	desc := handle.descriptor

	// Calculate output dimensions
	outputH := (desc.H+2*desc.PadH-desc.DilationH*(desc.R-1)-1)/desc.StrideH + 1
	outputW := (desc.W+2*desc.PadW-desc.DilationW*(desc.S-1)-1)/desc.StrideW + 1

	if outputH <= 0 || outputW <= 0 {
		return fmt.Errorf("invalid convolution parameters result in non-positive output dimensions")
	}

	return executeCutlassConv(desc, input, filter, output, outputH, outputW)
}

// CutlassSpmm performs Sparse Matrix-Dense Matrix Multiplication
func CutlassSpmm(sparseA, denseB, denseC *memory.Memory, M, N, K int, sparsity float32) error {
	if sparseA == nil || denseB == nil || denseC == nil {
		return fmt.Errorf("input matrices cannot be nil")
	}

	if M <= 0 || N <= 0 || K <= 0 {
		return fmt.Errorf("invalid matrix dimensions: M=%d, N=%d, K=%d", M, N, K)
	}

	if sparsity < 0 || sparsity > 1 {
		return fmt.Errorf("sparsity must be between 0 and 1, got %f", sparsity)
	}

	return executeCutlassSpmm(sparseA, denseB, denseC, M, N, K)
}

// CutlassRank2k performs rank-2k update: C = alpha*A*B^T + alpha*B*A^T + beta*C
func CutlassRank2k(A, B, C *memory.Memory, N, K int, alpha, beta float32) error {
	if A == nil || B == nil || C == nil {
		return fmt.Errorf("input matrices cannot be nil")
	}

	if N <= 0 || K <= 0 {
		return fmt.Errorf("invalid matrix dimensions: N=%d, K=%d", N, K)
	}
	if cuda.ShouldUseCuda() && cutlassNativeAvailable() {
		err := executeNativeCutlassRank2k(A, B, C, N, K, alpha, beta)
		if err == nil {
			return nil
		}
		if !errors.Is(err, errCUTLASSUnsupported) {
			return err
		}
	}

	return executeCutlassRank2k(A, B, C, N, K, alpha, beta)
}

// CutlassTrmm performs Triangular Matrix Multiplication
func CutlassTrmm(A, B *memory.Memory, M, N int, side, uplo, trans, diag string, alpha float32) error {
	if A == nil || B == nil {
		return fmt.Errorf("input matrices cannot be nil")
	}

	if M <= 0 || N <= 0 {
		return fmt.Errorf("invalid matrix dimensions: M=%d, N=%d", M, N)
	}

	// Validate parameters
	validSides := map[string]bool{"Left": true, "Right": true}
	validUplo := map[string]bool{"Upper": true, "Lower": true}
	validTrans := map[string]bool{"NoTrans": true, "Trans": true, "ConjTrans": true}
	validDiag := map[string]bool{"NonUnit": true, "Unit": true}

	if !validSides[side] || !validUplo[uplo] || !validTrans[trans] || !validDiag[diag] {
		return fmt.Errorf("invalid TRMM parameters")
	}
	if cuda.ShouldUseCuda() && cutlassNativeAvailable() {
		err := executeNativeCutlassTrmm(A, B, M, N, side, uplo, trans, diag, alpha)
		if err == nil {
			return nil
		}
		if !errors.Is(err, errCUTLASSUnsupported) {
			return err
		}
	}

	return executeCutlassTrmm(A, B, M, N, side, uplo, trans, diag, alpha)
}

// calculateGemmWorkspaceSize calculates required workspace size for GEMM
func calculateGemmWorkspaceSize(desc CutlassGemmDesc) int {
	// Workspace size depends on algorithm and split-K slices
	baseSize := desc.M * desc.N * getDataTypeSize(desc.DataType)

	if desc.SplitKSlices > 1 {
		return baseSize * desc.SplitKSlices
	}

	switch desc.Algorithm {
	case CutlassGemmSiMt:
		return baseSize * 2
	case CutlassGemmAnalytic:
		return baseSize / 2
	case CutlassGemmWmma, CutlassGemmTensorOp:
		return baseSize / 4 // Tensor operations are more efficient
	default:
		return baseSize
	}
}

// calculateConvWorkspaceSize calculates required workspace size for convolution
func calculateConvWorkspaceSize(desc CutlassConvDesc) int {
	// Workspace size for implicit GEMM convolution
	outputH := (desc.H+2*desc.PadH-desc.DilationH*(desc.R-1)-1)/desc.StrideH + 1
	outputW := (desc.W+2*desc.PadW-desc.DilationW*(desc.S-1)-1)/desc.StrideW + 1

	baseSize := desc.N * outputH * outputW * desc.K * getDataTypeSize(desc.DataType)

	switch desc.Mode {
	case CutlassConvForward:
		return baseSize
	case CutlassConvDgrad:
		return baseSize * 2 // Need extra space for gradient computation
	case CutlassConvWgrad:
		return baseSize * 3 // Weight gradients need the most workspace
	default:
		return baseSize
	}
}

// getDataTypeSize returns the size in bytes for each data type
func getDataTypeSize(dataType CutlassDataType) int {
	switch dataType {
	case CutlassFloat16, CutlassBFloat16:
		return 2
	case CutlassFloat32, CutlassInt32, CutlassTensorFloat32:
		return 4
	case CutlassFloat64, CutlassComplexFloat32:
		return 8
	case CutlassComplexFloat64:
		return 16
	case CutlassInt8:
		return 1
	case CutlassInt16:
		return 2
	default:
		return 4
	}
}

// GetCutlassVersion returns the simulated CUTLASS version
func GetCutlassVersion() string {
	return "3.4.0" // Latest CUTLASS version as of simulation
}

// GetOptimalGemmAlgorithm suggests optimal GEMM algorithm based on problem size
func GetOptimalGemmAlgorithm(M, N, K int, dataType CutlassDataType) CutlassGemmAlgorithm {
	// Simple heuristics for algorithm selection
	problemSize := int64(M) * int64(N) * int64(K)

	// For small problems, use default
	if problemSize < 1000000 {
		return CutlassGemmDefault
	}

	// For tensor-friendly data types and sizes, use tensor operations
	if (dataType == CutlassFloat16 || dataType == CutlassBFloat16 || dataType == CutlassTensorFloat32) &&
		M%16 == 0 && N%16 == 0 && K%16 == 0 {
		return CutlassGemmTensorOp
	}

	// For very large problems, use analytical
	if problemSize > 100000000 {
		return CutlassGemmAnalytic
	}

	// Default to SiMt for general cases
	return CutlassGemmSiMt
}

// Destroy cleans up GEMM handle resources
func (handle *CutlassGemmHandle) Destroy() error {
	if handle != nil && handle.native {
		return destroyNativeCutlassGemm(handle)
	}
	if handle.handle != nil {
		_ = handle.handle.Free()
		handle.handle = nil
	}
	if handle.workspace != nil {
		_ = handle.workspace.Free()
		handle.workspace = nil
	}
	return nil
}

// Destroy cleans up convolution handle resources
func (handle *CutlassConvHandle) Destroy() error {
	if handle.handle != nil {
		_ = handle.handle.Free()
		handle.handle = nil
	}
	if handle.workspace != nil {
		_ = handle.workspace.Free()
		handle.workspace = nil
	}
	return nil
}

func executeCutlassGemm(desc CutlassGemmDesc, A, B, C *memory.Memory) error {
	if desc.DataType == CutlassFloat64 {
		valuesA, err := readMathFloat64Memory(A, desc.M*desc.K)
		if err != nil {
			return err
		}
		valuesB, err := readMathFloat64Memory(B, desc.K*desc.N)
		if err != nil {
			return err
		}
		valuesC, err := readMathFloat64Memory(C, desc.M*desc.N)
		if err != nil {
			return err
		}
		for row := 0; row < desc.M; row++ {
			for col := 0; col < desc.N; col++ {
				sum := 0.0
				for inner := 0; inner < desc.K; inner++ {
					sum += cutlassMatrixValue64(valuesA, desc.M, desc.K, row, inner, desc.LayoutA, desc.OpA) * cutlassMatrixValue64(valuesB, desc.K, desc.N, inner, col, desc.LayoutB, desc.OpB)
				}
				index := cutlassMatrixIndex(row, col, desc.M, desc.N, desc.LayoutC)
				valuesC[index] = float64(desc.Alpha)*sum + float64(desc.Beta)*valuesC[index]
				valuesC[index] = applyCutlassEpilogue64(valuesC[index], desc.EpilogueOp)
			}
		}
		return writeMathFloat64Memory(C, valuesC)
	}
	if desc.DataType != CutlassFloat32 {
		return fmt.Errorf("deterministic CUTLASS path supports CutlassFloat32 and CutlassFloat64, got %d", desc.DataType)
	}
	valuesA, err := readMathFloat32Memory(A, desc.M*desc.K)
	if err != nil {
		return err
	}
	valuesB, err := readMathFloat32Memory(B, desc.K*desc.N)
	if err != nil {
		return err
	}
	valuesC, err := readMathFloat32Memory(C, desc.M*desc.N)
	if err != nil {
		return err
	}
	for row := 0; row < desc.M; row++ {
		for col := 0; col < desc.N; col++ {
			sum := 0.0
			for inner := 0; inner < desc.K; inner++ {
				sum += float64(cutlassMatrixValue32(valuesA, desc.M, desc.K, row, inner, desc.LayoutA, desc.OpA)) * float64(cutlassMatrixValue32(valuesB, desc.K, desc.N, inner, col, desc.LayoutB, desc.OpB))
			}
			index := cutlassMatrixIndex(row, col, desc.M, desc.N, desc.LayoutC)
			valuesC[index] = float32(float64(desc.Alpha)*sum + float64(desc.Beta)*float64(valuesC[index]))
			valuesC[index] = float32(applyCutlassEpilogue64(float64(valuesC[index]), desc.EpilogueOp))
		}
	}
	return writeMathFloat32Memory(C, valuesC)
}

func executeCutlassConv(desc CutlassConvDesc, input, filter, output *memory.Memory, outputH, outputW int) error {
	if desc.DataType != CutlassFloat32 {
		return fmt.Errorf("deterministic CUTLASS convolution currently supports CutlassFloat32, got %d", desc.DataType)
	}
	switch desc.Mode {
	case CutlassConvForward:
		inValues, err := readMathFloat32Memory(input, desc.N*desc.H*desc.W*desc.C)
		if err != nil {
			return err
		}
		filterValues, err := readMathFloat32Memory(filter, desc.K*desc.R*desc.S*desc.C)
		if err != nil {
			return err
		}
		outValues := make([]float32, desc.N*outputH*outputW*desc.K)
		for n := 0; n < desc.N; n++ {
			for oh := range outputH {
				for ow := range outputW {
					for k := 0; k < desc.K; k++ {
						sum := float32(0)
						for r := 0; r < desc.R; r++ {
							for s := 0; s < desc.S; s++ {
								for c := 0; c < desc.C; c++ {
									ih := oh*desc.StrideH - desc.PadH + r*maxInt(desc.DilationH, 1)
									iw := ow*desc.StrideW - desc.PadW + s*maxInt(desc.DilationW, 1)
									if ih < 0 || ih >= desc.H || iw < 0 || iw >= desc.W {
										continue
									}
									inputIndex := (((n*desc.H)+ih)*desc.W+iw)*desc.C + c
									filterIndex := (((k*desc.R)+r)*desc.S+s)*desc.C + c
									sum += inValues[inputIndex] * filterValues[filterIndex]
								}
							}
						}
						outIndex := (((n*outputH)+oh)*outputW+ow)*desc.K + k
						outValues[outIndex] = sum
					}
				}
			}
		}
		return writeMathFloat32Memory(output, outValues)
	case CutlassConvDgrad:
		gradOut, err := readMathFloat32Memory(input, desc.N*outputH*outputW*desc.K)
		if err != nil {
			return err
		}
		filterValues, err := readMathFloat32Memory(filter, desc.K*desc.R*desc.S*desc.C)
		if err != nil {
			return err
		}
		gradIn := make([]float32, desc.N*desc.H*desc.W*desc.C)
		for n := 0; n < desc.N; n++ {
			for oh := range outputH {
				for ow := range outputW {
					for k := 0; k < desc.K; k++ {
						grad := gradOut[(((n*outputH)+oh)*outputW+ow)*desc.K+k]
						for r := 0; r < desc.R; r++ {
							for s := 0; s < desc.S; s++ {
								for c := 0; c < desc.C; c++ {
									ih := oh*desc.StrideH - desc.PadH + r*maxInt(desc.DilationH, 1)
									iw := ow*desc.StrideW - desc.PadW + s*maxInt(desc.DilationW, 1)
									if ih < 0 || ih >= desc.H || iw < 0 || iw >= desc.W {
										continue
									}
									gradIn[(((n*desc.H)+ih)*desc.W+iw)*desc.C+c] += grad * filterValues[(((k*desc.R)+r)*desc.S+s)*desc.C+c]
								}
							}
						}
					}
				}
			}
		}
		return writeMathFloat32Memory(output, gradIn)
	case CutlassConvWgrad:
		inputValues, err := readMathFloat32Memory(input, desc.N*desc.H*desc.W*desc.C)
		if err != nil {
			return err
		}
		gradOut, err := readMathFloat32Memory(filter, desc.N*outputH*outputW*desc.K)
		if err != nil {
			return err
		}
		gradW := make([]float32, desc.K*desc.R*desc.S*desc.C)
		for n := 0; n < desc.N; n++ {
			for oh := range outputH {
				for ow := range outputW {
					for k := 0; k < desc.K; k++ {
						grad := gradOut[(((n*outputH)+oh)*outputW+ow)*desc.K+k]
						for r := 0; r < desc.R; r++ {
							for s := 0; s < desc.S; s++ {
								for c := 0; c < desc.C; c++ {
									ih := oh*desc.StrideH - desc.PadH + r*maxInt(desc.DilationH, 1)
									iw := ow*desc.StrideW - desc.PadW + s*maxInt(desc.DilationW, 1)
									if ih < 0 || ih >= desc.H || iw < 0 || iw >= desc.W {
										continue
									}
									gradW[(((k*desc.R)+r)*desc.S+s)*desc.C+c] += grad * inputValues[(((n*desc.H)+ih)*desc.W+iw)*desc.C+c]
								}
							}
						}
					}
				}
			}
		}
		return writeMathFloat32Memory(output, gradW)
	default:
		return fmt.Errorf("unsupported CUTLASS convolution mode: %d", desc.Mode)
	}
}

func executeCutlassSpmm(sparseA, denseB, denseC *memory.Memory, M, N, K int) error {
	aValues, err := readMathFloat32Memory(sparseA, M*K)
	if err != nil {
		return err
	}
	bValues, err := readMathFloat32Memory(denseB, K*N)
	if err != nil {
		return err
	}
	cValues := make([]float32, M*N)
	for row := range M {
		for col := range N {
			sum := float32(0)
			for inner := range K {
				sum += aValues[row*K+inner] * bValues[inner*N+col]
			}
			cValues[row*N+col] = sum
		}
	}
	return writeMathFloat32Memory(denseC, cValues)
}

func executeCutlassRank2k(A, B, C *memory.Memory, N, K int, alpha, beta float32) error {
	aValues, err := readMathFloat32Memory(A, N*K)
	if err != nil {
		return err
	}
	bValues, err := readMathFloat32Memory(B, N*K)
	if err != nil {
		return err
	}
	cValues, err := readMathFloat32Memory(C, N*N)
	if err != nil {
		return err
	}
	for row := range N {
		for col := range N {
			sum := float32(0)
			for inner := range K {
				sum += aValues[row*K+inner]*bValues[col*K+inner] + bValues[row*K+inner]*aValues[col*K+inner]
			}
			cValues[row*N+col] = alpha*sum + beta*cValues[row*N+col]
		}
	}
	return writeMathFloat32Memory(C, cValues)
}

func executeCutlassTrmm(A, B *memory.Memory, M, N int, side, uplo, trans, diag string, alpha float32) error {
	aValues, err := readMathFloat32Memory(A, M*M)
	if err != nil {
		return err
	}
	bValues, err := readMathFloat32Memory(B, M*N)
	if err != nil {
		return err
	}
	result := make([]float32, len(bValues))
	copy(result, bValues)
	if side == "Left" {
		for row := range M {
			for col := range N {
				sum := float32(0)
				for inner := range M {
					if !cutlassTriangularAllowed(row, inner, uplo) {
						continue
					}
					value := cutlassTriangularValue(aValues, M, row, inner, trans, diag)
					sum += value * bValues[inner*N+col]
				}
				result[row*N+col] = alpha * sum
			}
		}
	} else {
		for row := range M {
			for col := range N {
				sum := float32(0)
				for inner := range N {
					if !cutlassTriangularAllowed(col, inner, uplo) {
						continue
					}
					value := cutlassTriangularValue(aValues, N, col, inner, trans, diag)
					sum += bValues[row*N+inner] * value
				}
				result[row*N+col] = alpha * sum
			}
		}
	}
	return writeMathFloat32Memory(B, result)
}

func cutlassMatrixValue32(values []float32, rows, cols, row, col int, layout CutlassLayout, op CutlassOperation) float32 {
	if op != CutlassOpN {
		row, col = col, row
	}
	return values[cutlassMatrixIndex(row, col, rows, cols, layout)]
}

func cutlassMatrixValue64(values []float64, rows, cols, row, col int, layout CutlassLayout, op CutlassOperation) float64 {
	if op != CutlassOpN {
		row, col = col, row
	}
	return values[cutlassMatrixIndex(row, col, rows, cols, layout)]
}

func cutlassMatrixIndex(row, col, rows, cols int, layout CutlassLayout) int {
	if layout == CutlassColumnMajor {
		return col*rows + row
	}
	return row*cols + col
}

func applyCutlassEpilogue64(value float64, op CutlassEpilogueOp) float64 {
	switch op {
	case CutlassEpilogueRelu:
		if value < 0 {
			return 0
		}
		return value
	case CutlassEpilogueGelu:
		return 0.5 * value * (1 + math.Tanh(math.Sqrt(2/math.Pi)*(value+0.044715*math.Pow(value, 3))))
	case CutlassEpilogueSigmoid:
		return 1 / (1 + math.Exp(-value))
	default:
		return value
	}
}

func cutlassTriangularAllowed(row, col int, uplo string) bool {
	if uplo == "Upper" {
		return col >= row
	}
	return col <= row
}

func cutlassTriangularValue(values []float32, order, row, col int, trans, diag string) float32 {
	if diag == "Unit" && row == col {
		return 1
	}
	if trans == "Trans" || trans == "ConjTrans" {
		row, col = col, row
	}
	return values[row*order+col]
}

func maxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}
