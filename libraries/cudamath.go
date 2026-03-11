// Package libraries provides CUDA runtime library bindings for GoCUDA
// This implements CUDA Math API functionality for advanced mathematical functions
package libraries

import (
	"fmt"
	"math"
	"math/cmplx"
	"unsafe"

	"github.com/stitch1968/gocuda/memory"
)

// CUDA Math API - High-Performance Mathematical Functions Library

// Math operation types
type MathOperation int

const (
	// Basic operations
	MathOpAdd MathOperation = iota
	MathOpSub
	MathOpMul
	MathOpDiv
	MathOpFMA
	MathOpSqrt
	MathOpCbrt
	MathOpRsqrt
	MathOpRcp

	// Trigonometric functions
	MathOpSin
	MathOpCos
	MathOpTan
	MathOpAsin
	MathOpAcos
	MathOpAtan
	MathOpAtan2
	MathOpSincos
	MathOpSinpi
	MathOpCospi
	MathOpTanpi

	// Hyperbolic functions
	MathOpSinh
	MathOpCosh
	MathOpTanh
	MathOpAsinh
	MathOpAcosh
	MathOpAtanh

	// Exponential and logarithmic
	MathOpExp
	MathOpExp2
	MathOpExp10
	MathOpExpm1
	MathOpLog
	MathOpLog2
	MathOpLog10
	MathOpLog1p
	MathOpLogb
	MathOpPow
	MathOpPowi

	// Special functions
	MathOpErf
	MathOpErfc
	MathOpErfinv
	MathOpErfcinv
	MathOpGamma
	MathOpLgamma
	MathOpTgamma
	MathOpJ0
	MathOpJ1
	MathOpY0
	MathOpY1
	MathOpJn
	MathOpYn

	// Rounding and remainder
	MathOpCeil
	MathOpFloor
	MathOpTrunc
	MathOpRound
	MathOpRint
	MathOpNearbyint
	MathOpFmod
	MathOpRemainder
	MathOpRemquo
	MathOpModf
	MathOpFrexp
	MathOpLdexp

	// Comparison and classification
	MathOpFmax
	MathOpFmin
	MathOpFdim
	MathOpIsnan
	MathOpIsinf
	MathOpIsfinite
	MathOpSignbit
	MathOpCopysign

	// Complex operations
	MathOpCabs
	MathOpCarg
	MathOpConj
	MathOpCproj
	MathOpCreal
	MathOpCimag
)

// Math precision modes
type MathPrecision int

const (
	MathPrecisionFast     MathPrecision = iota // Fast, lower precision
	MathPrecisionDefault                       // Balanced precision/performance
	MathPrecisionAccurate                      // High precision
	MathPrecisionIEEE                          // IEEE 754 compliant
)

// Math data types
type MathDataType int

const (
	MathDataFloat32 MathDataType = iota
	MathDataFloat64
	MathDataComplexFloat32
	MathDataComplexFloat64
	MathDataHalf
	MathDataBFloat16
)

// Math configuration
type MathConfig struct {
	Precision   MathPrecision
	DataType    MathDataType
	VectorSize  int
	UseHardware bool // Use hardware-specific optimizations
	FastMath    bool // Enable fast math optimizations
	FlushToZero bool // Flush denormalized numbers to zero
	HandleNaN   bool // Special handling for NaN values
	HandleInf   bool // Special handling for infinity values
}

// Math context for batch operations
type MathContext struct {
	handle    *memory.Memory
	config    MathConfig
	workspace *memory.Memory
	streams   []*memory.Memory
}

// Math vector operation descriptor
type MathVectorOp struct {
	Operation MathOperation
	InputA    *memory.Memory
	InputB    *memory.Memory // Optional second input
	Output    *memory.Memory
	Size      int
	Config    MathConfig
}

// CreateMathContext creates a new CUDA Math API context
func CreateMathContext(config MathConfig) (*MathContext, error) {
	if err := ensureCudaReady(); err != nil {
		return nil, err
	}

	ctx := &MathContext{
		config: config,
	}

	// Allocate context handle
	var err error
	ctx.handle, err = memory.Alloc(4096)
	if err != nil {
		return nil, fmt.Errorf("failed to allocate math context handle: %v", err)
	}

	// Allocate workspace for intermediate computations
	workspaceSize := calculateMathWorkspaceSize(config)
	if workspaceSize > 0 {
		ctx.workspace, err = memory.Alloc(int64(workspaceSize))
		if err != nil {
			ctx.handle.Free()
			return nil, fmt.Errorf("failed to allocate math workspace: %v", err)
		}
	}

	// Create execution streams for parallel operations
	numStreams := 4
	ctx.streams = make([]*memory.Memory, numStreams)
	for i := range numStreams {
		ctx.streams[i], err = memory.Alloc(1024)
		if err != nil {
			// Clean up previously allocated streams
			for j := 0; j < i; j++ {
				ctx.streams[j].Free()
			}
			ctx.handle.Free()
			if ctx.workspace != nil {
				ctx.workspace.Free()
			}
			return nil, fmt.Errorf("failed to allocate stream %d: %v", i, err)
		}
	}

	return ctx, nil
}

// Elementary Math Functions

// VectorAdd performs element-wise addition: output[i] = a[i] + b[i]
func (ctx *MathContext) VectorAdd(a, b, output *memory.Memory, size int) error {
	return ctx.executeVectorOp(MathVectorOp{
		Operation: MathOpAdd,
		InputA:    a,
		InputB:    b,
		Output:    output,
		Size:      size,
		Config:    ctx.config,
	})
}

// VectorMul performs element-wise multiplication: output[i] = a[i] * b[i]
func (ctx *MathContext) VectorMul(a, b, output *memory.Memory, size int) error {
	return ctx.executeVectorOp(MathVectorOp{
		Operation: MathOpMul,
		InputA:    a,
		InputB:    b,
		Output:    output,
		Size:      size,
		Config:    ctx.config,
	})
}

// VectorFMA performs fused multiply-add: output[i] = a[i] * b[i] + c[i]
func (ctx *MathContext) VectorFMA(a, b, c, output *memory.Memory, size int) error {
	// For FMA, we use InputB as the multiplier and need a separate call for c
	err := ctx.executeVectorOp(MathVectorOp{
		Operation: MathOpFMA,
		InputA:    a,
		InputB:    b,
		Output:    output,
		Size:      size,
		Config:    ctx.config,
	})
	if err != nil {
		return err
	}

	// Add c to the result
	return ctx.VectorAdd(output, c, output, size)
}

// VectorSqrt performs element-wise square root: output[i] = sqrt(a[i])
func (ctx *MathContext) VectorSqrt(a, output *memory.Memory, size int) error {
	return ctx.executeVectorOp(MathVectorOp{
		Operation: MathOpSqrt,
		InputA:    a,
		Output:    output,
		Size:      size,
		Config:    ctx.config,
	})
}

// VectorRsqrt performs element-wise reciprocal square root: output[i] = 1/sqrt(a[i])
func (ctx *MathContext) VectorRsqrt(a, output *memory.Memory, size int) error {
	return ctx.executeVectorOp(MathVectorOp{
		Operation: MathOpRsqrt,
		InputA:    a,
		Output:    output,
		Size:      size,
		Config:    ctx.config,
	})
}

// Trigonometric Functions

// VectorSin performs element-wise sine: output[i] = sin(a[i])
func (ctx *MathContext) VectorSin(a, output *memory.Memory, size int) error {
	return ctx.executeVectorOp(MathVectorOp{
		Operation: MathOpSin,
		InputA:    a,
		Output:    output,
		Size:      size,
		Config:    ctx.config,
	})
}

// VectorCos performs element-wise cosine: output[i] = cos(a[i])
func (ctx *MathContext) VectorCos(a, output *memory.Memory, size int) error {
	return ctx.executeVectorOp(MathVectorOp{
		Operation: MathOpCos,
		InputA:    a,
		Output:    output,
		Size:      size,
		Config:    ctx.config,
	})
}

// VectorSinCos computes both sine and cosine: sin_out[i] = sin(a[i]), cos_out[i] = cos(a[i])
func (ctx *MathContext) VectorSinCos(a, sin_out, cos_out *memory.Memory, size int) error {
	// Compute sine
	err := ctx.VectorSin(a, sin_out, size)
	if err != nil {
		return err
	}

	// Compute cosine
	return ctx.VectorCos(a, cos_out, size)
}

// VectorTan performs element-wise tangent: output[i] = tan(a[i])
func (ctx *MathContext) VectorTan(a, output *memory.Memory, size int) error {
	return ctx.executeVectorOp(MathVectorOp{
		Operation: MathOpTan,
		InputA:    a,
		Output:    output,
		Size:      size,
		Config:    ctx.config,
	})
}

// Exponential and Logarithmic Functions

// VectorExp performs element-wise exponential: output[i] = exp(a[i])
func (ctx *MathContext) VectorExp(a, output *memory.Memory, size int) error {
	return ctx.executeVectorOp(MathVectorOp{
		Operation: MathOpExp,
		InputA:    a,
		Output:    output,
		Size:      size,
		Config:    ctx.config,
	})
}

// VectorLog performs element-wise natural logarithm: output[i] = log(a[i])
func (ctx *MathContext) VectorLog(a, output *memory.Memory, size int) error {
	return ctx.executeVectorOp(MathVectorOp{
		Operation: MathOpLog,
		InputA:    a,
		Output:    output,
		Size:      size,
		Config:    ctx.config,
	})
}

// VectorPow performs element-wise power: output[i] = pow(a[i], b[i])
func (ctx *MathContext) VectorPow(a, b, output *memory.Memory, size int) error {
	return ctx.executeVectorOp(MathVectorOp{
		Operation: MathOpPow,
		InputA:    a,
		InputB:    b,
		Output:    output,
		Size:      size,
		Config:    ctx.config,
	})
}

// Special Functions

// VectorErf performs element-wise error function: output[i] = erf(a[i])
func (ctx *MathContext) VectorErf(a, output *memory.Memory, size int) error {
	return ctx.executeVectorOp(MathVectorOp{
		Operation: MathOpErf,
		InputA:    a,
		Output:    output,
		Size:      size,
		Config:    ctx.config,
	})
}

// VectorGamma performs element-wise gamma function: output[i] = gamma(a[i])
func (ctx *MathContext) VectorGamma(a, output *memory.Memory, size int) error {
	return ctx.executeVectorOp(MathVectorOp{
		Operation: MathOpGamma,
		InputA:    a,
		Output:    output,
		Size:      size,
		Config:    ctx.config,
	})
}

// VectorBessel performs element-wise Bessel function J0: output[i] = j0(a[i])
func (ctx *MathContext) VectorBesselJ0(a, output *memory.Memory, size int) error {
	return ctx.executeVectorOp(MathVectorOp{
		Operation: MathOpJ0,
		InputA:    a,
		Output:    output,
		Size:      size,
		Config:    ctx.config,
	})
}

// Complex Math Functions

// VectorComplexAbs performs element-wise complex absolute value: output[i] = |a[i]|
func (ctx *MathContext) VectorComplexAbs(a, output *memory.Memory, size int) error {
	// Ensure we're working with complex data
	config := ctx.config
	if config.DataType != MathDataComplexFloat32 && config.DataType != MathDataComplexFloat64 {
		return fmt.Errorf("complex absolute value requires complex input data")
	}

	return ctx.executeVectorOp(MathVectorOp{
		Operation: MathOpCabs,
		InputA:    a,
		Output:    output,
		Size:      size,
		Config:    config,
	})
}

// VectorComplexArg performs element-wise complex argument: output[i] = arg(a[i])
func (ctx *MathContext) VectorComplexArg(a, output *memory.Memory, size int) error {
	config := ctx.config
	if config.DataType != MathDataComplexFloat32 && config.DataType != MathDataComplexFloat64 {
		return fmt.Errorf("complex argument requires complex input data")
	}

	return ctx.executeVectorOp(MathVectorOp{
		Operation: MathOpCarg,
		InputA:    a,
		Output:    output,
		Size:      size,
		Config:    config,
	})
}

// executeVectorOp executes a vector operation
func (ctx *MathContext) executeVectorOp(op MathVectorOp) error {
	if op.InputA == nil || op.Output == nil {
		return fmt.Errorf("input and output memories cannot be nil")
	}

	if op.Size <= 0 {
		return fmt.Errorf("vector size must be positive: %d", op.Size)
	}

	config := op.Config
	if config.DataType == 0 {
		config.DataType = ctx.config.DataType
	}

	switch config.DataType {
	case MathDataFloat32, MathDataHalf, MathDataBFloat16:
		return executeMathVectorFloat32(op)
	case MathDataFloat64:
		return executeMathVectorFloat64(op)
	case MathDataComplexFloat32:
		return executeMathVectorComplex64(op)
	case MathDataComplexFloat64:
		return executeMathVectorComplex128(op)
	default:
		return fmt.Errorf("unsupported math data type: %d", config.DataType)
	}
}

// Batch Operations for Multiple Vectors

// BatchVectorOps performs multiple vector operations in parallel
func (ctx *MathContext) BatchVectorOps(ops []MathVectorOp) error {
	if len(ops) == 0 {
		return fmt.Errorf("no operations provided")
	}

	// Group operations by type for better scheduling
	opGroups := make(map[MathOperation][]MathVectorOp)
	for _, op := range ops {
		opGroups[op.Operation] = append(opGroups[op.Operation], op)
	}

	// Execute operation groups in parallel using streams
	streamIdx := 0
	for opType, groupOps := range opGroups {
		for _, op := range groupOps {
			err := ctx.executeVectorOp(op)
			if err != nil {
				return fmt.Errorf("batch operation %v failed: %v", opType, err)
			}
			streamIdx = (streamIdx + 1) % len(ctx.streams)
		}
	}

	return nil
}

// Statistical and Reduction Operations

// VectorSum computes the sum of all elements in a vector
func (ctx *MathContext) VectorSum(a *memory.Memory, size int) (float64, error) {
	if a == nil {
		return 0, fmt.Errorf("input memory cannot be nil")
	}
	if size <= 0 {
		return 0, fmt.Errorf("vector size must be positive: %d", size)
	}

	switch ctx.config.DataType {
	case MathDataFloat64:
		values, err := readMathFloat64Memory(a, size)
		if err != nil {
			return 0, err
		}
		total := 0.0
		for _, value := range values {
			total += value
		}
		return total, nil
	case MathDataComplexFloat32:
		values, err := readMathComplex64Memory(a, size)
		if err != nil {
			return 0, err
		}
		total := 0.0
		for _, value := range values {
			total += float64(real(value))
		}
		return total, nil
	case MathDataComplexFloat64:
		values, err := readMathComplex128Memory(a, size)
		if err != nil {
			return 0, err
		}
		total := 0.0
		for _, value := range values {
			total += real(value)
		}
		return total, nil
	default:
		values, err := readMathFloat32Memory(a, size)
		if err != nil {
			return 0, err
		}
		total := 0.0
		for _, value := range values {
			total += float64(value)
		}
		return total, nil
	}
}

// VectorMax finds the maximum element in a vector
func (ctx *MathContext) VectorMax(a *memory.Memory, size int) (float64, int, error) {
	if a == nil {
		return 0, 0, fmt.Errorf("input memory cannot be nil")
	}
	if size <= 0 {
		return 0, 0, fmt.Errorf("vector size must be positive: %d", size)
	}

	switch ctx.config.DataType {
	case MathDataFloat64:
		values, err := readMathFloat64Memory(a, size)
		if err != nil {
			return 0, 0, err
		}
		maxValue, maxIndex := values[0], 0
		for index := 1; index < len(values); index++ {
			if values[index] > maxValue {
				maxValue = values[index]
				maxIndex = index
			}
		}
		return maxValue, maxIndex, nil
	case MathDataComplexFloat32, MathDataComplexFloat64:
		return 0, 0, fmt.Errorf("vector max is not defined for complex math data types")
	default:
		values, err := readMathFloat32Memory(a, size)
		if err != nil {
			return 0, 0, err
		}
		maxValue, maxIndex := values[0], 0
		for index := 1; index < len(values); index++ {
			if values[index] > maxValue {
				maxValue = values[index]
				maxIndex = index
			}
		}
		return float64(maxValue), maxIndex, nil
	}
}

// VectorNorm computes the L2 norm of a vector
func (ctx *MathContext) VectorNorm(a *memory.Memory, size int) (float64, error) {
	if a == nil {
		return 0, fmt.Errorf("input memory cannot be nil")
	}
	if size <= 0 {
		return 0, fmt.Errorf("vector size must be positive: %d", size)
	}

	switch ctx.config.DataType {
	case MathDataFloat64:
		values, err := readMathFloat64Memory(a, size)
		if err != nil {
			return 0, err
		}
		total := 0.0
		for _, value := range values {
			total += value * value
		}
		return math.Sqrt(total), nil
	case MathDataComplexFloat32:
		values, err := readMathComplex64Memory(a, size)
		if err != nil {
			return 0, err
		}
		total := 0.0
		for _, value := range values {
			magnitude := cmplx.Abs(complex128(value))
			total += magnitude * magnitude
		}
		return math.Sqrt(total), nil
	case MathDataComplexFloat64:
		values, err := readMathComplex128Memory(a, size)
		if err != nil {
			return 0, err
		}
		total := 0.0
		for _, value := range values {
			magnitude := cmplx.Abs(value)
			total += magnitude * magnitude
		}
		return math.Sqrt(total), nil
	default:
		values, err := readMathFloat32Memory(a, size)
		if err != nil {
			return 0, err
		}
		total := 0.0
		for _, value := range values {
			floatValue := float64(value)
			total += floatValue * floatValue
		}
		return math.Sqrt(total), nil
	}
}

// Utility Functions

// calculateMathOpComplexity returns the computational complexity of a math operation
func calculateMathOpComplexity(op MathOperation, size int) int {
	baseComplexity := size

	switch op {
	case MathOpAdd, MathOpSub, MathOpMul:
		return baseComplexity
	case MathOpDiv, MathOpSqrt, MathOpRsqrt:
		return baseComplexity * 2
	case MathOpSin, MathOpCos, MathOpTan:
		return baseComplexity * 3
	case MathOpExp, MathOpLog:
		return baseComplexity * 4
	case MathOpPow:
		return baseComplexity * 5
	case MathOpErf, MathOpGamma:
		return baseComplexity * 6
	case MathOpJ0, MathOpJ1, MathOpY0, MathOpY1:
		return baseComplexity * 8
	case MathOpFMA:
		return baseComplexity * 2
	case MathOpCabs, MathOpCarg:
		return baseComplexity * 3
	default:
		return baseComplexity * 2
	}
}

// getMathKernelName returns the kernel name for a math operation
func getMathKernelName(op MathOperation) string {
	switch op {
	case MathOpAdd:
		return "mathVectorAdd"
	case MathOpMul:
		return "mathVectorMul"
	case MathOpSin:
		return "mathVectorSin"
	case MathOpCos:
		return "mathVectorCos"
	case MathOpExp:
		return "mathVectorExp"
	case MathOpLog:
		return "mathVectorLog"
	case MathOpSqrt:
		return "mathVectorSqrt"
	case MathOpPow:
		return "mathVectorPow"
	case MathOpErf:
		return "mathVectorErf"
	case MathOpGamma:
		return "mathVectorGamma"
	default:
		return "mathVectorGeneric"
	}
}

// calculateMathWorkspaceSize calculates workspace requirements
func calculateMathWorkspaceSize(config MathConfig) int {
	baseSize := 1024 * 1024 // 1MB base

	// Vector size affects workspace
	if config.VectorSize > 0 {
		baseSize += config.VectorSize * getMathDataTypeSize(config.DataType)
	}

	// High precision needs more workspace
	if config.Precision == MathPrecisionAccurate || config.Precision == MathPrecisionIEEE {
		baseSize *= 2
	}

	return baseSize
}

// getMathDataTypeSize returns size in bytes for math data types
func getMathDataTypeSize(dataType MathDataType) int {
	switch dataType {
	case MathDataFloat32:
		return 4
	case MathDataFloat64:
		return 8
	case MathDataComplexFloat32:
		return 8
	case MathDataComplexFloat64:
		return 16
	case MathDataHalf:
		return 2
	case MathDataBFloat16:
		return 2
	default:
		return 4
	}
}

func executeMathVectorFloat32(op MathVectorOp) error {
	inputA, err := readMathFloat32Memory(op.InputA, op.Size)
	if err != nil {
		return err
	}
	var inputB []float32
	if requiresBinaryMathInput(op.Operation) {
		if op.InputB == nil {
			return fmt.Errorf("operation %s requires a second input", mathOperationName(op.Operation))
		}
		inputB, err = readMathFloat32Memory(op.InputB, op.Size)
		if err != nil {
			return err
		}
	}
	output := make([]float32, op.Size)
	for index, value := range inputA {
		result, calcErr := applyFloat32MathOperation(op.Operation, value, inputBValue(inputB, index))
		if calcErr != nil {
			return calcErr
		}
		output[index] = result
	}
	return writeMathFloat32Memory(op.Output, output)
}

func executeMathVectorFloat64(op MathVectorOp) error {
	inputA, err := readMathFloat64Memory(op.InputA, op.Size)
	if err != nil {
		return err
	}
	var inputB []float64
	if requiresBinaryMathInput(op.Operation) {
		if op.InputB == nil {
			return fmt.Errorf("operation %s requires a second input", mathOperationName(op.Operation))
		}
		inputB, err = readMathFloat64Memory(op.InputB, op.Size)
		if err != nil {
			return err
		}
	}
	output := make([]float64, op.Size)
	for index, value := range inputA {
		result, calcErr := applyFloat64MathOperation(op.Operation, value, inputBValue(inputB, index))
		if calcErr != nil {
			return calcErr
		}
		output[index] = result
	}
	return writeMathFloat64Memory(op.Output, output)
}

func executeMathVectorComplex64(op MathVectorOp) error {
	inputA, err := readMathComplex64Memory(op.InputA, op.Size)
	if err != nil {
		return err
	}
	var inputB []complex64
	if requiresBinaryMathInput(op.Operation) {
		if op.InputB == nil {
			return fmt.Errorf("operation %s requires a second input", mathOperationName(op.Operation))
		}
		inputB, err = readMathComplex64Memory(op.InputB, op.Size)
		if err != nil {
			return err
		}
	}
	if op.Operation == MathOpCabs || op.Operation == MathOpCarg {
		output := make([]float32, op.Size)
		for index, value := range inputA {
			if op.Operation == MathOpCabs {
				output[index] = float32(cmplx.Abs(complex128(value)))
				continue
			}
			output[index] = float32(cmplx.Phase(complex128(value)))
		}
		return writeMathFloat32Memory(op.Output, output)
	}
	output := make([]complex64, op.Size)
	for index, value := range inputA {
		result, calcErr := applyComplex64MathOperation(op.Operation, value, inputBValue(inputB, index))
		if calcErr != nil {
			return calcErr
		}
		output[index] = result
	}
	return writeMathComplex64Memory(op.Output, output)
}

func executeMathVectorComplex128(op MathVectorOp) error {
	inputA, err := readMathComplex128Memory(op.InputA, op.Size)
	if err != nil {
		return err
	}
	var inputB []complex128
	if requiresBinaryMathInput(op.Operation) {
		if op.InputB == nil {
			return fmt.Errorf("operation %s requires a second input", mathOperationName(op.Operation))
		}
		inputB, err = readMathComplex128Memory(op.InputB, op.Size)
		if err != nil {
			return err
		}
	}
	if op.Operation == MathOpCabs || op.Operation == MathOpCarg {
		output := make([]float64, op.Size)
		for index, value := range inputA {
			if op.Operation == MathOpCabs {
				output[index] = cmplx.Abs(value)
				continue
			}
			output[index] = cmplx.Phase(value)
		}
		return writeMathFloat64Memory(op.Output, output)
	}
	output := make([]complex128, op.Size)
	for index, value := range inputA {
		result, calcErr := applyComplex128MathOperation(op.Operation, value, inputBValue(inputB, index))
		if calcErr != nil {
			return calcErr
		}
		output[index] = result
	}
	return writeMathComplex128Memory(op.Output, output)
}

func requiresBinaryMathInput(op MathOperation) bool {
	switch op {
	case MathOpAdd, MathOpSub, MathOpMul, MathOpDiv, MathOpFMA, MathOpPow, MathOpAtan2, MathOpFmod, MathOpRemainder, MathOpFmax, MathOpFmin, MathOpFdim, MathOpCopysign:
		return true
	default:
		return false
	}
}

func applyFloat32MathOperation(op MathOperation, a, b float32) (float32, error) {
	result, err := applyFloat64MathOperation(op, float64(a), float64(b))
	return float32(result), err
}

func applyFloat64MathOperation(op MathOperation, a, b float64) (float64, error) {
	switch op {
	case MathOpAdd:
		return a + b, nil
	case MathOpSub:
		return a - b, nil
	case MathOpMul, MathOpFMA:
		return a * b, nil
	case MathOpDiv:
		return a / b, nil
	case MathOpSqrt:
		return math.Sqrt(a), nil
	case MathOpRsqrt:
		return 1 / math.Sqrt(a), nil
	case MathOpRcp:
		return 1 / a, nil
	case MathOpSin:
		return math.Sin(a), nil
	case MathOpCos:
		return math.Cos(a), nil
	case MathOpTan:
		return math.Tan(a), nil
	case MathOpExp:
		return math.Exp(a), nil
	case MathOpLog:
		return math.Log(a), nil
	case MathOpPow:
		return math.Pow(a, b), nil
	case MathOpErf:
		return math.Erf(a), nil
	case MathOpGamma, MathOpTgamma:
		return math.Gamma(a), nil
	case MathOpLgamma:
		lgamma, _ := math.Lgamma(a)
		return lgamma, nil
	case MathOpJ0:
		return math.J0(a), nil
	case MathOpJ1:
		return math.J1(a), nil
	case MathOpY0:
		return math.Y0(a), nil
	case MathOpY1:
		return math.Y1(a), nil
	case MathOpCeil:
		return math.Ceil(a), nil
	case MathOpFloor:
		return math.Floor(a), nil
	case MathOpTrunc:
		return math.Trunc(a), nil
	case MathOpRound:
		return math.Round(a), nil
	case MathOpFmod:
		return math.Mod(a, b), nil
	case MathOpRemainder:
		return math.Remainder(a, b), nil
	case MathOpFmax:
		return math.Max(a, b), nil
	case MathOpFmin:
		return math.Min(a, b), nil
	case MathOpFdim:
		return math.Dim(a, b), nil
	case MathOpCopysign:
		return math.Copysign(a, b), nil
	default:
		return 0, fmt.Errorf("operation %s is not implemented for real-valued math data", mathOperationName(op))
	}
}

func applyComplex64MathOperation(op MathOperation, a, b complex64) (complex64, error) {
	result, err := applyComplex128MathOperation(op, complex128(a), complex128(b))
	return complex64(result), err
}

func applyComplex128MathOperation(op MathOperation, a, b complex128) (complex128, error) {
	switch op {
	case MathOpAdd:
		return a + b, nil
	case MathOpSub:
		return a - b, nil
	case MathOpMul, MathOpFMA:
		return a * b, nil
	case MathOpDiv:
		return a / b, nil
	default:
		return 0, fmt.Errorf("operation %s is not implemented for complex-valued math data", mathOperationName(op))
	}
}

func mathOperationName(op MathOperation) string {
	switch op {
	case MathOpAdd:
		return "add"
	case MathOpMul:
		return "mul"
	case MathOpFMA:
		return "fma"
	case MathOpSqrt:
		return "sqrt"
	case MathOpRsqrt:
		return "rsqrt"
	case MathOpSin:
		return "sin"
	case MathOpCos:
		return "cos"
	case MathOpTan:
		return "tan"
	case MathOpExp:
		return "exp"
	case MathOpLog:
		return "log"
	case MathOpPow:
		return "pow"
	case MathOpErf:
		return "erf"
	case MathOpGamma:
		return "gamma"
	case MathOpJ0:
		return "j0"
	case MathOpCabs:
		return "cabs"
	case MathOpCarg:
		return "carg"
	default:
		return fmt.Sprintf("op-%d", op)
	}
}

func readMathFloat32Memory(mem *memory.Memory, length int) ([]float32, error) {
	return readMathTypedMemory[float32](mem, length)
}

func writeMathFloat32Memory(mem *memory.Memory, values []float32) error {
	return writeMathTypedMemory(mem, values)
}

func readMathFloat64Memory(mem *memory.Memory, length int) ([]float64, error) {
	return readMathTypedMemory[float64](mem, length)
}

func writeMathFloat64Memory(mem *memory.Memory, values []float64) error {
	return writeMathTypedMemory(mem, values)
}

func readMathComplex64Memory(mem *memory.Memory, length int) ([]complex64, error) {
	return readMathTypedMemory[complex64](mem, length)
}

func writeMathComplex64Memory(mem *memory.Memory, values []complex64) error {
	return writeMathTypedMemory(mem, values)
}

func readMathComplex128Memory(mem *memory.Memory, length int) ([]complex128, error) {
	return readMathTypedMemory[complex128](mem, length)
}

func writeMathComplex128Memory(mem *memory.Memory, values []complex128) error {
	return writeMathTypedMemory(mem, values)
}

func readMathTypedMemory[T any](mem *memory.Memory, length int) ([]T, error) {
	if mem == nil {
		return nil, fmt.Errorf("memory cannot be nil")
	}
	if length < 0 {
		return nil, fmt.Errorf("length must be non-negative: %d", length)
	}
	values := make([]T, length)
	if length == 0 {
		return values, nil
	}
	var zero T
	byteSize := int(unsafe.Sizeof(zero)) * length
	if int64(byteSize) > mem.Size() {
		return nil, fmt.Errorf("allocation too small: need %d bytes, have %d", byteSize, mem.Size())
	}
	hostBytes := unsafe.Slice((*byte)(unsafe.Pointer(&values[0])), byteSize)
	if err := memory.CopyDeviceToHost(hostBytes, mem); err != nil {
		return nil, err
	}
	return values, nil
}

func writeMathTypedMemory[T any](mem *memory.Memory, values []T) error {
	if mem == nil {
		return fmt.Errorf("memory cannot be nil")
	}
	if len(values) == 0 {
		return nil
	}
	var zero T
	byteSize := int(unsafe.Sizeof(zero)) * len(values)
	if int64(byteSize) > mem.Size() {
		return fmt.Errorf("allocation too small: need %d bytes, have %d", byteSize, mem.Size())
	}
	host := make([]T, len(values))
	copy(host, values)
	hostBytes := unsafe.Slice((*byte)(unsafe.Pointer(&host[0])), byteSize)
	return memory.CopyHostToDevice(mem, hostBytes)
}

func inputBValue[T any](values []T, index int) T {
	if len(values) == 0 {
		var zero T
		return zero
	}
	return values[index]
}

// Destroy cleans up math context resources
func (ctx *MathContext) Destroy() error {
	if ctx.handle != nil {
		ctx.handle.Free()
		ctx.handle = nil
	}
	if ctx.workspace != nil {
		ctx.workspace.Free()
		ctx.workspace = nil
	}
	for i, stream := range ctx.streams {
		if stream != nil {
			stream.Free()
			ctx.streams[i] = nil
		}
	}
	return nil
}

// Convenience functions for common operations

// ComputeElementwise performs elementwise operations on vectors
func ComputeElementwise(op MathOperation, a, b, output *memory.Memory, size int) error {
	config := MathConfig{
		Precision:   MathPrecisionDefault,
		DataType:    MathDataFloat32,
		VectorSize:  size,
		UseHardware: true,
		FastMath:    false,
	}

	ctx, err := CreateMathContext(config)
	if err != nil {
		return err
	}
	defer ctx.Destroy()

	return ctx.executeVectorOp(MathVectorOp{
		Operation: op,
		InputA:    a,
		InputB:    b,
		Output:    output,
		Size:      size,
		Config:    config,
	})
}

// ComputeUnary performs unary operations on vectors
func ComputeUnary(op MathOperation, a, output *memory.Memory, size int) error {
	return ComputeElementwise(op, a, nil, output, size)
}

// FastMathOperations provides optimized math operations with reduced precision
func FastMathOperations(op MathOperation, a, b, output *memory.Memory, size int) error {
	config := MathConfig{
		Precision:   MathPrecisionFast,
		DataType:    MathDataFloat32,
		VectorSize:  size,
		UseHardware: true,
		FastMath:    true,
		FlushToZero: true,
		HandleNaN:   false,
		HandleInf:   false,
	}

	ctx, err := CreateMathContext(config)
	if err != nil {
		return err
	}
	defer ctx.Destroy()

	return ctx.executeVectorOp(MathVectorOp{
		Operation: op,
		InputA:    a,
		InputB:    b,
		Output:    output,
		Size:      size,
		Config:    config,
	})
}

// HighPrecisionMath provides IEEE 754 compliant high-precision operations
func HighPrecisionMath(op MathOperation, a, b, output *memory.Memory, size int) error {
	config := MathConfig{
		Precision:   MathPrecisionIEEE,
		DataType:    MathDataFloat64,
		VectorSize:  size,
		UseHardware: false,
		FastMath:    false,
		FlushToZero: false,
		HandleNaN:   true,
		HandleInf:   true,
	}

	ctx, err := CreateMathContext(config)
	if err != nil {
		return err
	}
	defer ctx.Destroy()

	return ctx.executeVectorOp(MathVectorOp{
		Operation: op,
		InputA:    a,
		InputB:    b,
		Output:    output,
		Size:      size,
		Config:    config,
	})
}
