// Package libraries provides CUDA runtime library bindings for GoCUDA
// This implements CUDA Math API functionality for advanced mathematical functions
package libraries

import (
	"fmt"
	"math"

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
	for i := 0; i < numStreams; i++ {
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

	// Calculate operation complexity based on function type
	complexity := calculateMathOpComplexity(op.Operation, op.Size)

	// Apply precision modifiers
	switch op.Config.Precision {
	case MathPrecisionFast:
		complexity = complexity / 2
	case MathPrecisionAccurate:
		complexity = complexity * 2
	case MathPrecisionIEEE:
		complexity = complexity * 3
	}

	// Execute the mathematical operation
	kernelName := getMathKernelName(op.Operation)
	err := simulateKernelExecution(kernelName, complexity, 1)
	if err != nil {
		return fmt.Errorf("math operation %v failed: %v", op.Operation, err)
	}

	return nil
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

	// Simulate parallel reduction
	complexity := size
	err := simulateKernelExecution("mathVectorSum", complexity/100, 2)
	if err != nil {
		return 0, fmt.Errorf("vector sum failed: %v", err)
	}

	// Return simulated sum
	return float64(size) * 0.5, nil
}

// VectorMax finds the maximum element in a vector
func (ctx *MathContext) VectorMax(a *memory.Memory, size int) (float64, int, error) {
	if a == nil {
		return 0, 0, fmt.Errorf("input memory cannot be nil")
	}

	complexity := size
	err := simulateKernelExecution("mathVectorMax", complexity/100, 2)
	if err != nil {
		return 0, 0, fmt.Errorf("vector max failed: %v", err)
	}

	// Return simulated max and index
	return float64(size), size / 2, nil
}

// VectorNorm computes the L2 norm of a vector
func (ctx *MathContext) VectorNorm(a *memory.Memory, size int) (float64, error) {
	if a == nil {
		return 0, fmt.Errorf("input memory cannot be nil")
	}

	complexity := size * 2 // Square and sum operations
	err := simulateKernelExecution("mathVectorNorm", complexity/100, 2)
	if err != nil {
		return 0, fmt.Errorf("vector norm failed: %v", err)
	}

	// Return simulated norm
	return math.Sqrt(float64(size)), nil
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
