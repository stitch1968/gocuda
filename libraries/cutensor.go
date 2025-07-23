// Package libraries provides CUDA runtime library bindings for GoCUDA
// This implements cuTENSOR functionality for tensor operations and contractions
package libraries

import (
	"fmt"

	"github.com/stitch1968/gocuda/memory"
)

// cuTENSOR - High-Performance Tensor Operations Library

// Tensor data types
type TensorDataType int

const (
	TensorFloat16 TensorDataType = iota
	TensorFloat32
	TensorFloat64
	TensorComplex32
	TensorComplex64
	TensorInt8
	TensorInt16
	TensorInt32
	TensorInt64
	TensorUInt8
	TensorUInt16
	TensorUInt32
	TensorUInt64
	TensorBFloat16
)

// Tensor memory layout
type TensorLayout int

const (
	TensorLayoutRowMajor TensorLayout = iota // C-style (last dimension contiguous)
	TensorLayoutColMajor                     // Fortran-style (first dimension contiguous)
	TensorLayoutCustom                       // Custom stride pattern
)

// Tensor operation types
type TensorOperation int

const (
	// Basic operations
	TensorOpAdd TensorOperation = iota
	TensorOpSub
	TensorOpMul
	TensorOpDiv
	TensorOpScale
	TensorOpCopy
	TensorOpTranspose
	TensorOpPermute
	TensorOpReduce

	// Contractions
	TensorOpContraction
	TensorOpBilinear
	TensorOpElementwise

	// Advanced operations
	TensorOpConvolution
	TensorOpGEMM
	TensorOpBatchedGEMM
	TensorOpTensorCore
)

// Tensor reduction operations
type TensorReduction int

const (
	TensorReduceSum TensorReduction = iota
	TensorReduceMax
	TensorReduceMin
	TensorReduceMean
	TensorReduceNorm1
	TensorReduceNorm2
	TensorReduceNormInf
	TensorReduceAny
	TensorReduceAll
)

// Tensor descriptor
type CuTensorDescriptor struct {
	dataType   TensorDataType
	layout     TensorLayout
	dimensions []int
	strides    []int
	baseOffset int64
	alignReq   int
}

// Tensor contraction descriptor
type ContractionDescriptor struct {
	TensorA   *CuTensorDescriptor
	TensorB   *CuTensorDescriptor
	TensorC   *CuTensorDescriptor
	ModesA    []int // Contraction modes for tensor A
	ModesB    []int // Contraction modes for tensor B
	ModesC    []int // Output modes for tensor C
	Alpha     float64
	Beta      float64
	Algorithm ContractionAlgorithm
	Workspace *memory.Memory
}

// Contraction algorithms
type ContractionAlgorithm int

const (
	ContractionAlgoDefault ContractionAlgorithm = iota
	ContractionAlgoGEMM
	ContractionAlgoTensorCore
	ContractionAlgoOptimal
	ContractionAlgoFastest
	ContractionAlgoLowestMemory
)

// cuTENSOR handle
type CuTensorHandle struct {
	handle       *memory.Memory
	workspace    *memory.Memory
	planCache    map[string]*memory.Memory
	descriptors  []*CuTensorDescriptor
	computeType  TensorDataType
	mathMode     TensorMathMode
	streamHandle *memory.Memory
}

// Tensor math modes
type TensorMathMode int

const (
	TensorMathDefault TensorMathMode = iota
	TensorMathTensorCore
	TensorMathFast
	TensorMathAccurate
)

// Tensor plan for optimized execution
type TensorPlan struct {
	handle        *memory.Memory
	operation     TensorOperation
	algorithm     ContractionAlgorithm
	workspaceSize int64
	executionTime float64
	memoryReq     int64
}

// CreateCuTensorHandle creates a new cuTENSOR handle
func CreateCuTensorHandle() (*CuTensorHandle, error) {
	handle := &CuTensorHandle{
		planCache:   make(map[string]*memory.Memory),
		descriptors: make([]*CuTensorDescriptor, 0),
		computeType: TensorFloat32,
		mathMode:    TensorMathDefault,
	}

	// Allocate main handle
	var err error
	handle.handle, err = memory.Alloc(8192)
	if err != nil {
		return nil, fmt.Errorf("failed to allocate cuTENSOR handle: %v", err)
	}

	// Allocate workspace for operations
	handle.workspace, err = memory.Alloc(64 * 1024 * 1024) // 64MB default workspace
	if err != nil {
		handle.handle.Free()
		return nil, fmt.Errorf("failed to allocate cuTENSOR workspace: %v", err)
	}

	// Create stream handle
	handle.streamHandle, err = memory.Alloc(1024)
	if err != nil {
		handle.handle.Free()
		handle.workspace.Free()
		return nil, fmt.Errorf("failed to allocate stream handle: %v", err)
	}

	return handle, nil
}

// CreateCuTensorDescriptor creates a tensor descriptor
func CreateCuTensorDescriptor(dataType TensorDataType, dimensions []int, layout TensorLayout) (*CuTensorDescriptor, error) {
	if len(dimensions) == 0 {
		return nil, fmt.Errorf("tensor must have at least one dimension")
	}

	// Calculate strides based on layout
	strides := make([]int, len(dimensions))
	elementSize := getTensorDataTypeSize(dataType)

	switch layout {
	case TensorLayoutRowMajor:
		strides[len(dimensions)-1] = 1
		for i := len(dimensions) - 2; i >= 0; i-- {
			strides[i] = strides[i+1] * dimensions[i+1]
		}
	case TensorLayoutColMajor:
		strides[0] = 1
		for i := 1; i < len(dimensions); i++ {
			strides[i] = strides[i-1] * dimensions[i-1]
		}
	default:
		// Custom layout - use default row major
		strides[len(dimensions)-1] = 1
		for i := len(dimensions) - 2; i >= 0; i-- {
			strides[i] = strides[i+1] * dimensions[i+1]
		}
	}

	// Scale strides by element size
	for i := range strides {
		strides[i] *= elementSize
	}

	desc := &CuTensorDescriptor{
		dataType:   dataType,
		layout:     layout,
		dimensions: make([]int, len(dimensions)),
		strides:    strides,
		baseOffset: 0,
		alignReq:   256, // 256-byte alignment for optimal performance
	}

	copy(desc.dimensions, dimensions)

	return desc, nil
}

// Tensor Contraction Operations

// TensorContraction performs general tensor contraction
func (handle *CuTensorHandle) TensorContraction(
	alpha float64,
	tensorA *memory.Memory, descA *CuTensorDescriptor, modesA []int,
	tensorB *memory.Memory, descB *CuTensorDescriptor, modesB []int,
	beta float64,
	tensorC *memory.Memory, descC *CuTensorDescriptor, modesC []int,
	algorithm ContractionAlgorithm) error {

	if tensorA == nil || tensorB == nil || tensorC == nil {
		return fmt.Errorf("tensor pointers cannot be nil")
	}

	if descA == nil || descB == nil || descC == nil {
		return fmt.Errorf("tensor descriptors cannot be nil")
	}

	// Validate contraction modes
	if len(modesA) != len(descA.dimensions) || len(modesB) != len(descB.dimensions) || len(modesC) != len(descC.dimensions) {
		return fmt.Errorf("mode count must match tensor dimensions")
	}

	// Create contraction descriptor
	contractionDesc := &ContractionDescriptor{
		TensorA:   descA,
		TensorB:   descB,
		TensorC:   descC,
		ModesA:    modesA,
		ModesB:    modesB,
		ModesC:    modesC,
		Alpha:     alpha,
		Beta:      beta,
		Algorithm: algorithm,
		Workspace: handle.workspace,
	}

	// Calculate operation complexity
	complexity := calculateContractionComplexity(contractionDesc)

	// Execute contraction
	kernelName := getContractionKernelName(algorithm)
	err := simulateKernelExecution(kernelName, complexity, 4)
	if err != nil {
		return fmt.Errorf("tensor contraction failed: %v", err)
	}

	return nil
}

// BatchedTensorContraction performs batched tensor contractions
func (handle *CuTensorHandle) BatchedTensorContraction(
	batchCount int,
	alpha float64,
	tensorA []*memory.Memory, descA *CuTensorDescriptor, modesA []int,
	tensorB []*memory.Memory, descB *CuTensorDescriptor, modesB []int,
	beta float64,
	tensorC []*memory.Memory, descC *CuTensorDescriptor, modesC []int,
	algorithm ContractionAlgorithm) error {

	if batchCount <= 0 {
		return fmt.Errorf("batch count must be positive: %d", batchCount)
	}

	if len(tensorA) != batchCount || len(tensorB) != batchCount || len(tensorC) != batchCount {
		return fmt.Errorf("tensor array sizes must match batch count")
	}

	// Execute each contraction in the batch
	for i := 0; i < batchCount; i++ {
		err := handle.TensorContraction(
			alpha, tensorA[i], descA, modesA,
			tensorB[i], descB, modesB,
			beta, tensorC[i], descC, modesC,
			algorithm)
		if err != nil {
			return fmt.Errorf("batched contraction failed at index %d: %v", i, err)
		}
	}

	return nil
}

// Element-wise Operations

// TensorElementwiseAdd performs element-wise addition
func (handle *CuTensorHandle) TensorElementwiseAdd(
	alpha float64, tensorA *memory.Memory, descA *CuTensorDescriptor,
	beta float64, tensorB *memory.Memory, descB *CuTensorDescriptor,
	gamma float64, tensorC *memory.Memory, descC *CuTensorDescriptor) error {

	if !validateTensorCompatibility(descA, descB, descC) {
		return fmt.Errorf("tensor dimensions are not compatible for element-wise operations")
	}

	elements := calculateTensorElements(descA)
	complexity := elements * 2 // Add and scale operations

	err := simulateKernelExecution("tensorElementwiseAdd", complexity/1000, 2)
	if err != nil {
		return fmt.Errorf("tensor elementwise add failed: %v", err)
	}

	return nil
}

// TensorElementwiseMul performs element-wise multiplication
func (handle *CuTensorHandle) TensorElementwiseMul(
	alpha float64, tensorA *memory.Memory, descA *CuTensorDescriptor,
	beta float64, tensorB *memory.Memory, descB *CuTensorDescriptor,
	gamma float64, tensorC *memory.Memory, descC *CuTensorDescriptor) error {

	if !validateTensorCompatibility(descA, descB, descC) {
		return fmt.Errorf("tensor dimensions are not compatible for element-wise operations")
	}

	elements := calculateTensorElements(descA)
	complexity := elements * 2 // Multiply and scale operations

	err := simulateKernelExecution("tensorElementwiseMul", complexity/1000, 2)
	if err != nil {
		return fmt.Errorf("tensor elementwise multiply failed: %v", err)
	}

	return nil
}

// Tensor Reduction Operations

// TensorReduce performs tensor reduction along specified modes
func (handle *CuTensorHandle) TensorReduce(
	alpha float64, tensorA *memory.Memory, descA *CuTensorDescriptor,
	beta float64, tensorC *memory.Memory, descC *CuTensorDescriptor,
	reduceModes []int, reductionOp TensorReduction) error {

	if tensorA == nil || tensorC == nil {
		return fmt.Errorf("tensor pointers cannot be nil")
	}

	if len(reduceModes) == 0 {
		return fmt.Errorf("must specify at least one reduction mode")
	}

	// Validate reduction modes
	for _, mode := range reduceModes {
		if mode < 0 || mode >= len(descA.dimensions) {
			return fmt.Errorf("invalid reduction mode: %d", mode)
		}
	}

	inputElements := calculateTensorElements(descA)
	outputElements := calculateTensorElements(descC)
	_ = inputElements / outputElements // reductionRatio for future use

	// Calculate complexity based on reduction type
	var complexity int
	switch reductionOp {
	case TensorReduceSum, TensorReduceMean:
		complexity = inputElements
	case TensorReduceMax, TensorReduceMin:
		complexity = inputElements
	case TensorReduceNorm1, TensorReduceNorm2:
		complexity = inputElements * 2
	case TensorReduceNormInf:
		complexity = inputElements
	default:
		complexity = inputElements
	}

	err := simulateKernelExecution("tensorReduce", complexity/1000, 3)
	if err != nil {
		return fmt.Errorf("tensor reduction failed: %v", err)
	}

	return nil
}

// Tensor Transformation Operations

// TensorPermute performs tensor permutation (transpose generalization)
func (handle *CuTensorHandle) TensorPermute(
	alpha float64, tensorA *memory.Memory, descA *CuTensorDescriptor,
	tensorC *memory.Memory, descC *CuTensorDescriptor,
	perm []int) error {

	if len(perm) != len(descA.dimensions) {
		return fmt.Errorf("permutation array length must match tensor dimensions")
	}

	// Validate permutation array
	used := make([]bool, len(perm))
	for _, p := range perm {
		if p < 0 || p >= len(perm) || used[p] {
			return fmt.Errorf("invalid permutation array")
		}
		used[p] = true
	}

	elements := calculateTensorElements(descA)
	complexity := elements * 2 // Read and write with different access patterns

	err := simulateKernelExecution("tensorPermute", complexity/1000, 3)
	if err != nil {
		return fmt.Errorf("tensor permutation failed: %v", err)
	}

	return nil
}

// TensorCopy performs tensor copy with potential layout conversion
func (handle *CuTensorHandle) TensorCopy(
	alpha float64, tensorA *memory.Memory, descA *CuTensorDescriptor,
	tensorC *memory.Memory, descC *CuTensorDescriptor) error {

	if !validateTensorSameSize(descA, descC) {
		return fmt.Errorf("tensors must have the same total size for copy operation")
	}

	elements := calculateTensorElements(descA)
	complexity := elements

	// Add complexity if layout conversion is needed
	if descA.layout != descC.layout {
		complexity *= 2
	}

	err := simulateKernelExecution("tensorCopy", complexity/1000, 2)
	if err != nil {
		return fmt.Errorf("tensor copy failed: %v", err)
	}

	return nil
}

// Plan-based Execution for Optimization

// CreateContractionPlan creates an optimized execution plan for tensor contraction
func (handle *CuTensorHandle) CreateContractionPlan(
	descA *CuTensorDescriptor, modesA []int,
	descB *CuTensorDescriptor, modesB []int,
	descC *CuTensorDescriptor, modesC []int,
	algorithm ContractionAlgorithm) (*TensorPlan, error) {

	// Generate plan key for caching
	planKey := generatePlanKey(descA, descB, descC, modesA, modesB, modesC, algorithm)

	// Check if plan exists in cache
	if cachedPlan, exists := handle.planCache[planKey]; exists {
		return &TensorPlan{handle: cachedPlan}, nil
	}

	// Create new plan
	planHandle, err := memory.Alloc(2048)
	if err != nil {
		return nil, fmt.Errorf("failed to allocate plan handle: %v", err)
	}

	plan := &TensorPlan{
		handle:    planHandle,
		operation: TensorOpContraction,
		algorithm: algorithm,
	}

	// Calculate optimal workspace size
	plan.workspaceSize = calculateOptimalWorkspaceSize(descA, descB, descC, algorithm)

	// Estimate execution time and memory requirements
	plan.executionTime = estimateContractionTime(descA, descB, descC, algorithm)
	plan.memoryReq = calculateMemoryRequirement(descA, descB, descC)

	// Cache the plan
	handle.planCache[planKey] = planHandle

	return plan, nil
}

// ExecuteContractionPlan executes a pre-compiled contraction plan
func (handle *CuTensorHandle) ExecuteContractionPlan(
	plan *TensorPlan,
	alpha float64,
	tensorA *memory.Memory, tensorB *memory.Memory,
	beta float64,
	tensorC *memory.Memory) error {

	if plan == nil || plan.handle == nil {
		return fmt.Errorf("invalid or destroyed plan")
	}

	// Ensure workspace is large enough
	if handle.workspace.Size() < plan.workspaceSize {
		// Reallocate larger workspace
		handle.workspace.Free()
		newWorkspace, err := memory.Alloc(plan.workspaceSize)
		if err != nil {
			return fmt.Errorf("failed to allocate larger workspace: %v", err)
		}
		handle.workspace = newWorkspace
	}

	// Execute the planned operation
	complexity := int(plan.executionTime * 1000) // Convert to complexity units
	kernelName := getContractionKernelName(plan.algorithm)

	err := simulateKernelExecution(kernelName, complexity, 4)
	if err != nil {
		return fmt.Errorf("plan execution failed: %v", err)
	}

	return nil
}

// Utility Functions

// calculateContractionComplexity estimates the computational complexity of a contraction
func calculateContractionComplexity(desc *ContractionDescriptor) int {
	// Basic complexity estimation based on tensor dimensions
	elementsA := calculateTensorElements(desc.TensorA)
	elementsB := calculateTensorElements(desc.TensorB)
	elementsC := calculateTensorElements(desc.TensorC)

	// Estimate based on the largest tensor and algorithm
	maxElements := elementsA
	if elementsB > maxElements {
		maxElements = elementsB
	}
	if elementsC > maxElements {
		maxElements = elementsC
	}

	baseComplexity := maxElements

	// Algorithm-specific complexity modifiers
	switch desc.Algorithm {
	case ContractionAlgoGEMM:
		return baseComplexity * 2
	case ContractionAlgoTensorCore:
		return baseComplexity / 2 // Hardware acceleration
	case ContractionAlgoOptimal:
		return baseComplexity * 3
	case ContractionAlgoFastest:
		return baseComplexity / 2
	case ContractionAlgoLowestMemory:
		return baseComplexity * 4
	default:
		return baseComplexity
	}
}

// calculateTensorElements calculates total number of elements in a tensor
func calculateTensorElements(desc *CuTensorDescriptor) int {
	if len(desc.dimensions) == 0 {
		return 0
	}

	elements := 1
	for _, dim := range desc.dimensions {
		elements *= dim
	}
	return elements
}

// validateTensorCompatibility checks if tensors are compatible for element-wise operations
func validateTensorCompatibility(descA, descB, descC *CuTensorDescriptor) bool {
	// Check if all tensors have same dimensions
	if len(descA.dimensions) != len(descB.dimensions) || len(descB.dimensions) != len(descC.dimensions) {
		return false
	}

	for i := range descA.dimensions {
		if descA.dimensions[i] != descB.dimensions[i] || descB.dimensions[i] != descC.dimensions[i] {
			return false
		}
	}

	return true
}

// validateTensorSameSize checks if tensors have the same total size
func validateTensorSameSize(descA, descB *CuTensorDescriptor) bool {
	return calculateTensorElements(descA) == calculateTensorElements(descB)
}

// getTensorDataTypeSize returns the size in bytes for tensor data types
func getTensorDataTypeSize(dataType TensorDataType) int {
	switch dataType {
	case TensorFloat16, TensorBFloat16:
		return 2
	case TensorFloat32:
		return 4
	case TensorFloat64:
		return 8
	case TensorComplex32:
		return 8
	case TensorComplex64:
		return 16
	case TensorInt8, TensorUInt8:
		return 1
	case TensorInt16, TensorUInt16:
		return 2
	case TensorInt32, TensorUInt32:
		return 4
	case TensorInt64, TensorUInt64:
		return 8
	default:
		return 4
	}
}

// getContractionKernelName returns the kernel name for a contraction algorithm
func getContractionKernelName(algorithm ContractionAlgorithm) string {
	switch algorithm {
	case ContractionAlgoGEMM:
		return "tensorContractionGEMM"
	case ContractionAlgoTensorCore:
		return "tensorContractionTensorCore"
	case ContractionAlgoOptimal:
		return "tensorContractionOptimal"
	case ContractionAlgoFastest:
		return "tensorContractionFast"
	case ContractionAlgoLowestMemory:
		return "tensorContractionMemOpt"
	default:
		return "tensorContractionDefault"
	}
}

// generatePlanKey generates a unique key for plan caching
func generatePlanKey(descA, descB, descC *CuTensorDescriptor, modesA, modesB, modesC []int, algorithm ContractionAlgorithm) string {
	return fmt.Sprintf("plan_%v_%v_%v_%v_%v_%v_%d",
		descA.dimensions, descB.dimensions, descC.dimensions,
		modesA, modesB, modesC, algorithm)
}

// calculateOptimalWorkspaceSize calculates the optimal workspace size for an operation
func calculateOptimalWorkspaceSize(descA, descB, descC *CuTensorDescriptor, algorithm ContractionAlgorithm) int64 {
	baseSize := int64(calculateTensorElements(descA) * getTensorDataTypeSize(descA.dataType))

	// Algorithm-specific workspace requirements
	switch algorithm {
	case ContractionAlgoOptimal:
		return baseSize * 4
	case ContractionAlgoTensorCore:
		return baseSize * 2
	case ContractionAlgoLowestMemory:
		return baseSize / 2
	default:
		return baseSize * 2
	}
}

// estimateContractionTime estimates execution time for a contraction
func estimateContractionTime(descA, descB, descC *CuTensorDescriptor, algorithm ContractionAlgorithm) float64 {
	complexity := float64(calculateTensorElements(descA) + calculateTensorElements(descB) + calculateTensorElements(descC))

	switch algorithm {
	case ContractionAlgoFastest:
		return complexity * 0.5e-9 // 0.5 ns per element
	case ContractionAlgoTensorCore:
		return complexity * 0.2e-9 // 0.2 ns per element (hardware acceleration)
	case ContractionAlgoOptimal:
		return complexity * 1.0e-9 // 1.0 ns per element
	default:
		return complexity * 2.0e-9 // 2.0 ns per element
	}
}

// calculateMemoryRequirement calculates total memory requirement for tensors
func calculateMemoryRequirement(descA, descB, descC *CuTensorDescriptor) int64 {
	sizeA := int64(calculateTensorElements(descA) * getTensorDataTypeSize(descA.dataType))
	sizeB := int64(calculateTensorElements(descB) * getTensorDataTypeSize(descB.dataType))
	sizeC := int64(calculateTensorElements(descC) * getTensorDataTypeSize(descC.dataType))

	return sizeA + sizeB + sizeC
}

// Destroy cleans up cuTENSOR handle resources
func (handle *CuTensorHandle) Destroy() error {
	// Free plan cache
	for _, plan := range handle.planCache {
		if plan != nil {
			plan.Free()
		}
	}
	handle.planCache = make(map[string]*memory.Memory)

	// Free workspace
	if handle.workspace != nil {
		handle.workspace.Free()
		handle.workspace = nil
	}

	// Free stream handle
	if handle.streamHandle != nil {
		handle.streamHandle.Free()
		handle.streamHandle = nil
	}

	// Free main handle
	if handle.handle != nil {
		handle.handle.Free()
		handle.handle = nil
	}

	return nil
}

// DestroyTensorPlan destroys a tensor plan and frees its resources
func (plan *TensorPlan) Destroy() error {
	if plan.handle != nil {
		plan.handle.Free()
		plan.handle = nil
	}
	return nil
}

// Convenience Functions

// SimpleContraction performs a simple tensor contraction with default settings
func SimpleContraction(
	alpha float64,
	tensorA *memory.Memory, dimA []int,
	tensorB *memory.Memory, dimB []int,
	beta float64,
	tensorC *memory.Memory, dimC []int) error {

	handle, err := CreateCuTensorHandle()
	if err != nil {
		return err
	}
	defer handle.Destroy()

	// Create descriptors
	descA, err := CreateCuTensorDescriptor(TensorFloat32, dimA, TensorLayoutRowMajor)
	if err != nil {
		return err
	}

	descB, err := CreateCuTensorDescriptor(TensorFloat32, dimB, TensorLayoutRowMajor)
	if err != nil {
		return err
	}

	descC, err := CreateCuTensorDescriptor(TensorFloat32, dimC, TensorLayoutRowMajor)
	if err != nil {
		return err
	}

	// Generate simple contraction modes (assume last mode of A contracts with first mode of B)
	modesA := make([]int, len(dimA))
	modesB := make([]int, len(dimB))
	modesC := make([]int, len(dimC))

	for i := range modesA {
		modesA[i] = i
	}
	for i := range modesB {
		modesB[i] = len(dimA) + i
	}
	for i := range modesC {
		modesC[i] = i
	}

	// Contract last mode of A with first mode of B
	if len(dimA) > 0 && len(dimB) > 0 {
		modesB[0] = modesA[len(modesA)-1]
	}

	return handle.TensorContraction(
		alpha, tensorA, descA, modesA,
		tensorB, descB, modesB,
		beta, tensorC, descC, modesC,
		ContractionAlgoDefault)
}

// MatrixMultiply performs matrix multiplication using tensor contraction
func MatrixMultiply(
	alpha float64,
	matA *memory.Memory, rowsA, colsA int,
	matB *memory.Memory, rowsB, colsB int,
	beta float64,
	matC *memory.Memory) error {

	if colsA != rowsB {
		return fmt.Errorf("matrix dimensions incompatible: A[%dx%d] * B[%dx%d]", rowsA, colsA, rowsB, colsB)
	}

	return SimpleContraction(
		alpha,
		matA, []int{rowsA, colsA},
		matB, []int{rowsB, colsB},
		beta,
		matC, []int{rowsA, colsB})
}

// TensorElementwiseOp performs element-wise operations on tensors
func TensorElementwiseOp(
	operation TensorOperation,
	alpha float64, tensorA *memory.Memory, dimA []int,
	beta float64, tensorB *memory.Memory, dimB []int,
	tensorC *memory.Memory) error {

	handle, err := CreateCuTensorHandle()
	if err != nil {
		return err
	}
	defer handle.Destroy()

	// Create descriptors
	descA, err := CreateCuTensorDescriptor(TensorFloat32, dimA, TensorLayoutRowMajor)
	if err != nil {
		return err
	}

	descB, err := CreateCuTensorDescriptor(TensorFloat32, dimB, TensorLayoutRowMajor)
	if err != nil {
		return err
	}

	descC, err := CreateCuTensorDescriptor(TensorFloat32, dimA, TensorLayoutRowMajor) // Assume output has same dims as A
	if err != nil {
		return err
	}

	switch operation {
	case TensorOpAdd:
		return handle.TensorElementwiseAdd(alpha, tensorA, descA, beta, tensorB, descB, 0.0, tensorC, descC)
	case TensorOpMul:
		return handle.TensorElementwiseMul(alpha, tensorA, descA, beta, tensorB, descB, 0.0, tensorC, descC)
	default:
		return fmt.Errorf("unsupported elementwise operation: %v", operation)
	}
}
