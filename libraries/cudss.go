// Package libraries provides CUDA runtime library bindings for GoCUDA
// This implements cuDSS functionality for direct sparse solver operations
package libraries

import (
	"errors"
	"fmt"
	"math"
	"runtime"
	"slices"
	"sync"

	cuda "github.com/stitch1968/gocuda"
	"github.com/stitch1968/gocuda/internal"
	"github.com/stitch1968/gocuda/memory"
	"github.com/stitch1968/gocuda/streams"
)

// cuDSS - CUDA Direct Sparse Solver Library

// DSS matrix formats
type DSSMatrixFormat int

const (
	DSSMatrixFormatCSR DSSMatrixFormat = iota
	DSSMatrixFormatCOO
	DSSMatrixFormatCSC
)

// DSS factorization algorithms
type DSSFactorization int

const (
	DSSFactorizationLU DSSFactorization = iota
	DSSFactorizationLDLT
	DSSFactorizationCholesky
	DSSFactorizationQR
)

// DSS ordering algorithms
type DSSOrdering int

const (
	DSSOrderingNone DSSOrdering = iota
	DSSOrderingAMD
	DSSOrderingMETIS
	DSSOrderingNDBox
	DSSOrderingRCM
	DSSOrderingFillReducing
)

// DSS refinement options
type DSSRefinement int

const (
	DSSRefinementNone DSSRefinement = iota
	DSSRefinementSingle
	DSSRefinementDouble
	DSSRefinementMixed
)

// DSS pivot type
type DSSPivotType int

const (
	DSSPivotNone DSSPivotType = iota
	DSSPivotPartial
	DSSPivotRook
	DSSPivotBunch
)

// DSS configuration structure
type DSSConfig struct {
	MatrixFormat   DSSMatrixFormat
	Factorization  DSSFactorization
	Ordering       DSSOrdering
	Refinement     DSSRefinement
	PivotType      DSSPivotType
	PivotThreshold float64
	Symmetry       bool
	Deterministic  bool
	UseGPU         bool
}

// DSSHandle serializes operations on one cuDSS solver. Do not copy it after use.
type DSSHandle struct {
	mu                sync.Mutex
	handle            *memory.Memory
	config            DSSConfig
	workspace         *memory.Memory
	n                 int
	nnz               int
	analyzed          bool
	rowOffsets        []int32
	columnIndices     []int32
	analyzedSymmetry  bool
	factored          bool
	destroyed         bool
	matrix            *DSSMatrix
	dense             []float64
	det               float64
	nativeHandle      uintptr
	nativeConfig      uintptr
	nativeData        uintptr
	native            bool
	deviceID          int
	strictNative      bool
	pendingMatrices   []*DSSMatrix
	pendingOperations []dssPendingOperation
}

// DSS matrix descriptor
type DSSMatrix struct {
	mu           sync.Mutex
	owner        *DSSHandle
	users        int
	retentions   []*memory.Retention
	handle       *memory.Memory
	rowPtr       *memory.Memory
	colInd       *memory.Memory
	values       *memory.Memory
	nativeValues *memory.Memory
	n            int
	nnz          int
	format       DSSMatrixFormat
	symmetry     bool
	nativeHandle uintptr
	nativeType   int
	native       bool
	deviceID     int
	destroyed    bool
}

var errCUDSSUnsupported = errors.New("cudss native path unsupported for requested parameters")

// ErrDSSNativeOperationUnsupported indicates that a strict native cuDSS handle
// cannot execute an operation without a host fallback.
var ErrDSSNativeOperationUnsupported = errors.New("native cuDSS operation is not supported")

// ErrDSSMatrixInUse indicates that a cuDSS handle retains the matrix.
var ErrDSSMatrixInUse = errors.New("cuDSS matrix is in use")

// ErrDSSInertiaUnavailable indicates that inertia is not measured by this wrapper.
var ErrDSSInertiaUnavailable = errors.New("cuDSS inertia is not available")

// DSSSolutionInfo reports the maximum residual and backward errors across
// right-hand sides. Iterations is -1 when not measured. Error, PivotGrowth and
// ConditionNumber are NaN when unavailable; InertiaAvailable describes Inertia.
type DSSSolutionInfo struct {
	Iterations         int
	Residual           float64
	Error              float64
	Determinant        float64
	Inertia            [3]int // positive, negative, zero eigenvalues
	InertiaAvailable   bool
	PivotGrowth        float64
	ConditionNumber    float64
	BackwardError      float64
	ComponentwiseError float64
}

// CreateDSSHandle creates a new cuDSS solver handle
func CreateDSSHandle(config DSSConfig) (*DSSHandle, error) {
	if err := ensureCudaReady(); err != nil {
		return nil, err
	}
	if cuda.ShouldUseCuda() && cudssNativeAvailable() {
		runtime.LockOSThread()
		defer runtime.UnlockOSThread()
		deviceID, err := internal.CurrentDevice()
		if err != nil {
			return nil, err
		}
		return createDSSHandleOnDevice(config, deviceID, false)
	}
	return createDeterministicDSSHandle(config)
}

// CreateNativeDSSHandleOnDevice creates a device-bound handle that rejects host fallback.
func CreateNativeDSSHandleOnDevice(config DSSConfig, deviceID int) (*DSSHandle, error) {
	if err := ensureCudaReady(); err != nil {
		return nil, err
	}
	if !cuda.ShouldUseCuda() || !cudssNativeAvailable() {
		return nil, streams.ErrNativeCUDAUnavailable
	}
	if deviceID < 0 || deviceID >= cuda.GetCudaDeviceCount() {
		return nil, fmt.Errorf("device ID %d out of range", deviceID)
	}
	if config.MatrixFormat != DSSMatrixFormatCSR ||
		(config.Factorization != DSSFactorizationLU && config.Factorization != DSSFactorizationCholesky) ||
		config.Ordering != DSSOrderingNone || config.Refinement != DSSRefinementNone ||
		config.PivotType != DSSPivotNone || config.PivotThreshold != 0 ||
		config.Symmetry != (config.Factorization == DSSFactorizationCholesky) {
		return nil, fmt.Errorf("%w: requested solver configuration", ErrDSSNativeOperationUnsupported)
	}
	return createDSSHandleOnDevice(config, deviceID, true)
}

func createDSSHandleOnDevice(config DSSConfig, deviceID int, strict bool) (*DSSHandle, error) {
	var handle *DSSHandle
	err := internal.RunOnDevice(deviceID, func() error {
		var createErr error
		handle, createErr = createNativeDSSHandle(config)
		return createErr
	})
	if err != nil {
		if handle != nil {
			err = errors.Join(err, internal.RunOnDevice(deviceID, func() error { return destroyNativeDSSHandle(handle) }))
		}
		return nil, err
	}
	handle.deviceID = deviceID
	handle.strictNative = strict
	return handle, nil
}

// Backend reports the handle backend; non-strict handles may still use host fallbacks.
func (handle *DSSHandle) Backend() LibraryImplementationMode {
	if handle == nil {
		return ""
	}
	handle.mu.Lock()
	defer handle.mu.Unlock()
	if handle.native {
		return ImplementationModeNativeCUDA
	}
	return ImplementationModeValidatedGo
}

func createDeterministicDSSHandle(config DSSConfig) (*DSSHandle, error) {
	handle := &DSSHandle{
		config:   config,
		factored: false,
	}

	// Allocate handle memory
	var err error
	handle.handle, err = memory.Alloc(8192) // DSS solver state
	if err != nil {
		return nil, fmt.Errorf("failed to allocate DSS handle: %v", err)
	}

	// Allocate workspace (size depends on configuration)
	workspaceSize := calculateDSSWorkspaceSize(config)
	if workspaceSize > 0 {
		handle.workspace, err = memory.Alloc(int64(workspaceSize))
		if err != nil {
			handle.handle.Free()
			return nil, fmt.Errorf("failed to allocate DSS workspace: %v", err)
		}
	}

	return handle, nil
}

// CreateDSSMatrix creates a DSS matrix from sparse data
func CreateDSSMatrix(n, nnz int, rowPtr, colInd, values *memory.Memory, format DSSMatrixFormat, symmetry bool) (*DSSMatrix, error) {
	if err := ensureCudaReady(); err != nil {
		return nil, err
	}

	if n <= 0 || nnz <= 0 {
		return nil, fmt.Errorf("invalid matrix dimensions: n=%d, nnz=%d", n, nnz)
	}

	if rowPtr == nil || colInd == nil || values == nil {
		return nil, fmt.Errorf("matrix data pointers cannot be nil")
	}
	retained, err := retainLibraryBuffers(rowPtr, colInd, values)
	if err != nil {
		return nil, err
	}
	defer func() { releaseLibraryBuffers(retained) }()
	if cuda.ShouldUseCuda() && cudssNativeAvailable() {
		if format != DSSMatrixFormatCSR {
			matrix, err := createDeterministicDSSMatrix(n, nnz, rowPtr, colInd, values, format, symmetry)
			if err != nil {
				return nil, err
			}
			matrix.retentions, retained = retained, nil
			return matrix, nil
		}
		deviceID := rowPtr.GetDeviceID()
		if err := validateDSSNativeMatrixBuffers(n, nnz, deviceID, rowPtr, colInd, values); err != nil {
			return nil, err
		}
		var matrix *DSSMatrix
		err = internal.RunOnDevice(deviceID, func() error {
			var createErr error
			matrix, createErr = createNativeDSSMatrix(n, nnz, rowPtr, colInd, values, format, symmetry)
			return createErr
		})
		if err != nil {
			if matrix != nil {
				err = errors.Join(err, internal.RunOnDevice(deviceID, func() error { return destroyNativeDSSMatrix(matrix) }))
			}
			return nil, err
		}
		matrix.deviceID = deviceID
		matrix.retentions, retained = retained, nil
		return matrix, nil
	}
	matrix, err := createDeterministicDSSMatrix(n, nnz, rowPtr, colInd, values, format, symmetry)
	if err != nil {
		return nil, err
	}
	matrix.retentions, retained = retained, nil
	return matrix, nil
}

func (matrix *DSSMatrix) retain(handle *DSSHandle) error {
	matrix.mu.Lock()
	defer matrix.mu.Unlock()
	if matrix.destroyed {
		return fmt.Errorf("cuDSS matrix has been destroyed")
	}
	if matrix.owner != nil && matrix.owner != handle {
		return ErrDSSMatrixInUse
	}
	matrix.owner = handle
	matrix.users++
	return nil
}

func (matrix *DSSMatrix) release(handle *DSSHandle) {
	matrix.mu.Lock()
	defer matrix.mu.Unlock()
	if matrix.owner != handle || matrix.users <= 0 {
		panic("invalid cuDSS matrix retention")
	}
	matrix.users--
	if matrix.users == 0 {
		matrix.owner = nil
	}
}

func validateDSSNativeBuffer(buffer *memory.Memory, deviceID int) error {
	if buffer == nil || buffer.Ptr() == nil {
		return fmt.Errorf("cuDSS buffer is nil or freed")
	}
	switch buffer.GetType() {
	case memory.TypeDevice, memory.TypeDeviceOnly, memory.TypeUnified:
	default:
		return fmt.Errorf("native cuDSS requires device or managed memory")
	}
	if buffer.GetDeviceID() != deviceID {
		return fmt.Errorf("cuDSS buffer device %d does not match device %d", buffer.GetDeviceID(), deviceID)
	}
	return nil
}

func validateDSSNativeMatrixBuffers(n, nnz, deviceID int, rowPtr, colInd, values *memory.Memory) error {
	if n <= 0 || nnz <= 0 || int64(n) > math.MaxInt32 || int64(nnz) > math.MaxInt32 {
		return fmt.Errorf("cuDSS dimensions exceed 32-bit CSR index capacity")
	}
	for _, buffer := range []*memory.Memory{rowPtr, colInd, values} {
		if err := validateDSSNativeBuffer(buffer, deviceID); err != nil {
			return err
		}
	}
	if uint64(rowPtr.Size())/4 < uint64(n)+1 || uint64(colInd.Size())/4 < uint64(nnz) {
		return fmt.Errorf("cuDSS CSR index buffer is too small")
	}
	if values.Size() != int64(nnz)*4 && values.Size() != int64(nnz)*8 {
		return fmt.Errorf("cuDSS values must contain exactly nnz float32 or float64 elements")
	}
	return nil
}

func (handle *DSSHandle) validateNativeMatrix(matrix *DSSMatrix) error {
	if !matrix.native {
		if handle.strictNative {
			return fmt.Errorf("%w: matrix does not use native cuDSS", ErrDSSNativeOperationUnsupported)
		}
		return nil
	}
	if !handle.native {
		return fmt.Errorf("native cuDSS matrix requires a native handle")
	}
	if matrix.deviceID != handle.deviceID {
		return fmt.Errorf("cuDSS matrix device %d does not match handle device %d", matrix.deviceID, handle.deviceID)
	}
	if handle.strictNative && matrix.symmetry != handle.config.Symmetry {
		return fmt.Errorf("%w: matrix symmetry does not match solver configuration", ErrDSSNativeOperationUnsupported)
	}
	return validateDSSNativeMatrixBuffers(matrix.n, matrix.nnz, handle.deviceID, matrix.rowPtr, matrix.colInd, matrix.values)
}

func createDeterministicDSSMatrix(n, nnz int, rowPtr, colInd, values *memory.Memory, format DSSMatrixFormat, symmetry bool) (*DSSMatrix, error) {
	matrix := &DSSMatrix{
		rowPtr:   rowPtr,
		colInd:   colInd,
		values:   values,
		n:        n,
		nnz:      nnz,
		format:   format,
		symmetry: symmetry,
	}

	// Allocate matrix handle
	var err error
	matrix.handle, err = memory.Alloc(4096) // Matrix descriptor
	if err != nil {
		return nil, fmt.Errorf("failed to allocate matrix handle: %v", err)
	}

	return matrix, nil
}

// Analyze performs symbolic analysis and records the CSR structure.
// A successful analysis invalidates any previous factorization.
func (handle *DSSHandle) Analyze(matrix *DSSMatrix) error {
	if handle == nil || matrix == nil {
		return fmt.Errorf("handle and matrix cannot be nil")
	}
	handle.mu.Lock()
	defer handle.mu.Unlock()
	if handle.destroyed {
		return fmt.Errorf("cuDSS handle has been destroyed")
	}
	if handle.hasPendingLocked() {
		if err := handle.synchronizeLocked(); err != nil {
			return err
		}
	}
	if err := matrix.retain(handle); err != nil {
		return err
	}
	retained := true
	defer func() {
		if retained {
			matrix.release(handle)
		}
	}()
	if err := handle.validateNativeMatrix(matrix); err != nil {
		return err
	}
	offsets, columns, err := readDSSCSRIndices(matrix)
	if err != nil {
		return err
	}
	handle.analyzed = false
	handle.factored = false
	handle.dense = nil
	handle.rowOffsets = nil
	handle.columnIndices = nil
	if handle.native && matrix.native {
		pending, err := handle.runNativePhaseLocked(func() error { return analyzeNativeDSS(handle, matrix) })
		if pending {
			handle.pendingMatrices = append(handle.pendingMatrices, matrix)
			retained = false
		}
		if err != nil {
			return err
		}
	}
	previous := handle.matrix
	handle.n = matrix.n
	handle.nnz = matrix.nnz
	handle.matrix = matrix
	retained = false
	if previous != nil {
		previous.release(handle)
	}
	handle.rowOffsets = offsets
	handle.columnIndices = columns
	handle.analyzedSymmetry = matrix.symmetry
	handle.analyzed = true
	return nil
}

func (handle *DSSHandle) validateAnalyzedMatrix(matrix *DSSMatrix) error {
	if handle.destroyed {
		return fmt.Errorf("cuDSS handle has been destroyed")
	}
	if !handle.analyzed {
		return fmt.Errorf("matrix must be analyzed before factorization")
	}
	if err := handle.validateNativeMatrix(matrix); err != nil {
		return err
	}
	if matrix.n != handle.n || matrix.nnz != handle.nnz || matrix.symmetry != handle.analyzedSymmetry {
		return fmt.Errorf("matrix structure differs from analyzed matrix")
	}
	offsets, columns, err := readDSSCSRIndices(matrix)
	if err != nil {
		return err
	}
	if !slices.Equal(offsets, handle.rowOffsets) || !slices.Equal(columns, handle.columnIndices) {
		return fmt.Errorf("matrix sparsity pattern differs from analyzed matrix")
	}
	return nil
}

// Factor performs numerical factorization using the analyzed CSR structure.
func (handle *DSSHandle) Factor(matrix *DSSMatrix) error {
	if handle == nil || matrix == nil {
		return fmt.Errorf("handle and matrix cannot be nil")
	}
	handle.mu.Lock()
	defer handle.mu.Unlock()
	return handle.factorLocked(matrix)
}

func (handle *DSSHandle) factorLocked(matrix *DSSMatrix) error {
	if handle.hasPendingLocked() {
		if err := handle.synchronizeLocked(); err != nil {
			return err
		}
	}
	if err := matrix.retain(handle); err != nil {
		return err
	}
	retained := true
	defer func() {
		if retained {
			matrix.release(handle)
		}
	}()
	if err := handle.validateAnalyzedMatrix(matrix); err != nil {
		return err
	}
	dense, err := dssDenseFromMatrix(matrix)
	if err != nil {
		return err
	}
	det, err := dssDeterminant(dense, matrix.n)
	if err != nil {
		return err
	}
	if handle.native && matrix.native {
		handle.factored = false
		pending, err := handle.runNativePhaseLocked(func() error {
			if err := refreshNativeDSSValues(matrix); err != nil {
				return err
			}
			return factorNativeDSS(handle, matrix)
		})
		if pending {
			handle.pendingMatrices = append(handle.pendingMatrices, matrix)
			retained = false
		}
		if err != nil {
			return err
		}
	}
	previous := handle.matrix
	handle.dense = dense
	handle.matrix = matrix
	retained = false
	if previous != nil {
		previous.release(handle)
	}
	handle.det = det
	handle.factored = true
	return nil
}

// Solve solves Ax = b using the last successful factorization. Call Refactor
// after modifying matrix values.
func (handle *DSSHandle) Solve(b, x *memory.Memory, nrhs int) (*DSSSolutionInfo, error) {
	if handle == nil {
		return nil, fmt.Errorf("nil cuDSS handle")
	}
	handle.mu.Lock()
	defer handle.mu.Unlock()
	return handle.solveLocked(b, x, nrhs)
}

func (handle *DSSHandle) solveLocked(b, x *memory.Memory, nrhs int) (*DSSSolutionInfo, error) {
	if handle.hasPendingLocked() {
		if err := handle.synchronizeLocked(); err != nil {
			return nil, err
		}
	}
	if err := handle.validateSolve(b, x, nrhs); err != nil {
		return nil, err
	}
	retained, err := retainLibraryBuffers(b, x)
	if err != nil {
		return nil, err
	}
	transferred := false
	defer func() {
		if !transferred {
			releaseLibraryBuffers(retained)
		}
	}()
	if handle.native && handle.matrix.native {
		for _, buffer := range []*memory.Memory{b, x} {
			if err := validateDSSNativeBuffer(buffer, handle.deviceID); err != nil {
				return nil, err
			}
		}
		var info *DSSSolutionInfo
		err = internal.RunOnDevice(handle.deviceID, func() error {
			var solveErr error
			info, transferred, solveErr = solveNativeDSS(handle, b, x, nrhs, retained)
			return solveErr
		})
		if err == nil {
			return info, nil
		}
		if handle.strictNative && errors.Is(err, errCUDSSUnsupported) {
			return nil, errors.Join(ErrDSSNativeOperationUnsupported, err)
		}
		if handle.strictNative || err != errCUDSSUnsupported {
			return nil, err
		}
	}
	bValues, err := readMathFloat32Memory(b, handle.n*nrhs)
	if err != nil {
		return nil, err
	}
	xValues := make([]float32, handle.n*nrhs)
	var info *DSSSolutionInfo
	for rhs := range nrhs {
		rhsValues := make([]float64, handle.n)
		for index := 0; index < handle.n; index++ {
			rhsValues[index] = float64(bValues[rhs*handle.n+index])
		}
		var solution []float64
		switch handle.config.Factorization {
		case DSSFactorizationCholesky:
			solution, err = dssSolveCholesky(handle.dense, handle.n, rhsValues)
		default:
			solution, err = dssSolveGaussian(handle.dense, handle.n, rhsValues)
		}
		if err != nil {
			return nil, err
		}
		for index := 0; index < handle.n; index++ {
			xValues[rhs*handle.n+index] = float32(solution[index])
			solution[index] = float64(xValues[rhs*handle.n+index])
		}
		current, err := dssSolutionDiagnostics(handle.dense, handle.n, solution, rhsValues, handle.det)
		if err != nil {
			return nil, err
		}
		info = mergeDSSDiagnostics(info, current)
	}
	if err := writeMathFloat32Memory(x, xValues); err != nil {
		return nil, err
	}
	return info, nil
}

func (handle *DSSHandle) validateSolve(b, x *memory.Memory, nrhs int) error {
	if handle == nil || !handle.factored {
		return fmt.Errorf("matrix must be factored before solving")
	}
	if b == nil || x == nil {
		return fmt.Errorf("solution vectors cannot be nil")
	}
	if nrhs <= 0 || handle.n <= 0 || uint64(nrhs) > uint64(^uint(0)>>1)/8/uint64(handle.n) {
		return fmt.Errorf("invalid number of right-hand sides: %d", nrhs)
	}
	if handle.matrix == nil {
		return fmt.Errorf("factored matrix is unavailable")
	}
	return handle.validateAnalyzedMatrix(handle.matrix)
}

// SolveMultiple solves multiple right-hand sides and returns diagnostics for each.
func (handle *DSSHandle) SolveMultiple(B, X *memory.Memory, nrhs int) ([]*DSSSolutionInfo, error) {
	if handle == nil {
		return nil, fmt.Errorf("nil cuDSS handle")
	}
	handle.mu.Lock()
	defer handle.mu.Unlock()
	if handle.hasPendingLocked() {
		if err := handle.synchronizeLocked(); err != nil {
			return nil, err
		}
	}
	if err := handle.validateSolve(B, X, nrhs); err != nil {
		return nil, err
	}
	retained, err := retainLibraryBuffers(B, X)
	if err != nil {
		return nil, err
	}
	defer releaseLibraryBuffers(retained)
	count := handle.n * nrhs
	native := handle.native && handle.matrix.native
	rhs, err := readDSSSolveValues(B, count, native)
	if err != nil {
		return nil, err
	}
	if _, err := handle.solveLocked(B, X, nrhs); err != nil {
		return nil, err
	}
	solutions, err := readDSSSolveValues(X, count, native)
	if err != nil {
		return nil, err
	}
	infos := make([]*DSSSolutionInfo, nrhs)
	for i := range nrhs {
		start := i * handle.n
		infos[i], err = dssSolutionDiagnostics(handle.dense, handle.n, solutions[start:start+handle.n], rhs[start:start+handle.n], handle.det)
		if err != nil {
			return nil, err
		}
	}
	return infos, nil
}

// GetDeterminant computes the determinant of the factored matrix
func (handle *DSSHandle) GetDeterminant() (float64, error) {
	if handle == nil {
		return 0, fmt.Errorf("nil cuDSS handle")
	}
	handle.mu.Lock()
	defer handle.mu.Unlock()
	if !handle.factored {
		return 0, fmt.Errorf("matrix must be factored to compute determinant")
	}

	return handle.det, nil
}

// GetInertia returns ErrDSSInertiaUnavailable for LDLT and Cholesky factors.
func (handle *DSSHandle) GetInertia() ([3]int, error) {
	if handle == nil {
		return [3]int{}, fmt.Errorf("nil cuDSS handle")
	}
	handle.mu.Lock()
	defer handle.mu.Unlock()
	if !handle.factored {
		return [3]int{}, fmt.Errorf("matrix must be factored to compute inertia")
	}

	// Only valid for LDLT and Cholesky factorizations
	if handle.config.Factorization != DSSFactorizationLDLT &&
		handle.config.Factorization != DSSFactorizationCholesky {
		return [3]int{}, fmt.Errorf("inertia only available for LDLT and Cholesky factorizations")
	}

	return [3]int{}, ErrDSSInertiaUnavailable
}

// Refactor updates the numerical factorization with new values. The CSR
// offsets and column order must match the last successful analysis.
func (handle *DSSHandle) Refactor(matrix *DSSMatrix) error {
	if handle == nil || matrix == nil {
		return fmt.Errorf("handle and matrix cannot be nil")
	}
	handle.mu.Lock()
	defer handle.mu.Unlock()
	if !handle.factored {
		return fmt.Errorf("matrix must be initially factored before refactoring")
	}
	return handle.factorLocked(matrix)
}

// calculateDSSWorkspaceSize calculates workspace requirements
func calculateDSSWorkspaceSize(config DSSConfig) int {
	baseSize := 1024 * 1024 // 1MB base workspace

	// Factorization type affects workspace
	switch config.Factorization {
	case DSSFactorizationLU:
		baseSize *= 2
	case DSSFactorizationLDLT:
		baseSize *= 1
	case DSSFactorizationCholesky:
		baseSize *= 1
	case DSSFactorizationQR:
		baseSize *= 3
	}

	// Ordering algorithms need different workspace
	switch config.Ordering {
	case DSSOrderingMETIS:
		baseSize *= 2
	case DSSOrderingNDBox:
		baseSize *= 3
	}

	// Refinement needs extra workspace
	if config.Refinement != DSSRefinementNone {
		baseSize += 512 * 1024
	}

	return baseSize
}

// Destroy cleans up DSS handle resources
func (handle *DSSHandle) Destroy() error {
	if handle == nil {
		return fmt.Errorf("nil cuDSS handle")
	}
	handle.mu.Lock()
	defer handle.mu.Unlock()
	if handle.destroyed {
		return nil
	}
	if err := handle.synchronizeLocked(); err != nil {
		return err
	}
	if handle.native {
		destroyed := false
		err := internal.RunOnDevice(handle.deviceID, func() error {
			if err := destroyNativeDSSHandle(handle); err != nil {
				return err
			}
			destroyed = true
			return nil
		})
		if err != nil {
			if destroyed {
				if handle.matrix != nil {
					handle.matrix.release(handle)
					handle.matrix = nil
				}
				handle.destroyed = true
			}
			return err
		}
	}
	if handle.handle != nil {
		if err := handle.handle.Free(); err != nil {
			return err
		}
		handle.handle = nil
	}
	if handle.workspace != nil {
		if err := handle.workspace.Free(); err != nil {
			return err
		}
		handle.workspace = nil
	}
	handle.factored = false
	handle.analyzed = false
	handle.rowOffsets = nil
	handle.columnIndices = nil
	if handle.matrix != nil {
		handle.matrix.release(handle)
	}
	handle.matrix = nil
	handle.dense = nil
	handle.destroyed = true
	return nil
}

func dssDenseFromMatrix(matrix *DSSMatrix) ([]float64, error) {
	rowPtr, colInd, err := readDSSCSRIndices(matrix)
	if err != nil {
		return nil, err
	}
	if uint64(matrix.n) > uint64(^uint(0)>>1)/8/uint64(matrix.n) {
		return nil, fmt.Errorf("dense matrix size exceeds supported range")
	}
	values, err := readDSSFloatValues(matrix.values, matrix.nnz)
	if err != nil {
		return nil, err
	}
	dense := make([]float64, matrix.n*matrix.n)
	switch matrix.format {
	case DSSMatrixFormatCSR:
		for row := 0; row < matrix.n; row++ {
			for idx := rowPtr[row]; idx < rowPtr[row+1]; idx++ {
				col := colInd[idx]
				dense[row*matrix.n+int(col)] = values[idx]
				if matrix.symmetry && row != int(col) {
					dense[int(col)*matrix.n+row] = values[idx]
				}
			}
		}
	default:
		return nil, fmt.Errorf("deterministic cuDSS currently supports CSR matrices only")
	}
	return dense, nil
}

func readDSSCSRIndices(matrix *DSSMatrix) ([]int32, []int32, error) {
	if matrix != nil && matrix.destroyed {
		return nil, nil, fmt.Errorf("cuDSS matrix has been destroyed")
	}
	if matrix == nil || matrix.n <= 0 || matrix.n == int(^uint(0)>>1) || matrix.nnz <= 0 ||
		int64(matrix.n) > math.MaxInt32 || int64(matrix.nnz) > math.MaxInt32 {
		return nil, nil, fmt.Errorf("invalid cuDSS matrix dimensions")
	}
	if matrix.format != DSSMatrixFormatCSR {
		return nil, nil, fmt.Errorf("deterministic cuDSS currently supports CSR matrices only")
	}
	if matrix.rowPtr == nil || matrix.colInd == nil || matrix.values == nil ||
		matrix.rowPtr.Ptr() == nil || matrix.colInd.Ptr() == nil || matrix.values.Ptr() == nil {
		return nil, nil, fmt.Errorf("cuDSS matrix contains nil or freed buffers")
	}
	offsets, err := readInt32Memory(matrix.rowPtr, matrix.n+1)
	if err != nil {
		return nil, nil, fmt.Errorf("cuDSS CSR row offsets: %w", err)
	}
	columns, err := readInt32Memory(matrix.colInd, matrix.nnz)
	if err != nil {
		return nil, nil, fmt.Errorf("cuDSS CSR column indices: %w", err)
	}
	if err := validateCSRIndices(offsets, columns, matrix.n, matrix.n, true); err != nil {
		return nil, nil, fmt.Errorf("cuDSS CSR: %w", err)
	}
	return offsets, columns, nil
}

func readDSSFloatValues(mem *memory.Memory, length int) ([]float64, error) {
	switch mem.Size() {
	case int64(length) * 4:
		values32, err := readMathFloat32Memory(mem, length)
		if err != nil {
			return nil, err
		}
		values64 := make([]float64, length)
		for index, value := range values32 {
			values64[index] = float64(value)
		}
		return values64, nil
	case int64(length) * 8:
		return readMathFloat64Memory(mem, length)
	default:
		return nil, fmt.Errorf("unsupported cuDSS value buffer size %d for %d elements", mem.Size(), length)
	}
}

func dssSolveGaussian(dense []float64, n int, rhs []float64) ([]float64, error) {
	a := append([]float64(nil), dense...)
	b := append([]float64(nil), rhs...)
	for pivot := range n {
		maxRow := pivot
		maxVal := math.Abs(a[pivot*n+pivot])
		for row := pivot + 1; row < n; row++ {
			candidate := math.Abs(a[row*n+pivot])
			if candidate > maxVal {
				maxVal = candidate
				maxRow = row
			}
		}
		if maxVal == 0 {
			return nil, fmt.Errorf("singular matrix")
		}
		if maxRow != pivot {
			for col := pivot; col < n; col++ {
				a[pivot*n+col], a[maxRow*n+col] = a[maxRow*n+col], a[pivot*n+col]
			}
			b[pivot], b[maxRow] = b[maxRow], b[pivot]
		}
		for row := pivot + 1; row < n; row++ {
			factor := a[row*n+pivot] / a[pivot*n+pivot]
			for col := pivot; col < n; col++ {
				a[row*n+col] -= factor * a[pivot*n+col]
			}
			b[row] -= factor * b[pivot]
		}
	}
	x := make([]float64, n)
	for row := n - 1; row >= 0; row-- {
		sum := b[row]
		for col := row + 1; col < n; col++ {
			sum -= a[row*n+col] * x[col]
		}
		x[row] = sum / a[row*n+row]
	}
	return x, nil
}

func dssSolveCholesky(dense []float64, n int, rhs []float64) ([]float64, error) {
	l := make([]float64, len(dense))
	for i := range n {
		for j := 0; j <= i; j++ {
			sum := dense[i*n+j]
			for k := 0; k < j; k++ {
				sum -= l[i*n+k] * l[j*n+k]
			}
			if i == j {
				if sum <= 0 {
					return nil, fmt.Errorf("matrix is not positive definite")
				}
				l[i*n+j] = math.Sqrt(sum)
			} else {
				l[i*n+j] = sum / l[j*n+j]
			}
		}
	}
	y := make([]float64, n)
	for i := range n {
		sum := rhs[i]
		for j := 0; j < i; j++ {
			sum -= l[i*n+j] * y[j]
		}
		y[i] = sum / l[i*n+i]
	}
	x := make([]float64, n)
	for i := n - 1; i >= 0; i-- {
		sum := y[i]
		for j := i + 1; j < n; j++ {
			sum -= l[j*n+i] * x[j]
		}
		x[i] = sum / l[i*n+i]
	}
	return x, nil
}

func dssDeterminant(dense []float64, n int) (float64, error) {
	a := append([]float64(nil), dense...)
	sign := 1.0
	for pivot := range n {
		maxRow := pivot
		maxVal := math.Abs(a[pivot*n+pivot])
		for row := pivot + 1; row < n; row++ {
			candidate := math.Abs(a[row*n+pivot])
			if candidate > maxVal {
				maxVal = candidate
				maxRow = row
			}
		}
		if maxVal == 0 {
			return 0, nil
		}
		if maxRow != pivot {
			sign *= -1
			for col := pivot; col < n; col++ {
				a[pivot*n+col], a[maxRow*n+col] = a[maxRow*n+col], a[pivot*n+col]
			}
		}
		for row := pivot + 1; row < n; row++ {
			factor := a[row*n+pivot] / a[pivot*n+pivot]
			for col := pivot; col < n; col++ {
				a[row*n+col] -= factor * a[pivot*n+col]
			}
		}
	}
	det := sign
	for i := range n {
		det *= a[i*n+i]
	}
	return det, nil
}

func readDSSSolveValues(mem *memory.Memory, count int, native bool) ([]float64, error) {
	if native {
		return readDSSFloatValues(mem, count)
	}
	values, err := readMathFloat32Memory(mem, count)
	if err != nil {
		return nil, err
	}
	result := make([]float64, len(values))
	for i, value := range values {
		result[i] = float64(value)
	}
	return result, nil
}

func dssSolutionDiagnostics(dense []float64, n int, x, b []float64, determinant float64) (*DSSSolutionInfo, error) {
	if n <= 0 || len(x) != n || len(b) != n || len(dense)/n != n || len(dense)%n != 0 {
		return nil, fmt.Errorf("invalid cuDSS diagnostic dimensions")
	}
	normA, normX, normB, maxResidual, componentwise := 0.0, 0.0, 0.0, 0.0, 0.0
	for i := range n {
		if !dssFinite(x[i]) || !dssFinite(b[i]) {
			return nil, fmt.Errorf("cuDSS solution or right-hand side contains non-finite values")
		}
		normX = math.Max(normX, math.Abs(x[i]))
		normB = math.Max(normB, math.Abs(b[i]))
	}
	for row := range n {
		sum, rowNorm, denominator := 0.0, 0.0, math.Abs(b[row])
		for col := range n {
			value := dense[row*n+col]
			if !dssFinite(value) {
				return nil, fmt.Errorf("cuDSS matrix contains non-finite values")
			}
			product := value * x[col]
			sum += product
			rowNorm += math.Abs(value)
			denominator += math.Abs(product)
		}
		residual := math.Abs(sum - b[row])
		if !dssFinite(sum) || !dssFinite(rowNorm) || !dssFinite(denominator) || !dssFinite(residual) {
			return nil, fmt.Errorf("cuDSS diagnostic computation overflowed")
		}
		normA = math.Max(normA, rowNorm)
		maxResidual = math.Max(maxResidual, residual)
		if denominator != 0 {
			componentwise = math.Max(componentwise, residual/denominator)
		}
	}
	normDenominator := normA*normX + normB
	if !dssFinite(normDenominator) {
		return nil, fmt.Errorf("cuDSS diagnostic computation overflowed")
	}
	backward := 0.0
	if normDenominator != 0 {
		backward = maxResidual / normDenominator
	}
	return &DSSSolutionInfo{
		Iterations:         -1,
		Residual:           maxResidual,
		Error:              math.NaN(),
		Determinant:        determinant,
		PivotGrowth:        math.NaN(),
		ConditionNumber:    math.NaN(),
		BackwardError:      backward,
		ComponentwiseError: componentwise,
	}, nil
}

func dssFinite(value float64) bool {
	return !math.IsNaN(value) && !math.IsInf(value, 0)
}

func dssResidual(dense []float64, n int, x, b []float64) float64 {
	maxResidual := 0.0
	for row := range n {
		sum := 0.0
		for col := range n {
			sum += dense[row*n+col] * x[col]
		}
		if residual := math.Abs(sum - b[row]); residual > maxResidual {
			maxResidual = residual
		}
	}
	return maxResidual
}

func mergeDSSDiagnostics(result, current *DSSSolutionInfo) *DSSSolutionInfo {
	if result == nil {
		return current
	}
	result.Residual = math.Max(result.Residual, current.Residual)
	result.BackwardError = math.Max(result.BackwardError, current.BackwardError)
	result.ComponentwiseError = math.Max(result.ComponentwiseError, current.ComponentwiseError)
	return result
}

// Destroy cleans up DSS matrix resources
func (matrix *DSSMatrix) Destroy() error {
	if matrix == nil {
		return fmt.Errorf("nil cuDSS matrix")
	}
	matrix.mu.Lock()
	defer matrix.mu.Unlock()
	if matrix.destroyed {
		return nil
	}
	if matrix.users != 0 {
		return ErrDSSMatrixInUse
	}
	if matrix.native {
		destroyed := false
		err := internal.RunOnDevice(matrix.deviceID, func() error {
			if err := destroyNativeDSSMatrix(matrix); err != nil {
				return err
			}
			destroyed = true
			return nil
		})
		if err != nil {
			if destroyed {
				matrix.destroyed = true
				releaseLibraryBuffers(matrix.retentions)
				matrix.retentions = nil
			}
			return err
		}
	}
	if matrix.handle != nil {
		if err := matrix.handle.Free(); err != nil {
			return err
		}
		matrix.handle = nil
	}
	matrix.destroyed = true
	releaseLibraryBuffers(matrix.retentions)
	matrix.retentions = nil
	return nil
}

// Convenience functions for common operations

// SolveDirect provides a simple interface for direct sparse solving
func SolveDirect(n, nnz int, rowPtr, colInd, values, b, x *memory.Memory) (result *DSSSolutionInfo, err error) {
	// Create default configuration
	config := DSSConfig{
		MatrixFormat:   DSSMatrixFormatCSR,
		Factorization:  DSSFactorizationLU,
		Ordering:       DSSOrderingAMD,
		Refinement:     DSSRefinementSingle,
		PivotType:      DSSPivotPartial,
		PivotThreshold: 0.1,
		Symmetry:       false,
		Deterministic:  false,
		UseGPU:         true,
	}

	// Create handle and matrix
	handle, err := CreateDSSHandle(config)
	if err != nil {
		return nil, err
	}
	var matrix *DSSMatrix
	defer func() {
		err = errors.Join(err, handle.Destroy())
		if matrix != nil {
			err = errors.Join(err, matrix.Destroy())
		}
		if err != nil {
			result = nil
		}
	}()
	matrix, err = CreateDSSMatrix(n, nnz, rowPtr, colInd, values, DSSMatrixFormatCSR, false)
	if err != nil {
		return nil, err
	}
	// Analyze, factor, and solve
	err = handle.Analyze(matrix)
	if err != nil {
		return nil, err
	}

	err = handle.Factor(matrix)
	if err != nil {
		return nil, err
	}

	return handle.Solve(b, x, 1)
}

// SolveSymmetric solves a symmetric positive definite system using Cholesky
func SolveSymmetric(n, nnz int, rowPtr, colInd, values, b, x *memory.Memory) (result *DSSSolutionInfo, err error) {
	config := DSSConfig{
		MatrixFormat:   DSSMatrixFormatCSR,
		Factorization:  DSSFactorizationCholesky,
		Ordering:       DSSOrderingAMD,
		Refinement:     DSSRefinementDouble,
		PivotType:      DSSPivotNone,
		PivotThreshold: 0.0,
		Symmetry:       true,
		Deterministic:  false,
		UseGPU:         true,
	}

	handle, err := CreateDSSHandle(config)
	if err != nil {
		return nil, err
	}
	var matrix *DSSMatrix
	defer func() {
		err = errors.Join(err, handle.Destroy())
		if matrix != nil {
			err = errors.Join(err, matrix.Destroy())
		}
		if err != nil {
			result = nil
		}
	}()
	matrix, err = CreateDSSMatrix(n, nnz, rowPtr, colInd, values, DSSMatrixFormatCSR, true)
	if err != nil {
		return nil, err
	}
	err = handle.Analyze(matrix)
	if err != nil {
		return nil, err
	}

	err = handle.Factor(matrix)
	if err != nil {
		return nil, err
	}

	return handle.Solve(b, x, 1)
}
