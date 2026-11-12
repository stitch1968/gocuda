// Package libraries provides CUDA runtime library bindings for GoCUDA
// This implements cuSPARSE functionality for sparse matrix operations
package libraries

import (
	"errors"
	"fmt"
	"math"
	"sync"
	"time"
	"unsafe"

	cuda "github.com/stitch1968/gocuda"
	"github.com/stitch1968/gocuda/internal"
	"github.com/stitch1968/gocuda/memory"
	"github.com/stitch1968/gocuda/streams"
)

// cuSPARSE - Sparse Matrix Operations Library

// SparseMatrix represents a sparse matrix
type SparseMatrix struct {
	rows    int
	cols    int
	nnz     int            // Number of non-zero elements
	values  *memory.Memory // Non-zero values
	rowPtrs *memory.Memory // Row pointers (CSR format)
	colInds *memory.Memory // Column indices
	format  MatrixFormat
}

// MatrixFormat represents different sparse matrix storage formats
type MatrixFormat int

const (
	MatrixFormatCSR MatrixFormat = iota // Compressed Sparse Row
	MatrixFormatCSC                     // Compressed Sparse Column
	MatrixFormatCOO                     // Coordinate format
	MatrixFormatELL                     // ELLPACK format
	MatrixFormatHYB                     // Hybrid ELL-COO format
)

// SparseContext manages cuSPARSE operations
type SparseContext struct {
	handle            unsafe.Pointer // Simulated or native handle
	native            bool
	deviceID          int
	mu                sync.Mutex
	destroyed         bool
	pending           []*memory.Retention
	pendingMatrices   []*SparseMatrix
	pendingWorkspaces []*memory.Memory
	pendingCleanups   []func() error
	strictNative      bool
}

// ErrNativeSparseOperationUnsupported indicates that a native-only sparse
// context cannot execute the requested operation on the GPU.
var ErrNativeSparseOperationUnsupported = errors.New("operation is not implemented by the native cuSPARSE backend")

func (ctx *SparseContext) requireHostOperation(name string) error {
	if ctx.strictNative {
		return fmt.Errorf("%s: %w", name, ErrNativeSparseOperationUnsupported)
	}
	return nil
}

// releasePending is called only after the CUDA default stream has completed.
func (ctx *SparseContext) releasePending() error {
	var errs []error
	remainingCleanups := ctx.pendingCleanups[:0]
	for _, cleanup := range ctx.pendingCleanups {
		completed := false
		err := internal.RunOnDevice(ctx.deviceID, func() error {
			if err := cleanup(); err != nil {
				return err
			}
			completed = true
			return nil
		})
		if !completed {
			remainingCleanups = append(remainingCleanups, cleanup)
		}
		if err != nil {
			errs = append(errs, err)
		}
	}
	ctx.pendingCleanups = remainingCleanups
	if len(remainingCleanups) > 0 {
		return errors.Join(errs...)
	}
	for _, held := range ctx.pending {
		held.Release()
	}
	ctx.pending = nil
	remaining := ctx.pendingMatrices[:0]
	for _, matrix := range ctx.pendingMatrices {
		if err := matrix.Destroy(); err != nil {
			remaining = append(remaining, matrix)
			errs = append(errs, err)
		}
	}
	ctx.pendingMatrices = remaining
	remainingWorkspaces := ctx.pendingWorkspaces[:0]
	for _, workspace := range ctx.pendingWorkspaces {
		if err := workspace.Free(); err != nil {
			remainingWorkspaces = append(remainingWorkspaces, workspace)
			errs = append(errs, err)
		}
	}
	ctx.pendingWorkspaces = remainingWorkspaces
	return errors.Join(errs...)
}

func (ctx *SparseContext) lock() error {
	if ctx == nil {
		return fmt.Errorf("nil sparse context")
	}
	ctx.mu.Lock()
	if ctx.destroyed {
		ctx.mu.Unlock()
		return fmt.Errorf("sparse context has been destroyed")
	}
	return nil
}

func runSparseOnDevice[T any](ctx *SparseContext, run func() (T, error), cleanup func(T) error) (T, error) {
	if !ctx.native {
		return run()
	}
	var result T
	err := internal.RunOnDevice(ctx.deviceID, func() error {
		var runErr error
		result, runErr = run()
		return runErr
	})
	if err != nil {
		err = errors.Join(err, cleanup(result))
		var zero T
		return zero, err
	}
	return result, nil
}

// CreateSparseContext creates a new cuSPARSE context
func CreateSparseContext() (*SparseContext, error) {
	if err := ensureCudaReady(); err != nil {
		return nil, err
	}
	if cuda.ShouldUseCuda() {
		return createNativeSparseContext()
	}

	return &SparseContext{
		handle: unsafe.Pointer(uintptr(time.Now().UnixNano())), // Simulated handle
	}, nil
}

// CreateNativeSparseContextOnDevice creates a device-bound, native-only
// context. Operations without native CUDA implementations return an error.
func CreateNativeSparseContextOnDevice(deviceID int) (*SparseContext, error) {
	if err := ensureCudaReady(); err != nil {
		return nil, err
	}
	if !cuda.ShouldUseCuda() {
		return nil, streams.ErrNativeCUDAUnavailable
	}
	if deviceID < 0 || deviceID >= cuda.GetCudaDeviceCount() {
		return nil, fmt.Errorf("device ID %d out of range", deviceID)
	}
	var ctx *SparseContext
	err := internal.RunOnDevice(deviceID, func() error {
		var createErr error
		ctx, createErr = createNativeSparseContext()
		return createErr
	})
	if err != nil {
		if ctx != nil {
			return nil, errors.Join(err, ctx.DestroyContext())
		}
		return nil, err
	}
	ctx.strictNative = true
	return ctx, nil
}

// SpMVBackend reports the execution backend for a given sparse format.
func (ctx *SparseContext) SpMVBackend(format MatrixFormat) (LibraryImplementationMode, error) {
	if err := ctx.lock(); err != nil {
		return "", err
	}
	defer ctx.mu.Unlock()
	switch format {
	case MatrixFormatCSR, MatrixFormatCOO:
		if ctx.native {
			return ImplementationModeNativeCUDA, nil
		}
		return ImplementationModeValidatedGo, nil
	default:
		return "", fmt.Errorf("unsupported matrix format for SpMV")
	}
}

// SpSVBackend reports the execution backend for a triangular sparse format.
func (ctx *SparseContext) SpSVBackend(format MatrixFormat) (LibraryImplementationMode, error) {
	if err := ctx.lock(); err != nil {
		return "", err
	}
	defer ctx.mu.Unlock()
	switch format {
	case MatrixFormatCSR:
		if ctx.native {
			return ImplementationModeNativeCUDA, nil
		}
		return ImplementationModeValidatedGo, nil
	case MatrixFormatCOO:
		if err := ctx.requireHostOperation("COO triangular solve"); err != nil {
			return "", err
		}
		return ImplementationModeValidatedGo, nil
	default:
		return "", fmt.Errorf("unsupported matrix format for triangular solve")
	}
}

// SpGEMMBackend reports the backend for sparse matrix multiplication.
func (ctx *SparseContext) SpGEMMBackend(formatA, formatB MatrixFormat) (LibraryImplementationMode, error) {
	if err := ctx.lock(); err != nil {
		return "", err
	}
	defer ctx.mu.Unlock()
	if (formatA != MatrixFormatCSR && formatA != MatrixFormatCOO) ||
		(formatB != MatrixFormatCSR && formatB != MatrixFormatCOO) {
		return "", fmt.Errorf("unsupported matrix format for sparse multiplication")
	}
	if ctx.native && formatA == MatrixFormatCSR && formatB == MatrixFormatCSR {
		return ImplementationModeNativeCUDA, nil
	}
	if err := ctx.requireHostOperation("sparse matrix multiplication"); err != nil {
		return "", err
	}
	return ImplementationModeValidatedGo, nil
}

// SparseToDenseBackend reports the backend used for a sparse matrix format.
func (ctx *SparseContext) SparseToDenseBackend(format MatrixFormat) (LibraryImplementationMode, error) {
	if err := ctx.lock(); err != nil {
		return "", err
	}
	defer ctx.mu.Unlock()
	if format != MatrixFormatCSR && format != MatrixFormatCOO {
		return "", fmt.Errorf("unsupported matrix format for sparse-to-dense conversion")
	}
	if ctx.native {
		return ImplementationModeNativeCUDA, nil
	}
	return ImplementationModeValidatedGo, nil
}

// DenseToSparseBackend reports the backend used for a dense-to-sparse format.
func (ctx *SparseContext) DenseToSparseBackend(format MatrixFormat) (LibraryImplementationMode, error) {
	if err := ctx.lock(); err != nil {
		return "", err
	}
	defer ctx.mu.Unlock()
	switch format {
	case MatrixFormatCSR:
		if ctx.native {
			return ImplementationModeNativeCUDA, nil
		}
		return ImplementationModeValidatedGo, nil
	case MatrixFormatCOO:
		if err := ctx.requireHostOperation("dense-to-COO conversion"); err != nil {
			return "", err
		}
		return ImplementationModeValidatedGo, nil
	default:
		return "", fmt.Errorf("unsupported matrix format for dense-to-sparse conversion")
	}
}

// CreateSparseMatrix allocates a sparse matrix with nnz entries. Callers must
// initialize its values and indices before executing operations; use
// CreateCSRMatrix or CreateCOOMatrix to construct validated matrices.
func (ctx *SparseContext) CreateSparseMatrix(rows, cols, nnz int, format MatrixFormat) (*SparseMatrix, error) {
	if err := ctx.lock(); err != nil {
		return nil, err
	}
	defer ctx.mu.Unlock()
	if rows <= 0 || cols <= 0 || nnz < 0 ||
		int64(rows) > math.MaxInt32 || int64(cols) > math.MaxInt32 || int64(nnz) > math.MaxInt32 ||
		rows == int(^uint(0)>>1) {
		return nil, fmt.Errorf("invalid matrix dimensions")
	}
	if format != MatrixFormatCSR && format != MatrixFormatCOO {
		return nil, fmt.Errorf("unsupported matrix format")
	}
	return runSparseOnDevice(ctx, func() (*SparseMatrix, error) {
		return allocateSparseMatrix(rows, cols, nnz, format)
	}, func(matrix *SparseMatrix) error {
		return matrix.Destroy()
	})
}

// CreateCSRMatrix creates a CSR matrix from zero-based row offsets, column
// indices and float32 values. Duplicate indices are summed by host operations.
func (ctx *SparseContext) CreateCSRMatrix(rows, cols int, rowOffsets, columnIndices []int32, values []float32) (*SparseMatrix, error) {
	if err := ctx.lock(); err != nil {
		return nil, err
	}
	defer ctx.mu.Unlock()
	if rows <= 0 || cols <= 0 || int64(rows) > math.MaxInt32 || int64(cols) > math.MaxInt32 ||
		rows == int(^uint(0)>>1) || len(rowOffsets) != rows+1 || len(columnIndices) != len(values) || int64(len(values)) > math.MaxInt32 {
		return nil, fmt.Errorf("invalid CSR matrix dimensions or buffer lengths")
	}
	if rowOffsets[0] != 0 || int64(rowOffsets[rows]) != int64(len(values)) {
		return nil, fmt.Errorf("CSR row offsets must start at zero and end at nnz")
	}
	for row := range rows {
		if rowOffsets[row] < 0 || rowOffsets[row] > rowOffsets[row+1] || int64(rowOffsets[row+1]) > int64(len(values)) {
			return nil, fmt.Errorf("invalid CSR row offsets at row %d", row)
		}
	}
	for i, col := range columnIndices {
		if col < 0 || int64(col) >= int64(cols) {
			return nil, fmt.Errorf("CSR column %d out of range at entry %d", col, i)
		}
	}
	return runSparseOnDevice(ctx, func() (*SparseMatrix, error) {
		return createSparseMatrixWithBuffers(rows, cols, values, rowOffsets, columnIndices, MatrixFormatCSR)
	}, func(matrix *SparseMatrix) error { return matrix.Destroy() })
}

// CreateCOOMatrix creates a COO matrix from zero-based row and column indices.
func (ctx *SparseContext) CreateCOOMatrix(rows, cols int, rowIndices, columnIndices []int32, values []float32) (*SparseMatrix, error) {
	if err := ctx.lock(); err != nil {
		return nil, err
	}
	defer ctx.mu.Unlock()
	if rows <= 0 || cols <= 0 || int64(rows) > math.MaxInt32 || int64(cols) > math.MaxInt32 ||
		len(rowIndices) != len(values) || len(columnIndices) != len(values) || int64(len(values)) > math.MaxInt32 {
		return nil, fmt.Errorf("invalid COO matrix dimensions or buffer lengths")
	}
	for i := range values {
		if rowIndices[i] < 0 || int64(rowIndices[i]) >= int64(rows) || columnIndices[i] < 0 || int64(columnIndices[i]) >= int64(cols) {
			return nil, fmt.Errorf("COO index out of range at entry %d", i)
		}
	}
	return runSparseOnDevice(ctx, func() (*SparseMatrix, error) {
		return createSparseMatrixWithBuffers(rows, cols, values, rowIndices, columnIndices, MatrixFormatCOO)
	}, func(matrix *SparseMatrix) error { return matrix.Destroy() })
}

func allocateSparseMatrix(rows, cols, nnz int, format MatrixFormat) (*SparseMatrix, error) {
	capacity := sparseMax(nnz, 1)
	values, err := memory.Alloc(int64(capacity) * 4) // float32 values
	if err != nil {
		return nil, fmt.Errorf("failed to allocate values memory: %w", err)
	}

	var rowPtrs, colInds *memory.Memory

	switch format {
	case MatrixFormatCSR:
		rowPtrs, err = memory.Alloc((int64(rows) + 1) * 4) // int32 row pointers
		if err != nil {
			return nil, errors.Join(fmt.Errorf("failed to allocate row pointers: %w", err), values.Free())
		}

		colInds, err = memory.Alloc(int64(capacity) * 4) // int32 column indices
		if err != nil {
			return nil, errors.Join(fmt.Errorf("failed to allocate column indices: %w", err), values.Free(), rowPtrs.Free())
		}

	case MatrixFormatCOO:
		// For COO format, we need row indices instead of row pointers
		rowPtrs, err = memory.Alloc(int64(capacity) * 4) // int32 row indices
		if err != nil {
			return nil, errors.Join(fmt.Errorf("failed to allocate row indices: %w", err), values.Free())
		}

		colInds, err = memory.Alloc(int64(capacity) * 4) // int32 column indices
		if err != nil {
			return nil, errors.Join(fmt.Errorf("failed to allocate column indices: %w", err), values.Free(), rowPtrs.Free())
		}

	}

	matrix := &SparseMatrix{
		rows:    rows,
		cols:    cols,
		nnz:     nnz,
		values:  values,
		rowPtrs: rowPtrs,
		colInds: colInds,
		format:  format,
	}
	if nnz == 0 && format == MatrixFormatCSR {
		if err := writeMathTypedMemory(matrix.rowPtrs, make([]int32, rows+1)); err != nil {
			return nil, errors.Join(err, matrix.Destroy())
		}
	}
	return matrix, nil
}

func createSparseMatrixWithBuffers(rows, cols int, values []float32, rowData []int32, colData []int32, format MatrixFormat) (*SparseMatrix, error) {
	if len(values) != len(colData) {
		return nil, fmt.Errorf("sparse buffer length mismatch")
	}
	if format == MatrixFormatCOO && len(values) != len(rowData) ||
		format == MatrixFormatCSR && (rows <= 0 || rows == int(^uint(0)>>1) || len(rowData) != rows+1) {
		return nil, fmt.Errorf("sparse buffer length mismatch")
	}
	allocationNNZ := sparseMax(len(values), 1)
	ctx := &SparseContext{}
	sparse, err := ctx.CreateSparseMatrix(rows, cols, allocationNNZ, format)
	if err != nil {
		return nil, err
	}
	sparse.nnz = len(values)
	if err := writeFloat32Memory(sparse.values, values, allocationNNZ); err != nil {
		return nil, errors.Join(err, sparse.Destroy())
	}
	if err := writeInt32Memory(sparse.colInds, colData, allocationNNZ); err != nil {
		return nil, errors.Join(err, sparse.Destroy())
	}
	if err := writeInt32Memory(sparse.rowPtrs, rowData, len(rowData)); err != nil {
		return nil, errors.Join(err, sparse.Destroy())
	}
	return sparse, nil
}

// SpMV computes y = alpha*A*x + beta*y. CSR and COO use cuSPARSE in CUDA mode.
// Native execution rejects aliased x/y and validates sparse metadata on the
// host, canonicalizing COO and duplicate CSR entries before the GPU operation.
func (ctx *SparseContext) SpMV(alpha float32, A *SparseMatrix, x *memory.Memory, beta float32, y *memory.Memory) error {
	if ctx == nil || A == nil || x == nil || y == nil {
		return fmt.Errorf("input matrices/vectors cannot be nil")
	}
	if err := ctx.lock(); err != nil {
		return err
	}
	defer ctx.mu.Unlock()
	retentions, err := retainSparseBuffers(A.values, A.rowPtrs, A.colInds, x, y)
	if err != nil {
		return err
	}
	defer func() {
		releaseSparseBuffers(retentions)
	}()
	rows, err := readSparseRows(A)
	if err != nil {
		return err
	}
	if uint64(A.cols) > uint64(x.Size())/4 || uint64(A.rows) > uint64(y.Size())/4 {
		return fmt.Errorf("sparse vector buffers are too small")
	}

	if !ctx.native {
		return sparseMVHost(alpha, A, rows, x, beta, y)
	}
	if err := ctx.validateNativeBuffers(A, x, y); err != nil {
		return err
	}
	transferred, err := ctx.runNativeMatrix(A, rows, retentions, func(matrix *SparseMatrix) error {
		return nativeSpMV(ctx, alpha, matrix, x, beta, y)
	})
	if transferred {
		retentions = nil
	}
	return err
}

// SpMM computes C = alpha*A*B + beta*C on the host. C must have enough
// capacity for the resulting nonzeros.
func (ctx *SparseContext) SpMM(alpha float32, A *SparseMatrix, B *SparseMatrix, beta float32, C *SparseMatrix) error {
	if err := ctx.lock(); err != nil {
		return err
	}
	defer ctx.mu.Unlock()
	if err := ctx.requireHostOperation("sparse matrix multiplication"); err != nil {
		return err
	}
	if A == nil || B == nil || C == nil {
		return fmt.Errorf("input matrices cannot be nil")
	}
	if A.cols != B.rows {
		return fmt.Errorf("incompatible matrix dimensions for multiplication")
	}
	if C.rows != A.rows || C.cols != B.cols {
		return fmt.Errorf("output matrix dimensions do not match multiplication result")
	}

	if ctx.native {
		return internal.RunOnDevice(ctx.deviceID, func() error {
			return sparseMMHost(alpha, A, B, beta, C)
		})
	}
	return sparseMMHost(alpha, A, B, beta, C)
}

// SpGEMM returns A*B in CSR format. CSR inputs use cuSPARSE in CUDA mode;
// other supported formats use host execution unless native-only mode is set.
func (ctx *SparseContext) SpGEMM(A, B *SparseMatrix) (*SparseMatrix, error) {
	if err := ctx.lock(); err != nil {
		return nil, err
	}
	defer ctx.mu.Unlock()
	if A == nil || B == nil {
		return nil, fmt.Errorf("input matrices cannot be nil")
	}
	if A.cols != B.rows {
		return nil, fmt.Errorf("incompatible matrix dimensions")
	}
	if ctx.native && A.format == MatrixFormatCSR && B.format == MatrixFormatCSR {
		return ctx.nativeSparseProduct(A, B)
	}
	if err := ctx.requireHostOperation("sparse matrix multiplication"); err != nil {
		return nil, err
	}

	return runSparseOnDevice(ctx, func() (*SparseMatrix, error) {
		return sparseMultiplyHost(A, B)
	}, func(matrix *SparseMatrix) error {
		return matrix.Destroy()
	})
}

// SpLU performs host-side sparse LU factorization without pivoting. It returns
// an error if a nonzero pivot cannot be used; the API does not expose a
// permutation matrix.
func (ctx *SparseContext) SpLU(A *SparseMatrix) (*SparseMatrix, *SparseMatrix, error) {
	if err := ctx.lock(); err != nil {
		return nil, nil, err
	}
	defer ctx.mu.Unlock()
	if err := ctx.requireHostOperation("sparse LU"); err != nil {
		return nil, nil, err
	}
	if A == nil {
		return nil, nil, fmt.Errorf("input matrix cannot be nil")
	}
	if A.rows != A.cols {
		return nil, nil, fmt.Errorf("matrix must be square for LU factorization")
	}

	type luResult struct {
		lower, upper *SparseMatrix
	}
	result, err := runSparseOnDevice(ctx, func() (luResult, error) {
		lower, upper, err := sparseLUHost(A)
		return luResult{lower, upper}, err
	}, func(result luResult) error {
		return errors.Join(result.lower.Destroy(), result.upper.Destroy())
	})
	return result.lower, result.upper, err
}

// SpSV solves a lower- or upper-triangular system with explicit nonzero
// diagonal entries. CSR uses cuSPARSE in CUDA mode; COO uses host execution.
func (ctx *SparseContext) SpSV(A *SparseMatrix, b, x *memory.Memory) error {
	if ctx == nil || A == nil || b == nil || x == nil {
		return fmt.Errorf("inputs cannot be nil")
	}
	if err := ctx.lock(); err != nil {
		return err
	}
	defer ctx.mu.Unlock()
	if A.format == MatrixFormatCOO {
		if err := ctx.requireHostOperation("COO triangular solve"); err != nil {
			return err
		}
	}
	retentions, err := retainSparseBuffers(A.values, A.rowPtrs, A.colInds, b, x)
	if err != nil {
		return err
	}
	defer func() { releaseSparseBuffers(retentions) }()
	if A.rows != A.cols {
		return fmt.Errorf("triangular matrix must be square")
	}
	rows, err := readSparseRows(A)
	if err != nil {
		return err
	}
	lower, err := sparseTriangleIsLower(rows)
	if err != nil {
		return err
	}
	if uint64(A.rows) > uint64(b.Size())/4 || uint64(A.rows) > uint64(x.Size())/4 {
		return fmt.Errorf("triangular vector buffers are too small")
	}
	if !ctx.native || A.format == MatrixFormatCOO {
		return sparseTriangularSolveRows(A, rows, b, x)
	}
	if err := ctx.validateNativeBuffers(A, b, x); err != nil {
		return err
	}
	transferred, err := ctx.runNativeMatrix(A, rows, retentions, func(matrix *SparseMatrix) error {
		return nativeSpSVCSR(ctx, matrix, b, x, lower)
	})
	if transferred {
		retentions = nil
	}
	return err
}

// DenseToSparse converts a row-major dense matrix to CSR or COO. CSR uses
// cuSPARSE in CUDA mode; COO uses host execution unless native-only mode is set.
func (ctx *SparseContext) DenseToSparse(dense *memory.Memory, rows, cols int, format MatrixFormat) (*SparseMatrix, error) {
	if err := ctx.lock(); err != nil {
		return nil, err
	}
	defer ctx.mu.Unlock()
	if dense == nil {
		return nil, fmt.Errorf("dense matrix cannot be nil")
	}
	if ctx.native && format == MatrixFormatCSR {
		return ctx.nativeDenseToSparse(dense, rows, cols)
	}
	if err := ctx.requireHostOperation("dense-to-sparse conversion"); err != nil {
		return nil, err
	}

	return runSparseOnDevice(ctx, func() (*SparseMatrix, error) {
		return denseToSparseFromHost(dense, rows, cols, format)
	}, func(matrix *SparseMatrix) error {
		return matrix.Destroy()
	})
}

// SparseToDense converts CSR or COO to row-major dense form. CUDA mode uses
// cuSPARSE after host-side index validation and canonicalization.
func (ctx *SparseContext) SparseToDense(sparse *SparseMatrix) (*memory.Memory, error) {
	if err := ctx.lock(); err != nil {
		return nil, err
	}
	defer ctx.mu.Unlock()
	if sparse == nil {
		return nil, fmt.Errorf("sparse matrix cannot be nil")
	}
	if !ctx.native {
		return sparseToDenseHost(sparse)
	}
	return ctx.nativeSparseToDense(sparse)
}

// Destroy cleans up the sparse matrix
func (sm *SparseMatrix) Destroy() error {
	if sm == nil {
		return nil
	}
	return errors.Join(sm.values.Free(), sm.rowPtrs.Free(), sm.colInds.Free())
}

// DestroyContext destroys the cuSPARSE context
func (ctx *SparseContext) DestroyContext() error {
	if ctx == nil {
		return nil
	}
	ctx.mu.Lock()
	defer ctx.mu.Unlock()
	if ctx.destroyed {
		return nil
	}
	if ctx.native {
		return internal.RunOnDevice(ctx.deviceID, func() error {
			if len(ctx.pending) > 0 || len(ctx.pendingMatrices) > 0 || len(ctx.pendingWorkspaces) > 0 || len(ctx.pendingCleanups) > 0 {
				if err := internal.CudaDefaultStreamSynchronizeOnDevice(ctx.deviceID); err != nil {
					return err
				}
				if err := ctx.releasePending(); err != nil {
					return err
				}
			}
			if err := destroyNativeSparseContext(ctx); err != nil {
				return err
			}
			ctx.handle = nil
			ctx.destroyed = true
			return nil
		})
	}
	// Cleanup simulation context
	ctx.handle = nil
	ctx.destroyed = true
	return nil
}

// GetMatrixInfo returns information about the sparse matrix
func (sm *SparseMatrix) GetMatrixInfo() (rows, cols, nnz int, format MatrixFormat) {
	return sm.rows, sm.cols, sm.nnz, sm.format
}

// Values returns the matrix-owned float32 value buffer. It remains valid until
// Destroy and must not be freed separately.
func (sm *SparseMatrix) Values() *memory.Memory {
	if sm == nil {
		return nil
	}
	return sm.values
}

// RowData returns matrix-owned CSR row offsets or COO row indices.
func (sm *SparseMatrix) RowData() *memory.Memory {
	if sm == nil {
		return nil
	}
	return sm.rowPtrs
}

// ColumnIndices returns the matrix-owned zero-based int32 column indices.
func (sm *SparseMatrix) ColumnIndices() *memory.Memory {
	if sm == nil {
		return nil
	}
	return sm.colInds
}

func denseToSparseFromHost(dense *memory.Memory, rows, cols int, format MatrixFormat) (*SparseMatrix, error) {
	if rows <= 0 || cols <= 0 || int64(rows) > math.MaxInt32 || int64(cols) > math.MaxInt32 ||
		uint64(rows)*uint64(cols) > uint64(^uint(0)>>1)/4 {
		return nil, fmt.Errorf("invalid dense matrix dimensions")
	}
	values, err := readMathFloat32Memory(dense, rows*cols)
	if err != nil {
		return nil, err
	}
	switch format {
	case MatrixFormatCSR:
		rowPtrs := make([]int32, rows+1)
		colInds := make([]int32, 0)
		nonZeroValues := make([]float32, 0)
		for row := range rows {
			rowPtrs[row] = int32(len(nonZeroValues))
			for col := range cols {
				value := values[row*cols+col]
				if value == 0 {
					continue
				}
				if len(nonZeroValues) >= math.MaxInt32 {
					return nil, fmt.Errorf("sparse matrix exceeds 32-bit index capacity")
				}
				nonZeroValues = append(nonZeroValues, value)
				colInds = append(colInds, int32(col))
			}
		}
		rowPtrs[rows] = int32(len(nonZeroValues))
		return createSparseMatrixWithBuffers(rows, cols, nonZeroValues, rowPtrs, colInds, MatrixFormatCSR)
	case MatrixFormatCOO:
		rowIdx := make([]int32, 0)
		colInds := make([]int32, 0)
		nonZeroValues := make([]float32, 0)
		for row := range rows {
			for col := range cols {
				value := values[row*cols+col]
				if value == 0 {
					continue
				}
				if len(nonZeroValues) >= math.MaxInt32 {
					return nil, fmt.Errorf("sparse matrix exceeds 32-bit index capacity")
				}
				nonZeroValues = append(nonZeroValues, value)
				rowIdx = append(rowIdx, int32(row))
				colInds = append(colInds, int32(col))
			}
		}
		return createSparseMatrixWithBuffers(rows, cols, nonZeroValues, rowIdx, colInds, MatrixFormatCOO)
	default:
		return nil, fmt.Errorf("unsupported matrix format")
	}
}

func sparseToDenseHost(sparse *SparseMatrix) (*memory.Memory, error) {
	rows, err := readSparseRows(sparse)
	if err != nil {
		return nil, err
	}
	if uint64(sparse.rows)*uint64(sparse.cols) > uint64(^uint(0)>>1)/4 {
		return nil, fmt.Errorf("dense matrix dimensions exceed supported size")
	}
	dense, err := memory.Alloc(int64(sparse.rows) * int64(sparse.cols) * 4)
	if err != nil {
		return nil, err
	}
	denseValues := make([]float32, sparse.rows*sparse.cols)
	for row, entries := range rows {
		for col, value := range entries {
			denseValues[row*sparse.cols+col] = value
		}
	}
	if err := writeFloat32Memory(dense, denseValues, len(denseValues)); err != nil {
		return nil, errors.Join(err, dense.Free())
	}
	return dense, nil
}

func copySparseMatrixData(dst, src *SparseMatrix) error {
	if dst.rows != src.rows || dst.cols != src.cols || dst.format != src.format {
		return fmt.Errorf("sparse matrix layouts do not match")
	}
	if dst.values.Size() < src.values.Size() || dst.colInds.Size() < src.colInds.Size() || dst.rowPtrs.Size() < src.rowPtrs.Size() {
		return fmt.Errorf("destination sparse buffers are too small")
	}
	srcValues, err := readFloat32Memory(src.values, sparseMax(src.nnz, 1))
	if err != nil {
		return err
	}
	if err := writeFloat32Memory(dst.values, srcValues[:src.nnz], sparseMax(src.nnz, 1)); err != nil {
		return err
	}
	srcCols, err := readInt32Memory(src.colInds, sparseMax(src.nnz, 1))
	if err != nil {
		return err
	}
	if err := writeInt32Memory(dst.colInds, srcCols[:src.nnz], sparseMax(src.nnz, 1)); err != nil {
		return err
	}
	rowLength := src.rows + 1
	if src.format == MatrixFormatCOO {
		rowLength = sparseMax(src.nnz, 1)
	}
	srcRows, err := readInt32Memory(src.rowPtrs, rowLength)
	if err != nil {
		return err
	}
	if err := writeInt32Memory(dst.rowPtrs, srcRows, rowLength); err != nil {
		return err
	}
	dst.nnz = src.nnz
	return nil
}

func readFloat32Memory(mem *memory.Memory, length int) ([]float32, error) {
	return readMathTypedMemory[float32](mem, length)
}

func writeFloat32Memory(mem *memory.Memory, values []float32, length int) error {
	if length < 0 || len(values) > length {
		return fmt.Errorf("invalid float buffer length")
	}
	host := make([]float32, length)
	copy(host, values)
	return writeMathTypedMemory(mem, host)
}

func readInt32Memory(mem *memory.Memory, length int) ([]int32, error) {
	return readMathTypedMemory[int32](mem, length)
}

func writeInt32Memory(mem *memory.Memory, values []int32, length int) error {
	if length < 0 || len(values) > length {
		return fmt.Errorf("invalid integer buffer length")
	}
	host := make([]int32, length)
	copy(host, values)
	return writeMathTypedMemory(mem, host)
}

func sparseMax(a, b int) int {
	if a > b {
		return a
	}
	return b
}
