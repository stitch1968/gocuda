package libraries

import (
	"errors"
	"fmt"

	"github.com/stitch1968/gocuda/internal"
	"github.com/stitch1968/gocuda/memory"
)

func retainSparseBuffers(buffers ...*memory.Memory) ([]*memory.Retention, error) {
	retentions := make([]*memory.Retention, 0, len(buffers))
	for _, buffer := range buffers {
		retention, err := buffer.Retain()
		if err != nil {
			releaseSparseBuffers(retentions)
			return nil, err
		}
		retentions = append(retentions, retention)
	}
	return retentions, nil
}

func releaseSparseBuffers(retentions []*memory.Retention) {
	for _, held := range retentions {
		held.Release()
	}
}

func (ctx *SparseContext) validateNativeBuffers(matrix *SparseMatrix, input, output *memory.Memory) error {
	if input.Ptr() == output.Ptr() {
		return fmt.Errorf("native sparse operation does not support aliased input and output")
	}
	if output.Ptr() == matrix.values.Ptr() || output.Ptr() == matrix.rowPtrs.Ptr() || output.Ptr() == matrix.colInds.Ptr() {
		return fmt.Errorf("native sparse output cannot alias sparse matrix buffers")
	}
	return ctx.validateNativeMemory(matrix.values, matrix.rowPtrs, matrix.colInds, input, output)
}

func (ctx *SparseContext) validateNativeMemory(buffers ...*memory.Memory) error {
	for _, buffer := range buffers {
		if buffer == nil || buffer.Ptr() == nil {
			return fmt.Errorf("native sparse buffer is nil or freed")
		}
		if buffer.GetType() != memory.TypeDevice && buffer.GetType() != memory.TypeDeviceOnly && buffer.GetType() != memory.TypeUnified {
			return fmt.Errorf("native sparse operation requires device or managed memory")
		}
		if buffer.GetDeviceID() != ctx.deviceID {
			return fmt.Errorf("native sparse buffers must belong to context device %d", ctx.deviceID)
		}
	}
	return nil
}

func (ctx *SparseContext) nativeSparseProduct(A, B *SparseMatrix) (*SparseMatrix, error) {
	retentions, err := retainSparseBuffers(A.values, A.rowPtrs, A.colInds, B.values, B.rowPtrs, B.colInds)
	if err != nil {
		return nil, err
	}
	defer func() { releaseSparseBuffers(retentions) }()
	if err := ctx.validateNativeMemory(A.values, A.rowPtrs, A.colInds, B.values, B.rowPtrs, B.colInds); err != nil {
		return nil, err
	}
	rowsA, err := readSparseRows(A)
	if err != nil {
		return nil, err
	}
	rowsB, err := readSparseRows(B)
	if err != nil {
		return nil, err
	}
	canonical := make([]*SparseMatrix, 0, 2)
	for _, input := range []struct {
		rows sparseRows
		cols int
	}{{rowsA, A.cols}, {rowsB, B.cols}} {
		matrix, err := runSparseOnDevice(ctx, func() (*SparseMatrix, error) {
			return sparseMatrixFromRows(input.rows, input.cols, MatrixFormatCSR)
		}, func(matrix *SparseMatrix) error { return matrix.Destroy() })
		if err != nil {
			for _, allocated := range canonical {
				if cleanupErr := allocated.Destroy(); cleanupErr != nil {
					ctx.pendingMatrices = append(ctx.pendingMatrices, allocated)
					err = errors.Join(err, cleanupErr)
				}
			}
			return nil, err
		}
		canonical = append(canonical, matrix)
	}
	var result *SparseMatrix
	err = internal.RunOnDevice(ctx.deviceID, func() error {
		var runErr error
		result, runErr = nativeSpGEMMCSR(ctx, canonical[0], canonical[1])
		return runErr
	})
	if err != nil {
		ctx.pending = append(ctx.pending, retentions...)
		retentions = nil
		ctx.pendingMatrices = append(ctx.pendingMatrices, canonical...)
		if result != nil {
			ctx.pendingMatrices = append(ctx.pendingMatrices, result)
		}
		return nil, err
	}
	cleanupErr := ctx.releasePending()
	for _, allocated := range canonical {
		if err := allocated.Destroy(); err != nil {
			ctx.pendingMatrices = append(ctx.pendingMatrices, allocated)
			cleanupErr = errors.Join(cleanupErr, err)
		}
	}
	if cleanupErr != nil {
		if err := result.Destroy(); err != nil {
			ctx.pendingMatrices = append(ctx.pendingMatrices, result)
			cleanupErr = errors.Join(cleanupErr, err)
		}
		return nil, cleanupErr
	}
	return result, nil
}

func (ctx *SparseContext) nativeSparseToDense(matrix *SparseMatrix) (*memory.Memory, error) {
	retentions, err := retainSparseBuffers(matrix.values, matrix.rowPtrs, matrix.colInds)
	if err != nil {
		return nil, err
	}
	defer func() { releaseSparseBuffers(retentions) }()
	if err := ctx.validateNativeMemory(matrix.values, matrix.rowPtrs, matrix.colInds); err != nil {
		return nil, err
	}
	rows, err := readSparseRows(matrix)
	if err != nil {
		return nil, err
	}
	if uint64(matrix.rows)*uint64(matrix.cols) > uint64(^uint(0)>>1)/4 {
		return nil, fmt.Errorf("dense matrix dimensions exceed supported size")
	}
	var result *memory.Memory
	transferred, err := ctx.runNativeMatrix(matrix, rows, retentions, func(canonical *SparseMatrix) error {
		var runErr error
		result, runErr = nativeSparseToDense(ctx, canonical)
		return runErr
	})
	if transferred {
		retentions = nil
		if result != nil {
			ctx.pendingWorkspaces = append(ctx.pendingWorkspaces, result)
		}
		return nil, err
	}
	if err != nil {
		if result != nil {
			if cleanupErr := result.Free(); cleanupErr != nil {
				ctx.pendingWorkspaces = append(ctx.pendingWorkspaces, result)
				err = errors.Join(err, cleanupErr)
			}
		}
		return nil, err
	}
	if result == nil {
		return nil, fmt.Errorf("native sparse-to-dense conversion returned no output")
	}
	return result, nil
}

func (ctx *SparseContext) nativeDenseToSparse(dense *memory.Memory, rows, cols int) (*SparseMatrix, error) {
	retentions, err := retainSparseBuffers(dense)
	if err != nil {
		return nil, err
	}
	defer func() { releaseSparseBuffers(retentions) }()
	if err := ctx.validateNativeMemory(dense); err != nil {
		return nil, err
	}
	if rows <= 0 || cols <= 0 || int64(rows) > int64(^uint32(0)>>1) || int64(cols) > int64(^uint32(0)>>1) ||
		uint64(rows)*uint64(cols) > uint64(^uint(0)>>1)/4 || uint64(rows)*uint64(cols) > uint64(dense.Size())/4 {
		return nil, fmt.Errorf("dense matrix dimensions are invalid or exceed its buffer")
	}
	if uint64(rows)*uint64(cols) > uint64(^uint32(0)>>1) {
		return nil, fmt.Errorf("native dense-to-CSR matrix may exceed 32-bit nonzero capacity")
	}
	var result *SparseMatrix
	err = internal.RunOnDevice(ctx.deviceID, func() error {
		var runErr error
		result, runErr = nativeDenseToSparseCSR(ctx, dense, rows, cols)
		return runErr
	})
	if err != nil {
		ctx.pending = append(ctx.pending, retentions...)
		retentions = nil
		if result != nil {
			ctx.pendingMatrices = append(ctx.pendingMatrices, result)
		}
		return nil, err
	}
	if result == nil {
		return nil, fmt.Errorf("native dense-to-sparse conversion returned no output")
	}
	if err := ctx.releasePending(); err != nil {
		if cleanupErr := result.Destroy(); cleanupErr != nil {
			ctx.pendingMatrices = append(ctx.pendingMatrices, result)
			return nil, errors.Join(err, cleanupErr)
		}
		return nil, err
	}
	return result, nil
}

// runNativeMatrix retains caller buffers on failure until a later successful
// synchronization or context destruction. The bool reports transferred leases.
func (ctx *SparseContext) runNativeMatrix(matrix *SparseMatrix, rows sparseRows, retentions []*memory.Retention, run func(*SparseMatrix) error) (bool, error) {
	uniqueEntries := 0
	for _, row := range rows {
		uniqueEntries += len(row)
	}
	var canonical *SparseMatrix
	if matrix.format == MatrixFormatCOO || uniqueEntries != matrix.nnz {
		var err error
		canonical, err = runSparseOnDevice(ctx, func() (*SparseMatrix, error) {
			return sparseMatrixFromRows(rows, matrix.cols, matrix.format)
		}, func(matrix *SparseMatrix) error { return matrix.Destroy() })
		if err != nil {
			return false, err
		}
	}
	nativeMatrix := matrix
	if canonical != nil {
		nativeMatrix = canonical
	}
	err := internal.RunOnDevice(ctx.deviceID, func() error { return run(nativeMatrix) })
	if err != nil {
		ctx.pending = append(ctx.pending, retentions...)
		if canonical != nil {
			ctx.pendingMatrices = append(ctx.pendingMatrices, canonical)
		}
		return true, err
	}
	cleanupErr := ctx.releasePending()
	if canonical != nil {
		if err := canonical.Destroy(); err != nil {
			ctx.pendingMatrices = append(ctx.pendingMatrices, canonical)
			cleanupErr = errors.Join(cleanupErr, err)
		}
	}
	return false, cleanupErr
}
