package libraries

import (
	"errors"
	"fmt"
	"math"
	"sort"

	"github.com/stitch1968/gocuda/memory"
)

type sparseRows []map[int]float32

func readSparseRows(matrix *SparseMatrix) (sparseRows, error) {
	if matrix == nil || matrix.rows <= 0 || matrix.cols <= 0 || matrix.nnz < 0 ||
		int64(matrix.rows) > math.MaxInt32 || int64(matrix.cols) > math.MaxInt32 || int64(matrix.nnz) > math.MaxInt32 {
		return nil, fmt.Errorf("invalid sparse matrix dimensions")
	}
	if matrix.values == nil || matrix.colInds == nil || matrix.rowPtrs == nil ||
		matrix.values.Ptr() == nil || matrix.colInds.Ptr() == nil || matrix.rowPtrs.Ptr() == nil {
		return nil, fmt.Errorf("sparse matrix contains nil or freed buffers")
	}
	if matrix.format == MatrixFormatCSR && (uint64(matrix.rows)+1 > uint64(matrix.rowPtrs.Size())/4 || matrix.rows == int(^uint(0)>>1)) {
		return nil, fmt.Errorf("CSR row offsets buffer is too small")
	}
	values, err := readMathTypedMemory[float32](matrix.values, matrix.nnz)
	if err != nil {
		return nil, fmt.Errorf("sparse values: %w", err)
	}
	columns, err := readMathTypedMemory[int32](matrix.colInds, matrix.nnz)
	if err != nil {
		return nil, fmt.Errorf("sparse column indices: %w", err)
	}
	rows := make(sparseRows, matrix.rows)
	for i := range rows {
		rows[i] = make(map[int]float32)
	}
	add := func(row, offset int) error {
		column := int(columns[offset])
		if column < 0 || column >= matrix.cols {
			return fmt.Errorf("sparse column %d out of range at entry %d", column, offset)
		}
		rows[row][column] += values[offset]
		return nil
	}
	switch matrix.format {
	case MatrixFormatCSR:
		offsets, err := readMathTypedMemory[int32](matrix.rowPtrs, matrix.rows+1)
		if err != nil {
			return nil, fmt.Errorf("CSR row offsets: %w", err)
		}
		if offsets[0] != 0 || int64(offsets[matrix.rows]) != int64(matrix.nnz) {
			return nil, fmt.Errorf("CSR row offsets must start at zero and end at nnz")
		}
		for row := range matrix.rows {
			start, end := int(offsets[row]), int(offsets[row+1])
			if start < 0 || end < start || end > matrix.nnz {
				return nil, fmt.Errorf("invalid CSR row offsets at row %d", row)
			}
			for index := start; index < end; index++ {
				if err := add(row, index); err != nil {
					return nil, err
				}
			}
		}
	case MatrixFormatCOO:
		indices, err := readMathTypedMemory[int32](matrix.rowPtrs, matrix.nnz)
		if err != nil {
			return nil, fmt.Errorf("COO row indices: %w", err)
		}
		for index, rowValue := range indices {
			row := int(rowValue)
			if row < 0 || row >= matrix.rows {
				return nil, fmt.Errorf("sparse row %d out of range at entry %d", row, index)
			}
			if err := add(row, index); err != nil {
				return nil, err
			}
		}
	default:
		return nil, fmt.Errorf("unsupported sparse matrix format %d", matrix.format)
	}
	return rows, nil
}

func sparseMatrixFromRows(rows sparseRows, columns int, format MatrixFormat) (*SparseMatrix, error) {
	values := make([]float32, 0)
	columnIndices := make([]int32, 0)
	rowData := make([]int32, 0)
	if format == MatrixFormatCSR {
		rowData = make([]int32, len(rows)+1)
	} else if format != MatrixFormatCOO {
		return nil, fmt.Errorf("unsupported sparse matrix format %d", format)
	}
	for row, entries := range rows {
		for col := range entries {
			if col < 0 || col >= columns {
				return nil, fmt.Errorf("sparse column %d out of range", col)
			}
		}
		for _, col := range sparseSortedColumns(entries) {
			if entries[col] == 0 {
				continue
			}
			if len(values) >= math.MaxInt32 {
				return nil, fmt.Errorf("sparse matrix exceeds 32-bit index capacity")
			}
			values = append(values, entries[col])
			columnIndices = append(columnIndices, int32(col))
			if format == MatrixFormatCOO {
				rowData = append(rowData, int32(row))
			}
		}
		if format == MatrixFormatCSR {
			rowData[row+1] = int32(len(values))
		}
	}
	return createSparseMatrixWithBuffers(len(rows), columns, values, rowData, columnIndices, format)
}

func sparseSortedColumns(entries map[int]float32) []int {
	keys := make([]int, 0, len(entries))
	for col := range entries {
		keys = append(keys, col)
	}
	sort.Ints(keys)
	return keys
}

func sparseMVHost(alpha float32, matrix *SparseMatrix, rows sparseRows, x *memory.Memory, beta float32, y *memory.Memory) error {
	xValues, err := readMathFloat32Memory(x, matrix.cols)
	if err != nil {
		return err
	}
	yValues, err := readMathFloat32Memory(y, matrix.rows)
	if err != nil {
		return err
	}
	for row, entries := range rows {
		var sum float32
		for _, col := range sparseSortedColumns(entries) {
			sum += entries[col] * xValues[col]
		}
		yValues[row] = alpha*sum + beta*yValues[row]
	}
	return writeMathFloat32Memory(y, yValues)
}

func sparseMultiplyRows(a, b sparseRows) sparseRows {
	result := make(sparseRows, len(a))
	for row, entries := range a {
		result[row] = make(map[int]float32)
		for _, inner := range sparseSortedColumns(entries) {
			for _, col := range sparseSortedColumns(b[inner]) {
				result[row][col] += entries[inner] * b[inner][col]
			}
		}
	}
	return result
}

func sparseMultiplyHost(A, B *SparseMatrix) (*SparseMatrix, error) {
	a, err := readSparseRows(A)
	if err != nil {
		return nil, err
	}
	b, err := readSparseRows(B)
	if err != nil {
		return nil, err
	}
	return sparseMatrixFromRows(sparseMultiplyRows(a, b), B.cols, MatrixFormatCSR)
}

func sparseMMHost(alpha float32, A, B *SparseMatrix, beta float32, C *SparseMatrix) error {
	a, err := readSparseRows(A)
	if err != nil {
		return err
	}
	b, err := readSparseRows(B)
	if err != nil {
		return err
	}
	c, err := readSparseRows(C)
	if err != nil {
		return err
	}
	product := sparseMultiplyRows(a, b)
	for row := range product {
		for col := range product[row] {
			product[row][col] *= alpha
		}
		for col, value := range c[row] {
			product[row][col] += beta * value
		}
	}
	result, err := sparseMatrixFromRows(product, C.cols, C.format)
	if err != nil {
		return err
	}
	copyErr := copySparseMatrixData(C, result)
	return errors.Join(copyErr, result.Destroy())
}

func sparseLUHost(matrix *SparseMatrix) (*SparseMatrix, *SparseMatrix, error) {
	work, err := readSparseRows(matrix)
	if err != nil {
		return nil, nil, err
	}
	lower := make(sparseRows, matrix.rows)
	for row := range lower {
		lower[row] = map[int]float32{row: 1}
	}
	for pivot := range matrix.rows {
		diagonal := work[pivot][pivot]
		if diagonal == 0 || math.IsNaN(float64(diagonal)) || math.IsInf(float64(diagonal), 0) {
			return nil, nil, fmt.Errorf("sparse LU requires a finite nonzero pivot at row %d; pivoting is not supported", pivot)
		}
		for row := pivot + 1; row < matrix.rows; row++ {
			value := work[row][pivot]
			if value == 0 {
				continue
			}
			factor := value / diagonal
			lower[row][pivot] = factor
			delete(work[row], pivot)
			for _, col := range sparseSortedColumns(work[pivot]) {
				if col > pivot {
					work[row][col] -= factor * work[pivot][col]
				}
			}
		}
	}
	L, err := sparseMatrixFromRows(lower, matrix.cols, MatrixFormatCSR)
	if err != nil {
		return nil, nil, err
	}
	U, err := sparseMatrixFromRows(work, matrix.cols, MatrixFormatCSR)
	if err != nil {
		return nil, nil, errors.Join(err, L.Destroy())
	}
	return L, U, nil
}

func sparseTriangleIsLower(rows sparseRows) (bool, error) {
	lower, upper := true, true
	for row, entries := range rows {
		for col, value := range entries {
			if value == 0 {
				continue
			}
			if col > row {
				lower = false
			}
			if col < row {
				upper = false
			}
		}
		diagonal := entries[row]
		if diagonal == 0 || math.IsNaN(float64(diagonal)) || math.IsInf(float64(diagonal), 0) {
			return false, fmt.Errorf("triangular matrix has an invalid diagonal at row %d", row)
		}
	}
	if !lower && !upper {
		return false, fmt.Errorf("sparse matrix must be triangular")
	}
	return lower, nil
}

func sparseTriangularSolveRows(matrix *SparseMatrix, rows sparseRows, b, x *memory.Memory) error {
	lower, err := sparseTriangleIsLower(rows)
	if err != nil {
		return err
	}
	rhs, err := readMathFloat32Memory(b, matrix.rows)
	if err != nil {
		return err
	}
	if x == nil || x.Ptr() == nil || uint64(matrix.rows) > uint64(x.Size())/4 {
		return fmt.Errorf("solution buffer is nil, freed, or too small")
	}
	solution := make([]float32, matrix.rows)
	for step := range matrix.rows {
		row := step
		if !lower {
			row = matrix.rows - 1 - step
		}
		diagonal := rows[row][row]
		value := rhs[row]
		for _, col := range sparseSortedColumns(rows[row]) {
			if col != row {
				value -= rows[row][col] * solution[col]
			}
		}
		solution[row] = value / diagonal
	}
	return writeMathFloat32Memory(x, solution)
}
