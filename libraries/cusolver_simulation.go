package libraries

import (
	"errors"
	"fmt"
	"math"

	"github.com/stitch1968/gocuda/memory"
)

func factorLU(values []float32, rows, cols int) ([]int32, error) {
	pivots := make([]int32, min(rows, cols))
	for column := range len(pivots) {
		pivot := column
		for row := column + 1; row < rows; row++ {
			if math.Abs(float64(values[column*rows+row])) > math.Abs(float64(values[column*rows+pivot])) {
				pivot = row
			}
		}
		if values[column*rows+pivot] == 0 {
			return nil, fmt.Errorf("LU factorization: singular matrix at pivot %d", column+1)
		}
		pivots[column] = int32(pivot + 1)
		if pivot != column {
			for j := range cols {
				values[j*rows+column], values[j*rows+pivot] = values[j*rows+pivot], values[j*rows+column]
			}
		}
		for row := column + 1; row < rows; row++ {
			index := column*rows + row
			values[index] /= values[column*rows+column]
			for j := column + 1; j < cols; j++ {
				values[j*rows+row] -= values[index] * values[j*rows+column]
			}
		}
	}
	return pivots, nil
}

func simulatedLUFactorization(matrix *memory.Memory, rows, cols int) (*LUInfo, error) {
	values, err := readMathFloat32Memory(matrix, rows*cols)
	if err != nil {
		return nil, err
	}
	pivots, err := factorLU(values, rows, cols)
	if err != nil {
		return nil, err
	}
	pivotMemory, err := memory.AllocOnDevice(int64(len(pivots))*4, matrix.GetDeviceID())
	if err != nil {
		return nil, fmt.Errorf("allocate LU pivots: %w", err)
	}
	if err := writeMathTypedMemory(pivotMemory, pivots); err != nil {
		return nil, errors.Join(err, pivotMemory.Free())
	}
	if err := writeMathFloat32Memory(matrix, values); err != nil {
		return nil, errors.Join(err, pivotMemory.Free())
	}
	return &LUInfo{ipiv: pivotMemory}, nil
}

func simulatedSolveLinearSystem(matrix, rhs *memory.Memory, n int) (*memory.Memory, error) {
	values, err := readMathFloat32Memory(matrix, n*n)
	if err != nil {
		return nil, err
	}
	solution, err := readMathFloat32Memory(rhs, n)
	if err != nil {
		return nil, err
	}
	pivots, err := factorLU(values, n, n)
	if err != nil {
		return nil, err
	}
	for i, pivot := range pivots {
		j := int(pivot) - 1
		solution[i], solution[j] = solution[j], solution[i]
	}
	for row := range n {
		for col := 0; col < row; col++ {
			solution[row] -= values[col*n+row] * solution[col]
		}
	}
	for row := n - 1; row >= 0; row-- {
		for col := row + 1; col < n; col++ {
			solution[row] -= values[col*n+row] * solution[col]
		}
		solution[row] /= values[row*n+row]
	}
	output, err := memory.AllocOnDevice(int64(n)*4, matrix.GetDeviceID())
	if err != nil {
		return nil, fmt.Errorf("allocate solver output: %w", err)
	}
	if err := writeMathFloat32Memory(output, solution); err != nil {
		return nil, errors.Join(err, output.Free())
	}
	if err := writeMathFloat32Memory(matrix, values); err != nil {
		return nil, errors.Join(err, output.Free())
	}
	return output, nil
}

func simulatedCholeskyFactorization(matrix *memory.Memory, n int) error {
	values, err := readMathFloat32Memory(matrix, n*n)
	if err != nil {
		return err
	}
	for col := range n {
		for row := 0; row < col; row++ {
			sum := values[col*n+row]
			for i := 0; i < row; i++ {
				sum -= values[row*n+i] * values[col*n+i]
			}
			values[col*n+row] = sum / values[row*n+row]
		}
		diagonal := values[col*n+col]
		for row := 0; row < col; row++ {
			value := values[col*n+row]
			diagonal -= value * value
		}
		if diagonal <= 0 || math.IsNaN(float64(diagonal)) {
			return fmt.Errorf("Cholesky factorization: matrix is not positive definite at pivot %d", col+1)
		}
		values[col*n+col] = float32(math.Sqrt(float64(diagonal)))
	}
	return writeMathFloat32Memory(matrix, values)
}
