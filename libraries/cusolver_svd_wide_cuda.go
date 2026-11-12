//go:build cuda

package libraries

import (
	"errors"
	"fmt"

	"github.com/stitch1968/gocuda/memory"
)

func nativeSVDWide(ctx *SolverContext, matrix *memory.Memory, rows, cols int, vectors bool) (result *SVDInfo, err error) {
	values, err := readMathFloat32Memory(matrix, rows*cols)
	if err != nil {
		return nil, err
	}
	transposedValues := make([]float32, len(values))
	for row := range rows {
		for col := range cols {
			transposedValues[row*cols+col] = values[col*rows+row]
		}
	}
	transposed, err := solverAllocateValues(ctx.deviceID, transposedValues)
	if err != nil {
		return nil, fmt.Errorf("allocate transposed SVD input: %w", err)
	}
	defer func() {
		err = errors.Join(err, transposed.Free())
		if err != nil && result != nil {
			err = errors.Join(err, result.Destroy())
			result = nil
		}
	}()
	decomposition, err := nativeSVDDecomposition(ctx, transposed, cols, rows, vectors)
	if err != nil {
		return nil, err
	}
	defer func() {
		err = errors.Join(err, decomposition.Destroy())
	}()
	if !vectors {
		result = decomposition
		decomposition = nil
		return result, nil
	}
	uTransposed, err := readMathFloat32Memory(decomposition.u, cols*cols)
	if err != nil {
		return nil, err
	}
	vtTransposed, err := readMathFloat32Memory(decomposition.vt, rows*rows)
	if err != nil {
		return nil, err
	}
	uValues := make([]float32, rows*rows)
	vtValues := make([]float32, cols*cols)
	for col := range rows {
		for row := range rows {
			uValues[col*rows+row] = vtTransposed[row*rows+col]
		}
	}
	for col := range cols {
		for row := range cols {
			vtValues[col*cols+row] = uTransposed[row*cols+col]
		}
	}
	result = &SVDInfo{}
	result.u, err = solverAllocateValues(ctx.deviceID, uValues)
	if err != nil {
		return result, err
	}
	result.vt, err = solverAllocateValues(ctx.deviceID, vtValues)
	if err != nil {
		return result, err
	}
	result.s = decomposition.s
	result.workspace = decomposition.workspace
	decomposition.s = nil
	decomposition.workspace = nil
	return result, nil
}
