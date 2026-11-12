package libraries

import (
	"errors"
	"fmt"
	"math"
	"unsafe"

	"github.com/stitch1968/gocuda/memory"
	"gonum.org/v1/gonum/mat"
)

func solverAllocateValues[T any](deviceID int, values []T) (*memory.Memory, error) {
	var zero T
	size := uint64(unsafe.Sizeof(zero))
	if len(values) == 0 || size == 0 || uint64(len(values)) > uint64(^uint(0)>>1)/size {
		return nil, fmt.Errorf("invalid solver output size")
	}
	output, err := memory.AllocOnDevice(int64(uint64(len(values))*size), deviceID)
	if err != nil {
		return nil, err
	}
	if err := writeMathTypedMemory(output, values); err != nil {
		return nil, errors.Join(err, output.Free())
	}
	return output, nil
}

func solverReadFiniteMatrix(matrix *memory.Memory, rows, cols int) ([]float64, error) {
	values, err := readMathFloat32Memory(matrix, rows*cols)
	if err != nil {
		return nil, err
	}
	result := make([]float64, len(values))
	for row := range rows {
		for col := range cols {
			value := float64(values[col*rows+row])
			if math.IsNaN(value) || math.IsInf(value, 0) {
				return nil, fmt.Errorf("matrix element [%d,%d] is not finite", row, col)
			}
			result[row*cols+col] = value
		}
	}
	return result, nil
}

func solverCheckedSquareSize(size int) (int, error) {
	maxElements := int(^uint(0)>>1) / 4
	if size <= 0 || size > maxElements/size {
		return 0, fmt.Errorf("solver output dimensions exceed supported size")
	}
	return size * size, nil
}

func simulatedQRFactorization(matrix *memory.Memory, rows, cols int) (*QRInfo, error) {
	values, err := readMathFloat32Memory(matrix, rows*cols)
	if err != nil {
		return nil, err
	}
	work := make([]float64, len(values))
	for i, value := range values {
		work[i] = float64(value)
		if math.IsNaN(work[i]) || math.IsInf(work[i], 0) {
			return nil, fmt.Errorf("matrix contains a non-finite value")
		}
	}
	tau := make([]float32, min(rows, cols))
	for col := range len(tau) {
		alpha := work[col*rows+col]
		norm := 0.0
		for row := col + 1; row < rows; row++ {
			norm = math.Hypot(norm, work[col*rows+row])
		}
		if norm == 0 {
			continue
		}
		beta := -math.Copysign(math.Hypot(alpha, norm), alpha)
		factor := (beta - alpha) / beta
		tau[col] = float32(factor)
		for row := col + 1; row < rows; row++ {
			work[col*rows+row] /= alpha - beta
		}
		work[col*rows+col] = beta
		for otherCol := col + 1; otherCol < cols; otherCol++ {
			dot := work[otherCol*rows+col]
			for row := col + 1; row < rows; row++ {
				dot += work[col*rows+row] * work[otherCol*rows+row]
			}
			dot *= factor
			work[otherCol*rows+col] -= dot
			for row := col + 1; row < rows; row++ {
				work[otherCol*rows+row] -= work[col*rows+row] * dot
			}
		}
	}
	for i := range values {
		values[i] = float32(work[i])
	}
	tauMemory, err := solverAllocateValues(matrix.GetDeviceID(), tau)
	if err != nil {
		return nil, fmt.Errorf("allocate QR scalars: %w", err)
	}
	if err := writeMathFloat32Memory(matrix, values); err != nil {
		return nil, errors.Join(err, tauMemory.Free())
	}
	return &QRInfo{tau: tauMemory}, nil
}

func simulatedSVDDecomposition(matrix *memory.Memory, rows, cols int, computeUV bool) (info *SVDInfo, err error) {
	var uCount, vCount int
	if computeUV {
		if uCount, err = solverCheckedSquareSize(rows); err != nil {
			return nil, err
		}
		if vCount, err = solverCheckedSquareSize(cols); err != nil {
			return nil, err
		}
	}
	values, err := solverReadFiniteMatrix(matrix, rows, cols)
	if err != nil {
		return nil, err
	}
	var decomposition mat.SVD
	kind := mat.SVDNone
	if computeUV {
		kind = mat.SVDFull
	}
	if !decomposition.Factorize(mat.NewDense(rows, cols, values), kind) {
		return nil, fmt.Errorf("SVD did not converge")
	}
	singular := decomposition.Values(nil)
	s := make([]float32, len(singular))
	for i, value := range singular {
		s[i] = float32(value)
	}
	info = &SVDInfo{}
	allocated := info
	defer func() {
		if err != nil {
			err = errors.Join(err, allocated.Destroy())
			info = nil
		}
	}()
	info.s, err = solverAllocateValues(matrix.GetDeviceID(), s)
	if err != nil || !computeUV {
		return info, err
	}
	var u, v mat.Dense
	decomposition.UTo(&u)
	decomposition.VTo(&v)
	uValues := make([]float32, uCount)
	vtValues := make([]float32, vCount)
	for col := range rows {
		for row := range rows {
			uValues[col*rows+row] = float32(u.At(row, col))
		}
	}
	for col := range cols {
		for row := range cols {
			vtValues[col*cols+row] = float32(v.At(col, row))
		}
	}
	info.u, err = solverAllocateValues(matrix.GetDeviceID(), uValues)
	if err != nil {
		return nil, err
	}
	info.vt, err = solverAllocateValues(matrix.GetDeviceID(), vtValues)
	if err != nil {
		return nil, err
	}
	return info, nil
}

func simulatedEigenvalues(matrix *memory.Memory, n int, vectors bool) (eigenvalues, eigenvectors *memory.Memory, err error) {
	values, err := solverReadFiniteMatrix(matrix, n, n)
	if err != nil {
		return nil, nil, err
	}
	symmetric := mat.NewSymDense(n, nil)
	for row := range n {
		for col := row; col < n; col++ {
			symmetric.SetSym(row, col, values[row*n+col])
		}
	}
	var decomposition mat.EigenSym
	if !decomposition.Factorize(symmetric, vectors) {
		return nil, nil, fmt.Errorf("symmetric eigenvalue decomposition did not converge")
	}
	rawValues := decomposition.Values(nil)
	packed := make([]Complex64, n)
	for i, value := range rawValues {
		packed[i] = Complex64{Real: float32(value)}
	}
	eigenvalues, err = solverAllocateValues(matrix.GetDeviceID(), packed)
	if err != nil || !vectors {
		return eigenvalues, nil, err
	}
	allocatedValues := eigenvalues
	defer func() {
		if err != nil {
			err = errors.Join(err, allocatedValues.Free())
			eigenvalues = nil
		}
	}()
	var q mat.Dense
	decomposition.VectorsTo(&q)
	result := make([]float32, n*n)
	for col := range n {
		for row := range n {
			result[col*n+row] = float32(q.At(row, col))
		}
	}
	eigenvectors, err = solverAllocateValues(matrix.GetDeviceID(), result)
	return eigenvalues, eigenvectors, err
}

func simulatedPseudoInverse(matrix *memory.Memory, rows, cols int) (*memory.Memory, error) {
	values, err := solverReadFiniteMatrix(matrix, rows, cols)
	if err != nil {
		return nil, err
	}
	var decomposition mat.SVD
	if !decomposition.Factorize(mat.NewDense(rows, cols, values), mat.SVDThin) {
		return nil, fmt.Errorf("pseudoinverse SVD did not converge")
	}
	singular := decomposition.Values(nil)
	var u, v mat.Dense
	decomposition.UTo(&u)
	decomposition.VTo(&v)
	tolerance := float64(math.Nextafter32(1, 2)-1) * float64(max(rows, cols)) * singular[0]
	result := make([]float32, rows*cols)
	for col := range rows {
		for row := range cols {
			var value float64
			for k, sigma := range singular {
				if sigma > tolerance {
					value += v.At(row, k) * u.At(col, k) / sigma
				}
			}
			result[col*cols+row] = float32(value)
		}
	}
	return solverAllocateValues(matrix.GetDeviceID(), result)
}
