package libraries

import (
	"math"
	"testing"

	"github.com/stitch1968/gocuda/memory"
)

func solverTestBuffer(t *testing.T, values []float32) *memory.Memory {
	t.Helper()
	buffer, err := memory.Alloc(int64(len(values)) * 4)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := buffer.Free(); err != nil {
			t.Error(err)
		}
	})
	if err := writeMathFloat32Memory(buffer, values); err != nil {
		t.Fatal(err)
	}
	return buffer
}

func testSolverLUAndSolve(t *testing.T, ctx *SolverContext) {
	t.Helper()
	// Column-major matrix [[0,2,1],[1,-2,-3],[2,3,1]].
	original := []float32{0, 1, 2, 2, -2, 3, 1, -3, 1}
	matrix := solverTestBuffer(t, original)
	info, err := ctx.LUFactorization(matrix, 3, 3)
	if err != nil {
		t.Fatal(err)
	}
	defer info.Destroy()
	values, err := readMathFloat32Memory(matrix, 9)
	if err != nil {
		t.Fatal(err)
	}
	pivots, err := readMathTypedMemory[int32](info.ipiv, 3)
	if err != nil {
		t.Fatal(err)
	}
	permuted := append([]float32(nil), original...)
	for row, pivot := range pivots {
		target := int(pivot) - 1
		if target < row || target >= 3 {
			t.Fatalf("invalid pivot %d at row %d", pivot, row)
		}
		for col := range 3 {
			permuted[col*3+row], permuted[col*3+target] = permuted[col*3+target], permuted[col*3+row]
		}
	}
	for row := range 3 {
		for col := range 3 {
			var product float32
			for k := range 3 {
				var l, u float32
				switch {
				case row == k:
					l = 1
				case row > k:
					l = values[k*3+row]
				}
				if k <= col {
					u = values[col*3+k]
				}
				product += l * u
			}
			if want := permuted[col*3+row]; math.Abs(float64(product-want)) > 1e-4 {
				t.Fatalf("LU reconstruction[%d,%d]=%v, want %v", row, col, product, want)
			}
		}
	}
	matrix = solverTestBuffer(t, original)
	rhs, err := memory.AllocDeviceOnly(12)
	if err != nil {
		t.Fatal(err)
	}
	defer rhs.Free()
	if err := writeMathFloat32Memory(rhs, []float32{3, 0, 7}); err != nil {
		t.Fatal(err)
	}
	solution, err := ctx.SolveLinearSystem(matrix, rhs, 3)
	if err != nil {
		t.Fatal(err)
	}
	defer solution.Free()
	result, err := readMathFloat32Memory(solution, 3)
	if err != nil {
		t.Fatal(err)
	}
	for i, want := range []float32{1, 2, -1} {
		if math.Abs(float64(result[i]-want)) > 1e-4 {
			t.Fatalf("solution[%d]=%v, want %v", i, result[i], want)
		}
	}
}

func testSolverCholesky(t *testing.T, ctx *SolverContext) {
	t.Helper()
	original := []float32{4, 2, 2, 2, 5, 1, 2, 1, 6}
	matrix := solverTestBuffer(t, original)
	if err := ctx.CholeskyFactorization(matrix, 3); err != nil {
		t.Fatal(err)
	}
	values, err := readMathFloat32Memory(matrix, 9)
	if err != nil {
		t.Fatal(err)
	}
	for row := range 3 {
		for col := range 3 {
			var product float32
			for k := 0; k <= min(row, col); k++ {
				product += values[row*3+k] * values[col*3+k]
			}
			if want := original[col*3+row]; math.Abs(float64(product-want)) > 1e-4 {
				t.Fatalf("Cholesky reconstruction[%d,%d]=%v, want %v", row, col, product, want)
			}
		}
	}
	if err := ctx.CholeskyFactorization(solverTestBuffer(t, []float32{1, 2, 2, 1}), 2); err == nil {
		t.Fatal("expected non-positive-definite matrix to fail")
	}
}

func TestSimulatedSolverNumericalResults(t *testing.T) {
	ctx := &SolverContext{}
	testSolverLUAndSolve(t, ctx)
	testSolverCholesky(t, ctx)
	singular := solverTestBuffer(t, []float32{1, 2, 2, 4})
	if _, err := ctx.LUFactorization(singular, 2, 2); err == nil {
		t.Fatal("expected singular LU factorization to fail")
	}
}

func TestSimulatedSolverRejectsInvalidInputs(t *testing.T) {
	ctx := &SolverContext{}
	matrix := solverTestBuffer(t, []float32{1, 0, 0, 1})
	if _, err := ctx.LUFactorization(matrix, 3, 3); err == nil {
		t.Fatal("expected undersized matrix to fail")
	}
	if _, err := ctx.SVDDecomposition(matrix, -1, 2, true); err == nil {
		t.Fatal("expected invalid SVD dimensions to fail")
	}
	invalid := solverTestBuffer(t, []float32{float32(math.NaN())})
	if _, err := ctx.QRFactorization(invalid, 1, 1); err == nil {
		t.Fatal("expected non-finite QR input to fail")
	}
	if _, err := ctx.SVDDecomposition(invalid, 1, 1, true); err == nil {
		t.Fatal("expected non-finite SVD input to fail")
	}
	if _, _, err := ctx.Eigenvalues(invalid, 1, true); err == nil {
		t.Fatal("expected non-finite eigenvalue input to fail")
	}
	if _, err := ctx.PseudoInverse(invalid, 1, 1); err == nil {
		t.Fatal("expected non-finite pseudoinverse input to fail")
	}
}
