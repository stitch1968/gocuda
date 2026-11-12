package libraries

import (
	"math"
	"testing"
)

func testSolverRectangularLU(t *testing.T, ctx *SolverContext) {
	t.Helper()
	const rows, cols = 3, 2
	original := []float32{0, 1, 2, 2, -2, 3}
	matrix := solverTestBuffer(t, original)
	info, err := ctx.LUFactorization(matrix, rows, cols)
	if err != nil {
		t.Fatal(err)
	}
	defer info.Destroy()
	values, err := readMathFloat32Memory(matrix, rows*cols)
	if err != nil {
		t.Fatal(err)
	}
	pivots, err := readMathTypedMemory[int32](info.ipiv, cols)
	if err != nil {
		t.Fatal(err)
	}
	permuted := append([]float32(nil), original...)
	for row, pivot := range pivots {
		target := int(pivot) - 1
		if target < row || target >= rows {
			t.Fatalf("invalid pivot %d at row %d", pivot, row)
		}
		for col := range cols {
			permuted[col*rows+row], permuted[col*rows+target] = permuted[col*rows+target], permuted[col*rows+row]
		}
	}
	for row := range rows {
		for col := range cols {
			var product float32
			for k := range cols {
				var l, u float32
				switch {
				case row == k:
					l = 1
				case row > k:
					l = values[k*rows+row]
				}
				if k <= col {
					u = values[col*rows+k]
				}
				product += l * u
			}
			if want := permuted[col*rows+row]; math.Abs(float64(product-want)) > 1e-4 {
				t.Fatalf("rectangular LU[%d,%d]=%v, want %v", row, col, product, want)
			}
		}
	}
}

func TestSimulatedSolverRectangularLU(t *testing.T) {
	testSolverRectangularLU(t, &SolverContext{})
}
