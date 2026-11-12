package libraries

import (
	"math"
	"testing"

	"github.com/stitch1968/gocuda/memory"
)

func TestSimulatedSolverPseudoInverse(t *testing.T) {
	tests := []struct {
		name       string
		rows, cols int
		values     []float32
	}{
		{"tall", 3, 2, []float32{1, 3, 5, 2, 4, 7}},
		{"wide", 2, 3, []float32{1, 3, 2, 4, 5, 7}},
		{"rank-deficient", 3, 2, []float32{1, 2, 3, 2, 4, 6}},
		{"small-scale", 3, 2, []float32{1e-8, 3e-8, 5e-8, 2e-8, 4e-8, 7e-8}},
		{"zero", 2, 2, []float32{0, 0, 0, 0}},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			matrix, err := memory.AllocDeviceOnly(int64(len(tc.values) * 4))
			if err != nil {
				t.Fatal(err)
			}
			defer matrix.Free()
			if err := writeMathFloat32Memory(matrix, tc.values); err != nil {
				t.Fatal(err)
			}
			ctx := &SolverContext{}
			inverse, err := ctx.PseudoInverse(matrix, tc.rows, tc.cols)
			if err != nil {
				t.Fatal(err)
			}
			defer inverse.Free()
			result, err := readMathFloat32Memory(inverse, len(tc.values))
			if err != nil {
				t.Fatal(err)
			}
			var largest float64
			for _, value := range tc.values {
				largest = math.Max(largest, math.Abs(float64(value)))
			}
			tolerance := largest * 2e-4
			for row := range tc.rows {
				for col := range tc.cols {
					var product float64
					for i := range tc.cols {
						for j := range tc.rows {
							product += float64(tc.values[i*tc.rows+row]) * float64(result[j*tc.cols+i]) * float64(tc.values[col*tc.rows+j])
						}
					}
					want := float64(tc.values[col*tc.rows+row])
					if math.Abs(product-want) > tolerance || math.IsNaN(product) {
						t.Fatalf("A*A+*A[%d,%d]=%g, want %g (tolerance %g)", row, col, product, want, tolerance)
					}
				}
			}
		})
	}
}
