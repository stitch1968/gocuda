//go:build cuda

package libraries

import (
	"math"
	"testing"

	cuda "github.com/stitch1968/gocuda"
	"github.com/stitch1968/gocuda/memory"
)

func TestNativeSolverPseudoInverseRectangularAndScaled(t *testing.T) {
	if err := cuda.Initialize(); err != nil {
		t.Fatal(err)
	}
	if !cuda.ShouldUseCuda() {
		t.Skip("CUDA hardware unavailable")
	}
	ctx, err := CreateNativeSolverContextOnDevice(0)
	if err != nil {
		t.Fatal(err)
	}
	defer ctx.DestroyContext()
	for _, scale := range []float32{1, 1e-8} {
		values := []float32{scale, 3 * scale, 5 * scale, 2 * scale, 4 * scale, 7 * scale}
		matrix, err := memory.AllocDeviceOnlyOnDevice(24, 0)
		if err != nil {
			t.Fatal(err)
		}
		if err := writeMathFloat32Memory(matrix, values); err != nil {
			t.Fatal(err)
		}
		inverse, err := ctx.PseudoInverse(matrix, 3, 2)
		if err != nil {
			t.Fatal(err)
		}
		result, err := readMathFloat32Memory(inverse, 6)
		if err != nil {
			t.Fatal(err)
		}
		for row := range 2 {
			for col := range 2 {
				var product float64
				for k := range 3 {
					product += float64(result[k*2+row]) * float64(values[col*3+k])
				}
				want := 0.0
				if row == col {
					want = 1
				}
				if math.Abs(product-want) > 2e-4 {
					t.Fatalf("scale %g: A+*A[%d,%d]=%g, want %g", scale, row, col, product, want)
				}
			}
		}
		if err := inverse.Free(); err != nil {
			t.Fatal(err)
		}
		if err := matrix.Free(); err != nil {
			t.Fatal(err)
		}
	}
}
