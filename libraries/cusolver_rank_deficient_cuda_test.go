//go:build cuda

package libraries

import (
	"math"
	"testing"

	cuda "github.com/stitch1968/gocuda"
	"github.com/stitch1968/gocuda/memory"
)

func TestNativeSolverPseudoInverseRankDeficient(t *testing.T) {
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
	values := []float32{1, 2, 3, 2, 4, 6}
	matrix, err := memory.AllocDeviceOnlyOnDevice(24, 0)
	if err != nil {
		t.Fatal(err)
	}
	defer matrix.Free()
	if err := writeMathFloat32Memory(matrix, values); err != nil {
		t.Fatal(err)
	}
	inverse, err := ctx.PseudoInverse(matrix, 3, 2)
	if err != nil {
		t.Fatal(err)
	}
	defer inverse.Free()
	result, err := readMathFloat32Memory(inverse, 6)
	if err != nil {
		t.Fatal(err)
	}
	for row := range 3 {
		for col := range 2 {
			var product float64
			for i := range 2 {
				for j := range 3 {
					product += float64(values[i*3+row]) * float64(result[j*2+i]) * float64(values[col*3+j])
				}
			}
			if want := float64(values[col*3+row]); math.Abs(product-want) > 2e-4 {
				t.Fatalf("A*A+*A[%d,%d]=%g, want %g", row, col, product, want)
			}
		}
	}
}
