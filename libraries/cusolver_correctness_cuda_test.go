//go:build cuda

package libraries

import (
	"testing"

	cuda "github.com/stitch1968/gocuda"
)

func TestNativeSolverNumericalResults(t *testing.T) {
	if err := cuda.Initialize(); err != nil {
		t.Fatal(err)
	}
	if !cuda.ShouldUseCuda() {
		t.Skip("CUDA hardware unavailable")
	}
	device, err := cuda.NewContext(0)
	if err != nil {
		t.Fatal(err)
	}
	if err := device.Run(func() error {
		ctx, err := CreateSolverContext()
		if err != nil {
			return err
		}
		defer ctx.DestroyContext()
		testSolverLUAndSolve(t, ctx)
		testSolverCholesky(t, ctx)
		testSolverRectangularLU(t, ctx)
		testSolverQR(t, ctx, 3, 2, []float32{1, 3, 5, 2, 4, 7})
		testSolverQR(t, ctx, 2, 3, []float32{1, 3, 2, 4, 5, 7})
		testSolverSVD(t, ctx, 3, 2, []float32{1, 3, 5, 2, 4, 7})
		testSolverSVD(t, ctx, 2, 3, []float32{1, 3, 2, 4, 5, 7})
		testSolverEigen(t, ctx)
		return nil
	}); err != nil {
		t.Fatal(err)
	}
}
