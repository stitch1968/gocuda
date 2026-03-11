//go:build cuda

package libraries

import (
	"math"
	"testing"

	cuda "github.com/stitch1968/gocuda"
	"github.com/stitch1968/gocuda/memory"
)

func TestAmgXNativeBackendSelection(t *testing.T) {
	if err := cuda.Initialize(); err != nil {
		t.Fatalf("Initialize failed: %v", err)
	}
	if !cuda.ShouldUseCuda() {
		t.Skip("CUDA runtime not available")
	}
	if !amgxNativeAvailable() {
		t.Skip("AmgX native backend not available")
	}
	handle, err := CreateAmgXHandle(AmgXConfig{Solver: AmgXSolverAMG, Precision: AmgXPrecisionDouble, MaxIterations: 50, Tolerance: 1e-12, RelativeTolerance: 1e-12, MaxLevels: 8, Cycle: AmgXCycleV, Interpolation: AmgXInterpolationClassical, Smoother: AmgXSmootherJacobi})
	if err != nil {
		t.Fatalf("CreateAmgXHandle failed: %v", err)
	}
	defer handle.Destroy()
	if !handle.native {
		t.Fatal("expected native AmgX handle in CUDA build")
	}
}

func TestAmgXNativeSolve(t *testing.T) {
	if err := cuda.Initialize(); err != nil {
		t.Fatalf("Initialize failed: %v", err)
	}
	if !cuda.ShouldUseCuda() {
		t.Skip("CUDA runtime not available")
	}
	if !amgxNativeAvailable() {
		t.Skip("AmgX native backend not available")
	}

	rowPtr, _ := memory.Alloc(3 * 4)
	colInd, _ := memory.Alloc(4 * 4)
	values, _ := memory.Alloc(4 * 8)
	b, _ := memory.Alloc(2 * 8)
	x, _ := memory.Alloc(2 * 8)
	defer rowPtr.Free()
	defer colInd.Free()
	defer values.Free()
	defer b.Free()
	defer x.Free()
	_ = writeInt32Memory(rowPtr, []int32{0, 2, 4}, 3)
	_ = writeInt32Memory(colInd, []int32{0, 1, 0, 1}, 4)
	_ = writeMathFloat64Memory(values, []float64{4, 1, 1, 3})
	_ = writeMathFloat64Memory(b, []float64{6, 7})
	_ = writeMathFloat64Memory(x, []float64{0, 0})

	handle, err := CreateAmgXHandle(AmgXConfig{Solver: AmgXSolverAMG, Precision: AmgXPrecisionDouble, MaxIterations: 50, Tolerance: 1e-12, RelativeTolerance: 1e-12, MaxLevels: 8, Cycle: AmgXCycleV, Interpolation: AmgXInterpolationClassical, Smoother: AmgXSmootherJacobi})
	if err != nil {
		t.Fatalf("CreateAmgXHandle failed: %v", err)
	}
	defer handle.Destroy()
	matrix, err := CreateAmgXMatrix(2, 4, rowPtr, colInd, values, AmgXModeDevice)
	if err != nil {
		t.Fatalf("CreateAmgXMatrix failed: %v", err)
	}
	defer matrix.Destroy()
	bVec, err := CreateAmgXVector(2, b, AmgXModeDevice)
	if err != nil {
		t.Fatalf("CreateAmgXVector(rhs) failed: %v", err)
	}
	defer bVec.Destroy()
	xVec, err := CreateAmgXVector(2, x, AmgXModeDevice)
	if err != nil {
		t.Fatalf("CreateAmgXVector(solution) failed: %v", err)
	}
	defer xVec.Destroy()

	if err := handle.Setup(matrix); err != nil {
		t.Fatalf("Setup failed: %v", err)
	}
	info, err := handle.Solve(bVec, xVec)
	if err != nil {
		t.Fatalf("Solve failed: %v", err)
	}
	if info.AbsoluteResidual > 1e-6 {
		t.Fatalf("unexpected residual: %v", info.AbsoluteResidual)
	}
	valuesOut, err := readMathFloat64Memory(x, 2)
	if err != nil {
		t.Fatalf("read solution failed: %v", err)
	}
	if math.Abs(valuesOut[0]-1) > 1e-5 || math.Abs(valuesOut[1]-2) > 1e-5 {
		t.Fatalf("unexpected solution: %v", valuesOut)
	}
}
