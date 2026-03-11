package libraries

import (
	"math"
	"testing"

	"github.com/stitch1968/gocuda/memory"
)

func TestAmgXSolveAndUpdateMatrix(t *testing.T) {
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
	writeInt32Memory(rowPtr, []int32{0, 2, 4}, 3)
	writeInt32Memory(colInd, []int32{0, 1, 0, 1}, 4)
	writeMathFloat64Memory(values, []float64{4, 1, 1, 3})
	writeMathFloat64Memory(b, []float64{6, 7})
	writeMathFloat64Memory(x, []float64{0, 0})

	handle, err := CreateAmgXHandle(AmgXConfig{Solver: AmgXSolverAMG, Precision: AmgXPrecisionDouble, MaxIterations: 50, Tolerance: 1e-12, RelativeTolerance: 1e-12, MaxLevels: 8})
	if err != nil {
		t.Fatalf("CreateAmgXHandle failed: %v", err)
	}
	defer handle.Destroy()
	matrix, err := CreateAmgXMatrix(2, 4, rowPtr, colInd, values, AmgXModeDevice)
	if err != nil {
		t.Fatalf("CreateAmgXMatrix failed: %v", err)
	}
	defer matrix.Destroy()
	bVec, _ := CreateAmgXVector(2, b, AmgXModeDevice)
	xVec, _ := CreateAmgXVector(2, x, AmgXModeDevice)
	defer bVec.Destroy()
	defer xVec.Destroy()

	if err := handle.Setup(matrix); err != nil {
		t.Fatalf("Setup failed: %v", err)
	}
	info, err := handle.Solve(bVec, xVec)
	if err != nil {
		t.Fatalf("Solve failed: %v", err)
	}
	if info.AbsoluteResidual > 1e-10 {
		t.Fatalf("unexpected residual: %v", info.AbsoluteResidual)
	}
	valuesOut, _ := readMathFloat64Memory(x, 2)
	assertFloat64Slice(t, valuesOut, []float64{1, 2}, 1e-9)

	gridComplexity, err := handle.GetGridComplexity()
	if err != nil || gridComplexity < 1 {
		t.Fatalf("unexpected grid complexity result: %v %v", gridComplexity, err)
	}

	x2, _ := memory.Alloc(2 * 8)
	defer x2.Free()
	writeMathFloat64Memory(x2, []float64{0, 0})
	x2Vec, _ := CreateAmgXVector(2, x2, AmgXModeDevice)
	defer x2Vec.Destroy()
	infos, err := handle.SolveMultiple([]*AmgXVector{bVec}, []*AmgXVector{x2Vec})
	if err != nil {
		t.Fatalf("SolveMultiple failed: %v", err)
	}
	if len(infos) != 1 || infos[0] == nil {
		t.Fatalf("unexpected solve-multiple info: %v", infos)
	}

	updatedValues, _ := memory.Alloc(4 * 8)
	defer updatedValues.Free()
	writeMathFloat64Memory(updatedValues, []float64{5, 1, 1, 3})
	updatedMatrix, err := CreateAmgXMatrix(2, 4, rowPtr, colInd, updatedValues, AmgXModeDevice)
	if err != nil {
		t.Fatalf("Create updated matrix failed: %v", err)
	}
	defer updatedMatrix.Destroy()
	if err := handle.UpdateMatrix(updatedMatrix, true); err != nil {
		t.Fatalf("UpdateMatrix failed: %v", err)
	}
	writeMathFloat64Memory(x, []float64{0, 0})
	updatedInfo, err := handle.Solve(bVec, xVec)
	if err != nil {
		t.Fatalf("Solve after update failed: %v", err)
	}
	if updatedInfo.AbsoluteResidual > 1e-10 {
		t.Fatalf("unexpected updated residual: %v", updatedInfo.AbsoluteResidual)
	}
	updatedSolution, _ := readMathFloat64Memory(x, 2)
	assertFloat64Slice(t, updatedSolution, []float64{11.0 / 14.0, 29.0 / 14.0}, 1e-9)
}

func assertFloat64Slice(t *testing.T, got, want []float64, tolerance float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("unexpected length: got %d want %d", len(got), len(want))
	}
	for index := range got {
		if math.Abs(got[index]-want[index]) > tolerance {
			t.Fatalf("unexpected value[%d]: got %v want %v", index, got[index], want[index])
		}
	}
}
