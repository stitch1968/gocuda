package libraries

import (
	"math"
	"testing"

	"github.com/stitch1968/gocuda/memory"
)

func TestCuDSSFactorizationStateAndValues(t *testing.T) {
	allocate := func(size int64) *memory.Memory {
		t.Helper()
		buffer, err := memory.Alloc(size)
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() {
			if err := buffer.Free(); err != nil {
				t.Error(err)
			}
		})
		return buffer
	}
	rows, columns, values := allocate(12), allocate(8), allocate(8)
	rhs, output := allocate(8), allocate(8)
	if err := writeMathFloat32Memory(output, []float32{0, 0}); err != nil {
		t.Fatal(err)
	}
	if err := writeInt32Memory(rows, []int32{0, 1, 2}, 3); err != nil {
		t.Fatal(err)
	}
	if err := writeInt32Memory(columns, []int32{0, 1}, 2); err != nil {
		t.Fatal(err)
	}
	if err := writeMathFloat32Memory(values, []float32{2, 4}); err != nil {
		t.Fatal(err)
	}
	matrix, err := CreateDSSMatrix(2, 2, rows, columns, values, DSSMatrixFormatCSR, false)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := matrix.Destroy(); err != nil {
			t.Error(err)
		}
	})
	handle, err := CreateDSSHandle(DSSConfig{MatrixFormat: DSSMatrixFormatCSR, Factorization: DSSFactorizationLU})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := handle.Destroy(); err != nil {
			t.Error(err)
		}
	})
	if err := handle.Factor(matrix); err == nil {
		t.Fatal("factorization before analysis must fail")
	}
	if _, err := handle.Solve(rhs, output, 1); err == nil {
		t.Fatal("solve before factorization must fail")
	}
	if err := handle.Analyze(matrix); err != nil {
		t.Fatal(err)
	}
	if err := writeMathFloat32Memory(values, []float32{4, 8}); err != nil {
		t.Fatal(err)
	}
	if err := handle.Factor(matrix); err != nil {
		t.Fatal(err)
	}
	checkSolution := func(b []float32) {
		t.Helper()
		if err := writeMathFloat32Memory(rhs, b); err != nil {
			t.Fatal(err)
		}
		if _, err := handle.Solve(rhs, output, 1); err != nil {
			t.Fatal(err)
		}
		got, err := readMathFloat32Memory(output, 2)
		if err != nil || math.Abs(float64(got[0]-1)) > 1e-5 || math.Abs(float64(got[1]-1)) > 1e-5 {
			t.Fatalf("solution = %v, error = %v, want [1 1]", got, err)
		}
	}
	checkSolution([]float32{4, 8})
	if err := writeInt32Memory(columns, []int32{1, 0}, 2); err != nil {
		t.Fatal(err)
	}
	if err := handle.Refactor(matrix); err == nil {
		t.Fatal("refactorization with a changed sparsity pattern must fail")
	}
	if err := handle.Factor(matrix); err == nil {
		t.Fatal("factorization with a changed sparsity pattern must fail")
	}
	if _, err := handle.Solve(rhs, output, 1); err == nil {
		t.Fatal("solve with a changed sparsity pattern must fail")
	}
	if err := writeInt32Memory(columns, []int32{0, 1}, 2); err != nil {
		t.Fatal(err)
	}
	if err := writeInt32Memory(rows, []int32{0, 2, 2}, 3); err != nil {
		t.Fatal(err)
	}
	if err := handle.Refactor(matrix); err == nil {
		t.Fatal("refactorization with changed row offsets must fail")
	}
	if err := writeInt32Memory(rows, []int32{0, 1, 2}, 3); err != nil {
		t.Fatal(err)
	}
	if err := writeMathFloat32Memory(values, []float32{8, 4}); err != nil {
		t.Fatal(err)
	}
	if err := handle.Refactor(matrix); err != nil {
		t.Fatal(err)
	}
	checkSolution([]float32{8, 4})
	if _, err := handle.SolveMultiple(rhs, output, -1); err == nil {
		t.Fatal("negative right-hand-side count must fail")
	}
	if err := handle.Analyze(matrix); err != nil {
		t.Fatal(err)
	}
	if _, err := handle.Solve(rhs, output, 1); err == nil {
		t.Fatal("reanalysis must invalidate previous factors")
	}
	if _, err := handle.GetDeterminant(); err == nil {
		t.Fatal("reanalysis must invalidate previous determinant")
	}
	if err := handle.Destroy(); err != nil {
		t.Fatal(err)
	}
	if err := handle.Analyze(matrix); err == nil {
		t.Fatal("destroyed handle must reject analysis")
	}
	if err := handle.Factor(matrix); err == nil {
		t.Fatal("destroyed handle must reject factorization")
	}
}

func TestCuDSSMatrixDoesNotOwnFloat64Values(t *testing.T) {
	allocateIndices := func(indices []int32) *memory.Memory {
		t.Helper()
		buffer, err := memory.Alloc(int64(len(indices)) * 4)
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() {
			if err := buffer.Free(); err != nil {
				t.Error(err)
			}
		})
		if err := writeInt32Memory(buffer, indices, len(indices)); err != nil {
			t.Fatal(err)
		}
		return buffer
	}
	rows := allocateIndices([]int32{0, 1, 2})
	columns := allocateIndices([]int32{0, 1})
	values, err := memory.Alloc(16)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := values.Free(); err != nil {
			t.Error(err)
		}
	})
	if err := writeMathFloat64Memory(values, []float64{2, 4}); err != nil {
		t.Fatal(err)
	}
	matrix, err := CreateDSSMatrix(2, 2, rows, columns, values, DSSMatrixFormatCSR, false)
	if err != nil {
		t.Fatal(err)
	}
	if err := matrix.Destroy(); err != nil {
		t.Fatal(err)
	}
	if _, _, err := readDSSCSRIndices(matrix); err == nil {
		t.Fatal("destroyed matrix must reject further use")
	}
	if values.Ptr() == nil {
		t.Fatal("destroying the matrix must not free caller-owned values")
	}
	if err := writeMathFloat64Memory(values, []float64{3, 5}); err != nil {
		t.Fatal(err)
	}
}
