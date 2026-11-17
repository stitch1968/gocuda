package libraries

import (
	"errors"
	"math"
	"testing"

	"github.com/stitch1968/gocuda/memory"
)

func TestCuDSSSolutionDiagnostics(t *testing.T) {
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
	rows, columns, values := allocate(12), allocate(16), allocate(16)
	rhs, output := allocate(16), allocate(16)
	for _, entry := range []struct {
		buffer *memory.Memory
		data   []int32
	}{{rows, []int32{0, 2, 4}}, {columns, []int32{0, 1, 0, 1}}} {
		if err := writeInt32Memory(entry.buffer, entry.data, len(entry.data)); err != nil {
			t.Fatal(err)
		}
	}
	if err := writeMathFloat32Memory(values, []float32{3, 1, 1, 2}); err != nil {
		t.Fatal(err)
	}
	if err := writeMathFloat32Memory(rhs, []float32{3, 1, 1, 1}); err != nil {
		t.Fatal(err)
	}
	if err := writeMathFloat32Memory(output, []float32{0, 0, 0, 0}); err != nil {
		t.Fatal(err)
	}
	matrix, err := CreateDSSMatrix(2, 4, rows, columns, values, DSSMatrixFormatCSR, false)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := matrix.Destroy(); err != nil {
			t.Error(err)
		}
	})
	handle, err := CreateDSSHandle(DSSConfig{MatrixFormat: DSSMatrixFormatCSR, Factorization: DSSFactorizationLU, Refinement: DSSRefinementSingle})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := handle.Destroy(); err != nil {
			t.Error(err)
		}
	})
	if err := handle.Analyze(matrix); err != nil {
		t.Fatal(err)
	}
	if err := handle.Factor(matrix); err != nil {
		t.Fatal(err)
	}
	infos, err := handle.SolveMultiple(rhs, output, 2)
	if err != nil {
		t.Fatal(err)
	}
	got, err := readMathFloat32Memory(output, 4)
	if err != nil {
		t.Fatal(err)
	}
	if len(infos) != 2 {
		t.Fatalf("got %d diagnostic records, want 2", len(infos))
	}
	if infos[0].Residual != 0 || infos[0].BackwardError != 0 || infos[0].ComponentwiseError != 0 {
		t.Fatalf("exact solve diagnostics = %+v", infos[0])
	}
	x0, x1 := float64(got[2]), float64(got[3])
	r0, r1 := math.Abs(3*x0+x1-1), math.Abs(x0+2*x1-1)
	residual := math.Max(r0, r1)
	normwise := residual / (4*math.Max(math.Abs(x0), math.Abs(x1)) + 1)
	componentwise := math.Max(r0/(3*math.Abs(x0)+math.Abs(x1)+1), r1/(math.Abs(x0)+2*math.Abs(x1)+1))
	if residual == 0 || math.Abs(infos[1].Residual-residual) > 1e-14 ||
		math.Abs(infos[1].BackwardError-normwise) > 1e-14 ||
		math.Abs(infos[1].ComponentwiseError-componentwise) > 1e-14 {
		t.Fatalf("rounded solve diagnostics = %+v, want residual %g, backward %g, componentwise %g",
			infos[1], residual, normwise, componentwise)
	}
	if infos[1].Iterations != -1 || infos[1].InertiaAvailable ||
		!math.IsNaN(infos[1].Error) || !math.IsNaN(infos[1].PivotGrowth) || !math.IsNaN(infos[1].ConditionNumber) {
		t.Fatalf("unmeasured diagnostics must be marked unavailable: %+v", infos[1])
	}
	summary, err := handle.Solve(rhs, output, 2)
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(summary.Residual-residual) > 1e-14 || math.Abs(summary.BackwardError-normwise) > 1e-14 ||
		math.Abs(summary.ComponentwiseError-componentwise) > 1e-14 {
		t.Fatalf("aggregate diagnostics = %+v", summary)
	}
}

func TestCuDSSUnavailableInertiaAndNonfiniteDiagnostics(t *testing.T) {
	for _, factorization := range []DSSFactorization{DSSFactorizationLDLT, DSSFactorizationCholesky} {
		handle := &DSSHandle{factored: true, config: DSSConfig{Factorization: factorization}}
		if _, err := handle.GetInertia(); !errors.Is(err, ErrDSSInertiaUnavailable) {
			t.Fatalf("inertia error = %v, want ErrDSSInertiaUnavailable", err)
		}
	}
	if _, err := dssSolutionDiagnostics([]float64{math.NaN()}, 1, []float64{1}, []float64{1}, 0); err == nil {
		t.Fatal("expected non-finite matrix to fail")
	}
	if _, err := dssSolutionDiagnostics([]float64{1}, 1, []float64{math.Inf(1)}, []float64{1}, 0); err == nil {
		t.Fatal("expected non-finite solution to fail")
	}
	info, err := dssSolutionDiagnostics([]float64{0}, 1, []float64{0}, []float64{0}, 0)
	if err != nil || info.Residual != 0 || info.BackwardError != 0 || info.ComponentwiseError != 0 {
		t.Fatalf("zero-system diagnostics = %+v, error = %v", info, err)
	}
}
