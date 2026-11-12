package libraries

import (
	"math"
	"testing"
)

func testSolverQR(t *testing.T, ctx *SolverContext, rows, cols int, original []float32) {
	t.Helper()
	matrix := solverTestBuffer(t, original)
	info, err := ctx.QRFactorization(matrix, rows, cols)
	if err != nil {
		t.Fatal(err)
	}
	defer info.Destroy()
	if info.Tau() == nil {
		t.Fatal("expected QR scalar allocation")
	}
	compact, err := readMathFloat32Memory(matrix, rows*cols)
	if err != nil {
		t.Fatal(err)
	}
	tau, err := readMathFloat32Memory(info.Tau(), min(rows, cols))
	if err != nil {
		t.Fatal(err)
	}
	reconstructed := make([]float32, rows*cols)
	for col := range cols {
		for row := 0; row <= col && row < rows; row++ {
			reconstructed[col*rows+row] = compact[col*rows+row]
		}
	}
	for k := min(rows, cols) - 1; k >= 0; k-- {
		for col := range cols {
			dot := reconstructed[col*rows+k]
			for row := k + 1; row < rows; row++ {
				dot += compact[k*rows+row] * reconstructed[col*rows+row]
			}
			dot *= tau[k]
			reconstructed[col*rows+k] -= dot
			for row := k + 1; row < rows; row++ {
				reconstructed[col*rows+row] -= compact[k*rows+row] * dot
			}
		}
	}
	for i := range original {
		if math.Abs(float64(reconstructed[i]-original[i])) > 2e-4 {
			t.Fatalf("QR reconstruction[%d]=%v, want %v", i, reconstructed[i], original[i])
		}
	}
}

func testSolverSVD(t *testing.T, ctx *SolverContext, rows, cols int, original []float32) {
	t.Helper()
	matrix := solverTestBuffer(t, original)
	info, err := ctx.SVDDecomposition(matrix, rows, cols, true)
	if err != nil {
		t.Fatal(err)
	}
	defer info.Destroy()
	if info.SingularValues() == nil || info.U() == nil || info.VT() == nil {
		t.Fatal("full SVD must expose singular values and vectors")
	}
	singular, err := readMathFloat32Memory(info.SingularValues(), min(rows, cols))
	if err != nil {
		t.Fatal(err)
	}
	u, err := readMathFloat32Memory(info.U(), rows*rows)
	if err != nil {
		t.Fatal(err)
	}
	vt, err := readMathFloat32Memory(info.VT(), cols*cols)
	if err != nil {
		t.Fatal(err)
	}
	for i, value := range singular {
		if value < 0 || i > 0 && value > singular[i-1] {
			t.Fatalf("singular values not non-negative and descending: %v", singular)
		}
	}
	for row := range rows {
		for col := range cols {
			var value float32
			for k, sigma := range singular {
				value += u[k*rows+row] * sigma * vt[col*cols+k]
			}
			if want := original[col*rows+row]; math.Abs(float64(value-want)) > 2e-4 {
				t.Fatalf("SVD reconstruction[%d,%d]=%v, want %v", row, col, value, want)
			}
		}
	}
	withoutVectors, err := ctx.SVDDecomposition(solverTestBuffer(t, original), rows, cols, false)
	if err != nil {
		t.Fatal(err)
	}
	defer withoutVectors.Destroy()
	if withoutVectors.U() != nil || withoutVectors.VT() != nil {
		t.Fatal("SVD without vectors must not allocate them")
	}
	onlyValues, err := readMathFloat32Memory(withoutVectors.SingularValues(), len(singular))
	if err != nil {
		t.Fatal(err)
	}
	for i := range singular {
		if math.Abs(float64(onlyValues[i]-singular[i])) > 2e-4 {
			t.Fatalf("SVD singular value[%d]=%v, want %v", i, onlyValues[i], singular[i])
		}
	}
}

func testSolverEigen(t *testing.T, ctx *SolverContext) {
	t.Helper()
	// cuSOLVER's symmetric eigensolver uses the upper triangle.
	upper := []float32{4, 99, 99, 2, 5, 99, 2, 1, 6}
	symmetric := []float32{4, 2, 2, 2, 5, 1, 2, 1, 6}
	matrix := solverTestBuffer(t, upper)
	valuesMemory, vectorsMemory, err := ctx.Eigenvalues(matrix, 3, true)
	if err != nil {
		t.Fatal(err)
	}
	defer valuesMemory.Free()
	defer vectorsMemory.Free()
	values, err := readMathTypedMemory[Complex64](valuesMemory, 3)
	if err != nil {
		t.Fatal(err)
	}
	vectors, err := readMathFloat32Memory(vectorsMemory, 9)
	if err != nil {
		t.Fatal(err)
	}
	for col := range 3 {
		if values[col].Imag != 0 || col > 0 && values[col].Real < values[col-1].Real {
			t.Fatalf("unexpected eigenvalues: %v", values)
		}
		for row := range 3 {
			var product float32
			for k := range 3 {
				product += symmetric[k*3+row] * vectors[col*3+k]
			}
			want := values[col].Real * vectors[col*3+row]
			if math.Abs(float64(product-want)) > 2e-4 {
				t.Fatalf("eigenvector residual[%d,%d] = %v, want %v", row, col, product, want)
			}
		}
	}
	onlyValues, noVectors, err := ctx.Eigenvalues(solverTestBuffer(t, upper), 3, false)
	if err != nil {
		t.Fatal(err)
	}
	defer onlyValues.Free()
	if noVectors != nil {
		t.Fatal("eigenvalues without vectors must return nil vectors")
	}
	other, err := readMathTypedMemory[Complex64](onlyValues, 3)
	if err != nil {
		t.Fatal(err)
	}
	for i := range values {
		if math.Abs(float64(values[i].Real-other[i].Real)) > 2e-4 {
			t.Fatalf("eigenvalue[%d]=%v, want %v", i, other[i], values[i])
		}
	}
}

func TestSimulatedSolverDecompositions(t *testing.T) {
	ctx := &SolverContext{}
	testSolverQR(t, ctx, 3, 2, []float32{1, 3, 5, 2, 4, 7})
	testSolverQR(t, ctx, 2, 3, []float32{1, 3, 2, 4, 5, 7})
	testSolverQR(t, ctx, 2, 2, []float32{0, 0, 0, 0})
	testSolverSVD(t, ctx, 3, 2, []float32{1, 3, 5, 2, 4, 7})
	testSolverSVD(t, ctx, 2, 3, []float32{1, 3, 2, 4, 5, 7})
	testSolverEigen(t, ctx)
}
