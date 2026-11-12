package libraries

import (
	"math"
	"testing"
)

func sparseTestCSR(t *testing.T, ctx *SparseContext, rows, cols int, offsets, columns []int32, values []float32) *SparseMatrix {
	t.Helper()
	matrix, err := ctx.CreateCSRMatrix(rows, cols, offsets, columns, values)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := matrix.Destroy(); err != nil {
			t.Error(err)
		}
	})
	return matrix
}

func sparseTestDense(t *testing.T, ctx *SparseContext, matrix *SparseMatrix, want []float32) {
	t.Helper()
	dense, err := ctx.SparseToDense(matrix)
	if err != nil {
		t.Fatal(err)
	}
	defer dense.Free()
	values, err := readMathFloat32Memory(dense, len(want))
	if err != nil {
		t.Fatal(err)
	}
	for i := range want {
		if math.Abs(float64(values[i]-want[i])) > 1e-5 {
			t.Fatalf("dense[%d] = %v, want %v", i, values[i], want[i])
		}
	}
}

func testSparseNumerics(t *testing.T, ctx *SparseContext) {
	t.Helper()
	a := sparseTestCSR(t, ctx, 2, 3, []int32{0, 2, 3}, []int32{0, 2, 1}, []float32{1, 2, 3})
	b := sparseTestCSR(t, ctx, 3, 2, []int32{0, 1, 2, 3}, []int32{1, 0, 1}, []float32{4, 5, 6})
	x := solverTestBuffer(t, []float32{2, 3, 4})
	y := solverTestBuffer(t, []float32{5, 7})
	if err := ctx.SpMV(2, a, x, -1, y); err != nil {
		t.Fatal(err)
	}
	values, err := readMathFloat32Memory(y, 2)
	if err != nil {
		t.Fatal(err)
	}
	if values[0] != 15 || values[1] != 11 {
		t.Fatalf("CSR SpMV = %v, want [15 11]", values)
	}
	duplicate := sparseTestCSR(t, ctx, 2, 3, []int32{0, 3, 4}, []int32{0, 2, 2, 1}, []float32{1, 1, 1, 3})
	if err := writeMathFloat32Memory(y, []float32{5, 7}); err != nil {
		t.Fatal(err)
	}
	if err := ctx.SpMV(2, duplicate, x, -1, y); err != nil {
		t.Fatal(err)
	}
	values, err = readMathFloat32Memory(y, 2)
	if err != nil {
		t.Fatal(err)
	}
	if values[0] != 15 || values[1] != 11 {
		t.Fatalf("duplicate CSR SpMV = %v, want [15 11]", values)
	}
	coo, err := ctx.CreateCOOMatrix(2, 3, []int32{0, 0, 0, 1}, []int32{0, 2, 2, 1}, []float32{1, 1, 1, 3})
	if err != nil {
		t.Fatal(err)
	}
	defer coo.Destroy()
	if err := writeMathFloat32Memory(y, []float32{5, 7}); err != nil {
		t.Fatal(err)
	}
	if err := ctx.SpMV(2, coo, x, -1, y); err != nil {
		t.Fatal(err)
	}
	values, err = readMathFloat32Memory(y, 2)
	if err != nil {
		t.Fatal(err)
	}
	if values[0] != 15 || values[1] != 11 {
		t.Fatalf("COO SpMV = %v, want [15 11]", values)
	}
	sparseTestDense(t, ctx, coo, []float32{1, 0, 2, 0, 3, 0})

	product, err := ctx.SpGEMM(a, b)
	if err != nil {
		t.Fatal(err)
	}
	defer product.Destroy()
	sparseTestDense(t, ctx, product, []float32{0, 16, 15, 0})
	duplicateProduct, err := ctx.SpGEMM(duplicate, b)
	if err != nil {
		t.Fatal(err)
	}
	defer duplicateProduct.Destroy()
	sparseTestDense(t, ctx, duplicateProduct, []float32{0, 16, 15, 0})
	cancelA := sparseTestCSR(t, ctx, 1, 2, []int32{0, 2}, []int32{0, 1}, []float32{1, 1})
	cancelB := sparseTestCSR(t, ctx, 2, 1, []int32{0, 1, 2}, []int32{0, 0}, []float32{1, -1})
	cancelProduct, err := ctx.SpGEMM(cancelA, cancelB)
	if err != nil {
		t.Fatal(err)
	}
	defer cancelProduct.Destroy()
	sparseTestDense(t, ctx, cancelProduct, []float32{0})
	mixedProduct, err := ctx.SpGEMM(coo, b)
	if err != nil {
		t.Fatal(err)
	}
	defer mixedProduct.Destroy()
	sparseTestDense(t, ctx, mixedProduct, []float32{0, 16, 15, 0})
	c := sparseTestCSR(t, ctx, 2, 2, []int32{0, 2, 4}, []int32{0, 1, 0, 1}, []float32{1, 2, 3, 4})
	if err := ctx.SpMM(2, a, b, 0.5, c); err != nil {
		t.Fatal(err)
	}
	sparseTestDense(t, ctx, c, []float32{0.5, 33, 31.5, 2})
	cCOO, err := ctx.CreateCOOMatrix(2, 2, []int32{0, 0, 1, 1}, []int32{0, 1, 0, 1}, []float32{1, 2, 3, 4})
	if err != nil {
		t.Fatal(err)
	}
	defer cCOO.Destroy()
	if err := ctx.SpMM(2, a, b, 0.5, cCOO); err != nil {
		t.Fatal(err)
	}
	sparseTestDense(t, ctx, cCOO, []float32{0.5, 33, 31.5, 2})
	cSmall := sparseTestCSR(t, ctx, 2, 2, []int32{0, 1, 2}, []int32{0, 1}, []float32{1, 1})
	if err := ctx.SpMM(1, a, b, 1, cSmall); err == nil {
		t.Fatal("expected sparse multiplication to reject insufficient output capacity")
	}
	sparseTestDense(t, ctx, cSmall, []float32{1, 0, 0, 1})
	if _, _, nnz, _ := cSmall.GetMatrixInfo(); nnz != 2 {
		t.Fatalf("failed multiplication changed output nnz to %d", nnz)
	}

	empty := sparseTestCSR(t, ctx, 2, 2, []int32{0, 0, 0}, nil, nil)
	emptyProduct, err := ctx.SpGEMM(empty, empty)
	if err != nil {
		t.Fatal(err)
	}
	defer emptyProduct.Destroy()
	sparseTestDense(t, ctx, emptyProduct, []float32{0, 0, 0, 0})
	emptyX := solverTestBuffer(t, []float32{1, 2})
	emptyY := solverTestBuffer(t, []float32{3, 4})
	if err := ctx.SpMV(1, empty, emptyX, 2, emptyY); err != nil {
		t.Fatal(err)
	}
	emptyValues, err := readMathFloat32Memory(emptyY, 2)
	if err != nil {
		t.Fatal(err)
	}
	if emptyValues[0] != 6 || emptyValues[1] != 8 {
		t.Fatalf("empty CSR SpMV = %v, want [6 8]", emptyValues)
	}
	emptyAllocated, err := ctx.CreateSparseMatrix(2, 2, 0, MatrixFormatCSR)
	if err != nil {
		t.Fatal(err)
	}
	defer emptyAllocated.Destroy()
	if err := ctx.SpMV(1, emptyAllocated, emptyX, 2, emptyY); err != nil {
		t.Fatalf("new empty CSR matrix: %v", err)
	}
	emptyValues, err = readMathFloat32Memory(emptyY, 2)
	if err != nil {
		t.Fatal(err)
	}
	if emptyValues[0] != 12 || emptyValues[1] != 16 {
		t.Fatalf("new empty CSR SpMV = %v, want [12 16]", emptyValues)
	}

	for _, format := range []MatrixFormat{MatrixFormatCSR, MatrixFormatCOO} {
		dense := solverTestBuffer(t, []float32{0, 1e-20, -2, 0})
		sparse, err := ctx.DenseToSparse(dense, 2, 2, format)
		if err != nil {
			t.Fatal(err)
		}
		sparseTestDense(t, ctx, sparse, []float32{0, 1e-20, -2, 0})
		_, _, nnz, _ := sparse.GetMatrixInfo()
		if nnz != 2 {
			t.Fatalf("small nonzero entry was lost; nnz = %d", nnz)
		}
		sparseValues, err := readMathFloat32Memory(sparse.values, nnz)
		if err != nil || sparseValues[0] != float32(1e-20) || sparseValues[1] != -2 {
			t.Fatalf("dense-to-sparse values = %v, error = %v", sparseValues, err)
		}
		if err := sparse.Destroy(); err != nil {
			t.Fatal(err)
		}
		unsorted := sparseTestCSR(t, ctx, 2, 3, []int32{0, 2, 3}, []int32{2, 0, 1}, []float32{2, 1, 3})
		sparseTestDense(t, ctx, unsorted, []float32{1, 0, 2, 0, 3, 0})
	}
	for _, test := range []struct {
		rows, cols int
		values     []float32
		nnz        int
	}{
		{2, 3, []float32{0, 0, 0, 0, 0, 0}, 0},
		{2, 3, []float32{1, 0, -2, 0, 3, 0}, 3},
		{3, 1, []float32{0, -1, 0}, 1},
	} {
		dense := solverTestBuffer(t, test.values)
		sparse, err := ctx.DenseToSparse(dense, test.rows, test.cols, MatrixFormatCSR)
		if err != nil {
			t.Fatal(err)
		}
		rows, cols, nnz, format := sparse.GetMatrixInfo()
		if rows != test.rows || cols != test.cols || nnz != test.nnz || format != MatrixFormatCSR {
			t.Fatalf("dense-to-CSR dimensions = %d x %d, nnz = %d, format = %d", rows, cols, nnz, format)
		}
		sparseTestDense(t, ctx, sparse, test.values)
		if err := sparse.Destroy(); err != nil {
			t.Fatal(err)
		}
	}
	special := solverTestBuffer(t, []float32{float32(math.NaN()), float32(math.Inf(1)), float32(math.Inf(-1)), 0})
	specialSparse, err := ctx.DenseToSparse(special, 2, 2, MatrixFormatCSR)
	if err != nil {
		t.Fatal(err)
	}
	defer specialSparse.Destroy()
	specialValues, err := readMathFloat32Memory(specialSparse.values, specialSparse.nnz)
	if err != nil || len(specialValues) != 3 || !math.IsNaN(float64(specialValues[0])) ||
		!math.IsInf(float64(specialValues[1]), 1) || !math.IsInf(float64(specialValues[2]), -1) {
		t.Fatalf("dense-to-CSR special values = %v, error = %v", specialValues, err)
	}
}

func testSparseLUAndTriangularSolve(t *testing.T, ctx *SparseContext) {
	t.Helper()
	matrix := sparseTestCSR(t, ctx, 3, 3, []int32{0, 2, 5, 7}, []int32{0, 1, 0, 1, 2, 1, 2}, []float32{2, 1, 4, 3, 1, 2, 4})
	lower, upper, err := ctx.SpLU(matrix)
	if err != nil {
		t.Fatal(err)
	}
	defer lower.Destroy()
	defer upper.Destroy()
	lDense, err := ctx.SparseToDense(lower)
	if err != nil {
		t.Fatal(err)
	}
	defer lDense.Free()
	uDense, err := ctx.SparseToDense(upper)
	if err != nil {
		t.Fatal(err)
	}
	defer uDense.Free()
	l, err := readMathFloat32Memory(lDense, 9)
	if err != nil {
		t.Fatal(err)
	}
	u, err := readMathFloat32Memory(uDense, 9)
	if err != nil {
		t.Fatal(err)
	}
	want := []float32{2, 1, 0, 4, 3, 1, 0, 2, 4}
	for row := range 3 {
		for col := range 3 {
			var got float32
			for k := range 3 {
				got += l[row*3+k] * u[k*3+col]
			}
			if math.Abs(float64(got-want[row*3+col])) > 1e-5 {
				t.Fatalf("sparse LU[%d,%d]=%v, want %v", row, col, got, want[row*3+col])
			}
		}
	}
	triangular := sparseTestCSR(t, ctx, 3, 3, []int32{0, 1, 3, 6}, []int32{0, 0, 1, 0, 1, 2}, []float32{2, 1, 3, 4, -1, 2})
	rhs := solverTestBuffer(t, []float32{2, 7, 8})
	out := solverTestBuffer(t, []float32{0, 0, 0})
	if err := ctx.SpSV(triangular, rhs, out); err != nil {
		t.Fatal(err)
	}
	solution, err := readMathFloat32Memory(out, 3)
	if err != nil {
		t.Fatal(err)
	}
	for i, value := range []float32{1, 2, 3} {
		if solution[i] != value {
			t.Fatalf("triangular solution[%d]=%v, want %v", i, solution[i], value)
		}
	}
	needsPivot := sparseTestCSR(t, ctx, 2, 2, []int32{0, 1, 2}, []int32{1, 0}, []float32{1, 1})
	if _, _, err := ctx.SpLU(needsPivot); err == nil {
		t.Fatal("expected sparse LU to reject unsupported pivoting")
	}
	if err := ctx.SpSV(matrix, rhs, out); err == nil {
		t.Fatal("expected non-triangular solve to fail")
	}
	upperTriangular := sparseTestCSR(t, ctx, 2, 2, []int32{0, 2, 3}, []int32{0, 1, 1}, []float32{2, 1, 3})
	upperRHS := solverTestBuffer(t, []float32{4, 6})
	upperOut := solverTestBuffer(t, []float32{0, 0})
	if err := ctx.SpSV(upperTriangular, upperRHS, upperOut); err != nil {
		t.Fatal(err)
	}
	upperSolution, err := readMathFloat32Memory(upperOut, 2)
	if err != nil {
		t.Fatal(err)
	}
	if upperSolution[0] != 1 || upperSolution[1] != 2 {
		t.Fatalf("upper triangular solution = %v, want [1 2]", upperSolution)
	}
	unsorted := sparseTestCSR(t, ctx, 2, 2, []int32{0, 1, 3}, []int32{0, 1, 0}, []float32{2, 3, 1})
	unsortedRHS := solverTestBuffer(t, []float32{2, 7})
	unsortedOut := solverTestBuffer(t, []float32{0, 0})
	if err := ctx.SpSV(unsorted, unsortedRHS, unsortedOut); err != nil {
		t.Fatal(err)
	}
	unsortedSolution, err := readMathFloat32Memory(unsortedOut, 2)
	if err != nil {
		t.Fatal(err)
	}
	if unsortedSolution[0] != 1 || unsortedSolution[1] != 2 {
		t.Fatalf("unsorted triangular solution = %v, want [1 2]", unsortedSolution)
	}
	coo, err := ctx.CreateCOOMatrix(2, 2, []int32{1, 0, 1}, []int32{0, 0, 1}, []float32{1, 2, 3})
	if err != nil {
		t.Fatal(err)
	}
	defer coo.Destroy()
	cooRHS := solverTestBuffer(t, []float32{2, 7})
	cooOut := solverTestBuffer(t, []float32{0, 0})
	if err := ctx.SpSV(coo, cooRHS, cooOut); err != nil {
		t.Fatal(err)
	}
	cooSolution, err := readMathFloat32Memory(cooOut, 2)
	if err != nil {
		t.Fatal(err)
	}
	if cooSolution[0] != 1 || cooSolution[1] != 2 {
		t.Fatalf("COO triangular solution = %v, want [1 2]", cooSolution)
	}
}

func TestSparseNumerics(t *testing.T) {
	ctx, err := CreateSparseContext()
	if err != nil {
		t.Fatal(err)
	}
	defer ctx.DestroyContext()
	testSparseNumerics(t, ctx)
	testSparseLUAndTriangularSolve(t, ctx)
}

func TestSparseRejectsInvalidIndices(t *testing.T) {
	ctx := &SparseContext{}
	if _, err := ctx.CreateCSRMatrix(2, 3, []int32{0, 2, 1}, []int32{0}, []float32{1}); err == nil {
		t.Fatal("expected invalid CSR offsets to fail")
	}
	if _, err := ctx.CreateCOOMatrix(2, 3, []int32{2}, []int32{0}, []float32{1}); err == nil {
		t.Fatal("expected invalid COO row to fail")
	}
	if _, err := ctx.CreateCSRMatrix(1, 1, []int32{0, 1}, []int32{1}, []float32{1}); err == nil {
		t.Fatal("expected invalid CSR column to fail")
	}
	matrix, err := ctx.CreateCSRMatrix(1, 1, []int32{0, 1}, []int32{0}, []float32{1})
	if err != nil {
		t.Fatal(err)
	}
	defer matrix.Destroy()
	if err := writeMathTypedMemory(matrix.colInds, []int32{2}); err != nil {
		t.Fatal(err)
	}
	if _, err := ctx.SparseToDense(matrix); err == nil {
		t.Fatal("expected malformed sparse buffer to fail")
	}
}
