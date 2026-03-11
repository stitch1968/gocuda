package libraries

import (
	"testing"

	"github.com/stitch1968/gocuda/memory"
)

func TestDenseSparseRoundTripCSR(t *testing.T) {
	dense, err := memory.Alloc(4 * 4 * 4)
	if err != nil {
		t.Fatal(err)
	}
	defer dense.Free()

	values, err := memory.View[float32](dense, 16)
	if err != nil {
		t.Fatal(err)
	}
	copy(values, []float32{
		1, 0, 2, 0,
		0, 0, 0, 3,
		4, 0, 5, 0,
		0, 0, 0, 0,
	})

	sparse, err := denseToSparseFromHost(dense, 4, 4, MatrixFormatCSR)
	if err != nil {
		t.Fatal(err)
	}
	defer sparse.Destroy()

	if sparse.nnz != 5 {
		t.Fatalf("expected 5 non-zero values, got %d", sparse.nnz)
	}

	roundTrip, err := sparseToDenseHost(sparse)
	if err != nil {
		t.Fatal(err)
	}
	defer roundTrip.Free()

	roundTripValues, err := memory.View[float32](roundTrip, 16)
	if err != nil {
		t.Fatal(err)
	}
	for index, expected := range values {
		if roundTripValues[index] != expected {
			t.Fatalf("round-trip mismatch at %d: got %v want %v", index, roundTripValues[index], expected)
		}
	}
}

func TestSparseMatrixMultiplySimulationPath(t *testing.T) {
	ctx, err := CreateSparseContext()
	if err != nil {
		t.Fatal(err)
	}
	defer ctx.DestroyContext()
	A, err := createSparseMatrixWithBuffers(
		2,
		3,
		[]float32{1, 2, 3},
		[]int32{0, 2, 3},
		[]int32{0, 2, 1},
		MatrixFormatCSR,
	)
	if err != nil {
		t.Fatal(err)
	}
	defer A.Destroy()
	B, err := createSparseMatrixWithBuffers(
		3,
		2,
		[]float32{4, 5, 6},
		[]int32{0, 1, 2, 3},
		[]int32{1, 0, 1},
		MatrixFormatCSR,
	)
	if err != nil {
		t.Fatal(err)
	}
	defer B.Destroy()

	C, err := ctx.SpGEMM(A, B)
	if err != nil {
		t.Fatal(err)
	}
	defer C.Destroy()

	if C.rows != 2 || C.cols != 2 {
		t.Fatalf("unexpected output dimensions: %dx%d", C.rows, C.cols)
	}
	if C.nnz <= 0 {
		t.Fatalf("expected non-zero result, got nnz=%d", C.nnz)
	}
}
