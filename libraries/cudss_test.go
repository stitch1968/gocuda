package libraries

import (
	"math"
	"testing"

	"github.com/stitch1968/gocuda/memory"
)

func TestCuDSSSolveAndDeterminant(t *testing.T) {
	rowPtr, _ := memory.Alloc(3 * 4)
	colInd, _ := memory.Alloc(4 * 4)
	values, _ := memory.Alloc(4 * 4)
	b, _ := memory.Alloc(2 * 4)
	x, _ := memory.Alloc(2 * 4)
	defer rowPtr.Free()
	defer colInd.Free()
	defer values.Free()
	defer b.Free()
	defer x.Free()

	writeInt32Memory(rowPtr, []int32{0, 2, 4}, 3)
	writeInt32Memory(colInd, []int32{0, 1, 0, 1}, 4)
	writeMathFloat32Memory(values, []float32{4, 1, 2, 3})
	writeMathFloat32Memory(b, []float32{1, 2})

	matrix, err := CreateDSSMatrix(2, 4, rowPtr, colInd, values, DSSMatrixFormatCSR, false)
	if err != nil {
		t.Fatalf("CreateDSSMatrix failed: %v", err)
	}

	defer matrix.Destroy()
	handle, err := CreateDSSHandle(DSSConfig{MatrixFormat: DSSMatrixFormatCSR, Factorization: DSSFactorizationLU})
	if err != nil {
		t.Fatalf("CreateDSSHandle failed: %v", err)
	}
	defer handle.Destroy()

	if err := handle.Analyze(matrix); err != nil {
		t.Fatalf("Analyze failed: %v", err)
	}
	if err := handle.Factor(matrix); err != nil {
		t.Fatalf("Factor failed: %v", err)
	}
	info, err := handle.Solve(b, x, 1)
	if err != nil {
		t.Fatalf("Solve failed: %v", err)
	}
	if info.Residual > 1e-5 {
		t.Fatalf("unexpected residual: %v", info.Residual)
	}
	xValues, err := readMathFloat32Memory(x, 2)
	if err != nil {
		t.Fatalf("read solution failed: %v", err)
	}
	if math.Abs(float64(xValues[0]-0.1)) > 1e-4 || math.Abs(float64(xValues[1]-0.6)) > 1e-4 {
		t.Fatalf("unexpected solution: %v", xValues)
	}
	det, err := handle.GetDeterminant()
	if err != nil {
		t.Fatalf("GetDeterminant failed: %v", err)
	}
	if math.Abs(det-10) > 1e-6 {
		t.Fatalf("unexpected determinant: got %v want 10", det)
	}
}

func TestCuDSSRejectsMalformedCSR(t *testing.T) {
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
	validRows, validColumns := []int32{0, 2, 4}, []int32{0, 1, 0, 1}
	if err := writeInt32Memory(rows, validRows, 3); err != nil {
		t.Fatal(err)
	}
	if err := writeInt32Memory(columns, validColumns, 4); err != nil {
		t.Fatal(err)
	}
	if err := writeMathFloat32Memory(values, []float32{4, 1, 2, 3}); err != nil {
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
	handle, err := CreateDSSHandle(DSSConfig{MatrixFormat: DSSMatrixFormatCSR, Factorization: DSSFactorizationLU})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := handle.Destroy(); err != nil {
			t.Error(err)
		}
	})
	for _, tc := range []struct {
		name          string
		rows, columns []int32
	}{
		{"nonzero start", []int32{1, 2, 4}, validColumns},
		{"decreasing offsets", []int32{0, 3, 2}, validColumns},
		{"negative offset", []int32{0, -1, 4}, validColumns},
		{"offset past end", []int32{0, 5, 4}, validColumns},
		{"negative column", validRows, []int32{-1, 1, 0, 1}},
		{"column out of range", validRows, []int32{0, 2, 0, 1}},
		{"duplicate column", validRows, []int32{0, 0, 0, 1}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if err := writeInt32Memory(rows, tc.rows, 3); err != nil {
				t.Fatal(err)
			}
			if err := writeInt32Memory(columns, tc.columns, 4); err != nil {
				t.Fatal(err)
			}
			if err := handle.Analyze(matrix); err == nil {
				t.Fatal("expected invalid CSR analysis to fail")
			}
			if err := handle.Factor(matrix); err == nil {
				t.Fatal("expected invalid CSR factorization to fail")
			}
		})
	}
	short := &DSSMatrix{n: 2, nnz: 4, rowPtr: allocate(8), colInd: columns, values: values, format: DSSMatrixFormatCSR}
	if _, _, err := readDSSCSRIndices(short); err == nil {
		t.Fatal("expected short row offset buffer to fail")
	}
	var nilHandle *DSSHandle
	if err := nilHandle.Analyze(matrix); err == nil {
		t.Fatal("expected nil handle analysis to fail")
	}
	if err := nilHandle.Factor(matrix); err == nil {
		t.Fatal("expected nil handle factorization to fail")
	}
	if err := nilHandle.Refactor(matrix); err == nil {
		t.Fatal("expected nil handle refactorization to fail")
	}
	if err := writeInt32Memory(rows, validRows, 3); err != nil {
		t.Fatal(err)
	}
	if err := writeInt32Memory(columns, []int32{1, 0, 1, 0}, 4); err != nil {
		t.Fatal(err)
	}
	if _, _, err := readDSSCSRIndices(matrix); err != nil {
		t.Fatalf("unsorted CSR columns should be accepted: %v", err)
	}
	if err := writeInt32Memory(columns, validColumns, 4); err != nil {
		t.Fatal(err)
	}
	if err := handle.Analyze(matrix); err != nil {
		t.Fatal(err)
	}
	if err := handle.Factor(matrix); err != nil {
		t.Fatal(err)
	}
	if err := writeInt32Memory(columns, []int32{0, 2, 0, 1}, 4); err != nil {
		t.Fatal(err)
	}
	if err := handle.Refactor(matrix); err == nil {
		t.Fatal("expected invalid CSR refactorization to fail")
	}
}
