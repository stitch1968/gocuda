//go:build cuda

package libraries

import (
	"errors"
	"reflect"
	"testing"

	cuda "github.com/stitch1968/gocuda"
	"github.com/stitch1968/gocuda/memory"
)

func TestNativeSparseStrictPolicyOnEachDevice(t *testing.T) {
	if err := cuda.Initialize(); err != nil {
		t.Fatal(err)
	}
	if !cuda.ShouldUseCuda() {
		t.Skip("CUDA hardware unavailable")
	}
	devices, err := cuda.GetDevices()
	if err != nil {
		t.Fatal(err)
	}
	for deviceID := range devices {
		t.Run(devices[deviceID].Name, func(t *testing.T) {
			ctx, err := CreateNativeSparseContextOnDevice(deviceID)
			if err != nil {
				t.Fatal(err)
			}
			defer ctx.DestroyContext()
			if got, err := ctx.SpMVBackend(MatrixFormatCSR); err != nil || got != ImplementationModeNativeCUDA {
				t.Fatalf("strict CSR backend = %q, error = %v", got, err)
			}
			if got, err := ctx.SpMVBackend(MatrixFormatCOO); err != nil || got != ImplementationModeNativeCUDA {
				t.Fatalf("strict COO backend = %q, error = %v", got, err)
			}
			if got, err := ctx.SpSVBackend(MatrixFormatCSR); err != nil || got != ImplementationModeNativeCUDA {
				t.Fatalf("strict CSR solve backend = %q, error = %v", got, err)
			}
			if _, err := ctx.SpSVBackend(MatrixFormatCOO); !errors.Is(err, ErrNativeSparseOperationUnsupported) {
				t.Fatalf("strict COO solve backend error = %v", err)
			}
			if got, err := ctx.SpGEMMBackend(MatrixFormatCSR, MatrixFormatCSR); err != nil || got != ImplementationModeNativeCUDA {
				t.Fatalf("strict CSR product backend = %q, error = %v", got, err)
			}
			if _, err := ctx.SpGEMMBackend(MatrixFormatCOO, MatrixFormatCSR); !errors.Is(err, ErrNativeSparseOperationUnsupported) {
				t.Fatalf("strict mixed product backend error = %v", err)
			}
			matrix, err := ctx.CreateCSRMatrix(2, 2, []int32{0, 2, 3}, []int32{0, 1, 1}, []float32{2, 1, 3})
			if err != nil {
				t.Fatal(err)
			}
			defer matrix.Destroy()
			if matrix.values.GetDeviceID() != deviceID || matrix.rowPtrs.GetDeviceID() != deviceID {
				t.Fatal("strict sparse matrix was allocated on the wrong device")
			}
			x, err := memory.AllocDeviceOnlyOnDevice(8, deviceID)
			if err != nil {
				t.Fatal(err)
			}
			defer x.Free()
			y, err := memory.AllocDeviceOnlyOnDevice(8, deviceID)
			if err != nil {
				t.Fatal(err)
			}
			defer y.Free()
			if err := writeMathFloat32Memory(x, []float32{1, 2}); err != nil {
				t.Fatal(err)
			}
			if err := writeMathFloat32Memory(y, []float32{0, 0}); err != nil {
				t.Fatal(err)
			}
			if err := ctx.SpMV(1, matrix, x, 0, matrix.Values()); err == nil {
				t.Fatal("expected sparse matrix/output alias to fail")
			}
			if err := ctx.SpMV(1, matrix, x, 0, y); err != nil {
				t.Fatal(err)
			}
			values, err := readMathFloat32Memory(y, 2)
			if err != nil {
				t.Fatal(err)
			}
			if values[0] != 4 || values[1] != 6 {
				t.Fatalf("native CSR SpMV = %v, want [4 6]", values)
			}
			coo, err := ctx.CreateCOOMatrix(2, 2, []int32{1, 0, 0, 0}, []int32{1, 1, 1, 0}, []float32{3, 0.25, 0.75, 2})
			if err != nil {
				t.Fatal(err)
			}
			defer coo.Destroy()
			if err := ctx.SpMV(1, coo, x, 0, y); err != nil {
				t.Fatal(err)
			}
			values, err = readMathFloat32Memory(y, 2)
			if err != nil {
				t.Fatal(err)
			}
			if values[0] != 4 || values[1] != 6 {
				t.Fatalf("native unsorted, duplicate COO SpMV = %v, want [4 6]", values)
			}
			empty, err := ctx.CreateCOOMatrix(2, 2, nil, nil, nil)
			if err != nil {
				t.Fatal(err)
			}
			defer empty.Destroy()
			if err := ctx.SpMV(1, empty, x, 2, y); err != nil {
				t.Fatal(err)
			}
			values, err = readMathFloat32Memory(y, 2)
			if err != nil {
				t.Fatal(err)
			}
			if values[0] != 8 || values[1] != 12 {
				t.Fatalf("native empty COO SpMV = %v, want [8 12]", values)
			}
			product, err := ctx.SpGEMM(matrix, matrix)
			if err != nil {
				t.Fatal(err)
			}
			defer product.Destroy()
			if product.values.GetDeviceID() != deviceID || product.rowPtrs.GetDeviceID() != deviceID {
				t.Fatal("strict sparse product was allocated on the wrong device")
			}
			productRows, err := readSparseRows(product)
			if err != nil {
				t.Fatal(err)
			}
			if productRows[0][0] != 4 || productRows[0][1] != 5 || productRows[1][1] != 9 {
				t.Fatalf("native sparse product = %v, want [[4 5] [0 9]]", productRows)
			}
			if _, err := ctx.SpGEMM(coo, matrix); !errors.Is(err, ErrNativeSparseOperationUnsupported) {
				t.Fatalf("strict mixed-format SpGEMM error = %v", err)
			}
			if err := ctx.SpMM(1, matrix, matrix, 0, matrix); !errors.Is(err, ErrNativeSparseOperationUnsupported) {
				t.Fatalf("strict SpMM error = %v", err)
			}
			if _, _, err := ctx.SpLU(matrix); !errors.Is(err, ErrNativeSparseOperationUnsupported) {
				t.Fatalf("strict sparse LU error = %v", err)
			}
			if err := writeMathFloat32Memory(x, []float32{4, 6}); err != nil {
				t.Fatal(err)
			}
			if err := ctx.SpSV(matrix, x, x); err == nil {
				t.Fatal("expected native triangular input/output alias to fail")
			}
			if err := ctx.SpSV(matrix, x, y); err != nil {
				t.Fatal(err)
			}
			values, err = readMathFloat32Memory(y, 2)
			if err != nil {
				t.Fatal(err)
			}
			if values[0] != 1 || values[1] != 2 {
				t.Fatalf("native upper triangular solve = %v, want [1 2]", values)
			}
			duplicateTriangle, err := ctx.CreateCSRMatrix(2, 2, []int32{0, 2, 4}, []int32{0, 0, 0, 1}, []float32{1, 1, 1, 3})
			if err != nil {
				t.Fatal(err)
			}
			defer duplicateTriangle.Destroy()
			if err := writeMathFloat32Memory(x, []float32{2, 7}); err != nil {
				t.Fatal(err)
			}
			if err := ctx.SpSV(duplicateTriangle, x, y); err != nil {
				t.Fatal(err)
			}
			values, err = readMathFloat32Memory(y, 2)
			if err != nil {
				t.Fatal(err)
			}
			if values[0] != 1 || values[1] != 2 {
				t.Fatalf("native duplicate lower triangular solve = %v, want [1 2]", values)
			}
			if err := ctx.SpSV(coo, x, y); !errors.Is(err, ErrNativeSparseOperationUnsupported) {
				t.Fatalf("strict COO sparse solve error = %v", err)
			}
			if got, err := ctx.DenseToSparseBackend(MatrixFormatCSR); err != nil || got != ImplementationModeNativeCUDA {
				t.Fatalf("strict dense-to-CSR backend = %q, error = %v", got, err)
			}
			if _, err := ctx.DenseToSparseBackend(MatrixFormatCOO); !errors.Is(err, ErrNativeSparseOperationUnsupported) {
				t.Fatalf("strict dense-to-COO backend error = %v", err)
			}
			if _, err := ctx.DenseToSparse(x, 1, 2, MatrixFormatCOO); !errors.Is(err, ErrNativeSparseOperationUnsupported) {
				t.Fatalf("strict dense-to-COO error = %v", err)
			}
			if _, err := ctx.DenseToSparse(x, 2, 2, MatrixFormatCSR); err == nil {
				t.Fatal("expected dense-to-CSR to reject a short input buffer")
			}
			if _, err := ctx.DenseToSparse(x, 0, 2, MatrixFormatCSR); err == nil {
				t.Fatal("expected dense-to-CSR to reject invalid dimensions")
			}
			converted, err := ctx.DenseToSparse(x, 1, 2, MatrixFormatCSR)
			if err != nil {
				t.Fatal(err)
			}
			defer converted.Destroy()
			if converted.values.GetDeviceID() != deviceID || converted.rowPtrs.GetDeviceID() != deviceID {
				t.Fatal("strict converted sparse matrix was allocated on the wrong device")
			}
			convertedRows, err := readSparseRows(converted)
			if err != nil || converted.nnz != 2 || convertedRows[0][0] != 2 || convertedRows[0][1] != 7 {
				t.Fatalf("strict dense-to-CSR output = %v, nnz = %d, error = %v", convertedRows, converted.nnz, err)
			}
			for _, sparse := range []*SparseMatrix{matrix, coo} {
				if got, err := ctx.SparseToDenseBackend(sparse.format); err != nil || got != ImplementationModeNativeCUDA {
					t.Fatalf("strict sparse-to-dense backend = %q, error = %v", got, err)
				}
				dense, err := ctx.SparseToDense(sparse)
				if err != nil {
					t.Fatal(err)
				}
				values, readErr := readMathFloat32Memory(dense, 4)
				freeErr := dense.Free()
				if readErr != nil || freeErr != nil {
					t.Fatalf("read/free dense result: %v / %v", readErr, freeErr)
				}
				want := []float32{2, 1, 0, 3}
				if !reflect.DeepEqual(values, want) {
					t.Fatalf("strict dense output = %v, want %v", values, want)
				}
			}
		})
	}
}
