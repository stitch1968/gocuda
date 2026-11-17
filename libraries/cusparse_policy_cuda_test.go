//go:build cuda

package libraries

import (
	"errors"
	"reflect"
	"testing"

	cuda "github.com/stitch1968/gocuda"
	"github.com/stitch1968/gocuda/memory"
)

func TestNativeSparseMVTransposeOnEachDevice(t *testing.T) {
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
			csr, err := ctx.CreateCSRMatrix(2, 3, []int32{0, 3, 4}, []int32{2, 0, 2, 1}, []float32{1, 1, 1, 3})
			if err != nil {
				t.Fatal(err)
			}
			defer csr.Destroy()
			coo, err := ctx.CreateCOOMatrix(2, 3, []int32{1, 0, 0, 0}, []int32{1, 2, 0, 2}, []float32{3, 1, 1, 1})
			if err != nil {
				t.Fatal(err)
			}
			defer coo.Destroy()
			input, err := memory.AllocDeviceOnlyOnDevice(8, deviceID)
			if err != nil {
				t.Fatal(err)
			}
			defer input.Free()
			output, err := memory.AllocDeviceOnlyOnDevice(16, deviceID)
			if err != nil {
				t.Fatal(err)
			}
			defer output.Free()
			if err := writeMathFloat32Memory(input, []float32{4, 5}); err != nil {
				t.Fatal(err)
			}
			for _, matrix := range []*SparseMatrix{csr, coo} {
				if got, err := ctx.SpMVBackend(matrix.format); err != nil || got != ImplementationModeNativeCUDA {
					t.Fatalf("transpose backend = %q, error = %v", got, err)
				}
				if err := writeMathFloat32Memory(output, []float32{1, 2, 3, 99}); err != nil {
					t.Fatal(err)
				}
				if err := ctx.SpMVEx(2, matrix, input, -1, output, BLASOpT); err != nil {
					t.Fatal(err)
				}
				got, err := readMathFloat32Memory(output, 4)
				if err != nil || !reflect.DeepEqual(got, []float32{7, 28, 13, 99}) {
					t.Fatalf("native transpose output = %v, error = %v", got, err)
				}
				if err := ctx.SpMVEx(1, matrix, output, 0, output, BLASOpT); err == nil {
					t.Fatal("expected aliased native vectors to fail")
				}
				if err := ctx.SpMVEx(1, matrix, input, 0, matrix.values, BLASOpT); err == nil {
					t.Fatal("expected sparse buffer alias to fail")
				}
				uninitialized, err := memory.AllocDeviceOnlyOnDevice(12, deviceID)
				if err != nil {
					t.Fatal(err)
				}
				defer uninitialized.Free()
				if err := ctx.SpMVEx(1, matrix, input, 0, uninitialized, BLASOpT); err != nil {
					t.Fatal(err)
				}
				got, err = readMathFloat32Memory(uninitialized, 3)
				if err != nil || !reflect.DeepEqual(got, []float32{4, 15, 8}) {
					t.Fatalf("native zero-beta transpose output = %v, error = %v", got, err)
				}
			}
			if len(devices) > 1 {
				other, err := memory.AllocDeviceOnlyOnDevice(8, (deviceID+1)%len(devices))
				if err != nil {
					t.Fatal(err)
				}
				defer other.Free()
				if err := ctx.SpMVEx(1, csr, other, 0, output, BLASOpT); err == nil {
					t.Fatal("expected cross-device input to fail")
				}
				got, err := readMathFloat32Memory(output, 4)
				if err != nil || !reflect.DeepEqual(got, []float32{7, 28, 13, 99}) {
					t.Fatalf("rejected transpose changed output: %v, error = %v", got, err)
				}
			}
		})
	}
}

func TestNativeSparseDenseOnEachDevice(t *testing.T) {
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
			csr, err := ctx.CreateCSRMatrix(2, 2, []int32{0, 2, 3}, []int32{0, 1, 1}, []float32{2, 1, 3})
			if err != nil {
				t.Fatal(err)
			}
			defer csr.Destroy()
			coo, err := ctx.CreateCOOMatrix(2, 2, []int32{1, 0, 0, 0}, []int32{1, 1, 1, 0}, []float32{3, 0.25, 0.75, 2})
			if err != nil {
				t.Fatal(err)
			}
			defer coo.Destroy()
			input, err := memory.AllocDeviceOnlyOnDevice(16, deviceID)
			if err != nil {
				t.Fatal(err)
			}
			defer input.Free()
			if err := writeMathFloat32Memory(input, []float32{1, 2, 3, 4}); err != nil {
				t.Fatal(err)
			}
			paddedInput, err := memory.AllocDeviceOnlyOnDevice(24, deviceID)
			if err != nil {
				t.Fatal(err)
			}
			defer paddedInput.Free()
			if err := writeMathFloat32Memory(paddedInput, []float32{1, 3, 111, 2, 4, 112}); err != nil {
				t.Fatal(err)
			}
			batchedInput, err := memory.AllocDeviceOnlyOnDevice(32, deviceID)
			if err != nil {
				t.Fatal(err)
			}
			defer batchedInput.Free()
			if err := writeMathFloat32Memory(batchedInput, []float32{1, 2, 3, 4, 2, 0, 1, -1}); err != nil {
				t.Fatal(err)
			}
			for _, matrix := range []*SparseMatrix{csr, coo} {
				if got, err := ctx.SpMMDenseBackend(matrix.format); err != nil || got != ImplementationModeNativeCUDA {
					t.Fatalf("strict sparse-dense backend = %q, error = %v", got, err)
				}
				output, err := memory.AllocDeviceOnlyOnDevice(16, deviceID)
				if err != nil {
					t.Fatal(err)
				}
				defer output.Free()
				if err := ctx.SpMMDense(1, matrix, input, 2, 0, output); err != nil {
					t.Fatal(err)
				}
				if err := ctx.SpMMDense(1, matrix, input, 2, 2, output); err != nil {
					t.Fatal(err)
				}
				got, err := readMathFloat32Memory(output, 4)
				if err != nil || !reflect.DeepEqual(got, []float32{15, 24, 27, 36}) {
					t.Fatalf("strict sparse-dense output = %v, error = %v", got, err)
				}
				paddedOutput, err := memory.AllocDeviceOnlyOnDevice(24, deviceID)
				if err != nil {
					t.Fatal(err)
				}
				defer paddedOutput.Free()
				if err := writeMathFloat32Memory(paddedOutput, []float32{1, 2, 91, 3, 4, 92}); err != nil {
					t.Fatal(err)
				}
				if err := ctx.SpMMDenseEx(2, matrix, paddedInput, 2, 3, 3, BLASOpT, BLASOpT, -1, paddedOutput); err != nil {
					t.Fatal(err)
				}
				got, err = readMathFloat32Memory(paddedOutput, 6)
				if err != nil || !reflect.DeepEqual(got, []float32{3, 6, 91, 17, 24, 92}) {
					t.Fatalf("strict transposed sparse-dense output = %v, error = %v", got, err)
				}
				batchedOutput, err := memory.AllocDeviceOnlyOnDevice(32, deviceID)
				if err != nil {
					t.Fatal(err)
				}
				defer batchedOutput.Free()
				if err := ctx.SpMMDenseBatched(1, matrix, batchedInput, 2, 2, 2, BLASOpN, BLASOpN, 2, 4, 4, 0, batchedOutput); err != nil {
					t.Fatal(err)
				}
				got, err = readMathFloat32Memory(batchedOutput, 8)
				if err != nil || !reflect.DeepEqual(got, []float32{5, 8, 9, 12, 5, -1, 3, -3}) {
					t.Fatalf("strict batched sparse-dense output = %v, error = %v", got, err)
				}
			}
			host, err := memory.AllocWithTypeAndStream(nil, 16, memory.TypePinned)
			if err != nil {
				t.Fatal(err)
			}
			defer host.Free()
			if err := ctx.SpMMDense(1, csr, host, 2, 0, input); err == nil {
				t.Fatal("expected pinned input to be rejected by the native backend")
			}
			if len(devices) > 1 {
				other, err := memory.AllocDeviceOnlyOnDevice(16, (deviceID+1)%len(devices))
				if err != nil {
					t.Fatal(err)
				}
				defer other.Free()
				if err := ctx.SpMMDense(1, csr, other, 2, 0, input); err == nil {
					t.Fatal("expected a different-device input to be rejected")
				}
				otherBatch, err := memory.AllocDeviceOnlyOnDevice(32, (deviceID+1)%len(devices))
				if err != nil {
					t.Fatal(err)
				}
				defer otherBatch.Free()
				if err := ctx.SpMMDenseBatched(1, csr, otherBatch, 2, 2, 2, BLASOpN, BLASOpN, 2, 4, 4, 0, batchedInput); err == nil {
					t.Fatal("expected a different-device batch to be rejected")
				}
				got, err := readMathFloat32Memory(batchedInput, 8)
				if err != nil || !reflect.DeepEqual(got, []float32{1, 2, 3, 4, 2, 0, 1, -1}) {
					t.Fatalf("rejected batch changed output: %v, error %v", got, err)
				}
			}
		})
	}
}

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
			if got, err := ctx.SpSVBackend(MatrixFormatCOO); err != nil || got != ImplementationModeNativeCUDA {
				t.Fatalf("strict COO solve backend = %q, error = %v", got, err)
			}
			if got, err := ctx.SpGEMMBackend(MatrixFormatCSR, MatrixFormatCSR); err != nil || got != ImplementationModeNativeCUDA {
				t.Fatalf("strict CSR product backend = %q, error = %v", got, err)
			}
			if got, err := ctx.SpGEMMBackend(MatrixFormatCOO, MatrixFormatCSR); err != nil || got != ImplementationModeNativeCUDA {
				t.Fatalf("strict mixed product backend = %q, error = %v", got, err)
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
			for _, inputs := range [][2]*SparseMatrix{{coo, matrix}, {matrix, coo}, {coo, coo}, {empty, coo}} {
				mixed, err := ctx.SpGEMM(inputs[0], inputs[1])
				if err != nil {
					t.Fatal(err)
				}
				mixedRows, readErr := readSparseRows(mixed)
				mixedDevice := mixed.values.GetDeviceID()
				mixedFormat := mixed.format
				destroyErr := mixed.Destroy()
				if readErr != nil || destroyErr != nil {
					t.Fatalf("strict mixed product read/destroy: %v / %v", readErr, destroyErr)
				}
				if mixedDevice != deviceID || mixedFormat != MatrixFormatCSR {
					t.Fatal("strict mixed product has incorrect device or format")
				}
				want := []float32{4, 5, 0, 9}
				if inputs[0] == empty {
					want = []float32{0, 0, 0, 0}
				}
				got := []float32{mixedRows[0][0], mixedRows[0][1], mixedRows[1][0], mixedRows[1][1]}
				if !reflect.DeepEqual(got, want) {
					t.Fatalf("strict mixed product = %v, want %v", got, want)
				}
			}
			if got, err := ctx.SpMMBackend(MatrixFormatCOO, MatrixFormatCSR, MatrixFormatCOO); err != nil || got != ImplementationModeNativeCUDA {
				t.Fatalf("strict scaled product backend = %q, error = %v", got, err)
			}
			scaled, err := ctx.CreateCOOMatrix(2, 2, []int32{0, 0, 1, 1}, []int32{0, 1, 0, 1}, []float32{1, 2, 3, 4})
			if err != nil {
				t.Fatal(err)
			}
			defer scaled.Destroy()
			if err := ctx.SpMM(2, coo, matrix, 0.5, scaled); err != nil {
				t.Fatal(err)
			}
			scaledRows, err := readSparseRows(scaled)
			if err != nil {
				t.Fatal(err)
			}
			if got := []float32{scaledRows[0][0], scaledRows[0][1], scaledRows[1][0], scaledRows[1][1]}; !reflect.DeepEqual(got, []float32{8.5, 11, 1.5, 20}) {
				t.Fatalf("strict scaled product = %v", got)
			}
			if len(devices) > 1 {
				otherCtx, err := CreateNativeSparseContextOnDevice((deviceID + 1) % len(devices))
				if err != nil {
					t.Fatal(err)
				}
				defer otherCtx.DestroyContext()
				other, err := otherCtx.CreateCSRMatrix(2, 2, []int32{0, 1, 2}, []int32{0, 1}, []float32{1, 1})
				if err != nil {
					t.Fatal(err)
				}
				defer other.Destroy()
				if err := ctx.SpMM(1, matrix, other, 1, scaled); err == nil {
					t.Fatal("expected sparse product to reject an input on a different device")
				}
				if err := ctx.SpMM(1, matrix, matrix, 1, other); err == nil {
					t.Fatal("expected sparse product to reject an output on a different device")
				}
				unchanged, err := readSparseRows(scaled)
				if err != nil || unchanged[0][0] != 8.5 || unchanged[0][1] != 11 || unchanged[1][0] != 1.5 || unchanged[1][1] != 20 {
					t.Fatalf("failed cross-device product changed output: %v, error %v", unchanged, err)
				}
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
			if err := writeMathFloat32Memory(x, []float32{4, 6}); err != nil {
				t.Fatal(err)
			}
			if err := ctx.SpSV(coo, x, y); err != nil {
				t.Fatal(err)
			}
			values, err = readMathFloat32Memory(y, 2)
			if err != nil || values[0] != 1 || values[1] != 2 {
				t.Fatalf("strict duplicate COO triangular solve = %v, error = %v; want [1 2]", values, err)
			}
			for _, tc := range []struct {
				matrix *SparseMatrix
				rhs    []float32
			}{
				{matrix, []float32{2, 7}},
				{duplicateTriangle, []float32{4, 6}},
				{coo, []float32{2, 7}},
			} {
				if err := writeMathFloat32Memory(x, tc.rhs); err != nil {
					t.Fatal(err)
				}
				if err := ctx.SpSVEx(tc.matrix, x, x, BLASOpT); err == nil {
					t.Fatal("expected aliased native transpose solve to fail")
				}
				if err := ctx.SpSVEx(tc.matrix, x, tc.matrix.values, BLASOpT); err == nil {
					t.Fatal("expected sparse buffer alias in transpose solve to fail")
				}
				if err := ctx.SpSVEx(tc.matrix, x, y, BLASOpT); err != nil {
					t.Fatal(err)
				}
				values, err = readMathFloat32Memory(y, 2)
				if err != nil || !reflect.DeepEqual(values, []float32{1, 2}) {
					t.Fatalf("strict transpose solve = %v, error = %v", values, err)
				}
			}
			if len(devices) > 1 {
				other, err := memory.AllocDeviceOnlyOnDevice(8, (deviceID+1)%len(devices))
				if err != nil {
					t.Fatal(err)
				}
				defer other.Free()
				if err := ctx.SpSVEx(matrix, other, y, BLASOpT); err == nil {
					t.Fatal("expected transpose solve on a different-device input to fail")
				}
				values, err = readMathFloat32Memory(y, 2)
				if err != nil || !reflect.DeepEqual(values, []float32{1, 2}) {
					t.Fatalf("rejected transpose solve changed output: %v, error = %v", values, err)
				}
			}
			if err := writeMathFloat32Memory(x, []float32{2, 7}); err != nil {
				t.Fatal(err)
			}
			if got, err := ctx.DenseToSparseBackend(MatrixFormatCSR); err != nil || got != ImplementationModeNativeCUDA {
				t.Fatalf("strict dense-to-CSR backend = %q, error = %v", got, err)
			}
			for _, format := range []MatrixFormat{MatrixFormatCSR, MatrixFormatCOO} {
				if got, err := ctx.DenseToSparseBackend(format); err != nil || got != ImplementationModeNativeCUDA {
					t.Fatalf("strict dense-to-sparse backend for %d = %q, error = %v", format, got, err)
				}
				if _, err := ctx.DenseToSparse(x, 2, 2, format); err == nil {
					t.Fatal("expected dense-to-sparse to reject a short input buffer")
				}
				if _, err := ctx.DenseToSparse(x, 0, 2, format); err == nil {
					t.Fatal("expected dense-to-sparse to reject invalid dimensions")
				}
				converted, err := ctx.DenseToSparse(x, 1, 2, format)
				if err != nil {
					t.Fatal(err)
				}
				defer converted.Destroy()
				if converted.values.GetDeviceID() != deviceID || converted.rowPtrs.GetDeviceID() != deviceID ||
					converted.colInds.GetDeviceID() != deviceID {
					t.Fatal("strict converted sparse matrix was allocated on the wrong device")
				}
				convertedRows, err := readSparseRows(converted)
				if err != nil || converted.nnz != 2 || converted.format != format || convertedRows[0][0] != 2 || convertedRows[0][1] != 7 {
					t.Fatalf("strict dense-to-sparse output for %d = %v, nnz = %d, error = %v", format, convertedRows, converted.nnz, err)
				}
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
