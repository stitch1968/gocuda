//go:build cuda

package libraries

/*
#cgo linux CFLAGS: -I/usr/local/cuda/include -I/opt/cuda/include
#cgo windows CFLAGS: -I"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.1/include" -I"D:/NVIDIA/include"
#cgo linux LDFLAGS: -L/usr/local/cuda/lib64 -L/opt/cuda/lib64 -lcublas
#cgo windows LDFLAGS: -L${SRCDIR}/../lib_mingw -lcublas

#include <cublas_v2.h>

static cublasStatus_t createCublasHandleWrapper(cublasHandle_t* handle) {
	return cublasCreate(handle);
}

static cublasStatus_t destroyCublasHandleWrapper(cublasHandle_t handle) {
	return cublasDestroy(handle);
}

static cublasStatus_t cutlassStrmmWrapper(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo, cublasOperation_t transa, cublasDiagType_t diag, int m, int n, const float* alpha, const float* A, int lda, const float* B, int ldb, float* C, int ldc) {
	return cublasStrmm(handle, side, uplo, transa, diag, m, n, alpha, A, lda, B, ldb, C, ldc);
}
*/
import "C"

import (
	"fmt"
	"unsafe"

	"github.com/stitch1968/gocuda/memory"
)

func cublasHandleToUintptr(handle C.cublasHandle_t) uintptr {
	return uintptr(unsafe.Pointer(handle))
}

func cublasHandleFromUintptr(handle uintptr) C.cublasHandle_t {
	return (C.cublasHandle_t)(unsafe.Pointer(handle))
}

func cutlassNativeAvailable() bool {
	return true
}

func createNativeCutlassGemm(desc CutlassGemmDesc) (*CutlassGemmHandle, error) {
	if desc.DataType != CutlassFloat32 && desc.DataType != CutlassFloat64 {
		return nil, errCUTLASSUnsupported
	}
	if desc.LayoutA != CutlassRowMajor || desc.LayoutB != CutlassRowMajor || desc.LayoutC != CutlassRowMajor {
		return nil, errCUTLASSUnsupported
	}
	if desc.OpA != CutlassOpN || desc.OpB != CutlassOpN {
		return nil, errCUTLASSUnsupported
	}
	if desc.EpilogueOp != CutlassEpilogueLinearCombination {
		return nil, errCUTLASSUnsupported
	}

	var nativeHandle C.cublasHandle_t
	if status := C.createCublasHandleWrapper(&nativeHandle); status != C.CUBLAS_STATUS_SUCCESS {
		return nil, cutlassCublasError("cublasCreate", status)
	}

	handle, err := createDeterministicCutlassGemm(desc)
	if err != nil {
		_ = C.destroyCublasHandleWrapper(nativeHandle)
		return nil, err
	}
	handle.nativeHandle = cublasHandleToUintptr(nativeHandle)
	handle.native = true
	return handle, nil
}

func executeNativeCutlassGemm(handle *CutlassGemmHandle, A, B, output *memory.Memory) error {
	if handle == nil || !handle.native {
		return errCUTLASSUnsupported
	}
	desc := handle.descriptor
	if desc.DataType != CutlassFloat32 && desc.DataType != CutlassFloat64 {
		return errCUTLASSUnsupported
	}
	if desc.LayoutA != CutlassRowMajor || desc.LayoutB != CutlassRowMajor || desc.LayoutC != CutlassRowMajor {
		return errCUTLASSUnsupported
	}
	if desc.OpA != CutlassOpN || desc.OpB != CutlassOpN {
		return errCUTLASSUnsupported
	}
	if desc.EpilogueOp != CutlassEpilogueLinearCombination {
		return errCUTLASSUnsupported
	}

	nativeHandle := cublasHandleFromUintptr(handle.nativeHandle)
	m := C.int(desc.M)
	n := C.int(desc.N)
	k := C.int(desc.K)
	lda := C.int(desc.K)
	ldb := C.int(desc.N)
	ldc := C.int(desc.N)

	// Row-major C = A*B is equivalent to column-major C^T = B^T*A^T.
	if desc.DataType == CutlassFloat32 {
		alpha := C.float(desc.Alpha)
		beta := C.float(desc.Beta)
		status := C.cublasSgemm(
			nativeHandle,
			C.CUBLAS_OP_N,
			C.CUBLAS_OP_N,
			n,
			m,
			k,
			&alpha,
			(*C.float)(B.Ptr()),
			ldb,
			(*C.float)(A.Ptr()),
			lda,
			&beta,
			(*C.float)(output.Ptr()),
			ldc,
		)
		if status != C.CUBLAS_STATUS_SUCCESS {
			return cutlassCublasError("cublasSgemm", status)
		}
		return nil
	}

	alpha := C.double(desc.Alpha)
	beta := C.double(desc.Beta)
	status := C.cublasDgemm(
		nativeHandle,
		C.CUBLAS_OP_N,
		C.CUBLAS_OP_N,
		n,
		m,
		k,
		&alpha,
		(*C.double)(B.Ptr()),
		ldb,
		(*C.double)(A.Ptr()),
		lda,
		&beta,
		(*C.double)(output.Ptr()),
		ldc,
	)
	if status != C.CUBLAS_STATUS_SUCCESS {
		return cutlassCublasError("cublasDgemm", status)
	}
	return nil
}

func executeNativeCutlassRank2k(A, B, CMem *memory.Memory, N, K int, alpha, beta float32) error {
	if A == nil || B == nil || CMem == nil {
		return errCUTLASSUnsupported
	}
	if beta != 0 {
		return errCUTLASSUnsupported
	}
	return withCutlassCublasHandle(func(nativeHandle C.cublasHandle_t) error {
		rows := C.int(N)
		inner := C.int(K)
		lda := C.int(K)
		ldb := C.int(K)
		ldc := C.int(N)
		zero := C.float(0)
		one := C.float(1)
		alphaC := C.float(alpha)

		status := C.cublasSgemm(
			nativeHandle,
			C.CUBLAS_OP_T,
			C.CUBLAS_OP_N,
			rows,
			rows,
			inner,
			&alphaC,
			(*C.float)(B.Ptr()),
			ldb,
			(*C.float)(A.Ptr()),
			lda,
			&zero,
			(*C.float)(CMem.Ptr()),
			ldc,
		)
		if status != C.CUBLAS_STATUS_SUCCESS {
			return cutlassCublasError("cublasSgemm(rank2k_abt)", status)
		}

		status = C.cublasSgemm(
			nativeHandle,
			C.CUBLAS_OP_T,
			C.CUBLAS_OP_N,
			rows,
			rows,
			inner,
			&alphaC,
			(*C.float)(A.Ptr()),
			lda,
			(*C.float)(B.Ptr()),
			ldb,
			&one,
			(*C.float)(CMem.Ptr()),
			ldc,
		)
		if status != C.CUBLAS_STATUS_SUCCESS {
			return cutlassCublasError("cublasSgemm(rank2k_bat)", status)
		}
		return nil
	})
}

func executeNativeCutlassTrmm(A, B *memory.Memory, M, N int, side, uplo, trans, diag string, alpha float32) error {
	if A == nil || B == nil {
		return errCUTLASSUnsupported
	}
	return withCutlassCublasHandle(func(nativeHandle C.cublasHandle_t) error {
		temp, err := memory.Alloc(B.Size())
		if err != nil {
			return err
		}
		defer temp.Free()

		mCol := C.int(N)
		nCol := C.int(M)
		lda := C.int(M)
		if side == "Right" {
			lda = C.int(N)
		}
		ldb := C.int(N)
		ldc := C.int(N)
		alphaC := C.float(alpha)

		status := C.cutlassStrmmWrapper(
			nativeHandle,
			cutlassCublasSide(side),
			cutlassCublasFill(uplo),
			cutlassCublasOperation(trans),
			cutlassCublasDiag(diag),
			mCol,
			nCol,
			&alphaC,
			(*C.float)(A.Ptr()),
			lda,
			(*C.float)(B.Ptr()),
			ldb,
			(*C.float)(temp.Ptr()),
			ldc,
		)
		if status != C.CUBLAS_STATUS_SUCCESS {
			return cutlassCublasError("cublasStrmm", status)
		}
		return memory.CopyDeviceToDevice(B, temp)
	})
}

func executeNativeCutlassSpmm(sparseA, denseB, denseC *memory.Memory, M, N, K int) error {
	if sparseA == nil || denseB == nil || denseC == nil {
		return errCUTLASSUnsupported
	}
	return withCutlassCublasHandle(func(nativeHandle C.cublasHandle_t) error {
		return cutlassRunRowMajorSgemm(nativeHandle, M, N, K, sparseA, denseB, denseC, 1, 0, "cublasSgemm(spmm)")
	})
}

func executeNativeCutlassConv(desc CutlassConvDesc, input, filter, output *memory.Memory, outputH, outputW int) error {
	if desc.DataType != CutlassFloat32 {
		return errCUTLASSUnsupported
	}
	patchSize := desc.R * desc.S * desc.C
	rows := desc.N * outputH * outputW
	if patchSize <= 0 || rows <= 0 || desc.K <= 0 {
		return errCUTLASSUnsupported
	}

	switch desc.Mode {
	case CutlassConvForward:
		inputValues, err := readMathFloat32Memory(input, desc.N*desc.H*desc.W*desc.C)
		if err != nil {
			return err
		}
		filterValues, err := readMathFloat32Memory(filter, desc.K*patchSize)
		if err != nil {
			return err
		}
		inputCols := cutlassConvIm2Col(desc, inputValues, outputH, outputW)
		filterCols := cutlassTransposeMatrix(filterValues, desc.K, patchSize)

		inputColsMem, err := cutlassAllocAndWriteFloat32(inputCols)
		if err != nil {
			return err
		}
		defer inputColsMem.Free()
		filterColsMem, err := cutlassAllocAndWriteFloat32(filterCols)
		if err != nil {
			return err
		}
		defer filterColsMem.Free()

		return withCutlassCublasHandle(func(nativeHandle C.cublasHandle_t) error {
			return cutlassRunRowMajorSgemm(nativeHandle, rows, desc.K, patchSize, inputColsMem, filterColsMem, output, 1, 0, "cublasSgemm(conv_forward)")
		})

	case CutlassConvDgrad:
		gradOutValues, err := readMathFloat32Memory(input, rows*desc.K)
		if err != nil {
			return err
		}
		filterValues, err := readMathFloat32Memory(filter, desc.K*patchSize)
		if err != nil {
			return err
		}

		gradOutMem, err := cutlassAllocAndWriteFloat32(gradOutValues)
		if err != nil {
			return err
		}
		defer gradOutMem.Free()
		filterMem, err := cutlassAllocAndWriteFloat32(filterValues)
		if err != nil {
			return err
		}
		defer filterMem.Free()
		colGradMem, err := memory.Alloc(int64(rows * patchSize * 4))
		if err != nil {
			return err
		}
		defer colGradMem.Free()

		if err := withCutlassCublasHandle(func(nativeHandle C.cublasHandle_t) error {
			return cutlassRunRowMajorSgemm(nativeHandle, rows, patchSize, desc.K, gradOutMem, filterMem, colGradMem, 1, 0, "cublasSgemm(conv_dgrad)")
		}); err != nil {
			return err
		}

		colGradValues, err := readMathFloat32Memory(colGradMem, rows*patchSize)
		if err != nil {
			return err
		}
		gradIn := cutlassConvCol2Im(desc, colGradValues, outputH, outputW)
		return writeMathFloat32Memory(output, gradIn)

	case CutlassConvWgrad:
		inputValues, err := readMathFloat32Memory(input, desc.N*desc.H*desc.W*desc.C)
		if err != nil {
			return err
		}
		gradOutValues, err := readMathFloat32Memory(filter, rows*desc.K)
		if err != nil {
			return err
		}

		inputCols := cutlassConvIm2Col(desc, inputValues, outputH, outputW)
		gradOutT := cutlassTransposeMatrix(gradOutValues, rows, desc.K)

		gradOutTMem, err := cutlassAllocAndWriteFloat32(gradOutT)
		if err != nil {
			return err
		}
		defer gradOutTMem.Free()
		inputColsMem, err := cutlassAllocAndWriteFloat32(inputCols)
		if err != nil {
			return err
		}
		defer inputColsMem.Free()

		return withCutlassCublasHandle(func(nativeHandle C.cublasHandle_t) error {
			return cutlassRunRowMajorSgemm(nativeHandle, desc.K, patchSize, rows, gradOutTMem, inputColsMem, output, 1, 0, "cublasSgemm(conv_wgrad)")
		})

	default:
		return errCUTLASSUnsupported
	}
}

func destroyNativeCutlassGemm(handle *CutlassGemmHandle) error {
	if handle.nativeHandle != 0 {
		if status := C.destroyCublasHandleWrapper(cublasHandleFromUintptr(handle.nativeHandle)); status != C.CUBLAS_STATUS_SUCCESS {
			return cutlassCublasError("cublasDestroy", status)
		}
		handle.nativeHandle = 0
	}
	handle.native = false
	if handle.handle != nil {
		_ = handle.handle.Free()
		handle.handle = nil
	}
	if handle.workspace != nil {
		_ = handle.workspace.Free()
		handle.workspace = nil
	}
	return nil
}

func cutlassCublasError(name string, status C.cublasStatus_t) error {
	return fmt.Errorf("%s failed with status %d", name, int(status))
}

func withCutlassCublasHandle(fn func(C.cublasHandle_t) error) (err error) {
	var nativeHandle C.cublasHandle_t
	if status := C.createCublasHandleWrapper(&nativeHandle); status != C.CUBLAS_STATUS_SUCCESS {
		return cutlassCublasError("cublasCreate", status)
	}
	defer func() {
		if status := C.destroyCublasHandleWrapper(nativeHandle); status != C.CUBLAS_STATUS_SUCCESS && err == nil {
			err = cutlassCublasError("cublasDestroy", status)
		}
	}()
	return fn(nativeHandle)
}

func cutlassRunRowMajorSgemm(nativeHandle C.cublasHandle_t, m, n, k int, left, right, output *memory.Memory, alpha, beta float32, opName string) error {
	alphaC := C.float(alpha)
	betaC := C.float(beta)
	status := C.cublasSgemm(
		nativeHandle,
		C.CUBLAS_OP_N,
		C.CUBLAS_OP_N,
		C.int(n),
		C.int(m),
		C.int(k),
		&alphaC,
		(*C.float)(right.Ptr()),
		C.int(n),
		(*C.float)(left.Ptr()),
		C.int(k),
		&betaC,
		(*C.float)(output.Ptr()),
		C.int(n),
	)
	if status != C.CUBLAS_STATUS_SUCCESS {
		return cutlassCublasError(opName, status)
	}
	return nil
}

func cutlassAllocAndWriteFloat32(values []float32) (*memory.Memory, error) {
	mem, err := memory.Alloc(int64(len(values) * 4))
	if err != nil {
		return nil, err
	}
	if err := writeMathFloat32Memory(mem, values); err != nil {
		_ = mem.Free()
		return nil, err
	}
	return mem, nil
}

func cutlassConvIm2Col(desc CutlassConvDesc, inputValues []float32, outputH, outputW int) []float32 {
	patchSize := desc.R * desc.S * desc.C
	rows := desc.N * outputH * outputW
	cols := make([]float32, rows*patchSize)
	rowIndex := 0
	for n := 0; n < desc.N; n++ {
		for oh := 0; oh < outputH; oh++ {
			for ow := 0; ow < outputW; ow++ {
				colIndex := 0
				for r := 0; r < desc.R; r++ {
					for s := 0; s < desc.S; s++ {
						for c := 0; c < desc.C; c++ {
							ih := oh*desc.StrideH - desc.PadH + r*maxInt(desc.DilationH, 1)
							iw := ow*desc.StrideW - desc.PadW + s*maxInt(desc.DilationW, 1)
							if ih >= 0 && ih < desc.H && iw >= 0 && iw < desc.W {
								inputIndex := (((n*desc.H)+ih)*desc.W+iw)*desc.C + c
								cols[rowIndex*patchSize+colIndex] = inputValues[inputIndex]
							}
							colIndex++
						}
					}
				}
				rowIndex++
			}
		}
	}
	return cols
}

func cutlassConvCol2Im(desc CutlassConvDesc, colValues []float32, outputH, outputW int) []float32 {
	patchSize := desc.R * desc.S * desc.C
	gradIn := make([]float32, desc.N*desc.H*desc.W*desc.C)
	rowIndex := 0
	for n := 0; n < desc.N; n++ {
		for oh := 0; oh < outputH; oh++ {
			for ow := 0; ow < outputW; ow++ {
				colIndex := 0
				for r := 0; r < desc.R; r++ {
					for s := 0; s < desc.S; s++ {
						for c := 0; c < desc.C; c++ {
							ih := oh*desc.StrideH - desc.PadH + r*maxInt(desc.DilationH, 1)
							iw := ow*desc.StrideW - desc.PadW + s*maxInt(desc.DilationW, 1)
							if ih >= 0 && ih < desc.H && iw >= 0 && iw < desc.W {
								gradIndex := (((n*desc.H)+ih)*desc.W+iw)*desc.C + c
								gradIn[gradIndex] += colValues[rowIndex*patchSize+colIndex]
							}
							colIndex++
						}
					}
				}
				rowIndex++
			}
		}
	}
	return gradIn
}

func cutlassTransposeMatrix(values []float32, rows, cols int) []float32 {
	transposed := make([]float32, len(values))
	for row := 0; row < rows; row++ {
		for col := 0; col < cols; col++ {
			transposed[col*rows+row] = values[row*cols+col]
		}
	}
	return transposed
}

func cutlassCublasSide(side string) C.cublasSideMode_t {
	if side == "Left" {
		return C.CUBLAS_SIDE_RIGHT
	}
	return C.CUBLAS_SIDE_LEFT
}

func cutlassCublasFill(uplo string) C.cublasFillMode_t {
	if uplo == "Upper" {
		return C.CUBLAS_FILL_MODE_LOWER
	}
	return C.CUBLAS_FILL_MODE_UPPER
}

func cutlassCublasOperation(trans string) C.cublasOperation_t {
	switch trans {
	case "Trans", "ConjTrans":
		return C.CUBLAS_OP_T
	default:
		return C.CUBLAS_OP_N
	}
}

func cutlassCublasDiag(diag string) C.cublasDiagType_t {
	if diag == "Unit" {
		return C.CUBLAS_DIAG_UNIT
	}
	return C.CUBLAS_DIAG_NON_UNIT
}

var _ unsafe.Pointer
