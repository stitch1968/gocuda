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
	handle.nativeHandle = uintptr(nativeHandle)
	handle.native = true
	return handle, nil
}

func executeNativeCutlassGemm(handle *CutlassGemmHandle, A, B, C *memory.Memory) error {
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

	nativeHandle := C.cublasHandle_t(handle.nativeHandle)
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
			(*C.float)(C.Ptr()),
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
		(*C.double)(C.Ptr()),
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
			C.CUBLAS_OP_N,
			C.CUBLAS_OP_T,
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
			C.CUBLAS_OP_N,
			C.CUBLAS_OP_T,
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

func destroyNativeCutlassGemm(handle *CutlassGemmHandle) error {
	if handle.nativeHandle != 0 {
		if status := C.destroyCublasHandleWrapper(C.cublasHandle_t(handle.nativeHandle)); status != C.CUBLAS_STATUS_SUCCESS {
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
