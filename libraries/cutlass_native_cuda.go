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

var _ unsafe.Pointer
