//go:build cuda

package libraries

/*
#cgo linux CFLAGS: -I/usr/local/cuda/include -I/opt/cuda/include -I/usr/local/include -I/usr/local/cudss/include -I/opt/cudss/include
#cgo windows CFLAGS: -I"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.1/include" -I"D:/NVIDIA/include"
#cgo linux LDFLAGS: -L/usr/local/lib -L/usr/local/cudss/lib -L/opt/cudss/lib -lcudss -lcublas
#cgo windows LDFLAGS: -L${SRCDIR}/../lib_mingw -lcudss -lcublas

#include <cuda_runtime.h>
#include <cudss.h>
*/
import "C"

import (
	"fmt"
	"unsafe"

	"github.com/stitch1968/gocuda/memory"
)

func cudssNativeAvailable() bool {
	return true
}

func createNativeDSSHandle(config DSSConfig) (*DSSHandle, error) {
	var handle C.cudssHandle_t
	if status := C.cudssCreate(&handle); status != C.CUDSS_STATUS_SUCCESS {
		return nil, cudssError("cudssCreate", status)
	}

	var nativeConfig C.cudssConfig_t
	if status := C.cudssConfigCreate(&nativeConfig); status != C.CUDSS_STATUS_SUCCESS {
		_ = C.cudssDestroy(handle)
		return nil, cudssError("cudssConfigCreate", status)
	}

	var nativeData C.cudssData_t
	if status := C.cudssDataCreate(handle, &nativeData); status != C.CUDSS_STATUS_SUCCESS {
		_ = C.cudssConfigDestroy(nativeConfig)
		_ = C.cudssDestroy(handle)
		return nil, cudssError("cudssDataCreate", status)
	}

	if err := configureNativeDSSHandle(nativeConfig, config); err != nil {
		_ = C.cudssDataDestroy(handle, nativeData)
		_ = C.cudssConfigDestroy(nativeConfig)
		_ = C.cudssDestroy(handle)
		return nil, err
	}

	return &DSSHandle{
		config:       config,
		nativeHandle: uintptr(handle),
		nativeConfig: uintptr(nativeConfig),
		nativeData:   uintptr(nativeData),
		native:       true,
	}, nil
}

func createNativeDSSMatrix(n, nnz int, rowPtr, colInd, values *memory.Memory, format DSSMatrixFormat, symmetry bool) (*DSSMatrix, error) {
	if format != DSSMatrixFormatCSR {
		return createDeterministicDSSMatrix(n, nnz, rowPtr, colInd, values, format, symmetry)
	}

	matrixType := nativeDSSMatrixType(symmetry)
	rowEnd := unsafe.Add(rowPtr.Ptr(), unsafe.Sizeof(int32(0)))
	var matrix C.cudssMatrix_t
	if status := C.cudssMatrixCreateCsr(
		&matrix,
		C.int64_t(n),
		C.int64_t(n),
		C.int64_t(nnz),
		rowPtr.Ptr(),
		rowEnd,
		colInd.Ptr(),
		values.Ptr(),
		C.CUDA_R_32I,
		C.CUDA_R_32F,
		matrixType,
		C.CUDSS_MVIEW_FULL,
		C.CUDSS_BASE_ZERO,
	); status != C.CUDSS_STATUS_SUCCESS {
		return nil, cudssError("cudssMatrixCreateCsr", status)
	}

	return &DSSMatrix{
		rowPtr:       rowPtr,
		colInd:       colInd,
		values:       values,
		n:            n,
		nnz:          nnz,
		format:       format,
		symmetry:     symmetry,
		nativeHandle: uintptr(matrix),
		nativeType:   int(matrixType),
		native:       true,
	}, nil
}

func analyzeNativeDSS(handle *DSSHandle, matrix *DSSMatrix) error {
	if matrix.format != DSSMatrixFormatCSR {
		return errCUDSSUnsupported
	}
	if status := C.cudssExecute(
		C.cudssHandle_t(handle.nativeHandle),
		C.int(C.CUDSS_PHASE_ANALYSIS),
		C.cudssConfig_t(handle.nativeConfig),
		C.cudssData_t(handle.nativeData),
		C.cudssMatrix_t(matrix.nativeHandle),
		nil,
		nil,
	); status != C.CUDSS_STATUS_SUCCESS {
		return cudssError("cudssExecute(analysis)", status)
	}
	handle.n = matrix.n
	handle.nnz = matrix.nnz
	handle.matrix = matrix
	return nil
}

func factorNativeDSS(handle *DSSHandle, matrix *DSSMatrix) error {
	if matrix.format != DSSMatrixFormatCSR {
		return errCUDSSUnsupported
	}
	if status := C.cudssExecute(
		C.cudssHandle_t(handle.nativeHandle),
		C.int(C.CUDSS_PHASE_FACTORIZATION),
		C.cudssConfig_t(handle.nativeConfig),
		C.cudssData_t(handle.nativeData),
		C.cudssMatrix_t(matrix.nativeHandle),
		nil,
		nil,
	); status != C.CUDSS_STATUS_SUCCESS {
		return cudssError("cudssExecute(factorization)", status)
	}
	handle.matrix = matrix
	return nil
}

func solveNativeDSS(handle *DSSHandle, b, x *memory.Memory, nrhs int) (*DSSSolutionInfo, error) {
	if handle.matrix == nil || !handle.matrix.native {
		return nil, errCUDSSUnsupported
	}
	if nrhs <= 0 {
		return nil, fmt.Errorf("number of right-hand sides must be positive: %d", nrhs)
	}

	rhsMatrix, err := createNativeDSSDenseMatrix(handle.n, nrhs, b)
	if err != nil {
		return nil, err
	}
	defer C.cudssMatrixDestroy(rhsMatrix)

	solutionMatrix, err := createNativeDSSDenseMatrix(handle.n, nrhs, x)
	if err != nil {
		return nil, err
	}
	defer C.cudssMatrixDestroy(solutionMatrix)

	if status := C.cudssExecute(
		C.cudssHandle_t(handle.nativeHandle),
		C.int(C.CUDSS_PHASE_SOLVE),
		C.cudssConfig_t(handle.nativeConfig),
		C.cudssData_t(handle.nativeData),
		C.cudssMatrix_t(handle.matrix.nativeHandle),
		solutionMatrix,
		rhsMatrix,
	); status != C.CUDSS_STATUS_SUCCESS {
		return nil, cudssError("cudssExecute(solve)", status)
	}

	xValues, err := readMathFloat32Memory(x, handle.n*nrhs)
	if err != nil {
		return nil, err
	}
	bValues, err := readMathFloat32Memory(b, handle.n*nrhs)
	if err != nil {
		return nil, err
	}
	maxResidual := 0.0
	for rhs := 0; rhs < nrhs; rhs++ {
		solution := make([]float64, handle.n)
		rhsValues := make([]float64, handle.n)
		for index := 0; index < handle.n; index++ {
			solution[index] = float64(xValues[rhs*handle.n+index])
			rhsValues[index] = float64(bValues[rhs*handle.n+index])
		}
		residual := dssResidual(handle.dense, handle.n, solution, rhsValues)
		if residual > maxResidual {
			maxResidual = residual
		}
	}

	return &DSSSolutionInfo{
		Iterations:         1,
		Residual:           maxResidual,
		Error:              maxResidual,
		Determinant:        handle.det,
		PivotGrowth:        1.0,
		ConditionNumber:    0,
		BackwardError:      maxResidual,
		ComponentwiseError: maxResidual,
	}, nil
}

func refactorNativeDSS(handle *DSSHandle, matrix *DSSMatrix) error {
	if handle.matrix == nil || !matrix.native {
		return errCUDSSUnsupported
	}
	rowEnd := unsafe.Add(matrix.rowPtr.Ptr(), unsafe.Sizeof(int32(0)))
	if status := C.cudssMatrixSetCsrPointers(C.cudssMatrix_t(matrix.nativeHandle), matrix.rowPtr.Ptr(), rowEnd, matrix.colInd.Ptr(), matrix.values.Ptr()); status != C.CUDSS_STATUS_SUCCESS {
		return cudssError("cudssMatrixSetCsrPointers", status)
	}
	handle.matrix = matrix
	dense, err := dssDenseFromMatrix(matrix)
	if err != nil {
		return err
	}
	handle.dense = dense
	handle.det, _ = dssDeterminant(dense, matrix.n)
	if status := C.cudssExecute(
		C.cudssHandle_t(handle.nativeHandle),
		C.int(C.CUDSS_PHASE_FACTORIZATION),
		C.cudssConfig_t(handle.nativeConfig),
		C.cudssData_t(handle.nativeData),
		C.cudssMatrix_t(matrix.nativeHandle),
		nil,
		nil,
	); status != C.CUDSS_STATUS_SUCCESS {
		return cudssError("cudssExecute(refactorization)", status)
	}
	return nil
}

func destroyNativeDSSHandle(handle *DSSHandle) error {
	if handle.nativeData != 0 {
		if status := C.cudssDataDestroy(C.cudssHandle_t(handle.nativeHandle), C.cudssData_t(handle.nativeData)); status != C.CUDSS_STATUS_SUCCESS {
			return cudssError("cudssDataDestroy", status)
		}
		handle.nativeData = 0
	}
	if handle.nativeConfig != 0 {
		if status := C.cudssConfigDestroy(C.cudssConfig_t(handle.nativeConfig)); status != C.CUDSS_STATUS_SUCCESS {
			return cudssError("cudssConfigDestroy", status)
		}
		handle.nativeConfig = 0
	}
	if handle.nativeHandle != 0 {
		if status := C.cudssDestroy(C.cudssHandle_t(handle.nativeHandle)); status != C.CUDSS_STATUS_SUCCESS {
			return cudssError("cudssDestroy", status)
		}
		handle.nativeHandle = 0
	}
	handle.native = false
	handle.factored = false
	return nil
}

func destroyNativeDSSMatrix(matrix *DSSMatrix) error {
	if matrix.nativeHandle != 0 {
		if status := C.cudssMatrixDestroy(C.cudssMatrix_t(matrix.nativeHandle)); status != C.CUDSS_STATUS_SUCCESS {
			return cudssError("cudssMatrixDestroy", status)
		}
		matrix.nativeHandle = 0
	}
	matrix.native = false
	return nil
}

func configureNativeDSSHandle(nativeConfig C.cudssConfig_t, config DSSConfig) error {
	if config.Deterministic {
		enabled := C.int(1)
		if status := C.cudssConfigSet(nativeConfig, C.CUDSS_CONFIG_DETERMINISTIC_MODE, unsafe.Pointer(&enabled), C.size_t(unsafe.Sizeof(enabled))); status != C.CUDSS_STATUS_SUCCESS {
			return cudssError("cudssConfigSet(CUDSS_CONFIG_DETERMINISTIC_MODE)", status)
		}
	}
	return nil
}

func createNativeDSSDenseMatrix(rows, cols int, values *memory.Memory) (C.cudssMatrix_t, error) {
	var matrix C.cudssMatrix_t
	if status := C.cudssMatrixCreateDn(&matrix, C.int64_t(rows), C.int64_t(cols), C.int64_t(rows), values.Ptr(), C.CUDA_R_32F, C.CUDSS_LAYOUT_COL_MAJOR); status != C.CUDSS_STATUS_SUCCESS {
		return nil, cudssError("cudssMatrixCreateDn", status)
	}
	return matrix, nil
}

func nativeDSSMatrixType(symmetry bool) C.cudssMatrixType_t {
	if symmetry {
		return C.CUDSS_MTYPE_SPD
	}
	return C.CUDSS_MTYPE_GENERAL
}

func cudssError(operation string, status C.cudssStatus_t) error {
	return fmt.Errorf("%s failed: %s (%d)", operation, cudssStatusString(status), int(status))
}

func cudssStatusString(status C.cudssStatus_t) string {
	switch status {
	case C.CUDSS_STATUS_SUCCESS:
		return "success"
	case C.CUDSS_STATUS_NOT_INITIALIZED:
		return "not initialized"
	case C.CUDSS_STATUS_ALLOC_FAILED:
		return "allocation failed"
	case C.CUDSS_STATUS_INVALID_VALUE:
		return "invalid value"
	case C.CUDSS_STATUS_NOT_SUPPORTED:
		return "not supported"
	case C.CUDSS_STATUS_EXECUTION_FAILED:
		return "execution failed"
	case C.CUDSS_STATUS_INTERNAL_ERROR:
		return "internal error"
	default:
		return "unknown error"
	}
}
