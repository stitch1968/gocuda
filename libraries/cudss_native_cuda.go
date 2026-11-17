//go:build cuda

package libraries

/*
#cgo linux CFLAGS: -I/usr/local/cuda/include -I/opt/cuda/include -I/usr/local/include -I/usr/local/cudss/include -I/opt/cudss/include
#cgo windows CFLAGS: -I"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.1/include" -I"C:/Program Files/NVIDIA cuDSS/v0.7/include" -I"D:/NVIDIA/include"
#cgo linux LDFLAGS: -L/usr/local/lib -L/usr/local/cudss/lib -L/opt/cudss/lib -lcudss -lcublas
#cgo windows LDFLAGS: -L${SRCDIR}/../lib_mingw -lcudss -lcublas

#include <cuda_runtime.h>
#include <cudss.h>
*/
import "C"

import (
	"errors"
	"fmt"
	"unsafe"

	"github.com/stitch1968/gocuda/internal"
	"github.com/stitch1968/gocuda/memory"
)

type cudssNativeBuffer struct {
	mem    *memory.Memory
	target *memory.Memory
	values []float64
	width  int
	owns   bool
}

type cudssNativeSolveResources struct {
	rhsMatrix      C.cudssMatrix_t
	solutionMatrix C.cudssMatrix_t
	b              *cudssNativeBuffer
	x              *cudssNativeBuffer
}

func (resources *cudssNativeSolveResources) cleanup() error {
	var errs []error
	if resources.rhsMatrix != nil {
		if status := C.cudssMatrixDestroy(resources.rhsMatrix); status != C.CUDSS_STATUS_SUCCESS {
			errs = append(errs, cudssError("cudssMatrixDestroy(rhs)", status))
		} else {
			resources.rhsMatrix = nil
		}
	}
	if resources.solutionMatrix != nil {
		if status := C.cudssMatrixDestroy(resources.solutionMatrix); status != C.CUDSS_STATUS_SUCCESS {
			errs = append(errs, cudssError("cudssMatrixDestroy(solution)", status))
		} else {
			resources.solutionMatrix = nil
		}
	}
	if len(errs) != 0 {
		return errors.Join(errs...)
	}
	if err := resources.b.release(); err != nil {
		errs = append(errs, err)
	}
	if err := resources.x.release(); err != nil {
		errs = append(errs, err)
	}
	return errors.Join(errs...)
}

func cudssHandleToUintptr(handle C.cudssHandle_t) uintptr {
	return uintptr(unsafe.Pointer(handle))
}

func cudssHandleFromUintptr(handle uintptr) C.cudssHandle_t {
	return (C.cudssHandle_t)(unsafe.Pointer(handle))
}

func cudssConfigToUintptr(config C.cudssConfig_t) uintptr {
	return uintptr(unsafe.Pointer(config))
}

func cudssConfigFromUintptr(config uintptr) C.cudssConfig_t {
	return (C.cudssConfig_t)(unsafe.Pointer(config))
}

func cudssDataToUintptr(data C.cudssData_t) uintptr {
	return uintptr(unsafe.Pointer(data))
}

func cudssDataFromUintptr(data uintptr) C.cudssData_t {
	return (C.cudssData_t)(unsafe.Pointer(data))
}

func cudssMatrixToUintptr(matrix C.cudssMatrix_t) uintptr {
	return uintptr(unsafe.Pointer(matrix))
}

func cudssMatrixFromUintptr(matrix uintptr) C.cudssMatrix_t {
	return (C.cudssMatrix_t)(unsafe.Pointer(matrix))
}

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
		nativeHandle: cudssHandleToUintptr(handle),
		nativeConfig: cudssConfigToUintptr(nativeConfig),
		nativeData:   cudssDataToUintptr(nativeData),
		native:       true,
	}, nil
}

func createNativeDSSMatrix(n, nnz int, rowPtr, colInd, values *memory.Memory, format DSSMatrixFormat, symmetry bool) (*DSSMatrix, error) {
	if format != DSSMatrixFormatCSR {
		return createDeterministicDSSMatrix(n, nnz, rowPtr, colInd, values, format, symmetry)
	}

	nativeValues, err := cudssPrepareFloat64Buffer(values, nnz)
	if err != nil {
		return nil, err
	}
	if nativeValues.owns {
		defer func() {
			if nativeValues.mem != nil {
				_ = nativeValues.mem.Free()
			}
		}()
	}

	matrixType := nativeDSSMatrixType(symmetry)
	var matrix C.cudssMatrix_t
	if status := C.cudssMatrixCreateCsr(
		&matrix,
		C.int64_t(n),
		C.int64_t(n),
		C.int64_t(nnz),
		rowPtr.Ptr(),
		nil,
		colInd.Ptr(),
		nativeValues.mem.Ptr(),
		C.CUDA_R_32I,
		C.CUDA_R_64F,
		matrixType,
		C.CUDSS_MVIEW_FULL,
		C.CUDSS_BASE_ZERO,
	); status != C.CUDSS_STATUS_SUCCESS {
		return nil, cudssError("cudssMatrixCreateCsr", status)
	}
	if nativeValues.owns {
		defer func() {
			nativeValues.mem = nil
		}()
	}

	var ownedValues *memory.Memory
	if nativeValues.owns {
		ownedValues = nativeValues.mem
	}
	return &DSSMatrix{
		rowPtr:       rowPtr,
		colInd:       colInd,
		values:       values,
		nativeValues: ownedValues,
		n:            n,
		nnz:          nnz,
		format:       format,
		symmetry:     symmetry,
		nativeHandle: cudssMatrixToUintptr(matrix),
		nativeType:   int(matrixType),
		native:       true,
	}, nil
}

func refreshNativeDSSValues(matrix *DSSMatrix) error {
	if matrix.nativeValues == nil {
		return nil
	}
	values, err := readDSSFloatValues(matrix.values, matrix.nnz)
	if err != nil {
		return err
	}
	return writeMathFloat64Memory(matrix.nativeValues, values)
}

func analyzeNativeDSS(handle *DSSHandle, matrix *DSSMatrix) error {
	if matrix.format != DSSMatrixFormatCSR {
		return errCUDSSUnsupported
	}
	if status := C.cudssExecute(
		cudssHandleFromUintptr(handle.nativeHandle),
		C.int(C.CUDSS_PHASE_ANALYSIS),
		cudssConfigFromUintptr(handle.nativeConfig),
		cudssDataFromUintptr(handle.nativeData),
		cudssMatrixFromUintptr(matrix.nativeHandle),
		nil,
		nil,
	); status != C.CUDSS_STATUS_SUCCESS {
		return cudssError("cudssExecute(analysis)", status)
	}
	return nil
}

func factorNativeDSS(handle *DSSHandle, matrix *DSSMatrix) error {
	if matrix.format != DSSMatrixFormatCSR {
		return errCUDSSUnsupported
	}
	if status := C.cudssExecute(
		cudssHandleFromUintptr(handle.nativeHandle),
		C.int(C.CUDSS_PHASE_FACTORIZATION),
		cudssConfigFromUintptr(handle.nativeConfig),
		cudssDataFromUintptr(handle.nativeData),
		cudssMatrixFromUintptr(matrix.nativeHandle),
		nil,
		nil,
	); status != C.CUDSS_STATUS_SUCCESS {
		return cudssError("cudssExecute(factorization)", status)
	}
	return nil
}

func solveNativeDSS(handle *DSSHandle, b, x *memory.Memory, nrhs int, retained []*memory.Retention) (info *DSSSolutionInfo, deferred bool, err error) {
	if handle.matrix == nil || !handle.matrix.native {
		return nil, false, errCUDSSUnsupported
	}
	if nrhs <= 0 {
		return nil, false, fmt.Errorf("number of right-hand sides must be positive: %d", nrhs)
	}
	resources := &cudssNativeSolveResources{}
	defer func() {
		if deferred {
			return
		}
		if cleanupErr := resources.cleanup(); cleanupErr != nil {
			handle.pendingOperations = append(handle.pendingOperations, dssPendingOperation{cleanup: resources.cleanup, retentions: retained})
			deferred = true
			info = nil
			err = errors.Join(err, cleanupErr)
		}
	}()
	elementCount := handle.n * nrhs
	resources.b, err = cudssPrepareFloat64Buffer(b, elementCount)
	if err != nil {
		return nil, false, err
	}
	resources.x, err = cudssPrepareFloat64Buffer(x, elementCount)
	if err != nil {
		return nil, false, err
	}
	resources.rhsMatrix, err = createNativeDSSDenseMatrix(handle.n, nrhs, resources.b.mem)
	if err != nil {
		return nil, false, err
	}
	resources.solutionMatrix, err = createNativeDSSDenseMatrix(handle.n, nrhs, resources.x.mem)
	if err != nil {
		return nil, false, err
	}
	status := C.cudssExecute(
		cudssHandleFromUintptr(handle.nativeHandle),
		C.int(C.CUDSS_PHASE_SOLVE),
		cudssConfigFromUintptr(handle.nativeConfig),
		cudssDataFromUintptr(handle.nativeData),
		cudssMatrixFromUintptr(handle.matrix.nativeHandle),
		resources.solutionMatrix,
		resources.rhsMatrix,
	)
	var executeErr error
	if status != C.CUDSS_STATUS_SUCCESS {
		executeErr = cudssError("cudssExecute(solve)", status)
	}
	if syncErr := internal.CudaDefaultStreamSynchronizeOnDevice(handle.deviceID); syncErr != nil {
		handle.pendingOperations = append(handle.pendingOperations, dssPendingOperation{cleanup: resources.cleanup, retentions: retained})
		return nil, true, errors.Join(executeErr, syncErr)
	}
	if executeErr != nil {
		return nil, false, executeErr
	}

	xValues, err := readMathFloat64Memory(resources.x.mem, elementCount)
	if err != nil {
		return nil, false, err
	}
	if err := resources.x.writeBack(xValues); err != nil {
		return nil, false, err
	}
	if resources.x.width == 4 {
		for i, value := range xValues {
			xValues[i] = float64(float32(value))
		}
	}
	for rhs := 0; rhs < nrhs; rhs++ {
		start := rhs * handle.n
		current, err := dssSolutionDiagnostics(handle.dense, handle.n,
			xValues[start:start+handle.n], resources.b.values[start:start+handle.n], handle.det)
		if err != nil {
			return nil, false, err
		}
		info = mergeDSSDiagnostics(info, current)
	}
	return info, false, nil
}

func destroyNativeDSSHandle(handle *DSSHandle) error {
	if handle.nativeData != 0 {
		if status := C.cudssDataDestroy(cudssHandleFromUintptr(handle.nativeHandle), cudssDataFromUintptr(handle.nativeData)); status != C.CUDSS_STATUS_SUCCESS {
			return cudssError("cudssDataDestroy", status)
		}
		handle.nativeData = 0
	}
	if handle.nativeConfig != 0 {
		if status := C.cudssConfigDestroy(cudssConfigFromUintptr(handle.nativeConfig)); status != C.CUDSS_STATUS_SUCCESS {
			return cudssError("cudssConfigDestroy", status)
		}
		handle.nativeConfig = 0
	}
	if handle.nativeHandle != 0 {
		if status := C.cudssDestroy(cudssHandleFromUintptr(handle.nativeHandle)); status != C.CUDSS_STATUS_SUCCESS {
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
		if status := C.cudssMatrixDestroy(cudssMatrixFromUintptr(matrix.nativeHandle)); status != C.CUDSS_STATUS_SUCCESS {
			return cudssError("cudssMatrixDestroy", status)
		}
		matrix.nativeHandle = 0
	}
	if matrix.nativeValues != nil {
		if err := matrix.nativeValues.Free(); err != nil {
			return err
		}
		matrix.nativeValues = nil
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
	if status := C.cudssMatrixCreateDn(&matrix, C.int64_t(rows), C.int64_t(cols), C.int64_t(rows), values.Ptr(), C.CUDA_R_64F, C.CUDSS_LAYOUT_COL_MAJOR); status != C.CUDSS_STATUS_SUCCESS {
		return nil, cudssError("cudssMatrixCreateDn", status)
	}
	return matrix, nil
}

func cudssPrepareFloat64Buffer(mem *memory.Memory, length int) (*cudssNativeBuffer, error) {
	switch mem.Size() {
	case int64(length) * 4:
		values32, err := readMathFloat32Memory(mem, length)
		if err != nil {
			return nil, err
		}
		values64 := make([]float64, length)
		for index, value := range values32 {
			values64[index] = float64(value)
		}
		nativeMem, err := memory.Alloc(int64(length) * 8)
		if err != nil {
			return nil, err
		}
		if err := writeMathFloat64Memory(nativeMem, values64); err != nil {
			return nil, errors.Join(err, nativeMem.Free())
		}
		return &cudssNativeBuffer{mem: nativeMem, target: mem, values: values64, width: 4, owns: true}, nil
	case int64(length) * 8:
		values64, err := readMathFloat64Memory(mem, length)
		if err != nil {
			return nil, err
		}
		return &cudssNativeBuffer{mem: mem, target: mem, values: values64, width: 8}, nil
	default:
		return nil, fmt.Errorf("unsupported cuDSS buffer size %d for %d elements", mem.Size(), length)
	}
}

func (buffer *cudssNativeBuffer) release() error {
	if buffer != nil && buffer.owns && buffer.mem != nil {
		if err := buffer.mem.Free(); err != nil {
			return err
		}
		buffer.mem = nil
	}
	return nil
}

func (buffer *cudssNativeBuffer) writeBack(values []float64) error {
	if buffer == nil {
		return nil
	}
	switch buffer.width {
	case 4:
		values32 := make([]float32, len(values))
		for index, value := range values {
			values32[index] = float32(value)
		}
		return writeMathFloat32Memory(buffer.target, values32)
	case 8:
		return writeMathFloat64Memory(buffer.target, values)
	default:
		return fmt.Errorf("unsupported cuDSS write-back width %d", buffer.width)
	}
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
