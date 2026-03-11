//go:build cuda

package libraries

/*
#cgo linux CFLAGS: -I/usr/local/cuda/include -I/opt/cuda/include
#cgo windows CFLAGS: -I"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.1/include" -I"D:/NVIDIA/include"
#cgo linux LDFLAGS: -L/usr/local/cuda/lib64 -L/opt/cuda/lib64 -lcusparse
#cgo windows LDFLAGS: -L${SRCDIR}/../lib_mingw -lcusparse

#include <cusparse.h>

static cusparseStatus_t createSparseHandle(cusparseHandle_t* handle) {
	return cusparseCreate(handle);
}

static cusparseStatus_t destroySparseHandle(cusparseHandle_t handle) {
	return cusparseDestroy(handle);
}
*/
import "C"

import (
	"fmt"
	"unsafe"

	"github.com/stitch1968/gocuda/memory"
)

func createNativeSparseContext() (*SparseContext, error) {
	var handle C.cusparseHandle_t
	if status := C.createSparseHandle(&handle); status != C.CUSPARSE_STATUS_SUCCESS {
		return nil, cusparseError("cusparseCreate", status)
	}
	return &SparseContext{handle: unsafe.Pointer(handle), native: true}, nil
}

func nativeSpMVCSR(ctx *SparseContext, alpha float32, A *SparseMatrix, x *memory.Memory, beta float32, y *memory.Memory) error {
	var matA C.cusparseSpMatDescr_t
	status := C.cusparseCreateCsr(
		&matA,
		C.int64_t(A.rows),
		C.int64_t(A.cols),
		C.int64_t(A.nnz),
		A.rowPtrs.Ptr(),
		A.colInds.Ptr(),
		A.values.Ptr(),
		C.CUSPARSE_INDEX_32I,
		C.CUSPARSE_INDEX_32I,
		C.CUSPARSE_INDEX_BASE_ZERO,
		C.CUDA_R_32F,
	)
	if status != C.CUSPARSE_STATUS_SUCCESS {
		return cusparseError("cusparseCreateCsr", status)
	}
	defer C.cusparseDestroySpMat(matA)

	var vecX C.cusparseDnVecDescr_t
	status = C.cusparseCreateDnVec(&vecX, C.int64_t(A.cols), x.Ptr(), C.CUDA_R_32F)
	if status != C.CUSPARSE_STATUS_SUCCESS {
		return cusparseError("cusparseCreateDnVec(x)", status)
	}
	defer C.cusparseDestroyDnVec(vecX)

	var vecY C.cusparseDnVecDescr_t
	status = C.cusparseCreateDnVec(&vecY, C.int64_t(A.rows), y.Ptr(), C.CUDA_R_32F)
	if status != C.CUSPARSE_STATUS_SUCCESS {
		return cusparseError("cusparseCreateDnVec(y)", status)
	}
	defer C.cusparseDestroyDnVec(vecY)

	alphaValue := C.float(alpha)
	betaValue := C.float(beta)
	var bufferSize C.size_t
	status = C.cusparseSpMV_bufferSize(
		C.cusparseHandle_t(ctx.handle),
		C.CUSPARSE_OPERATION_NON_TRANSPOSE,
		unsafePointer(&alphaValue),
		matA,
		vecX,
		unsafePointer(&betaValue),
		vecY,
		C.CUDA_R_32F,
		C.CUSPARSE_SPMV_ALG_DEFAULT,
		&bufferSize,
	)
	if status != C.CUSPARSE_STATUS_SUCCESS {
		return cusparseError("cusparseSpMV_bufferSize", status)
	}

	var buffer *memory.Memory
	if bufferSize > 0 {
		var err error
		buffer, err = memory.Alloc(int64(bufferSize))
		if err != nil {
			return fmt.Errorf("failed to allocate cuSPARSE buffer: %v", err)
		}
		defer buffer.Free()
	}

	status = C.cusparseSpMV(
		C.cusparseHandle_t(ctx.handle),
		C.CUSPARSE_OPERATION_NON_TRANSPOSE,
		unsafePointer(&alphaValue),
		matA,
		vecX,
		unsafePointer(&betaValue),
		vecY,
		C.CUDA_R_32F,
		C.CUSPARSE_SPMV_ALG_DEFAULT,
		memoryPtrOrNil(buffer),
	)
	if status != C.CUSPARSE_STATUS_SUCCESS {
		return cusparseError("cusparseSpMV", status)
	}
	return nil
}

func nativeSpGEMM(ctx *SparseContext, A, B *SparseMatrix) (*SparseMatrix, error) {
	denseA, err := sparseToDenseHost(A)
	if err != nil {
		return nil, err
	}
	defer denseA.Free()
	denseB, err := sparseToDenseHost(B)
	if err != nil {
		return nil, err
	}
	defer denseB.Free()
	denseC, err := memory.Alloc(int64(A.rows * B.cols * 4))
	if err != nil {
		return nil, err
	}
	defer denseC.Free()
	if err := multiplyDenseMatrices(denseA, denseB, denseC, A.rows, A.cols, B.cols); err != nil {
		return nil, err
	}
	return denseToSparseFromHost(denseC, A.rows, B.cols, MatrixFormatCSR)
}

func nativeSparseLU(ctx *SparseContext, A *SparseMatrix) (*SparseMatrix, *SparseMatrix, error) {
	denseA, err := sparseToDenseHost(A)
	if err != nil {
		return nil, nil, err
	}
	defer denseA.Free()
	solverCtx, err := createNativeSolverContext()
	if err != nil {
		return nil, nil, err
	}
	defer destroyNativeSolverContext(solverCtx)
	luInfo, err := nativeLUFactorization(solverCtx, denseA, A.rows, A.cols)
	if err != nil {
		return nil, nil, err
	}
	defer luInfo.Destroy()
	return splitDenseLU(denseA, A.rows)
}

func nativeSparseSolve(ctx *SparseContext, A *SparseMatrix, b, x *memory.Memory) error {
	denseA, err := sparseToDenseHost(A)
	if err != nil {
		return err
	}
	defer denseA.Free()
	solverCtx, err := createNativeSolverContext()
	if err != nil {
		return err
	}
	defer destroyNativeSolverContext(solverCtx)
	solution, err := nativeSolveLinearSystem(solverCtx, denseA, b, A.rows)
	if err != nil {
		return err
	}
	defer solution.Free()
	src, err := memory.View[float32](solution, A.rows)
	if err != nil {
		return err
	}
	dst, err := memory.View[float32](x, A.rows)
	if err != nil {
		return err
	}
	copy(dst, src)
	return nil
}

func destroyNativeSparseContext(ctx *SparseContext) error {
	status := C.destroySparseHandle(C.cusparseHandle_t(ctx.handle))
	if status != C.CUSPARSE_STATUS_SUCCESS {
		return cusparseError("cusparseDestroy", status)
	}
	return nil
}

func multiplyDenseMatrices(A, B, C *memory.Memory, rowsA, colsA, colsB int) error {
	aValues, err := memory.View[float32](A, rowsA*colsA)
	if err != nil {
		return err
	}
	bValues, err := memory.View[float32](B, colsA*colsB)
	if err != nil {
		return err
	}
	cValues, err := memory.View[float32](C, rowsA*colsB)
	if err != nil {
		return err
	}
	for index := range cValues {
		cValues[index] = 0
	}
	for row := 0; row < rowsA; row++ {
		for col := 0; col < colsB; col++ {
			accum := float32(0)
			for inner := 0; inner < colsA; inner++ {
				accum += aValues[row*colsA+inner] * bValues[inner*colsB+col]
			}
			cValues[row*colsB+col] = accum
		}
	}
	return nil
}

func splitDenseLU(denseLU *memory.Memory, n int) (*SparseMatrix, *SparseMatrix, error) {
	values, err := memory.View[float32](denseLU, n*n)
	if err != nil {
		return nil, nil, err
	}
	lDense, err := memory.Alloc(int64(n * n * 4))
	if err != nil {
		return nil, nil, err
	}
	defer lDense.Free()
	uDense, err := memory.Alloc(int64(n * n * 4))
	if err != nil {
		return nil, nil, err
	}
	defer uDense.Free()
	lValues, err := memory.View[float32](lDense, n*n)
	if err != nil {
		return nil, nil, err
	}
	uValues, err := memory.View[float32](uDense, n*n)
	if err != nil {
		return nil, nil, err
	}
	for row := 0; row < n; row++ {
		for col := 0; col < n; col++ {
			value := values[row*n+col]
			switch {
			case row > col:
				lValues[row*n+col] = value
				uValues[row*n+col] = 0
			case row == col:
				lValues[row*n+col] = 1
				uValues[row*n+col] = value
			default:
				lValues[row*n+col] = 0
				uValues[row*n+col] = value
			}
		}
	}
	L, err := denseToSparseFromHost(lDense, n, n, MatrixFormatCSR)
	if err != nil {
		return nil, nil, err
	}
	U, err := denseToSparseFromHost(uDense, n, n, MatrixFormatCSR)
	if err != nil {
		_ = L.Destroy()
		return nil, nil, err
	}
	return L, U, nil
}

func memoryPtrOrNil(mem *memory.Memory) unsafe.Pointer {
	if mem == nil {
		return nil
	}
	return mem.Ptr()
}

func unsafePointer[T any](value *T) unsafe.Pointer {
	return unsafe.Pointer(value)
}

func cusparseError(operation string, status C.cusparseStatus_t) error {
	return fmt.Errorf("%s failed: %s (%d)", operation, cusparseStatusString(status), int(status))
}

func cusparseStatusString(status C.cusparseStatus_t) string {
	switch status {
	case C.CUSPARSE_STATUS_SUCCESS:
		return "success"
	case C.CUSPARSE_STATUS_NOT_INITIALIZED:
		return "not initialized"
	case C.CUSPARSE_STATUS_ALLOC_FAILED:
		return "allocation failed"
	case C.CUSPARSE_STATUS_INVALID_VALUE:
		return "invalid value"
	case C.CUSPARSE_STATUS_ARCH_MISMATCH:
		return "architecture mismatch"
	case C.CUSPARSE_STATUS_MAPPING_ERROR:
		return "mapping error"
	case C.CUSPARSE_STATUS_EXECUTION_FAILED:
		return "execution failed"
	case C.CUSPARSE_STATUS_INTERNAL_ERROR:
		return "internal error"
	case C.CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
		return "matrix type not supported"
	case C.CUSPARSE_STATUS_ZERO_PIVOT:
		return "zero pivot"
	case C.CUSPARSE_STATUS_NOT_SUPPORTED:
		return "not supported"
	default:
		return "unknown error"
	}
}
