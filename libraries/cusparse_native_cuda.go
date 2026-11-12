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

static cusparseStatus_t setSparseTriangle(cusparseSpMatDescr_t matrix, int lower) {
	cusparseFillMode_t fill = lower ? CUSPARSE_FILL_MODE_LOWER : CUSPARSE_FILL_MODE_UPPER;
	cusparseDiagType_t diagonal = CUSPARSE_DIAG_TYPE_NON_UNIT;
	cusparseStatus_t status = cusparseSpMatSetAttribute(matrix, CUSPARSE_SPMAT_FILL_MODE, &fill, sizeof(fill));
	if (status != CUSPARSE_STATUS_SUCCESS) return status;
	return cusparseSpMatSetAttribute(matrix, CUSPARSE_SPMAT_DIAG_TYPE, &diagonal, sizeof(diagonal));
}
*/
import "C"

import (
	"errors"
	"fmt"
	"runtime"
	"unsafe"

	"github.com/stitch1968/gocuda/internal"
	"github.com/stitch1968/gocuda/memory"
)

func createNativeSparseContext() (*SparseContext, error) {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	deviceID, err := internal.CurrentDevice()
	if err != nil {
		return nil, err
	}
	var handle C.cusparseHandle_t
	err = internal.RunOnDevice(deviceID, func() error {
		if status := C.createSparseHandle(&handle); status != C.CUSPARSE_STATUS_SUCCESS {
			return cusparseError("cusparseCreate", status)
		}
		return nil
	})
	if err != nil {
		if handle != nil {
			cleanupErr := internal.RunOnDevice(deviceID, func() error {
				if status := C.destroySparseHandle(handle); status != C.CUSPARSE_STATUS_SUCCESS {
					return cusparseError("cusparseDestroy", status)
				}
				return nil
			})
			return nil, errors.Join(err, cleanupErr)
		}
		return nil, err
	}
	return &SparseContext{handle: unsafe.Pointer(handle), native: true, deviceID: deviceID}, nil
}

func createNativeSparseMatrixDescriptor(A *SparseMatrix) (C.cusparseSpMatDescr_t, error) {
	var matA C.cusparseSpMatDescr_t
	var status C.cusparseStatus_t
	var descriptorName string
	switch A.format {
	case MatrixFormatCSR:
		descriptorName = "cusparseCreateCsr"
		status = C.cusparseCreateCsr(
			&matA, C.int64_t(A.rows), C.int64_t(A.cols), C.int64_t(A.nnz),
			A.rowPtrs.Ptr(), A.colInds.Ptr(), A.values.Ptr(),
			C.CUSPARSE_INDEX_32I, C.CUSPARSE_INDEX_32I, C.CUSPARSE_INDEX_BASE_ZERO, C.CUDA_R_32F,
		)
	case MatrixFormatCOO:
		descriptorName = "cusparseCreateCoo"
		status = C.cusparseCreateCoo(
			&matA, C.int64_t(A.rows), C.int64_t(A.cols), C.int64_t(A.nnz),
			A.rowPtrs.Ptr(), A.colInds.Ptr(), A.values.Ptr(),
			C.CUSPARSE_INDEX_32I, C.CUSPARSE_INDEX_BASE_ZERO, C.CUDA_R_32F,
		)
	default:
		return nil, fmt.Errorf("unsupported native sparse matrix format %d", A.format)
	}
	if status != C.CUSPARSE_STATUS_SUCCESS {
		return nil, cusparseError(descriptorName, status)
	}
	return matA, nil
}

func allocateNativeSparseWorkspace(size C.size_t) (*memory.Memory, error) {
	if size == 0 {
		return nil, nil
	}
	if uint64(size) > uint64(^uint(0)>>1) {
		return nil, fmt.Errorf("cuSPARSE workspace size exceeds supported range")
	}
	workspace, err := memory.Alloc(int64(size))
	if err != nil {
		return nil, fmt.Errorf("allocate cuSPARSE workspace: %w", err)
	}
	return workspace, nil
}

func (ctx *SparseContext) finishWorkspace(workspace *memory.Memory, operationErr *error) {
	if workspace == nil {
		return
	}
	if *operationErr != nil {
		ctx.pendingWorkspaces = append(ctx.pendingWorkspaces, workspace)
		return
	}
	if err := workspace.Free(); err != nil {
		ctx.pendingWorkspaces = append(ctx.pendingWorkspaces, workspace)
		*operationErr = errors.Join(*operationErr, err)
	}
}

func (ctx *SparseContext) finishNativeResource(operationErr *error, cleanup func() error) {
	if *operationErr != nil {
		ctx.pendingCleanups = append(ctx.pendingCleanups, cleanup)
		return
	}
	if err := cleanup(); err != nil {
		ctx.pendingCleanups = append(ctx.pendingCleanups, cleanup)
		*operationErr = errors.Join(*operationErr, err)
	}
}

func nativeSpMV(ctx *SparseContext, alpha float32, A *SparseMatrix, x *memory.Memory, beta float32, y *memory.Memory) (err error) {
	matA, err := createNativeSparseMatrixDescriptor(A)
	if err != nil {
		return err
	}
	defer func() {
		ctx.finishNativeResource(&err, func() error {
			if status := C.cusparseDestroySpMat(matA); status != C.CUSPARSE_STATUS_SUCCESS {
				return cusparseError("cusparseDestroySpMat", status)
			}
			return nil
		})
	}()

	var vecX C.cusparseDnVecDescr_t
	status := C.cusparseCreateDnVec(&vecX, C.int64_t(A.cols), x.Ptr(), C.CUDA_R_32F)
	if status != C.CUSPARSE_STATUS_SUCCESS {
		return cusparseError("cusparseCreateDnVec(x)", status)
	}
	defer func() {
		ctx.finishNativeResource(&err, func() error {
			if status := C.cusparseDestroyDnVec(vecX); status != C.CUSPARSE_STATUS_SUCCESS {
				return cusparseError("cusparseDestroyDnVec(x)", status)
			}
			return nil
		})
	}()

	var vecY C.cusparseDnVecDescr_t
	status = C.cusparseCreateDnVec(&vecY, C.int64_t(A.rows), y.Ptr(), C.CUDA_R_32F)
	if status != C.CUSPARSE_STATUS_SUCCESS {
		return cusparseError("cusparseCreateDnVec(y)", status)
	}
	defer func() {
		ctx.finishNativeResource(&err, func() error {
			if status := C.cusparseDestroyDnVec(vecY); status != C.CUSPARSE_STATUS_SUCCESS {
				return cusparseError("cusparseDestroyDnVec(y)", status)
			}
			return nil
		})
	}()

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

	buffer, err := allocateNativeSparseWorkspace(bufferSize)
	if err != nil {
		return err
	}
	defer func() { ctx.finishWorkspace(buffer, &err) }()

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
	return internal.CudaDefaultStreamSynchronizeOnDevice(ctx.deviceID)
}

func nativeSpSVCSR(ctx *SparseContext, A *SparseMatrix, b, x *memory.Memory, lower bool) (err error) {
	matA, err := createNativeSparseMatrixDescriptor(A)
	if err != nil {
		return err
	}
	defer func() {
		ctx.finishNativeResource(&err, func() error {
			if status := C.cusparseDestroySpMat(matA); status != C.CUSPARSE_STATUS_SUCCESS {
				return cusparseError("cusparseDestroySpMat", status)
			}
			return nil
		})
	}()
	lowerFlag := C.int(0)
	if lower {
		lowerFlag = 1
	}
	if status := C.setSparseTriangle(matA, lowerFlag); status != C.CUSPARSE_STATUS_SUCCESS {
		return cusparseError("cusparseSpMatSetAttribute", status)
	}
	var vecB C.cusparseDnVecDescr_t
	status := C.cusparseCreateDnVec(&vecB, C.int64_t(A.rows), b.Ptr(), C.CUDA_R_32F)
	if status != C.CUSPARSE_STATUS_SUCCESS {
		return cusparseError("cusparseCreateDnVec(b)", status)
	}
	defer func() {
		ctx.finishNativeResource(&err, func() error {
			if status := C.cusparseDestroyDnVec(vecB); status != C.CUSPARSE_STATUS_SUCCESS {
				return cusparseError("cusparseDestroyDnVec(b)", status)
			}
			return nil
		})
	}()
	var vecX C.cusparseDnVecDescr_t
	status = C.cusparseCreateDnVec(&vecX, C.int64_t(A.rows), x.Ptr(), C.CUDA_R_32F)
	if status != C.CUSPARSE_STATUS_SUCCESS {
		return cusparseError("cusparseCreateDnVec(x)", status)
	}
	defer func() {
		ctx.finishNativeResource(&err, func() error {
			if status := C.cusparseDestroyDnVec(vecX); status != C.CUSPARSE_STATUS_SUCCESS {
				return cusparseError("cusparseDestroyDnVec(x)", status)
			}
			return nil
		})
	}()
	var descriptor C.cusparseSpSVDescr_t
	status = C.cusparseSpSV_createDescr(&descriptor)
	if status != C.CUSPARSE_STATUS_SUCCESS {
		return cusparseError("cusparseSpSV_createDescr", status)
	}
	defer func() {
		ctx.finishNativeResource(&err, func() error {
			if status := C.cusparseSpSV_destroyDescr(descriptor); status != C.CUSPARSE_STATUS_SUCCESS {
				return cusparseError("cusparseSpSV_destroyDescr", status)
			}
			return nil
		})
	}()
	alpha := C.float(1)
	var bufferSize C.size_t
	status = C.cusparseSpSV_bufferSize(C.cusparseHandle_t(ctx.handle), C.CUSPARSE_OPERATION_NON_TRANSPOSE,
		unsafePointer(&alpha), matA, vecB, vecX, C.CUDA_R_32F, C.CUSPARSE_SPSV_ALG_DEFAULT, descriptor, &bufferSize)
	if status != C.CUSPARSE_STATUS_SUCCESS {
		return cusparseError("cusparseSpSV_bufferSize", status)
	}
	buffer, err := allocateNativeSparseWorkspace(bufferSize)
	if err != nil {
		return err
	}
	defer func() { ctx.finishWorkspace(buffer, &err) }()
	status = C.cusparseSpSV_analysis(C.cusparseHandle_t(ctx.handle), C.CUSPARSE_OPERATION_NON_TRANSPOSE,
		unsafePointer(&alpha), matA, vecB, vecX, C.CUDA_R_32F, C.CUSPARSE_SPSV_ALG_DEFAULT, descriptor, memoryPtrOrNil(buffer))
	if status != C.CUSPARSE_STATUS_SUCCESS {
		return cusparseError("cusparseSpSV_analysis", status)
	}
	status = C.cusparseSpSV_solve(C.cusparseHandle_t(ctx.handle), C.CUSPARSE_OPERATION_NON_TRANSPOSE,
		unsafePointer(&alpha), matA, vecB, vecX, C.CUDA_R_32F, C.CUSPARSE_SPSV_ALG_DEFAULT, descriptor)
	if status != C.CUSPARSE_STATUS_SUCCESS {
		return cusparseError("cusparseSpSV_solve", status)
	}
	return internal.CudaDefaultStreamSynchronizeOnDevice(ctx.deviceID)
}

func nativeSpGEMMCSR(ctx *SparseContext, A, B *SparseMatrix) (output *SparseMatrix, err error) {
	defer func() {
		if err != nil && output != nil {
			ctx.pendingMatrices = append(ctx.pendingMatrices, output)
			output = nil
		}
	}()
	matA, err := createNativeSparseMatrixDescriptor(A)
	if err != nil {
		return nil, err
	}
	defer func() {
		ctx.finishNativeResource(&err, func() error {
			if status := C.cusparseDestroySpMat(matA); status != C.CUSPARSE_STATUS_SUCCESS {
				return cusparseError("cusparseDestroySpMat(A)", status)
			}
			return nil
		})
	}()
	matB, err := createNativeSparseMatrixDescriptor(B)
	if err != nil {
		return nil, err
	}
	defer func() {
		ctx.finishNativeResource(&err, func() error {
			if status := C.cusparseDestroySpMat(matB); status != C.CUSPARSE_STATUS_SUCCESS {
				return cusparseError("cusparseDestroySpMat(B)", status)
			}
			return nil
		})
	}()
	var matC C.cusparseSpMatDescr_t
	status := C.cusparseCreateCsr(&matC, C.int64_t(A.rows), C.int64_t(B.cols), 0,
		nil, nil, nil, C.CUSPARSE_INDEX_32I, C.CUSPARSE_INDEX_32I, C.CUSPARSE_INDEX_BASE_ZERO, C.CUDA_R_32F)
	if status != C.CUSPARSE_STATUS_SUCCESS {
		return nil, cusparseError("cusparseCreateCsr(C)", status)
	}
	defer func() {
		ctx.finishNativeResource(&err, func() error {
			if status := C.cusparseDestroySpMat(matC); status != C.CUSPARSE_STATUS_SUCCESS {
				return cusparseError("cusparseDestroySpMat(C)", status)
			}
			return nil
		})
	}()
	var descriptor C.cusparseSpGEMMDescr_t
	status = C.cusparseSpGEMM_createDescr(&descriptor)
	if status != C.CUSPARSE_STATUS_SUCCESS {
		return nil, cusparseError("cusparseSpGEMM_createDescr", status)
	}
	defer func() {
		ctx.finishNativeResource(&err, func() error {
			if status := C.cusparseSpGEMM_destroyDescr(descriptor); status != C.CUSPARSE_STATUS_SUCCESS {
				return cusparseError("cusparseSpGEMM_destroyDescr", status)
			}
			return nil
		})
	}()
	alpha, beta := C.float(1), C.float(0)
	var size1 C.size_t
	status = C.cusparseSpGEMM_workEstimation(C.cusparseHandle_t(ctx.handle),
		C.CUSPARSE_OPERATION_NON_TRANSPOSE, C.CUSPARSE_OPERATION_NON_TRANSPOSE, unsafePointer(&alpha),
		matA, matB, unsafePointer(&beta), matC, C.CUDA_R_32F, C.CUSPARSE_SPGEMM_DEFAULT, descriptor, &size1, nil)
	if status != C.CUSPARSE_STATUS_SUCCESS {
		return nil, cusparseError("cusparseSpGEMM_workEstimation(size)", status)
	}
	work1, err := allocateNativeSparseWorkspace(size1)
	if err != nil {
		return nil, err
	}
	defer func() { ctx.finishWorkspace(work1, &err) }()
	status = C.cusparseSpGEMM_workEstimation(C.cusparseHandle_t(ctx.handle),
		C.CUSPARSE_OPERATION_NON_TRANSPOSE, C.CUSPARSE_OPERATION_NON_TRANSPOSE, unsafePointer(&alpha),
		matA, matB, unsafePointer(&beta), matC, C.CUDA_R_32F, C.CUSPARSE_SPGEMM_DEFAULT, descriptor, &size1, memoryPtrOrNil(work1))
	if status != C.CUSPARSE_STATUS_SUCCESS {
		return nil, cusparseError("cusparseSpGEMM_workEstimation", status)
	}
	var size2 C.size_t
	status = C.cusparseSpGEMM_compute(C.cusparseHandle_t(ctx.handle),
		C.CUSPARSE_OPERATION_NON_TRANSPOSE, C.CUSPARSE_OPERATION_NON_TRANSPOSE, unsafePointer(&alpha),
		matA, matB, unsafePointer(&beta), matC, C.CUDA_R_32F, C.CUSPARSE_SPGEMM_DEFAULT, descriptor, &size2, nil)
	if status != C.CUSPARSE_STATUS_SUCCESS {
		return nil, cusparseError("cusparseSpGEMM_compute(size)", status)
	}
	work2, err := allocateNativeSparseWorkspace(size2)
	if err != nil {
		return nil, err
	}
	defer func() { ctx.finishWorkspace(work2, &err) }()
	status = C.cusparseSpGEMM_compute(C.cusparseHandle_t(ctx.handle),
		C.CUSPARSE_OPERATION_NON_TRANSPOSE, C.CUSPARSE_OPERATION_NON_TRANSPOSE, unsafePointer(&alpha),
		matA, matB, unsafePointer(&beta), matC, C.CUDA_R_32F, C.CUSPARSE_SPGEMM_DEFAULT, descriptor, &size2, memoryPtrOrNil(work2))
	if status != C.CUSPARSE_STATUS_SUCCESS {
		return nil, cusparseError("cusparseSpGEMM_compute", status)
	}
	var rows, cols, nnz C.int64_t
	status = C.cusparseSpMatGetSize(matC, &rows, &cols, &nnz)
	if status != C.CUSPARSE_STATUS_SUCCESS {
		return nil, cusparseError("cusparseSpMatGetSize", status)
	}
	if int64(rows) != int64(A.rows) || int64(cols) != int64(B.cols) || nnz < 0 || int64(nnz) > int64(^uint32(0)>>1) {
		return nil, fmt.Errorf("cuSPARSE returned invalid product dimensions or nonzero count")
	}
	output, err = allocateSparseMatrix(A.rows, B.cols, int(nnz), MatrixFormatCSR)
	if err != nil {
		return nil, err
	}
	status = C.cusparseCsrSetPointers(matC, output.rowPtrs.Ptr(), output.colInds.Ptr(), output.values.Ptr())
	if status != C.CUSPARSE_STATUS_SUCCESS {
		return output, cusparseError("cusparseCsrSetPointers", status)
	}
	status = C.cusparseSpGEMM_copy(C.cusparseHandle_t(ctx.handle),
		C.CUSPARSE_OPERATION_NON_TRANSPOSE, C.CUSPARSE_OPERATION_NON_TRANSPOSE, unsafePointer(&alpha),
		matA, matB, unsafePointer(&beta), matC, C.CUDA_R_32F, C.CUSPARSE_SPGEMM_DEFAULT, descriptor)
	if status != C.CUSPARSE_STATUS_SUCCESS {
		return output, cusparseError("cusparseSpGEMM_copy", status)
	}
	if err := internal.CudaDefaultStreamSynchronizeOnDevice(ctx.deviceID); err != nil {
		return output, err
	}
	return output, nil
}

func nativeSparseToDense(ctx *SparseContext, matrix *SparseMatrix) (output *memory.Memory, err error) {
	matA, err := createNativeSparseMatrixDescriptor(matrix)
	if err != nil {
		return nil, err
	}
	defer func() {
		ctx.finishNativeResource(&err, func() error {
			if status := C.cusparseDestroySpMat(matA); status != C.CUSPARSE_STATUS_SUCCESS {
				return cusparseError("cusparseDestroySpMat", status)
			}
			return nil
		})
	}()
	output, err = memory.Alloc(int64(matrix.rows) * int64(matrix.cols) * 4)
	if err != nil {
		return nil, err
	}
	var dense C.cusparseDnMatDescr_t
	status := C.cusparseCreateDnMat(&dense, C.int64_t(matrix.rows), C.int64_t(matrix.cols), C.int64_t(matrix.cols),
		output.Ptr(), C.CUDA_R_32F, C.CUSPARSE_ORDER_ROW)
	if status != C.CUSPARSE_STATUS_SUCCESS {
		return output, cusparseError("cusparseCreateDnMat", status)
	}
	defer func() {
		ctx.finishNativeResource(&err, func() error {
			if status := C.cusparseDestroyDnMat(dense); status != C.CUSPARSE_STATUS_SUCCESS {
				return cusparseError("cusparseDestroyDnMat", status)
			}
			return nil
		})
	}()
	var bufferSize C.size_t
	status = C.cusparseSparseToDense_bufferSize(C.cusparseHandle_t(ctx.handle), matA, dense,
		C.CUSPARSE_SPARSETODENSE_ALG_DEFAULT, &bufferSize)
	if status != C.CUSPARSE_STATUS_SUCCESS {
		return output, cusparseError("cusparseSparseToDense_bufferSize", status)
	}
	workspace, err := allocateNativeSparseWorkspace(bufferSize)
	if err != nil {
		return output, err
	}
	defer func() { ctx.finishWorkspace(workspace, &err) }()
	status = C.cusparseSparseToDense(C.cusparseHandle_t(ctx.handle), matA, dense,
		C.CUSPARSE_SPARSETODENSE_ALG_DEFAULT, memoryPtrOrNil(workspace))
	if status != C.CUSPARSE_STATUS_SUCCESS {
		return output, cusparseError("cusparseSparseToDense", status)
	}
	if err := internal.CudaDefaultStreamSynchronizeOnDevice(ctx.deviceID); err != nil {
		return output, err
	}
	return output, nil
}

func nativeDenseToSparseCSR(ctx *SparseContext, input *memory.Memory, rows, cols int) (output *SparseMatrix, err error) {
	var analysis *SparseMatrix
	defer func() {
		if err == nil && analysis != nil {
			if cleanupErr := analysis.Destroy(); cleanupErr != nil {
				err = cleanupErr
			} else {
				analysis = nil
			}
		}
		if err != nil {
			if analysis != nil {
				ctx.pendingMatrices = append(ctx.pendingMatrices, analysis)
			}
			if output != nil {
				ctx.pendingMatrices = append(ctx.pendingMatrices, output)
				output = nil
			}
		}
	}()
	analysis, err = allocateSparseMatrix(rows, cols, 0, MatrixFormatCSR)
	if err != nil {
		return nil, err
	}
	var dense C.cusparseDnMatDescr_t
	status := C.cusparseCreateDnMat(&dense, C.int64_t(rows), C.int64_t(cols), C.int64_t(cols),
		input.Ptr(), C.CUDA_R_32F, C.CUSPARSE_ORDER_ROW)
	if status != C.CUSPARSE_STATUS_SUCCESS {
		return nil, cusparseError("cusparseCreateDnMat", status)
	}
	defer func() {
		ctx.finishNativeResource(&err, func() error {
			if status := C.cusparseDestroyDnMat(dense); status != C.CUSPARSE_STATUS_SUCCESS {
				return cusparseError("cusparseDestroyDnMat", status)
			}
			return nil
		})
	}()
	var sparse C.cusparseSpMatDescr_t
	status = C.cusparseCreateCsr(&sparse, C.int64_t(rows), C.int64_t(cols), 0,
		analysis.rowPtrs.Ptr(), nil, nil, C.CUSPARSE_INDEX_32I, C.CUSPARSE_INDEX_32I,
		C.CUSPARSE_INDEX_BASE_ZERO, C.CUDA_R_32F)
	if status != C.CUSPARSE_STATUS_SUCCESS {
		return nil, cusparseError("cusparseCreateCsr", status)
	}
	defer func() {
		ctx.finishNativeResource(&err, func() error {
			if status := C.cusparseDestroySpMat(sparse); status != C.CUSPARSE_STATUS_SUCCESS {
				return cusparseError("cusparseDestroySpMat", status)
			}
			return nil
		})
	}()
	var bufferSize C.size_t
	status = C.cusparseDenseToSparse_bufferSize(C.cusparseHandle_t(ctx.handle), dense, sparse,
		C.CUSPARSE_DENSETOSPARSE_ALG_DEFAULT, &bufferSize)
	if status != C.CUSPARSE_STATUS_SUCCESS {
		return nil, cusparseError("cusparseDenseToSparse_bufferSize", status)
	}
	workspace, err := allocateNativeSparseWorkspace(bufferSize)
	if err != nil {
		return nil, err
	}
	defer func() { ctx.finishWorkspace(workspace, &err) }()
	status = C.cusparseDenseToSparse_analysis(C.cusparseHandle_t(ctx.handle), dense, sparse,
		C.CUSPARSE_DENSETOSPARSE_ALG_DEFAULT, memoryPtrOrNil(workspace))
	if status != C.CUSPARSE_STATUS_SUCCESS {
		return nil, cusparseError("cusparseDenseToSparse_analysis", status)
	}
	var outputRows, outputCols, nnz C.int64_t
	status = C.cusparseSpMatGetSize(sparse, &outputRows, &outputCols, &nnz)
	if status != C.CUSPARSE_STATUS_SUCCESS {
		return nil, cusparseError("cusparseSpMatGetSize", status)
	}
	if int64(outputRows) != int64(rows) || int64(outputCols) != int64(cols) || nnz < 0 || int64(nnz) > int64(^uint32(0)>>1) {
		return nil, fmt.Errorf("cuSPARSE returned invalid dense-to-sparse dimensions or nonzero count")
	}
	output, err = allocateSparseMatrix(rows, cols, int(nnz), MatrixFormatCSR)
	if err != nil {
		return nil, err
	}
	analysis.rowPtrs, output.rowPtrs = output.rowPtrs, analysis.rowPtrs
	status = C.cusparseCsrSetPointers(sparse, output.rowPtrs.Ptr(), output.colInds.Ptr(), output.values.Ptr())
	if status != C.CUSPARSE_STATUS_SUCCESS {
		return output, cusparseError("cusparseCsrSetPointers", status)
	}
	status = C.cusparseDenseToSparse_convert(C.cusparseHandle_t(ctx.handle), dense, sparse,
		C.CUSPARSE_DENSETOSPARSE_ALG_DEFAULT, memoryPtrOrNil(workspace))
	if status != C.CUSPARSE_STATUS_SUCCESS {
		return output, cusparseError("cusparseDenseToSparse_convert", status)
	}
	if err := internal.CudaDefaultStreamSynchronizeOnDevice(ctx.deviceID); err != nil {
		return output, err
	}
	return output, nil
}

func destroyNativeSparseContext(ctx *SparseContext) error {
	status := C.destroySparseHandle(C.cusparseHandle_t(ctx.handle))
	if status != C.CUSPARSE_STATUS_SUCCESS {
		return cusparseError("cusparseDestroy", status)
	}
	return nil
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
