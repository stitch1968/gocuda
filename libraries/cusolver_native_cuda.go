//go:build cuda

package libraries

/*
#cgo linux CFLAGS: -I/usr/local/cuda/include -I/opt/cuda/include
#cgo windows CFLAGS: -I"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.1/include" -I"D:/NVIDIA/include"
#cgo linux LDFLAGS: -L/usr/local/cuda/lib64 -L/opt/cuda/lib64 -lcusolver
#cgo windows LDFLAGS: -L${SRCDIR}/../lib_mingw -lcusolver

#include <cusolverDn.h>
#include <cuda_runtime.h>

static cusolverStatus_t createSolverHandle(cusolverDnHandle_t* handle) {
	return cusolverDnCreate(handle);
}

static cusolverStatus_t destroySolverHandle(cusolverDnHandle_t handle) {
	return cusolverDnDestroy(handle);
}

static cusolverStatus_t gesvdBufferSizeWrapper(cusolverDnHandle_t handle, int m, int n, int* lwork) {
	return cusolverDnSgesvd_bufferSize(handle, m, n, lwork);
}

static cusolverStatus_t gesvdWrapper(
	cusolverDnHandle_t handle,
	signed char jobu,
	signed char jobvt,
	int m,
	int n,
	float* A,
	int lda,
	float* S,
	float* U,
	int ldu,
	float* VT,
	int ldvt,
	float* work,
	int lwork,
	float* rwork,
	int* devInfo) {
	return cusolverDnSgesvd(handle, jobu, jobvt, m, n, A, lda, S, U, ldu, VT, ldvt, work, lwork, rwork, devInfo);
}

static cusolverStatus_t syevdBufferSizeWrapper(
	cusolverDnHandle_t handle,
	int jobz,
	int fillMode,
	int n,
	float* A,
	int lda,
	float* W,
	int* lwork) {
	return cusolverDnSsyevd(handle, (cusolverEigMode_t)jobz, (cublasFillMode_t)fillMode, n, A, lda, W, NULL, 0, NULL);
}

static cusolverStatus_t syevdBufferQueryWrapper(
	cusolverDnHandle_t handle,
	int jobz,
	int fillMode,
	int n,
	float* A,
	int lda,
	float* W,
	int* lwork) {
	return cusolverDnSsyevd_bufferSize(handle, (cusolverEigMode_t)jobz, (cublasFillMode_t)fillMode, n, A, lda, W, lwork);
}

static cusolverStatus_t syevdWrapper(
	cusolverDnHandle_t handle,
	int jobz,
	int fillMode,
	int n,
	float* A,
	int lda,
	float* W,
	float* work,
	int lwork,
	int* devInfo) {
	return cusolverDnSsyevd(handle, (cusolverEigMode_t)jobz, (cublasFillMode_t)fillMode, n, A, lda, W, work, lwork, devInfo);
}
*/
import "C"

import (
	"fmt"
	"math"
	"unsafe"

	"github.com/stitch1968/gocuda/memory"
)

func createNativeSolverContext() (*SolverContext, error) {
	var handle C.cusolverDnHandle_t
	if status := C.createSolverHandle(&handle); status != C.CUSOLVER_STATUS_SUCCESS {
		return nil, cusolverError("cusolverDnCreate", status)
	}
	return &SolverContext{handle: unsafe.Pointer(handle), native: true}, nil
}

func nativeQRFactorization(ctx *SolverContext, A *memory.Memory, m, n int) (*QRInfo, error) {
	minMN := min(m, n)
	tau, err := memory.Alloc(int64(minMN * 4))
	if err != nil {
		return nil, fmt.Errorf("failed to allocate tau: %v", err)
	}
	var lwork C.int
	status := C.cusolverDnSgeqrf_bufferSize(C.cusolverDnHandle_t(ctx.handle), C.int(m), C.int(n), (*C.float)(A.Ptr()), C.int(m), &lwork)
	if status != C.CUSOLVER_STATUS_SUCCESS {
		_ = tau.Free()
		return nil, cusolverError("cusolverDnSgeqrf_bufferSize", status)
	}
	workspace, err := memory.Alloc(int64(lwork) * 4)
	if err != nil {
		_ = tau.Free()
		return nil, fmt.Errorf("failed to allocate workspace: %v", err)
	}
	devInfo, err := memory.Alloc(4)
	if err != nil {
		_ = tau.Free()
		_ = workspace.Free()
		return nil, fmt.Errorf("failed to allocate devInfo: %v", err)
	}
	defer devInfo.Free()
	status = C.cusolverDnSgeqrf(C.cusolverDnHandle_t(ctx.handle), C.int(m), C.int(n), (*C.float)(A.Ptr()), C.int(m), (*C.float)(tau.Ptr()), (*C.float)(workspace.Ptr()), lwork, (*C.int)(devInfo.Ptr()))
	if status != C.CUSOLVER_STATUS_SUCCESS {
		_ = tau.Free()
		_ = workspace.Free()
		return nil, cusolverError("cusolverDnSgeqrf", status)
	}
	infoValue, err := readInt32(devInfo)
	if err != nil {
		_ = tau.Free()
		_ = workspace.Free()
		return nil, err
	}
	if infoValue != 0 {
		_ = tau.Free()
		_ = workspace.Free()
		return nil, fmt.Errorf("cusolverDnSgeqrf returned info=%d", infoValue)
	}
	return &QRInfo{tau: tau, workspace: workspace, info: int(infoValue)}, nil
}

func nativeSVDDecomposition(ctx *SolverContext, A *memory.Memory, m, n int, computeUV bool) (*SVDInfo, error) {
	minMN := min(m, n)
	s, err := memory.Alloc(int64(minMN * 4))
	if err != nil {
		return nil, fmt.Errorf("failed to allocate singular values: %v", err)
	}
	jobu := C.schar('N')
	jobvt := C.schar('N')
	var u, vt *memory.Memory
	ldu := C.int(m)
	ldvt := C.int(n)
	if computeUV {
		jobu = C.schar('A')
		jobvt = C.schar('A')
		u, err = memory.Alloc(int64(m * m * 4))
		if err != nil {
			_ = s.Free()
			return nil, fmt.Errorf("failed to allocate U matrix: %v", err)
		}
		vt, err = memory.Alloc(int64(n * n * 4))
		if err != nil {
			_ = s.Free()
			_ = u.Free()
			return nil, fmt.Errorf("failed to allocate VT matrix: %v", err)
		}
	} else {
		ldu = 1
		ldvt = 1
	}
	var lwork C.int
	status := C.gesvdBufferSizeWrapper(C.cusolverDnHandle_t(ctx.handle), C.int(m), C.int(n), &lwork)
	if status != C.CUSOLVER_STATUS_SUCCESS {
		cleanupSVD(s, u, vt, nil)
		return nil, cusolverError("cusolverDnSgesvd_bufferSize", status)
	}
	workspace, err := memory.Alloc(int64(lwork) * 4)
	if err != nil {
		cleanupSVD(s, u, vt, nil)
		return nil, fmt.Errorf("failed to allocate workspace: %v", err)
	}
	devInfo, err := memory.Alloc(4)
	if err != nil {
		cleanupSVD(s, u, vt, workspace)
		return nil, fmt.Errorf("failed to allocate devInfo: %v", err)
	}
	defer devInfo.Free()
	status = C.gesvdWrapper(
		C.cusolverDnHandle_t(ctx.handle),
		jobu,
		jobvt,
		C.int(m),
		C.int(n),
		(*C.float)(A.Ptr()),
		C.int(m),
		(*C.float)(s.Ptr()),
		pointerOrNil(u),
		ldu,
		pointerOrNil(vt),
		ldvt,
		(*C.float)(workspace.Ptr()),
		lwork,
		nil,
		(*C.int)(devInfo.Ptr()),
	)
	if status != C.CUSOLVER_STATUS_SUCCESS {
		cleanupSVD(s, u, vt, workspace)
		return nil, cusolverError("cusolverDnSgesvd", status)
	}
	infoValue, err := readInt32(devInfo)
	if err != nil {
		cleanupSVD(s, u, vt, workspace)
		return nil, err
	}
	if infoValue != 0 {
		cleanupSVD(s, u, vt, workspace)
		return nil, fmt.Errorf("cusolverDnSgesvd returned info=%d", infoValue)
	}
	return &SVDInfo{s: s, u: u, vt: vt, workspace: workspace, info: int(infoValue)}, nil
}

func nativeLUFactorization(ctx *SolverContext, A *memory.Memory, m, n int) (*LUInfo, error) {
	minMN := min(m, n)
	ipiv, err := memory.Alloc(int64(minMN * 4))
	if err != nil {
		return nil, fmt.Errorf("failed to allocate pivot indices: %v", err)
	}
	var lwork C.int
	status := C.cusolverDnSgetrf_bufferSize(C.cusolverDnHandle_t(ctx.handle), C.int(m), C.int(n), (*C.float)(A.Ptr()), C.int(m), &lwork)
	if status != C.CUSOLVER_STATUS_SUCCESS {
		_ = ipiv.Free()
		return nil, cusolverError("cusolverDnSgetrf_bufferSize", status)
	}
	workspace, err := memory.Alloc(int64(lwork) * 4)
	if err != nil {
		_ = ipiv.Free()
		return nil, fmt.Errorf("failed to allocate workspace: %v", err)
	}
	devInfo, err := memory.Alloc(4)
	if err != nil {
		_ = ipiv.Free()
		_ = workspace.Free()
		return nil, fmt.Errorf("failed to allocate devInfo: %v", err)
	}
	defer devInfo.Free()
	status = C.cusolverDnSgetrf(C.cusolverDnHandle_t(ctx.handle), C.int(m), C.int(n), (*C.float)(A.Ptr()), C.int(m), (*C.float)(workspace.Ptr()), (*C.int)(ipiv.Ptr()), (*C.int)(devInfo.Ptr()))
	if status != C.CUSOLVER_STATUS_SUCCESS {
		_ = ipiv.Free()
		_ = workspace.Free()
		return nil, cusolverError("cusolverDnSgetrf", status)
	}
	infoValue, err := readInt32(devInfo)
	if err != nil {
		_ = ipiv.Free()
		_ = workspace.Free()
		return nil, err
	}
	if infoValue != 0 {
		_ = ipiv.Free()
		_ = workspace.Free()
		return nil, fmt.Errorf("cusolverDnSgetrf returned info=%d", infoValue)
	}
	return &LUInfo{ipiv: ipiv, workspace: workspace, info: int(infoValue)}, nil
}

func nativeSolveLinearSystem(ctx *SolverContext, A, b *memory.Memory, n int) (*memory.Memory, error) {
	luInfo, err := nativeLUFactorization(ctx, A, n, n)
	if err != nil {
		return nil, err
	}
	defer luInfo.Destroy()

	x, err := memory.Alloc(int64(n * 4))
	if err != nil {
		return nil, fmt.Errorf("failed to allocate solution vector: %v", err)
	}
	bValues, err := memory.View[float32](b, n)
	if err != nil {
		_ = x.Free()
		return nil, err
	}
	xValues, err := memory.View[float32](x, n)
	if err != nil {
		_ = x.Free()
		return nil, err
	}
	copy(xValues, bValues)

	devInfo, err := memory.Alloc(4)
	if err != nil {
		_ = x.Free()
		return nil, fmt.Errorf("failed to allocate devInfo: %v", err)
	}
	defer devInfo.Free()
	status := C.cusolverDnSgetrs(C.cusolverDnHandle_t(ctx.handle), C.CUBLAS_OP_N, C.int(n), 1, (*C.float)(A.Ptr()), C.int(n), (*C.int)(luInfo.ipiv.Ptr()), (*C.float)(x.Ptr()), C.int(n), (*C.int)(devInfo.Ptr()))
	if status != C.CUSOLVER_STATUS_SUCCESS {
		_ = x.Free()
		return nil, cusolverError("cusolverDnSgetrs", status)
	}
	infoValue, err := readInt32(devInfo)
	if err != nil {
		_ = x.Free()
		return nil, err
	}
	if infoValue != 0 {
		_ = x.Free()
		return nil, fmt.Errorf("cusolverDnSgetrs returned info=%d", infoValue)
	}
	return x, nil
}

func nativeEigenvalues(ctx *SolverContext, A *memory.Memory, n int, computeVectors bool) (*memory.Memory, *memory.Memory, error) {
	// Use the symmetric eigensolver path. This expects callers to provide a
	// symmetric matrix when targeting the native CUDA backend.
	eigenvals, err := memory.Alloc(int64(n * 8))
	if err != nil {
		return nil, nil, fmt.Errorf("failed to allocate eigenvalues: %v", err)
	}
	realEigenvals, err := memory.Alloc(int64(n * 4))
	if err != nil {
		_ = eigenvals.Free()
		return nil, nil, fmt.Errorf("failed to allocate real eigenvalues: %v", err)
	}
	defer realEigenvals.Free()

	var eigenvecs *memory.Memory
	jobz := C.int(C.CUSOLVER_EIG_MODE_NOVECTOR)
	if computeVectors {
		jobz = C.int(C.CUSOLVER_EIG_MODE_VECTOR)
		eigenvecs, err = memory.Alloc(int64(n * n * 4))
		if err != nil {
			_ = eigenvals.Free()
			return nil, nil, fmt.Errorf("failed to allocate eigenvectors: %v", err)
		}
		copyMatrix, copyErr := copyFloat32Matrix(A, n*n)
		if copyErr != nil {
			_ = eigenvals.Free()
			_ = eigenvecs.Free()
			return nil, nil, copyErr
		}
		_ = eigenvecs.Free()
		eigenvecs = copyMatrix
	}
	target := A
	if computeVectors {
		target = eigenvecs
	}
	var lwork C.int
	status := C.syevdBufferQueryWrapper(C.cusolverDnHandle_t(ctx.handle), jobz, C.int(C.CUBLAS_FILL_MODE_UPPER), C.int(n), (*C.float)(target.Ptr()), C.int(n), (*C.float)(realEigenvals.Ptr()), &lwork)
	if status != C.CUSOLVER_STATUS_SUCCESS {
		_ = eigenvals.Free()
		if eigenvecs != nil {
			_ = eigenvecs.Free()
		}
		return nil, nil, cusolverError("cusolverDnSsyevd_bufferSize", status)
	}
	workspace, err := memory.Alloc(int64(lwork) * 4)
	if err != nil {
		_ = eigenvals.Free()
		if eigenvecs != nil {
			_ = eigenvecs.Free()
		}
		return nil, nil, fmt.Errorf("failed to allocate workspace: %v", err)
	}
	defer workspace.Free()
	devInfo, err := memory.Alloc(4)
	if err != nil {
		_ = eigenvals.Free()
		if eigenvecs != nil {
			_ = eigenvecs.Free()
		}
		return nil, nil, fmt.Errorf("failed to allocate devInfo: %v", err)
	}
	defer devInfo.Free()
	status = C.syevdWrapper(C.cusolverDnHandle_t(ctx.handle), jobz, C.int(C.CUBLAS_FILL_MODE_UPPER), C.int(n), (*C.float)(target.Ptr()), C.int(n), (*C.float)(realEigenvals.Ptr()), (*C.float)(workspace.Ptr()), lwork, (*C.int)(devInfo.Ptr()))
	if status != C.CUSOLVER_STATUS_SUCCESS {
		_ = eigenvals.Free()
		if eigenvecs != nil {
			_ = eigenvecs.Free()
		}
		return nil, nil, cusolverError("cusolverDnSsyevd", status)
	}
	infoValue, err := readInt32(devInfo)
	if err != nil {
		_ = eigenvals.Free()
		if eigenvecs != nil {
			_ = eigenvecs.Free()
		}
		return nil, nil, err
	}
	if infoValue != 0 {
		_ = eigenvals.Free()
		if eigenvecs != nil {
			_ = eigenvecs.Free()
		}
		return nil, nil, fmt.Errorf("cusolverDnSsyevd returned info=%d", infoValue)
	}
	packed, err := memory.View[Complex64](eigenvals, n)
	if err != nil {
		_ = eigenvals.Free()
		if eigenvecs != nil {
			_ = eigenvecs.Free()
		}
		return nil, nil, err
	}
	realValues, err := memory.View[float32](realEigenvals, n)
	if err != nil {
		_ = eigenvals.Free()
		if eigenvecs != nil {
			_ = eigenvecs.Free()
		}
		return nil, nil, err
	}
	for i := range realValues {
		packed[i] = Complex64{Real: realValues[i]}
	}
	return eigenvals, eigenvecs, nil
}

func nativeCholeskyFactorization(ctx *SolverContext, A *memory.Memory, n int) error {
	var lwork C.int
	status := C.cusolverDnSpotrf_bufferSize(C.cusolverDnHandle_t(ctx.handle), C.CUBLAS_FILL_MODE_UPPER, C.int(n), (*C.float)(A.Ptr()), C.int(n), &lwork)
	if status != C.CUSOLVER_STATUS_SUCCESS {
		return cusolverError("cusolverDnSpotrf_bufferSize", status)
	}
	workspace, err := memory.Alloc(int64(lwork) * 4)
	if err != nil {
		return fmt.Errorf("failed to allocate workspace: %v", err)
	}
	defer workspace.Free()
	devInfo, err := memory.Alloc(4)
	if err != nil {
		return fmt.Errorf("failed to allocate devInfo: %v", err)
	}
	defer devInfo.Free()
	status = C.cusolverDnSpotrf(C.cusolverDnHandle_t(ctx.handle), C.CUBLAS_FILL_MODE_UPPER, C.int(n), (*C.float)(A.Ptr()), C.int(n), (*C.float)(workspace.Ptr()), lwork, (*C.int)(devInfo.Ptr()))
	if status != C.CUSOLVER_STATUS_SUCCESS {
		return cusolverError("cusolverDnSpotrf", status)
	}
	infoValue, err := readInt32(devInfo)
	if err != nil {
		return err
	}
	if infoValue != 0 {
		return fmt.Errorf("cusolverDnSpotrf returned info=%d", infoValue)
	}
	return nil
}

func nativePseudoInverse(ctx *SolverContext, A *memory.Memory, m, n int) (*memory.Memory, error) {
	copyA, err := copyFloat32Matrix(A, m*n)
	if err != nil {
		return nil, err
	}
	defer copyA.Free()
	svdInfo, err := nativeSVDDecomposition(ctx, copyA, m, n, true)
	if err != nil {
		return nil, err
	}
	defer svdInfo.Destroy()

	pinv, err := memory.Alloc(int64(n * m * 4))
	if err != nil {
		return nil, fmt.Errorf("failed to allocate pseudoinverse: %v", err)
	}
	if err := fillPseudoInverseFromSVD(pinv, svdInfo, m, n); err != nil {
		_ = pinv.Free()
		return nil, err
	}
	return pinv, nil
}

func fillPseudoInverseFromSVD(pinv *memory.Memory, svdInfo *SVDInfo, m, n int) error {
	sValues, err := memory.View[float32](svdInfo.s, min(m, n))
	if err != nil {
		return err
	}
	uValues, err := memory.View[float32](svdInfo.u, m*m)
	if err != nil {
		return err
	}
	vtValues, err := memory.View[float32](svdInfo.vt, n*n)
	if err != nil {
		return err
	}
	pinvValues, err := memory.View[float32](pinv, n*m)
	if err != nil {
		return err
	}
	for i := range pinvValues {
		pinvValues[i] = 0
	}
	tolerance := float32(1e-6)
	for row := 0; row < n; row++ {
		for col := 0; col < m; col++ {
			accum := float32(0)
			for k := 0; k < min(m, n); k++ {
				sigma := sValues[k]
				if float32(math.Abs(float64(sigma))) <= tolerance {
					continue
				}
				v := vtValues[k*n+row]
				u := uValues[col*m+k]
				accum += v * (1 / sigma) * u
			}
			pinvValues[row*m+col] = accum
		}
	}
	return nil
}

func copyFloat32Matrix(src *memory.Memory, length int) (*memory.Memory, error) {
	dst, err := memory.Alloc(int64(length * 4))
	if err != nil {
		return nil, fmt.Errorf("failed to allocate matrix copy: %v", err)
	}
	srcValues, err := memory.View[float32](src, length)
	if err != nil {
		_ = dst.Free()
		return nil, err
	}
	dstValues, err := memory.View[float32](dst, length)
	if err != nil {
		_ = dst.Free()
		return nil, err
	}
	copy(dstValues, srcValues)
	return dst, nil
}

func pointerOrNil(mem *memory.Memory) *C.float {
	if mem == nil {
		return nil
	}
	return (*C.float)(mem.Ptr())
}

func cleanupSVD(s, u, vt, workspace *memory.Memory) {
	if s != nil {
		_ = s.Free()
	}
	if u != nil {
		_ = u.Free()
	}
	if vt != nil {
		_ = vt.Free()
	}
	if workspace != nil {
		_ = workspace.Free()
	}
}

func readInt32(mem *memory.Memory) (int32, error) {
	values, err := memory.View[int32](mem, 1)
	if err != nil {
		return 0, err
	}
	return values[0], nil
}

func destroyNativeSolverContext(ctx *SolverContext) error {
	status := C.destroySolverHandle(C.cusolverDnHandle_t(ctx.handle))
	if status != C.CUSOLVER_STATUS_SUCCESS {
		return cusolverError("cusolverDnDestroy", status)
	}
	return nil
}

func cusolverError(operation string, status C.cusolverStatus_t) error {
	return fmt.Errorf("%s failed: %s (%d)", operation, cusolverStatusString(status), int(status))
}

func cusolverStatusString(status C.cusolverStatus_t) string {
	switch status {
	case C.CUSOLVER_STATUS_SUCCESS:
		return "success"
	case C.CUSOLVER_STATUS_NOT_INITIALIZED:
		return "not initialized"
	case C.CUSOLVER_STATUS_ALLOC_FAILED:
		return "allocation failed"
	case C.CUSOLVER_STATUS_INVALID_VALUE:
		return "invalid value"
	case C.CUSOLVER_STATUS_ARCH_MISMATCH:
		return "architecture mismatch"
	case C.CUSOLVER_STATUS_MAPPING_ERROR:
		return "mapping error"
	case C.CUSOLVER_STATUS_EXECUTION_FAILED:
		return "execution failed"
	case C.CUSOLVER_STATUS_INTERNAL_ERROR:
		return "internal error"
	case C.CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
		return "matrix type not supported"
	case C.CUSOLVER_STATUS_NOT_SUPPORTED:
		return "not supported"
	case C.CUSOLVER_STATUS_ZERO_PIVOT:
		return "zero pivot"
	case C.CUSOLVER_STATUS_INVALID_LICENSE:
		return "invalid license"
	default:
		return "unknown error"
	}
}
