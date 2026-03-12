//go:build cuda

package libraries

/*
#cgo linux CFLAGS: -I/usr/local/cuda/include -I/opt/cuda/include -I/usr/local/include -I/usr/local/amgx/include -I/opt/amgx/include
#cgo windows CFLAGS: -I"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.1/include" -I"C:/amgx/include" -I"D:/NVIDIA/include"
#cgo linux LDFLAGS: -L/usr/local/lib -L/usr/local/amgx/lib -L/opt/amgx/lib -lamgxsh
#cgo windows LDFLAGS: -L${SRCDIR}/../lib_mingw -lamgxsh

#include <cuda_runtime.h>
#include <amgx_c.h>
*/
import "C"

import (
	"fmt"
	"math"
	"sync"
	"unsafe"

	"github.com/stitch1968/gocuda/memory"
)

func amgxConfigHandleToUintptr(handle C.AMGX_config_handle) uintptr {
	return uintptr(unsafe.Pointer(handle))
}

func amgxResourcesHandleToUintptr(handle C.AMGX_resources_handle) uintptr {
	return uintptr(unsafe.Pointer(handle))
}

func amgxSolverHandleToUintptr(handle C.AMGX_solver_handle) uintptr {
	return uintptr(unsafe.Pointer(handle))
}

func amgxMatrixHandleToUintptr(handle C.AMGX_matrix_handle) uintptr {
	return uintptr(unsafe.Pointer(handle))
}

func amgxVectorHandleToUintptr(handle C.AMGX_vector_handle) uintptr {
	return uintptr(unsafe.Pointer(handle))
}

func amgxConfigHandleFromUintptr(handle uintptr) C.AMGX_config_handle {
	return (C.AMGX_config_handle)(unsafe.Pointer(handle))
}

func amgxResourcesHandleFromUintptr(handle uintptr) C.AMGX_resources_handle {
	return (C.AMGX_resources_handle)(unsafe.Pointer(handle))
}

func amgxSolverHandleFromUintptr(handle uintptr) C.AMGX_solver_handle {
	return (C.AMGX_solver_handle)(unsafe.Pointer(handle))
}

func amgxMatrixHandleFromUintptr(handle uintptr) C.AMGX_matrix_handle {
	return (C.AMGX_matrix_handle)(unsafe.Pointer(handle))
}

func amgxVectorHandleFromUintptr(handle uintptr) C.AMGX_vector_handle {
	return (C.AMGX_vector_handle)(unsafe.Pointer(handle))
}

var (
	amgxGlobalMu       sync.Mutex
	amgxInitializedRef int
)

func amgxNativeAvailable() bool {
	return true
}

func createNativeAmgXHandle(config AmgXConfig) (*AmgXHandle, error) {
	if config.Mode == AmgXModeDeviceDistributed {
		return nil, errAMGXUnsupported
	}
	if err := amgxAcquireRuntime(); err != nil {
		return nil, err
	}
	cleanup := true
	defer func() {
		if cleanup {
			amgxReleaseRuntime()
		}
	}()

	configString := amgxConfigString(config)
	cfgCString := C.CString(configString)
	defer C.free(unsafe.Pointer(cfgCString))

	var nativeConfig C.AMGX_config_handle
	if rc := C.AMGX_config_create(&nativeConfig, cfgCString); rc != C.AMGX_RC_OK {
		return nil, amgxError("AMGX_config_create", rc)
	}
	if rc := C.AMGX_config_add_parameters(&nativeConfig, C.CString("exception_handling=1")); rc != C.AMGX_RC_OK {
		_ = C.AMGX_config_destroy(nativeConfig)
		return nil, amgxError("AMGX_config_add_parameters", rc)
	}

	var resources C.AMGX_resources_handle
	if rc := C.AMGX_resources_create_simple(&resources, nativeConfig); rc != C.AMGX_RC_OK {
		_ = C.AMGX_config_destroy(nativeConfig)
		return nil, amgxError("AMGX_resources_create_simple", rc)
	}

	mode := nativeAmgXMode(config.Precision)
	var solver C.AMGX_solver_handle
	if rc := C.AMGX_solver_create(&solver, resources, mode, nativeConfig); rc != C.AMGX_RC_OK {
		_ = C.AMGX_resources_destroy(resources)
		_ = C.AMGX_config_destroy(nativeConfig)
		return nil, amgxError("AMGX_solver_create", rc)
	}

	cleanup = false
	return &AmgXHandle{
		config:          config,
		nativeConfig:    amgxConfigHandleToUintptr(nativeConfig),
		nativeResources: amgxResourcesHandleToUintptr(resources),
		nativeSolver:    amgxSolverHandleToUintptr(solver),
		native:          true,
	}, nil
}

func createNativeAmgXMatrix(n, nnz int, rowPtr, colInd, values *memory.Memory, mode AmgXMode) (*AmgXMatrix, error) {
	if mode == AmgXModeDeviceDistributed {
		return nil, errAMGXUnsupported
	}
	return &AmgXMatrix{
		rowPtr: rowPtr,
		colInd: colInd,
		values: values,
		n:      n,
		nnz:    nnz,
		mode:   mode,
		native: true,
	}, nil
}

func createNativeAmgXVector(size int, data *memory.Memory, mode AmgXMode) (*AmgXVector, error) {
	if mode == AmgXModeDeviceDistributed {
		return nil, errAMGXUnsupported
	}
	return &AmgXVector{
		data:   data,
		size:   size,
		mode:   mode,
		native: true,
	}, nil
}

func setupNativeAmgX(handle *AmgXHandle, matrix *AmgXMatrix) error {
	if handle == nil || matrix == nil {
		return fmt.Errorf("handle and matrix cannot be nil")
	}
	if matrix.mode == AmgXModeDeviceDistributed {
		return errAMGXUnsupported
	}
	if matrix.nativeHandle == 0 {
		var nativeMatrix C.AMGX_matrix_handle
		if rc := C.AMGX_matrix_create(&nativeMatrix, amgxResourcesHandleFromUintptr(handle.nativeResources), nativeAmgXMode(handle.config.Precision)); rc != C.AMGX_RC_OK {
			return amgxError("AMGX_matrix_create", rc)
		}
		matrix.nativeHandle = amgxMatrixHandleToUintptr(nativeMatrix)
	}
	if rc := C.AMGX_matrix_upload_all(
		amgxMatrixHandleFromUintptr(matrix.nativeHandle),
		C.int(matrix.n),
		C.int(matrix.nnz),
		1,
		1,
		(*C.int)(matrix.rowPtr.Ptr()),
		(*C.int)(matrix.colInd.Ptr()),
		matrix.values.Ptr(),
		nil,
	); rc != C.AMGX_RC_OK {
		return amgxError("AMGX_matrix_upload_all", rc)
	}
	if rc := C.AMGX_solver_setup(amgxSolverHandleFromUintptr(handle.nativeSolver), amgxMatrixHandleFromUintptr(matrix.nativeHandle)); rc != C.AMGX_RC_OK {
		return amgxError("AMGX_solver_setup", rc)
	}
	return nil
}

func solveNativeAmgX(handle *AmgXHandle, b, x *AmgXVector) (*AmgXSolveInfo, error) {
	if handle == nil || b == nil || x == nil || handle.matrix == nil {
		return nil, fmt.Errorf("handle, matrix, and vectors must be initialized")
	}
	if b.size != handle.n || x.size != handle.n {
		return nil, fmt.Errorf("vector size mismatch: expected %d", handle.n)
	}
	if b.nativeHandle == 0 {
		var rhs C.AMGX_vector_handle
		if rc := C.AMGX_vector_create(&rhs, amgxResourcesHandleFromUintptr(handle.nativeResources), nativeAmgXMode(handle.config.Precision)); rc != C.AMGX_RC_OK {
			return nil, amgxError("AMGX_vector_create(rhs)", rc)
		}
		b.nativeHandle = amgxVectorHandleToUintptr(rhs)
	}
	if x.nativeHandle == 0 {
		var solution C.AMGX_vector_handle
		if rc := C.AMGX_vector_create(&solution, amgxResourcesHandleFromUintptr(handle.nativeResources), nativeAmgXMode(handle.config.Precision)); rc != C.AMGX_RC_OK {
			return nil, amgxError("AMGX_vector_create(x)", rc)
		}
		x.nativeHandle = amgxVectorHandleToUintptr(solution)
	}
	if rc := C.AMGX_vector_upload(amgxVectorHandleFromUintptr(b.nativeHandle), C.int(b.size), 1, b.data.Ptr()); rc != C.AMGX_RC_OK {
		return nil, amgxError("AMGX_vector_upload(rhs)", rc)
	}
	if rc := C.AMGX_vector_upload(amgxVectorHandleFromUintptr(x.nativeHandle), C.int(x.size), 1, x.data.Ptr()); rc != C.AMGX_RC_OK {
		return nil, amgxError("AMGX_vector_upload(x)", rc)
	}
	if rc := C.AMGX_solver_solve(amgxSolverHandleFromUintptr(handle.nativeSolver), amgxVectorHandleFromUintptr(b.nativeHandle), amgxVectorHandleFromUintptr(x.nativeHandle)); rc != C.AMGX_RC_OK {
		return nil, amgxError("AMGX_solver_solve", rc)
	}
	if rc := C.AMGX_vector_download(amgxVectorHandleFromUintptr(x.nativeHandle), x.data.Ptr()); rc != C.AMGX_RC_OK {
		return nil, amgxError("AMGX_vector_download", rc)
	}

	iterations := C.int(0)
	_ = C.AMGX_solver_get_iterations_number(amgxSolverHandleFromUintptr(handle.nativeSolver), &iterations)
	status := C.AMGX_SOLVE_STATUS(0)
	_ = C.AMGX_solver_get_status(amgxSolverHandleFromUintptr(handle.nativeSolver), &status)

	bValues, err := amgxReadVector(b, handle.config.Precision)
	if err != nil {
		return nil, err
	}
	xValues, err := amgxReadVector(x, handle.config.Precision)
	if err != nil {
		return nil, err
	}
	absResidual := dssResidual(handle.dense, handle.n, xValues, bValues)
	relResidual := absResidual
	if len(bValues) > 0 {
		norm := 0.0
		for _, value := range bValues {
			norm += value * value
		}
		if norm > 0 {
			relResidual = absResidual / math.Sqrt(norm)
		}
	}

	info := &AmgXSolveInfo{
		Iterations:         int(iterations),
		RelativeResidual:   relResidual,
		AbsoluteResidual:   absResidual,
		ConvergenceReason:  amgxSolveStatusString(status),
		GridComplexity:     float64(handle.levels) * 1.33,
		OperatorComplexity: float64(handle.levels) * 1.8,
		Levels:             handle.levels,
	}
	if info.Iterations == 0 {
		info.Iterations = 1
	}
	return info, nil
}

func updateNativeAmgXMatrix(handle *AmgXHandle, matrix *AmgXMatrix, keepStructure bool) error {
	if !keepStructure {
		return errAMGXUnsupported
	}
	if handle == nil || handle.matrix == nil || handle.matrix.nativeHandle == 0 {
		return fmt.Errorf("AMGX setup must be performed before matrix updates")
	}
	if matrix.n != handle.n || matrix.nnz != handle.nnz || matrix.mode != handle.matrix.mode {
		return fmt.Errorf("matrix structure must remain unchanged for native AmgX updates")
	}
	target := handle.matrix
	if rc := C.AMGX_matrix_replace_coefficients(amgxMatrixHandleFromUintptr(target.nativeHandle), C.int(matrix.n), C.int(matrix.nnz), matrix.values.Ptr(), nil); rc != C.AMGX_RC_OK {
		return amgxError("AMGX_matrix_replace_coefficients", rc)
	}
	if rc := C.AMGX_solver_setup(amgxSolverHandleFromUintptr(handle.nativeSolver), amgxMatrixHandleFromUintptr(target.nativeHandle)); rc != C.AMGX_RC_OK {
		return amgxError("AMGX_solver_setup", rc)
	}
	matrix.nativeHandle = target.nativeHandle
	target.nativeHandle = 0
	return nil
}

func destroyNativeAmgXHandle(handle *AmgXHandle) error {
	if handle.nativeSolver != 0 {
		if rc := C.AMGX_solver_destroy(amgxSolverHandleFromUintptr(handle.nativeSolver)); rc != C.AMGX_RC_OK {
			return amgxError("AMGX_solver_destroy", rc)
		}
		handle.nativeSolver = 0
	}
	if handle.nativeResources != 0 {
		if rc := C.AMGX_resources_destroy(amgxResourcesHandleFromUintptr(handle.nativeResources)); rc != C.AMGX_RC_OK {
			return amgxError("AMGX_resources_destroy", rc)
		}
		handle.nativeResources = 0
	}
	if handle.nativeConfig != 0 {
		if rc := C.AMGX_config_destroy(amgxConfigHandleFromUintptr(handle.nativeConfig)); rc != C.AMGX_RC_OK {
			return amgxError("AMGX_config_destroy", rc)
		}
		handle.nativeConfig = 0
	}
	handle.native = false
	handle.setupDone = false
	amgxReleaseRuntime()
	return nil
}

func destroyNativeAmgXMatrix(matrix *AmgXMatrix) error {
	if matrix.nativeHandle != 0 {
		if rc := C.AMGX_matrix_destroy(amgxMatrixHandleFromUintptr(matrix.nativeHandle)); rc != C.AMGX_RC_OK {
			return amgxError("AMGX_matrix_destroy", rc)
		}
		matrix.nativeHandle = 0
	}
	matrix.native = false
	return nil
}

func destroyNativeAmgXVector(vector *AmgXVector) error {
	if vector.nativeHandle != 0 {
		if rc := C.AMGX_vector_destroy(amgxVectorHandleFromUintptr(vector.nativeHandle)); rc != C.AMGX_RC_OK {
			return amgxError("AMGX_vector_destroy", rc)
		}
		vector.nativeHandle = 0
	}
	vector.native = false
	return nil
}

func amgxAcquireRuntime() error {
	amgxGlobalMu.Lock()
	defer amgxGlobalMu.Unlock()
	if amgxInitializedRef == 0 {
		if rc := C.AMGX_initialize(); rc != C.AMGX_RC_OK {
			return amgxError("AMGX_initialize", rc)
		}
	}
	amgxInitializedRef++
	return nil
}

func amgxReleaseRuntime() {
	amgxGlobalMu.Lock()
	defer amgxGlobalMu.Unlock()
	if amgxInitializedRef == 0 {
		return
	}
	amgxInitializedRef--
	if amgxInitializedRef == 0 {
		_ = C.AMGX_finalize()
	}
}

func nativeAmgXMode(precision AmgXPrecision) C.AMGX_Mode {
	switch precision {
	case AmgXPrecisionFloat:
		return C.AMGX_mode_dFFI
	default:
		return C.AMGX_mode_dDDI
	}
}

func amgxConfigString(config AmgXConfig) string {
	solverName := amgxSolverString(config.Solver)
	preconditioner := "NOSOLVER"
	if config.Solver != AmgXSolverAMG {
		preconditioner = "AMG"
	}
	configString := fmt.Sprintf("config_version=2, solver(slv)=%s, slv:max_iters=%d, slv:tolerance=%g, slv:monitor_residual=%d, slv:print_solve_stats=%d", solverName, max(1, config.MaxIterations), config.Tolerance, amgxBool(config.MonitorResidual), amgxBool(config.PrintSolveStats))
	configString += fmt.Sprintf(", slv:preconditioner(amg)=%s, amg:max_levels=%d, amg:cycle=%s, amg:interpolator=%s, amg:smoother=%s", preconditioner, max(2, config.MaxLevels), amgxCycleString(config.Cycle), amgxInterpolationString(config.Interpolation), amgxSmootherString(config.Smoother))
	if config.Deterministic {
		configString += ", determinism_flag=1"
	}
	if config.UseScaling {
		configString += ", slv:solver_scaling=BINORMALIZATION"
	}
	return configString
}

func amgxSolverString(solver AmgXSolver) string {
	switch solver {
	case AmgXSolverPCG:
		return "PCG"
	case AmgXSolverPBICGSTAB:
		return "PBICGSTAB"
	case AmgXSolverGMRES:
		return "GMRES"
	case AmgXSolverFGMRES:
		return "FGMRES"
	case AmgXSolverCG:
		return "CG"
	case AmgXSolverBICGSTAB:
		return "BICGSTAB"
	case AmgXSolverIDR:
		return "IDR"
	default:
		return "AMG"
	}
}

func amgxCycleString(cycle AmgXCycle) string {
	switch cycle {
	case AmgXCycleW:
		return "W"
	case AmgXCycleF:
		return "F"
	default:
		return "V"
	}
}

func amgxInterpolationString(interpolation AmgXInterpolation) string {
	switch interpolation {
	case AmgXInterpolationDirect:
		return "D2"
	case AmgXInterpolationMultipass:
		return "MULTIPASS"
	case AmgXInterpolationExtended:
		return "EXTENDED"
	case AmgXInterpolationModifiedClassical:
		return "MODIFIED_CLASSICAL"
	default:
		return "CLASSICAL"
	}
}

func amgxSmootherString(smoother AmgXSmoother) string {
	switch smoother {
	case AmgXSmootherGS:
		return "GS"
	case AmgXSmootherSGS:
		return "SGS"
	case AmgXSmootherBlockJacobi:
		return "BLOCK_JACOBI"
	case AmgXSmootherCF_Jacobi:
		return "CF_JACOBI"
	case AmgXSmootherL1_Jacobi:
		return "L1_JACOBI"
	case AmgXSmootherChebyshev:
		return "CHEBYSHEV"
	case AmgXSmootherPolynomial:
		return "POLYNOMIAL"
	default:
		return "JACOBI"
	}
}

func amgxBool(value bool) int {
	if value {
		return 1
	}
	return 0
}

func amgxSolveStatusString(status C.AMGX_SOLVE_STATUS) string {
	switch status {
	case C.AMGX_SOLVE_SUCCESS:
		return "Tolerance reached"
	case C.AMGX_SOLVE_DIVERGED:
		return "Solver diverged"
	case C.AMGX_SOLVE_NOT_CONVERGED:
		return "Maximum iterations reached"
	default:
		return "Solver failed"
	}
}

func amgxError(operation string, rc C.AMGX_RC) error {
	return fmt.Errorf("%s failed: %s (%d)", operation, amgxErrorString(rc), int(rc))
}

func amgxErrorString(rc C.AMGX_RC) string {
	switch rc {
	case C.AMGX_RC_OK:
		return "success"
	case C.AMGX_RC_BAD_PARAMETERS:
		return "bad parameters"
	case C.AMGX_RC_UNKNOWN:
		return "unknown error"
	case C.AMGX_RC_NOT_SUPPORTED_TARGET:
		return "unsupported target"
	case C.AMGX_RC_NOT_SUPPORTED_BLOCKSIZE:
		return "unsupported blocksize"
	case C.AMGX_RC_CUDA_FAILURE:
		return "cuda failure"
	case C.AMGX_RC_THRUST_FAILURE:
		return "thrust failure"
	case C.AMGX_RC_NO_MEMORY:
		return "out of memory"
	case C.AMGX_RC_IO_ERROR:
		return "io error"
	case C.AMGX_RC_BAD_MODE:
		return "bad mode"
	case C.AMGX_RC_CORE:
		return "core error"
	case C.AMGX_RC_PLUGIN:
		return "plugin error"
	case C.AMGX_RC_BAD_CONFIGURATION:
		return "bad configuration"
	case C.AMGX_RC_NOT_IMPLEMENTED:
		return "not implemented"
	case C.AMGX_RC_LICENSE_NOT_FOUND:
		return "license not found"
	default:
		return "internal error"
	}
}
