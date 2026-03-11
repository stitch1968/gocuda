//go:build cuda

package libraries

/*
#cgo linux CFLAGS: -I/usr/local/cuda/include -I/opt/cuda/include
#cgo windows CFLAGS: -I"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.1/include" -I"D:/NVIDIA/include"
#cgo linux LDFLAGS: -L/usr/local/cuda/lib64 -L/opt/cuda/lib64 -lcutensor
#cgo windows LDFLAGS: -L${SRCDIR}/../lib_mingw -lcutensor

#include <cuda_runtime.h>
#include <cutensor.h>
*/
import "C"

import (
	"fmt"
	"unsafe"

	"github.com/stitch1968/gocuda/memory"
)

func cutensorNativeAvailable() bool {
	return true
}

func createNativeCuTensorHandle() (*CuTensorHandle, error) {
	var nativeHandle C.cutensorHandle_t
	if status := C.cutensorCreate(&nativeHandle); status != C.CUTENSOR_STATUS_SUCCESS {
		return nil, cutensorError("cutensorCreate", status)
	}

	return &CuTensorHandle{
		planCache:    make(map[string]*TensorPlan),
		descriptors:  make([]*CuTensorDescriptor, 0),
		computeType:  TensorFloat32,
		mathMode:     TensorMathDefault,
		nativeHandle: uintptr(nativeHandle),
		native:       true,
	}, nil
}

func executeNativeTensorContraction(handle *CuTensorHandle, desc *ContractionDescriptor, tensorA, tensorB, tensorC *memory.Memory) error {
	plan, err := buildNativeContractionPlan(handle, desc.TensorA, desc.ModesA, desc.TensorB, desc.ModesB, desc.TensorC, desc.ModesC, desc.Algorithm)
	if err != nil {
		return err
	}
	defer plan.Destroy()

	return executeNativeContractionPlan(handle, plan, desc.Alpha, tensorA, tensorB, desc.Beta, tensorC)
}

func executeNativeTensorElementwise(handle *CuTensorHandle, descA *CuTensorDescriptor, tensorA *memory.Memory, descB *CuTensorDescriptor, tensorB *memory.Memory, descC *CuTensorDescriptor, tensorC *memory.Memory, alpha, beta, gamma float64, multiply bool) error {
	if len(descA.dimensions) != len(descB.dimensions) || len(descA.dimensions) != len(descC.dimensions) {
		return fmt.Errorf("elementwise tensor ranks must match")
	}
	computeDesc, err := cutensorComputeDescriptor(descC.dataType)
	if err != nil {
		return err
	}

	nativeDescA, err := cutensorCreateTensorDescriptor(handle, descA)
	if err != nil {
		return err
	}
	defer C.cutensorDestroyTensorDescriptor(nativeDescA)

	nativeDescB, err := cutensorCreateTensorDescriptor(handle, descB)
	if err != nil {
		return err
	}
	defer C.cutensorDestroyTensorDescriptor(nativeDescB)

	nativeDescC, err := cutensorCreateTensorDescriptor(handle, descC)
	if err != nil {
		return err
	}
	defer C.cutensorDestroyTensorDescriptor(nativeDescC)

	modes := cutensorNaturalModes(len(descA.dimensions))
	var opDesc C.cutensorOperationDescriptor_t
	opAB := C.cutensorOperator_t(C.CUTENSOR_OP_ADD)
	if multiply {
		opAB = C.cutensorOperator_t(C.CUTENSOR_OP_MUL)
	}
	if status := C.cutensorCreateElementwiseTrinary(
		C.cutensorHandle_t(handle.nativeHandle),
		&opDesc,
		nativeDescA,
		cutensorModePointer(modes),
		C.cutensorOperator_t(C.CUTENSOR_OP_IDENTITY),
		nativeDescB,
		cutensorModePointer(modes),
		C.cutensorOperator_t(C.CUTENSOR_OP_IDENTITY),
		nativeDescC,
		cutensorModePointer(modes),
		C.cutensorOperator_t(C.CUTENSOR_OP_IDENTITY),
		nativeDescC,
		cutensorModePointer(modes),
		opAB,
		C.cutensorOperator_t(C.CUTENSOR_OP_ADD),
		computeDesc,
	); status != C.CUTENSOR_STATUS_SUCCESS {
		return cutensorError("cutensorCreateElementwiseTrinary", status)
	}
	defer C.cutensorDestroyOperationDescriptor(opDesc)

	planPref, plan, workspaceSize, err := cutensorCreateContractionPlanResources(handle, opDesc)
	if err != nil {
		return err
	}
	defer C.cutensorDestroyPlanPreference(planPref)
	defer C.cutensorDestroyPlan(plan)

	workspacePtr, err := cutensorEnsureWorkspace(handle, workspaceSize)
	if err != nil {
		return err
	}

	switch descC.dataType {
	case TensorFloat32:
		alpha32 := float32(alpha)
		beta32 := float32(beta)
		gamma32 := float32(gamma)
		if status := C.cutensorElementwiseTrinaryExecute(
			C.cutensorHandle_t(handle.nativeHandle),
			plan,
			unsafe.Pointer(&alpha32),
			tensorA.Ptr(),
			unsafe.Pointer(&beta32),
			tensorB.Ptr(),
			unsafe.Pointer(&gamma32),
			tensorC.Ptr(),
			tensorC.Ptr(),
			nil,
		); status != C.CUTENSOR_STATUS_SUCCESS {
			return cutensorError("cutensorElementwiseTrinaryExecute", status)
		}
	case TensorFloat64:
		alpha64 := alpha
		beta64 := beta
		gamma64 := gamma
		if status := C.cutensorElementwiseTrinaryExecute(
			C.cutensorHandle_t(handle.nativeHandle),
			plan,
			unsafe.Pointer(&alpha64),
			tensorA.Ptr(),
			unsafe.Pointer(&beta64),
			tensorB.Ptr(),
			unsafe.Pointer(&gamma64),
			tensorC.Ptr(),
			tensorC.Ptr(),
			nil,
		); status != C.CUTENSOR_STATUS_SUCCESS {
			return cutensorError("cutensorElementwiseTrinaryExecute", status)
		}
	default:
		_ = workspacePtr
		return errCuTensorUnsupported
	}

	return nil
}

func executeNativeTensorReduce(handle *CuTensorHandle, descA *CuTensorDescriptor, tensorA *memory.Memory, descC *CuTensorDescriptor, tensorC *memory.Memory, reduceModes []int, reductionOp TensorReduction, alpha, beta float64) error {
	if len(reduceModes) == len(descA.dimensions) {
		return errCuTensorUnsupported
	}
	computeDesc, err := cutensorComputeDescriptor(descA.dataType)
	if err != nil {
		return err
	}
	reduceOp, err := cutensorReductionOperator(reductionOp)
	if err != nil {
		return err
	}

	nativeDescA, err := cutensorCreateTensorDescriptor(handle, descA)
	if err != nil {
		return err
	}
	defer C.cutensorDestroyTensorDescriptor(nativeDescA)

	nativeDescC, err := cutensorCreateTensorDescriptor(handle, descC)
	if err != nil {
		return err
	}
	defer C.cutensorDestroyTensorDescriptor(nativeDescC)

	modesA := cutensorNaturalModes(len(descA.dimensions))
	modesC, err := cutensorReductionOutputModes(descA, reduceModes)
	if err != nil {
		return err
	}

	var opDesc C.cutensorOperationDescriptor_t
	if status := C.cutensorCreateReduction(
		C.cutensorHandle_t(handle.nativeHandle),
		&opDesc,
		nativeDescA,
		cutensorModePointer(modesA),
		C.cutensorOperator_t(C.CUTENSOR_OP_IDENTITY),
		nativeDescC,
		cutensorModePointer(modesC),
		C.cutensorOperator_t(C.CUTENSOR_OP_IDENTITY),
		nativeDescC,
		cutensorModePointer(modesC),
		reduceOp,
		computeDesc,
	); status != C.CUTENSOR_STATUS_SUCCESS {
		return cutensorError("cutensorCreateReduction", status)
	}
	defer C.cutensorDestroyOperationDescriptor(opDesc)

	planPref, plan, workspaceSize, err := cutensorCreateContractionPlanResources(handle, opDesc)
	if err != nil {
		return err
	}
	defer C.cutensorDestroyPlanPreference(planPref)
	defer C.cutensorDestroyPlan(plan)

	workspacePtr, err := cutensorEnsureWorkspace(handle, workspaceSize)
	if err != nil {
		return err
	}

	switch descA.dataType {
	case TensorFloat32:
		alpha32 := float32(alpha)
		beta32 := float32(beta)
		if status := C.cutensorReduce(
			C.cutensorHandle_t(handle.nativeHandle),
			plan,
			unsafe.Pointer(&alpha32),
			tensorA.Ptr(),
			unsafe.Pointer(&beta32),
			tensorC.Ptr(),
			tensorC.Ptr(),
			workspacePtr,
			C.uint64_t(workspaceSize),
			nil,
		); status != C.CUTENSOR_STATUS_SUCCESS {
			return cutensorError("cutensorReduce", status)
		}
	case TensorFloat64:
		alpha64 := alpha
		beta64 := beta
		if status := C.cutensorReduce(
			C.cutensorHandle_t(handle.nativeHandle),
			plan,
			unsafe.Pointer(&alpha64),
			tensorA.Ptr(),
			unsafe.Pointer(&beta64),
			tensorC.Ptr(),
			tensorC.Ptr(),
			workspacePtr,
			C.uint64_t(workspaceSize),
			nil,
		); status != C.CUTENSOR_STATUS_SUCCESS {
			return cutensorError("cutensorReduce", status)
		}
	default:
		return errCuTensorUnsupported
	}

	return nil
}

func executeNativeTensorPermute(handle *CuTensorHandle, descA *CuTensorDescriptor, tensorA *memory.Memory, descC *CuTensorDescriptor, tensorC *memory.Memory, perm []int, alpha float64) error {
	computeDesc, err := cutensorComputeDescriptor(descA.dataType)
	if err != nil {
		return err
	}

	nativeDescA, err := cutensorCreateTensorDescriptor(handle, descA)
	if err != nil {
		return err
	}
	defer C.cutensorDestroyTensorDescriptor(nativeDescA)

	nativeDescC, err := cutensorCreateTensorDescriptor(handle, descC)
	if err != nil {
		return err
	}
	defer C.cutensorDestroyTensorDescriptor(nativeDescC)

	modesA := cutensorNaturalModes(len(descA.dimensions))
	modesC := cutensorPermutedModes(len(descA.dimensions), perm)

	var opDesc C.cutensorOperationDescriptor_t
	if status := C.cutensorCreatePermutation(
		C.cutensorHandle_t(handle.nativeHandle),
		&opDesc,
		nativeDescA,
		cutensorModePointer(modesA),
		C.cutensorOperator_t(C.CUTENSOR_OP_IDENTITY),
		nativeDescC,
		cutensorModePointer(modesC),
		computeDesc,
	); status != C.CUTENSOR_STATUS_SUCCESS {
		return cutensorError("cutensorCreatePermutation", status)
	}
	defer C.cutensorDestroyOperationDescriptor(opDesc)

	planPref, plan, _, err := cutensorCreateContractionPlanResources(handle, opDesc)
	if err != nil {
		return err
	}
	defer C.cutensorDestroyPlanPreference(planPref)
	defer C.cutensorDestroyPlan(plan)

	switch descA.dataType {
	case TensorFloat32:
		alpha32 := float32(alpha)
		if status := C.cutensorPermute(
			C.cutensorHandle_t(handle.nativeHandle),
			plan,
			unsafe.Pointer(&alpha32),
			tensorA.Ptr(),
			tensorC.Ptr(),
			nil,
		); status != C.CUTENSOR_STATUS_SUCCESS {
			return cutensorError("cutensorPermute", status)
		}
	case TensorFloat64:
		alpha64 := alpha
		if status := C.cutensorPermute(
			C.cutensorHandle_t(handle.nativeHandle),
			plan,
			unsafe.Pointer(&alpha64),
			tensorA.Ptr(),
			tensorC.Ptr(),
			nil,
		); status != C.CUTENSOR_STATUS_SUCCESS {
			return cutensorError("cutensorPermute", status)
		}
	default:
		return errCuTensorUnsupported
	}

	return nil
}

func createNativeContractionPlan(handle *CuTensorHandle, descA *CuTensorDescriptor, modesA []int, descB *CuTensorDescriptor, modesB []int, descC *CuTensorDescriptor, modesC []int, algorithm ContractionAlgorithm) (*TensorPlan, error) {
	return buildNativeContractionPlan(handle, descA, modesA, descB, modesB, descC, modesC, algorithm)
}

func executeNativeContractionPlan(handle *CuTensorHandle, plan *TensorPlan, alpha float64, tensorA, tensorB *memory.Memory, beta float64, tensorC *memory.Memory) error {
	if handle == nil || !handle.native || handle.nativeHandle == 0 {
		return errCuTensorUnsupported
	}
	if plan == nil || !plan.native || plan.nativeHandle == 0 {
		return errCuTensorUnsupported
	}

	workspacePtr, err := cutensorEnsureWorkspace(handle, uint64(plan.workspaceSize))
	if err != nil {
		return err
	}

	switch plan.descC.dataType {
	case TensorFloat32:
		alpha32 := float32(alpha)
		beta32 := float32(beta)
		if status := C.cutensorContract(
			C.cutensorHandle_t(handle.nativeHandle),
			C.cutensorPlan_t(plan.nativeHandle),
			unsafe.Pointer(&alpha32),
			tensorA.Ptr(),
			tensorB.Ptr(),
			unsafe.Pointer(&beta32),
			tensorC.Ptr(),
			tensorC.Ptr(),
			workspacePtr,
			C.uint64_t(plan.workspaceSize),
			nil,
		); status != C.CUTENSOR_STATUS_SUCCESS {
			return cutensorError("cutensorContract", status)
		}
	case TensorFloat64:
		alpha64 := alpha
		beta64 := beta
		if status := C.cutensorContract(
			C.cutensorHandle_t(handle.nativeHandle),
			C.cutensorPlan_t(plan.nativeHandle),
			unsafe.Pointer(&alpha64),
			tensorA.Ptr(),
			tensorB.Ptr(),
			unsafe.Pointer(&beta64),
			tensorC.Ptr(),
			tensorC.Ptr(),
			workspacePtr,
			C.uint64_t(plan.workspaceSize),
			nil,
		); status != C.CUTENSOR_STATUS_SUCCESS {
			return cutensorError("cutensorContract", status)
		}
	default:
		return errCuTensorUnsupported
	}

	return nil
}

func destroyNativeCuTensorHandle(handle *CuTensorHandle) error {
	if handle.nativeHandle != 0 {
		if status := C.cutensorDestroy(C.cutensorHandle_t(handle.nativeHandle)); status != C.CUTENSOR_STATUS_SUCCESS {
			return cutensorError("cutensorDestroy", status)
		}
		handle.nativeHandle = 0
	}
	handle.native = false
	return nil
}

func destroyNativeTensorPlan(plan *TensorPlan) error {
	if plan.nativeHandle != 0 {
		if status := C.cutensorDestroyPlan(C.cutensorPlan_t(plan.nativeHandle)); status != C.CUTENSOR_STATUS_SUCCESS {
			return cutensorError("cutensorDestroyPlan", status)
		}
		plan.nativeHandle = 0
	}
	if plan.nativePlanPref != 0 {
		if status := C.cutensorDestroyPlanPreference(C.cutensorPlanPreference_t(plan.nativePlanPref)); status != C.CUTENSOR_STATUS_SUCCESS {
			return cutensorError("cutensorDestroyPlanPreference", status)
		}
		plan.nativePlanPref = 0
	}
	if plan.nativeOpDesc != 0 {
		if status := C.cutensorDestroyOperationDescriptor(C.cutensorOperationDescriptor_t(plan.nativeOpDesc)); status != C.CUTENSOR_STATUS_SUCCESS {
			return cutensorError("cutensorDestroyOperationDescriptor", status)
		}
		plan.nativeOpDesc = 0
	}
	if plan.nativeDescA != 0 {
		if status := C.cutensorDestroyTensorDescriptor(C.cutensorTensorDescriptor_t(plan.nativeDescA)); status != C.CUTENSOR_STATUS_SUCCESS {
			return cutensorError("cutensorDestroyTensorDescriptor(A)", status)
		}
		plan.nativeDescA = 0
	}
	if plan.nativeDescB != 0 {
		if status := C.cutensorDestroyTensorDescriptor(C.cutensorTensorDescriptor_t(plan.nativeDescB)); status != C.CUTENSOR_STATUS_SUCCESS {
			return cutensorError("cutensorDestroyTensorDescriptor(B)", status)
		}
		plan.nativeDescB = 0
	}
	if plan.nativeDescC != 0 {
		if status := C.cutensorDestroyTensorDescriptor(C.cutensorTensorDescriptor_t(plan.nativeDescC)); status != C.CUTENSOR_STATUS_SUCCESS {
			return cutensorError("cutensorDestroyTensorDescriptor(C)", status)
		}
		plan.nativeDescC = 0
	}
	plan.native = false
	return nil
}

func buildNativeContractionPlan(handle *CuTensorHandle, descA *CuTensorDescriptor, modesA []int, descB *CuTensorDescriptor, modesB []int, descC *CuTensorDescriptor, modesC []int, algorithm ContractionAlgorithm) (*TensorPlan, error) {
	_ = algorithm
	computeDesc, err := cutensorComputeDescriptor(descC.dataType)
	if err != nil {
		return nil, err
	}
	if descA.dataType != descB.dataType || descA.dataType != descC.dataType {
		return nil, fmt.Errorf("tensor contraction requires matching data types")
	}
	if len(modesA) != len(descA.dimensions) || len(modesB) != len(descB.dimensions) || len(modesC) != len(descC.dimensions) {
		return nil, fmt.Errorf("mode count must match tensor dimensions")
	}

	nativeDescA, err := cutensorCreateTensorDescriptor(handle, descA)
	if err != nil {
		return nil, err
	}
	cleanupA := true
	defer func() {
		if cleanupA {
			_ = C.cutensorDestroyTensorDescriptor(nativeDescA)
		}
	}()

	nativeDescB, err := cutensorCreateTensorDescriptor(handle, descB)
	if err != nil {
		return nil, err
	}
	cleanupB := true
	defer func() {
		if cleanupB {
			_ = C.cutensorDestroyTensorDescriptor(nativeDescB)
		}
	}()

	nativeDescC, err := cutensorCreateTensorDescriptor(handle, descC)
	if err != nil {
		return nil, err
	}
	cleanupC := true
	defer func() {
		if cleanupC {
			_ = C.cutensorDestroyTensorDescriptor(nativeDescC)
		}
	}()

	modeA, err := cutensorModesFromInts(modesA)
	if err != nil {
		return nil, err
	}
	modeB, err := cutensorModesFromInts(modesB)
	if err != nil {
		return nil, err
	}
	modeC, err := cutensorModesFromInts(modesC)
	if err != nil {
		return nil, err
	}

	var opDesc C.cutensorOperationDescriptor_t
	if status := C.cutensorCreateContraction(
		C.cutensorHandle_t(handle.nativeHandle),
		&opDesc,
		nativeDescA,
		cutensorModePointer(modeA),
		C.cutensorOperator_t(C.CUTENSOR_OP_IDENTITY),
		nativeDescB,
		cutensorModePointer(modeB),
		C.cutensorOperator_t(C.CUTENSOR_OP_IDENTITY),
		nativeDescC,
		cutensorModePointer(modeC),
		C.cutensorOperator_t(C.CUTENSOR_OP_IDENTITY),
		nativeDescC,
		cutensorModePointer(modeC),
		computeDesc,
	); status != C.CUTENSOR_STATUS_SUCCESS {
		return nil, cutensorError("cutensorCreateContraction", status)
	}
	cleanupOp := true
	defer func() {
		if cleanupOp {
			_ = C.cutensorDestroyOperationDescriptor(opDesc)
		}
	}()

	planPref, planHandle, workspaceSize, err := cutensorCreateContractionPlanResources(handle, opDesc)
	if err != nil {
		return nil, err
	}

	cleanupA = false
	cleanupB = false
	cleanupC = false
	cleanupOp = false

	return &TensorPlan{
		operation:      TensorOpContraction,
		algorithm:      algorithm,
		workspaceSize:  int64(workspaceSize),
		descA:          descA,
		descB:          descB,
		descC:          descC,
		modesA:         append([]int(nil), modesA...),
		modesB:         append([]int(nil), modesB...),
		modesC:         append([]int(nil), modesC...),
		nativeHandle:   uintptr(planHandle),
		nativeOpDesc:   uintptr(opDesc),
		nativePlanPref: uintptr(planPref),
		nativeDescA:    uintptr(nativeDescA),
		nativeDescB:    uintptr(nativeDescB),
		nativeDescC:    uintptr(nativeDescC),
		native:         true,
		memoryReq:      calculateMemoryRequirement(descA, descB, descC),
	}, nil
}

func cutensorCreateTensorDescriptor(handle *CuTensorHandle, desc *CuTensorDescriptor) (C.cutensorTensorDescriptor_t, error) {
	dataType, err := cutensorCUDADataType(desc.dataType)
	if err != nil {
		return nil, err
	}
	if len(desc.dimensions) == 0 {
		return nil, fmt.Errorf("tensor must have at least one dimension")
	}

	extents := make([]C.int64_t, len(desc.dimensions))
	strides := make([]C.int64_t, len(desc.strides))
	elementSize := getTensorDataTypeSize(desc.dataType)
	for index, dim := range desc.dimensions {
		extents[index] = C.int64_t(dim)
	}
	for index, stride := range desc.strides {
		strides[index] = C.int64_t(stride / elementSize)
	}

	var nativeDesc C.cutensorTensorDescriptor_t
	if status := C.cutensorCreateTensorDescriptor(
		C.cutensorHandle_t(handle.nativeHandle),
		&nativeDesc,
		C.uint32_t(len(extents)),
		(*C.int64_t)(unsafe.Pointer(&extents[0])),
		(*C.int64_t)(unsafe.Pointer(&strides[0])),
		dataType,
		C.uint32_t(cutensorAlignment(desc.alignReq)),
	); status != C.CUTENSOR_STATUS_SUCCESS {
		return nil, cutensorError("cutensorCreateTensorDescriptor", status)
	}
	return nativeDesc, nil
}

func cutensorCreateContractionPlanResources(handle *CuTensorHandle, opDesc C.cutensorOperationDescriptor_t) (C.cutensorPlanPreference_t, C.cutensorPlan_t, uint64, error) {
	var planPref C.cutensorPlanPreference_t
	if status := C.cutensorCreatePlanPreference(
		C.cutensorHandle_t(handle.nativeHandle),
		&planPref,
		C.cutensorAlgo_t(C.CUTENSOR_ALGO_DEFAULT),
		C.cutensorJitMode_t(C.CUTENSOR_JIT_MODE_NONE),
	); status != C.CUTENSOR_STATUS_SUCCESS {
		return nil, nil, 0, cutensorError("cutensorCreatePlanPreference", status)
	}
	cleanupPref := true
	defer func() {
		if cleanupPref {
			_ = C.cutensorDestroyPlanPreference(planPref)
		}
	}()

	workspaceEstimate := C.uint64_t(0)
	if status := C.cutensorEstimateWorkspaceSize(
		C.cutensorHandle_t(handle.nativeHandle),
		opDesc,
		planPref,
		C.cutensorWorksizePreference_t(C.CUTENSOR_WORKSPACE_DEFAULT),
		&workspaceEstimate,
	); status != C.CUTENSOR_STATUS_SUCCESS {
		return nil, nil, 0, cutensorError("cutensorEstimateWorkspaceSize", status)
	}

	var plan C.cutensorPlan_t
	if status := C.cutensorCreatePlan(
		C.cutensorHandle_t(handle.nativeHandle),
		&plan,
		opDesc,
		planPref,
		workspaceEstimate,
	); status != C.CUTENSOR_STATUS_SUCCESS {
		return nil, nil, 0, cutensorError("cutensorCreatePlan", status)
	}
	cleanupPlan := true
	defer func() {
		if cleanupPlan {
			_ = C.cutensorDestroyPlan(plan)
		}
	}()

	workspaceRequired := C.uint64_t(0)
	if status := C.cutensorPlanGetAttribute(
		C.cutensorHandle_t(handle.nativeHandle),
		plan,
		C.cutensorPlanAttribute_t(C.CUTENSOR_PLAN_REQUIRED_WORKSPACE),
		unsafe.Pointer(&workspaceRequired),
		C.size_t(unsafe.Sizeof(workspaceRequired)),
	); status != C.CUTENSOR_STATUS_SUCCESS {
		return nil, nil, 0, cutensorError("cutensorPlanGetAttribute", status)
	}

	cleanupPref = false
	cleanupPlan = false
	return planPref, plan, uint64(workspaceRequired), nil
}

func cutensorEnsureWorkspace(handle *CuTensorHandle, required uint64) (unsafe.Pointer, error) {
	if required == 0 {
		return nil, nil
	}
	if handle.workspace == nil || handle.workspace.Size() < int64(required) {
		if handle.workspace != nil {
			if err := handle.workspace.Free(); err != nil {
				return nil, err
			}
		}
		workspace, err := memory.Alloc(int64(required))
		if err != nil {
			return nil, err
		}
		handle.workspace = workspace
	}
	return handle.workspace.Ptr(), nil
}

func cutensorNaturalModes(rank int) []C.int32_t {
	modes := make([]C.int32_t, rank)
	for index := range modes {
		modes[index] = C.int32_t(index)
	}
	return modes
}

func cutensorModesFromInts(modes []int) ([]C.int32_t, error) {
	result := make([]C.int32_t, len(modes))
	for index, mode := range modes {
		if mode < 0 {
			return nil, fmt.Errorf("invalid tensor mode: %d", mode)
		}
		result[index] = C.int32_t(mode)
	}
	return result, nil
}

func cutensorPermutedModes(rank int, perm []int) []C.int32_t {
	result := make([]C.int32_t, rank)
	for inputIndex, targetIndex := range perm {
		result[targetIndex] = C.int32_t(inputIndex)
	}
	return result
}

func cutensorReductionOutputModes(descA *CuTensorDescriptor, reduceModes []int) ([]C.int32_t, error) {
	reduceSet := make(map[int]bool, len(reduceModes))
	for _, mode := range reduceModes {
		if mode < 0 || mode >= len(descA.dimensions) {
			return nil, fmt.Errorf("invalid reduction mode: %d", mode)
		}
		reduceSet[mode] = true
	}
	modes := make([]C.int32_t, 0, len(descA.dimensions)-len(reduceModes))
	for index := range descA.dimensions {
		if !reduceSet[index] {
			modes = append(modes, C.int32_t(index))
		}
	}
	if len(modes) == 0 {
		return nil, errCuTensorUnsupported
	}
	return modes, nil
}

func cutensorModePointer(modes []C.int32_t) *C.int32_t {
	if len(modes) == 0 {
		return nil
	}
	return (*C.int32_t)(unsafe.Pointer(&modes[0]))
}

func cutensorComputeDescriptor(dataType TensorDataType) (C.cutensorComputeDescriptor_t, error) {
	switch dataType {
	case TensorFloat32:
		return C.cutensorComputeDescriptor_t(C.CUTENSOR_COMPUTE_DESC_32F), nil
	case TensorFloat64:
		return C.cutensorComputeDescriptor_t(C.CUTENSOR_COMPUTE_DESC_64F), nil
	default:
		return 0, errCuTensorUnsupported
	}
}

func cutensorCUDADataType(dataType TensorDataType) (C.cudaDataType_t, error) {
	switch dataType {
	case TensorFloat32:
		return C.cudaDataType_t(C.CUDA_R_32F), nil
	case TensorFloat64:
		return C.cudaDataType_t(C.CUDA_R_64F), nil
	default:
		return 0, errCuTensorUnsupported
	}
}

func cutensorReductionOperator(reductionOp TensorReduction) (C.cutensorOperator_t, error) {
	switch reductionOp {
	case TensorReduceSum, TensorReduceMean, TensorReduceNorm1, TensorReduceNorm2:
		return C.cutensorOperator_t(C.CUTENSOR_OP_ADD), nil
	case TensorReduceMax, TensorReduceNormInf:
		return C.cutensorOperator_t(C.CUTENSOR_OP_MAX), nil
	case TensorReduceMin:
		return C.cutensorOperator_t(C.CUTENSOR_OP_MIN), nil
	default:
		return 0, errCuTensorUnsupported
	}
}

func cutensorAlignment(alignment int) int {
	if alignment > 0 {
		return alignment
	}
	return 256
}

func cutensorError(operation string, status C.cutensorStatus_t) error {
	base := fmt.Errorf("%s failed: %s (%d)", operation, C.GoString(C.cutensorGetErrorString(status)), int(status))
	switch status {
	case C.CUTENSOR_STATUS_NOT_SUPPORTED, C.CUTENSOR_STATUS_ARCH_MISMATCH:
		return fmt.Errorf("%w: %v", errCuTensorUnsupported, base)
	default:
		return base
	}
}
