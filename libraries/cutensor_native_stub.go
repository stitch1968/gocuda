//go:build !cuda

package libraries

import "github.com/stitch1968/gocuda/memory"

func cutensorNativeAvailable() bool {
	return false
}

func createNativeCuTensorHandle() (*CuTensorHandle, error) {
	return nil, errCuTensorUnsupported
}

func executeNativeTensorContraction(handle *CuTensorHandle, desc *ContractionDescriptor, tensorA, tensorB, tensorC *memory.Memory) error {
	return errCuTensorUnsupported
}

func executeNativeTensorElementwise(handle *CuTensorHandle, descA *CuTensorDescriptor, tensorA *memory.Memory, descB *CuTensorDescriptor, tensorB *memory.Memory, descC *CuTensorDescriptor, tensorC *memory.Memory, alpha, beta, gamma float64, multiply bool) error {
	return errCuTensorUnsupported
}

func executeNativeTensorReduce(handle *CuTensorHandle, descA *CuTensorDescriptor, tensorA *memory.Memory, descC *CuTensorDescriptor, tensorC *memory.Memory, reduceModes []int, reductionOp TensorReduction, alpha, beta float64) error {
	return errCuTensorUnsupported
}

func executeNativeTensorPermute(handle *CuTensorHandle, descA *CuTensorDescriptor, tensorA *memory.Memory, descC *CuTensorDescriptor, tensorC *memory.Memory, perm []int, alpha float64) error {
	return errCuTensorUnsupported
}

func createNativeContractionPlan(handle *CuTensorHandle, descA *CuTensorDescriptor, modesA []int, descB *CuTensorDescriptor, modesB []int, descC *CuTensorDescriptor, modesC []int, algorithm ContractionAlgorithm) (*TensorPlan, error) {
	return nil, errCuTensorUnsupported
}

func executeNativeContractionPlan(handle *CuTensorHandle, plan *TensorPlan, alpha float64, tensorA, tensorB *memory.Memory, beta float64, tensorC *memory.Memory) error {
	return errCuTensorUnsupported
}

func destroyNativeCuTensorHandle(handle *CuTensorHandle) error {
	return nil
}

func destroyNativeTensorPlan(plan *TensorPlan) error {
	return nil
}
