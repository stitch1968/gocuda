//go:build !cuda || windows

package libraries

import (
	"fmt"

	"github.com/stitch1968/gocuda/memory"
)

func cudnnNativeAvailable() bool {
	return false
	}

func createNativeDNNHandle() (*DNNHandle, error) {
	return nil, fmt.Errorf("cuDNN native backend requires a non-Windows CUDA-tagged build")
}

func destroyNativeDNNHandle(handle *DNNHandle) error {
	return fmt.Errorf("cuDNN native backend requires a non-Windows CUDA-tagged build")
}

func createNativeTensorDescriptor() (*TensorDescriptor, error) {
	return nil, fmt.Errorf("cuDNN native backend requires a non-Windows CUDA-tagged build")
}

func setNativeTensorNdDescriptor(desc *TensorDescriptor, dataType DNNDataType, dimensions []int, strides []int) error {
	return fmt.Errorf("cuDNN native backend requires a non-Windows CUDA-tagged build")
}

func setNativeTensor4dDescriptor(desc *TensorDescriptor, format DNNTensorFormat, dataType DNNDataType, n, c, h, w int) error {
	return fmt.Errorf("cuDNN native backend requires a non-Windows CUDA-tagged build")
}

func destroyNativeTensorDescriptor(desc *TensorDescriptor) error {
	return fmt.Errorf("cuDNN native backend requires a non-Windows CUDA-tagged build")
}

func createNativeFilterDescriptor() (*FilterDescriptor, error) {
	return nil, fmt.Errorf("cuDNN native backend requires a non-Windows CUDA-tagged build")
}

func setNativeFilterNdDescriptor(desc *FilterDescriptor, dataType DNNDataType, format DNNTensorFormat, dimensions []int) error {
	return fmt.Errorf("cuDNN native backend requires a non-Windows CUDA-tagged build")
}

func setNativeFilter4dDescriptor(desc *FilterDescriptor, dataType DNNDataType, format DNNTensorFormat, k, c, h, w int) error {
	return fmt.Errorf("cuDNN native backend requires a non-Windows CUDA-tagged build")
}

func destroyNativeFilterDescriptor(desc *FilterDescriptor) error {
	return fmt.Errorf("cuDNN native backend requires a non-Windows CUDA-tagged build")
}

func createNativeConvolutionDescriptor() (*ConvolutionDescriptor, error) {
	return nil, fmt.Errorf("cuDNN native backend requires a non-Windows CUDA-tagged build")
}

func setNativeConvolution2dDescriptor(desc *ConvolutionDescriptor, padH, padW, strideH, strideW, dilationH, dilationW int, mode DNNConvolutionMode, dataType DNNDataType) error {
	return fmt.Errorf("cuDNN native backend requires a non-Windows CUDA-tagged build")
}

func nativeConvolutionForwardOutputDim(desc *ConvolutionDescriptor, inputDesc *TensorDescriptor, filterDesc *FilterDescriptor) (int, int, int, int, error) {
	return 0, 0, 0, 0, fmt.Errorf("cuDNN native backend requires a non-Windows CUDA-tagged build")
}

func destroyNativeConvolutionDescriptor(desc *ConvolutionDescriptor) error {
	return fmt.Errorf("cuDNN native backend requires a non-Windows CUDA-tagged build")
}

func createNativePoolingDescriptor() (*PoolingDescriptor, error) {
	return nil, fmt.Errorf("cuDNN native backend requires a non-Windows CUDA-tagged build")
}

func setNativePooling2dDescriptor(desc *PoolingDescriptor, mode DNNPoolingMode, windowH, windowW, padH, padW, strideH, strideW int) error {
	return fmt.Errorf("cuDNN native backend requires a non-Windows CUDA-tagged build")
}

func destroyNativePoolingDescriptor(desc *PoolingDescriptor) error {
	return fmt.Errorf("cuDNN native backend requires a non-Windows CUDA-tagged build")
}

func createNativeActivationDescriptor() (*ActivationDescriptor, error) {
	return nil, fmt.Errorf("cuDNN native backend requires a non-Windows CUDA-tagged build")
}

func setNativeActivationDescriptor(desc *ActivationDescriptor, mode DNNActivationMode, nanOpt DNNNanPropagation, coeff float64) error {
	return fmt.Errorf("cuDNN native backend requires a non-Windows CUDA-tagged build")
}

func destroyNativeActivationDescriptor(desc *ActivationDescriptor) error {
	return fmt.Errorf("cuDNN native backend requires a non-Windows CUDA-tagged build")
}

func nativeConvolutionForward(handle *DNNHandle, alpha float32, inputDesc *TensorDescriptor, inputData *memory.Memory, filterDesc *FilterDescriptor, filterData *memory.Memory, convDesc *ConvolutionDescriptor, beta float32, outputDesc *TensorDescriptor, outputData *memory.Memory) error {
	return fmt.Errorf("cuDNN native backend requires a non-Windows CUDA-tagged build")
}

func nativePoolingForward(handle *DNNHandle, poolingDesc *PoolingDescriptor, alpha float32, inputDesc *TensorDescriptor, inputData *memory.Memory, beta float32, outputDesc *TensorDescriptor, outputData *memory.Memory) error {
	return fmt.Errorf("cuDNN native backend requires a non-Windows CUDA-tagged build")
}

func nativeActivationForward(handle *DNNHandle, activationDesc *ActivationDescriptor, alpha float32, inputDesc *TensorDescriptor, inputData *memory.Memory, beta float32, outputDesc *TensorDescriptor, outputData *memory.Memory) error {
	return fmt.Errorf("cuDNN native backend requires a non-Windows CUDA-tagged build")
}

func nativeBatchNormForwardInference(handle *DNNHandle, mode DNNBatchNormMode, alpha, beta float32, inputDesc *TensorDescriptor, input *memory.Memory, outputDesc *TensorDescriptor, output *memory.Memory, bnScaleBiasDesc *TensorDescriptor, bnScale, bnBias, estimatedMean, estimatedVariance *memory.Memory, epsilon float64) error {
	return fmt.Errorf("cuDNN native backend requires a non-Windows CUDA-tagged build")
}