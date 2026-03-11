//go:build cuda && !windows

package libraries

/*
#cgo linux CFLAGS: -I/usr/local/cuda/include -I/opt/cuda/include
#cgo linux LDFLAGS: -L/usr/local/cuda/lib64 -L/opt/cuda/lib64 -lcudnn

#include <cuda_runtime.h>
#include <cudnn.h>

static cudnnStatus_t createDNNHandle(cudnnHandle_t* handle) { return cudnnCreate(handle); }
static cudnnStatus_t destroyDNNHandle(cudnnHandle_t handle) { return cudnnDestroy(handle); }
static cudnnStatus_t createTensorDescriptorWrapper(cudnnTensorDescriptor_t* desc) { return cudnnCreateTensorDescriptor(desc); }
static cudnnStatus_t destroyTensorDescriptorWrapper(cudnnTensorDescriptor_t desc) { return cudnnDestroyTensorDescriptor(desc); }
static cudnnStatus_t setTensorNdDescriptorWrapper(cudnnTensorDescriptor_t desc, int dataType, int nbDims, int* dimA, int* strideA) { return cudnnSetTensorNdDescriptor(desc, (cudnnDataType_t)dataType, nbDims, dimA, strideA); }
static cudnnStatus_t setTensor4dDescriptorWrapper(cudnnTensorDescriptor_t desc, int format, int dataType, int n, int c, int h, int w) { return cudnnSetTensor4dDescriptor(desc, (cudnnTensorFormat_t)format, (cudnnDataType_t)dataType, n, c, h, w); }
static cudnnStatus_t createFilterDescriptorWrapper(cudnnFilterDescriptor_t* desc) { return cudnnCreateFilterDescriptor(desc); }
static cudnnStatus_t destroyFilterDescriptorWrapper(cudnnFilterDescriptor_t desc) { return cudnnDestroyFilterDescriptor(desc); }
static cudnnStatus_t setFilterNdDescriptorWrapper(cudnnFilterDescriptor_t desc, int dataType, int format, int nbDims, int* filterDimA) { return cudnnSetFilterNdDescriptor(desc, (cudnnDataType_t)dataType, (cudnnTensorFormat_t)format, nbDims, filterDimA); }
static cudnnStatus_t setFilter4dDescriptorWrapper(cudnnFilterDescriptor_t desc, int dataType, int format, int k, int c, int h, int w) { return cudnnSetFilter4dDescriptor(desc, (cudnnDataType_t)dataType, (cudnnTensorFormat_t)format, k, c, h, w); }
static cudnnStatus_t createConvolutionDescriptorWrapper(cudnnConvolutionDescriptor_t* desc) { return cudnnCreateConvolutionDescriptor(desc); }
static cudnnStatus_t destroyConvolutionDescriptorWrapper(cudnnConvolutionDescriptor_t desc) { return cudnnDestroyConvolutionDescriptor(desc); }
static cudnnStatus_t setConvolution2dDescriptorWrapper(cudnnConvolutionDescriptor_t desc, int padH, int padW, int strideH, int strideW, int dilationH, int dilationW, int mode, int dataType) { return cudnnSetConvolution2dDescriptor(desc, padH, padW, strideH, strideW, dilationH, dilationW, (cudnnConvolutionMode_t)mode, (cudnnDataType_t)dataType); }
static cudnnStatus_t getConvolution2dForwardOutputDimWrapper(cudnnConvolutionDescriptor_t convDesc, cudnnTensorDescriptor_t inputDesc, cudnnFilterDescriptor_t filterDesc, int* n, int* c, int* h, int* w) { return cudnnGetConvolution2dForwardOutputDim(convDesc, inputDesc, filterDesc, n, c, h, w); }
static cudnnStatus_t getConvolutionForwardWorkspaceSizeWrapper(cudnnHandle_t handle, cudnnTensorDescriptor_t xDesc, cudnnFilterDescriptor_t wDesc, cudnnConvolutionDescriptor_t convDesc, cudnnTensorDescriptor_t yDesc, size_t* workspaceSize) { return cudnnGetConvolutionForwardWorkspaceSize(handle, xDesc, wDesc, convDesc, yDesc, CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM, workspaceSize); }
static cudnnStatus_t convolutionForwardWrapper(cudnnHandle_t handle, const float* alpha, cudnnTensorDescriptor_t xDesc, const void* x, cudnnFilterDescriptor_t wDesc, const void* w, cudnnConvolutionDescriptor_t convDesc, void* workSpace, size_t workSpaceSizeInBytes, const float* beta, cudnnTensorDescriptor_t yDesc, void* y) { return cudnnConvolutionForward(handle, alpha, xDesc, x, wDesc, w, convDesc, CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM, workSpace, workSpaceSizeInBytes, beta, yDesc, y); }
static cudnnStatus_t createPoolingDescriptorWrapper(cudnnPoolingDescriptor_t* desc) { return cudnnCreatePoolingDescriptor(desc); }
static cudnnStatus_t destroyPoolingDescriptorWrapper(cudnnPoolingDescriptor_t desc) { return cudnnDestroyPoolingDescriptor(desc); }
static cudnnStatus_t setPooling2dDescriptorWrapper(cudnnPoolingDescriptor_t desc, int mode, int windowH, int windowW, int padH, int padW, int strideH, int strideW) { return cudnnSetPooling2dDescriptor(desc, (cudnnPoolingMode_t)mode, CUDNN_PROPAGATE_NAN, windowH, windowW, padH, padW, strideH, strideW); }
static cudnnStatus_t poolingForwardWrapper(cudnnHandle_t handle, cudnnPoolingDescriptor_t poolingDesc, const float* alpha, cudnnTensorDescriptor_t xDesc, const void* x, const float* beta, cudnnTensorDescriptor_t yDesc, void* y) { return cudnnPoolingForward(handle, poolingDesc, alpha, xDesc, x, beta, yDesc, y); }
static cudnnStatus_t createActivationDescriptorWrapper(cudnnActivationDescriptor_t* desc) { return cudnnCreateActivationDescriptor(desc); }
static cudnnStatus_t destroyActivationDescriptorWrapper(cudnnActivationDescriptor_t desc) { return cudnnDestroyActivationDescriptor(desc); }
static cudnnStatus_t setActivationDescriptorWrapper(cudnnActivationDescriptor_t desc, int mode, int nanOpt, double coeff) { return cudnnSetActivationDescriptor(desc, (cudnnActivationMode_t)mode, (cudnnNanPropagation_t)nanOpt, coeff); }
static cudnnStatus_t activationForwardWrapper(cudnnHandle_t handle, cudnnActivationDescriptor_t activationDesc, const float* alpha, cudnnTensorDescriptor_t xDesc, const void* x, const float* beta, cudnnTensorDescriptor_t yDesc, void* y) { return cudnnActivationForward(handle, activationDesc, alpha, xDesc, x, beta, yDesc, y); }
static cudnnStatus_t batchNormForwardInferenceWrapper(cudnnHandle_t handle, int mode, const float* alpha, const float* beta, cudnnTensorDescriptor_t xDesc, const void* x, cudnnTensorDescriptor_t yDesc, void* y, cudnnTensorDescriptor_t bnScaleBiasMeanVarDesc, const void* bnScale, const void* bnBias, const void* estimatedMean, const void* estimatedVariance, double epsilon) { return cudnnBatchNormalizationForwardInference(handle, (cudnnBatchNormMode_t)mode, alpha, beta, xDesc, x, yDesc, y, bnScaleBiasMeanVarDesc, bnScale, bnBias, estimatedMean, estimatedVariance, epsilon); }
*/
import "C"

import (
	"fmt"
	"unsafe"

	"github.com/stitch1968/gocuda/memory"
)

func cudnnNativeAvailable() bool {
	return true
}

func createNativeDNNHandle() (*DNNHandle, error) {
	var handle C.cudnnHandle_t
	if status := C.createDNNHandle(&handle); status != C.CUDNN_STATUS_SUCCESS {
		return nil, cudnnError("cudnnCreate", status)
	}
	return &DNNHandle{handle: uintptr(handle), native: true}, nil
}

func destroyNativeDNNHandle(handle *DNNHandle) error {
	if status := C.destroyDNNHandle(C.cudnnHandle_t(handle.handle)); status != C.CUDNN_STATUS_SUCCESS {
		return cudnnError("cudnnDestroy", status)
	}
	return nil
}

func createNativeTensorDescriptor() (*TensorDescriptor, error) {
	var desc C.cudnnTensorDescriptor_t
	if status := C.createTensorDescriptorWrapper(&desc); status != C.CUDNN_STATUS_SUCCESS {
		return nil, cudnnError("cudnnCreateTensorDescriptor", status)
	}
	return &TensorDescriptor{handle: uintptr(desc), native: true}, nil
}

func setNativeTensorNdDescriptor(desc *TensorDescriptor, dataType DNNDataType, dimensions []int, strides []int) error {
	nativeType, err := nativeDNNDataType(dataType)
	if err != nil {
		return err
	}
	if len(dimensions) == 0 || len(strides) == 0 {
		return fmt.Errorf("dimensions and strides cannot be empty")
	}
	if status := C.setTensorNdDescriptorWrapper(C.cudnnTensorDescriptor_t(desc.handle), C.int(nativeType), C.int(len(dimensions)), (*C.int)(unsafe.Pointer(&dimensions[0])), (*C.int)(unsafe.Pointer(&strides[0]))); status != C.CUDNN_STATUS_SUCCESS {
		return cudnnError("cudnnSetTensorNdDescriptor", status)
	}
	return nil
}

func setNativeTensor4dDescriptor(desc *TensorDescriptor, format DNNTensorFormat, dataType DNNDataType, n, c, h, w int) error {
	nativeFormat, err := nativeTensorFormat(format)
	if err != nil {
		return err
	}
	nativeType, err := nativeDNNDataType(dataType)
	if err != nil {
		return err
	}
	if status := C.setTensor4dDescriptorWrapper(C.cudnnTensorDescriptor_t(desc.handle), C.int(nativeFormat), C.int(nativeType), C.int(n), C.int(c), C.int(h), C.int(w)); status != C.CUDNN_STATUS_SUCCESS {
		return cudnnError("cudnnSetTensor4dDescriptor", status)
	}
	return nil
}

func destroyNativeTensorDescriptor(desc *TensorDescriptor) error { if status := C.destroyTensorDescriptorWrapper(C.cudnnTensorDescriptor_t(desc.handle)); status != C.CUDNN_STATUS_SUCCESS { return cudnnError("cudnnDestroyTensorDescriptor", status) }; return nil }

func createNativeFilterDescriptor() (*FilterDescriptor, error) {
	var desc C.cudnnFilterDescriptor_t
	if status := C.createFilterDescriptorWrapper(&desc); status != C.CUDNN_STATUS_SUCCESS {
		return nil, cudnnError("cudnnCreateFilterDescriptor", status)
	}
	return &FilterDescriptor{handle: uintptr(desc), native: true}, nil
}

func setNativeFilterNdDescriptor(desc *FilterDescriptor, dataType DNNDataType, format DNNTensorFormat, dimensions []int) error {
	nativeType, err := nativeDNNDataType(dataType)
	if err != nil { return err }
	nativeFormat, err := nativeTensorFormat(format)
	if err != nil { return err }
	if len(dimensions) == 0 { return fmt.Errorf("dimensions cannot be empty") }
	if status := C.setFilterNdDescriptorWrapper(C.cudnnFilterDescriptor_t(desc.handle), C.int(nativeType), C.int(nativeFormat), C.int(len(dimensions)), (*C.int)(unsafe.Pointer(&dimensions[0]))); status != C.CUDNN_STATUS_SUCCESS { return cudnnError("cudnnSetFilterNdDescriptor", status) }
	return nil
}

func setNativeFilter4dDescriptor(desc *FilterDescriptor, dataType DNNDataType, format DNNTensorFormat, k, c, h, w int) error {
	nativeType, err := nativeDNNDataType(dataType)
	if err != nil { return err }
	nativeFormat, err := nativeTensorFormat(format)
	if err != nil { return err }
	if status := C.setFilter4dDescriptorWrapper(C.cudnnFilterDescriptor_t(desc.handle), C.int(nativeType), C.int(nativeFormat), C.int(k), C.int(c), C.int(h), C.int(w)); status != C.CUDNN_STATUS_SUCCESS { return cudnnError("cudnnSetFilter4dDescriptor", status) }
	return nil
}

func destroyNativeFilterDescriptor(desc *FilterDescriptor) error { if status := C.destroyFilterDescriptorWrapper(C.cudnnFilterDescriptor_t(desc.handle)); status != C.CUDNN_STATUS_SUCCESS { return cudnnError("cudnnDestroyFilterDescriptor", status) }; return nil }

func createNativeConvolutionDescriptor() (*ConvolutionDescriptor, error) {
	var desc C.cudnnConvolutionDescriptor_t
	if status := C.createConvolutionDescriptorWrapper(&desc); status != C.CUDNN_STATUS_SUCCESS { return nil, cudnnError("cudnnCreateConvolutionDescriptor", status) }
	return &ConvolutionDescriptor{handle: uintptr(desc), dilationH: 1, dilationW: 1, native: true}, nil
}

func setNativeConvolution2dDescriptor(desc *ConvolutionDescriptor, padH, padW, strideH, strideW, dilationH, dilationW int, mode DNNConvolutionMode, dataType DNNDataType) error {
	nativeMode, err := nativeConvolutionMode(mode)
	if err != nil { return err }
	nativeType, err := nativeDNNDataType(dataType)
	if err != nil { return err }
	if status := C.setConvolution2dDescriptorWrapper(C.cudnnConvolutionDescriptor_t(desc.handle), C.int(padH), C.int(padW), C.int(strideH), C.int(strideW), C.int(dilationH), C.int(dilationW), C.int(nativeMode), C.int(nativeType)); status != C.CUDNN_STATUS_SUCCESS { return cudnnError("cudnnSetConvolution2dDescriptor", status) }
	return nil
}

func nativeConvolutionForwardOutputDim(desc *ConvolutionDescriptor, inputDesc *TensorDescriptor, filterDesc *FilterDescriptor) (int, int, int, int, error) {
	var n, c, h, w C.int
	if status := C.getConvolution2dForwardOutputDimWrapper(C.cudnnConvolutionDescriptor_t(desc.handle), C.cudnnTensorDescriptor_t(inputDesc.handle), C.cudnnFilterDescriptor_t(filterDesc.handle), &n, &c, &h, &w); status != C.CUDNN_STATUS_SUCCESS { return 0, 0, 0, 0, cudnnError("cudnnGetConvolution2dForwardOutputDim", status) }
	return int(n), int(c), int(h), int(w), nil
}

func destroyNativeConvolutionDescriptor(desc *ConvolutionDescriptor) error { if status := C.destroyConvolutionDescriptorWrapper(C.cudnnConvolutionDescriptor_t(desc.handle)); status != C.CUDNN_STATUS_SUCCESS { return cudnnError("cudnnDestroyConvolutionDescriptor", status) }; return nil }

func createNativePoolingDescriptor() (*PoolingDescriptor, error) {
	var desc C.cudnnPoolingDescriptor_t
	if status := C.createPoolingDescriptorWrapper(&desc); status != C.CUDNN_STATUS_SUCCESS { return nil, cudnnError("cudnnCreatePoolingDescriptor", status) }
	return &PoolingDescriptor{handle: uintptr(desc), native: true}, nil
}

func setNativePooling2dDescriptor(desc *PoolingDescriptor, mode DNNPoolingMode, windowH, windowW, padH, padW, strideH, strideW int) error {
	nativeMode, err := nativePoolingMode(mode)
	if err != nil { return err }
	if status := C.setPooling2dDescriptorWrapper(C.cudnnPoolingDescriptor_t(desc.handle), C.int(nativeMode), C.int(windowH), C.int(windowW), C.int(padH), C.int(padW), C.int(strideH), C.int(strideW)); status != C.CUDNN_STATUS_SUCCESS { return cudnnError("cudnnSetPooling2dDescriptor", status) }
	return nil
}

func destroyNativePoolingDescriptor(desc *PoolingDescriptor) error { if status := C.destroyPoolingDescriptorWrapper(C.cudnnPoolingDescriptor_t(desc.handle)); status != C.CUDNN_STATUS_SUCCESS { return cudnnError("cudnnDestroyPoolingDescriptor", status) }; return nil }

func createNativeActivationDescriptor() (*ActivationDescriptor, error) {
	var desc C.cudnnActivationDescriptor_t
	if status := C.createActivationDescriptorWrapper(&desc); status != C.CUDNN_STATUS_SUCCESS { return nil, cudnnError("cudnnCreateActivationDescriptor", status) }
	return &ActivationDescriptor{handle: uintptr(desc), native: true}, nil
}

func setNativeActivationDescriptor(desc *ActivationDescriptor, mode DNNActivationMode, nanOpt DNNNanPropagation, coeff float64) error {
	nativeMode, err := nativeActivationMode(mode)
	if err != nil { return err }
	nativeNanOpt, err := nativeNanPropagation(nanOpt)
	if err != nil { return err }
	if status := C.setActivationDescriptorWrapper(C.cudnnActivationDescriptor_t(desc.handle), C.int(nativeMode), C.int(nativeNanOpt), C.double(coeff)); status != C.CUDNN_STATUS_SUCCESS { return cudnnError("cudnnSetActivationDescriptor", status) }
	return nil
}

func destroyNativeActivationDescriptor(desc *ActivationDescriptor) error { if status := C.destroyActivationDescriptorWrapper(C.cudnnActivationDescriptor_t(desc.handle)); status != C.CUDNN_STATUS_SUCCESS { return cudnnError("cudnnDestroyActivationDescriptor", status) }; return nil }

func nativeConvolutionForward(handle *DNNHandle, alpha float32, inputDesc *TensorDescriptor, inputData *memory.Memory, filterDesc *FilterDescriptor, filterData *memory.Memory, convDesc *ConvolutionDescriptor, beta float32, outputDesc *TensorDescriptor, outputData *memory.Memory) error {
	var workspaceSize C.size_t
	if status := C.getConvolutionForwardWorkspaceSizeWrapper(C.cudnnHandle_t(handle.handle), C.cudnnTensorDescriptor_t(inputDesc.handle), C.cudnnFilterDescriptor_t(filterDesc.handle), C.cudnnConvolutionDescriptor_t(convDesc.handle), C.cudnnTensorDescriptor_t(outputDesc.handle), &workspaceSize); status != C.CUDNN_STATUS_SUCCESS { return cudnnError("cudnnGetConvolutionForwardWorkspaceSize", status) }
	var workspace *memory.Memory
	var workspacePtr unsafe.Pointer
	if workspaceSize > 0 { allocated, err := memory.Alloc(int64(workspaceSize)); if err != nil { return fmt.Errorf("failed to allocate cuDNN workspace: %v", err) }; workspace = allocated; defer workspace.Free(); workspacePtr = workspace.Ptr() }
	alphaValue := C.float(alpha)
	betaValue := C.float(beta)
	if status := C.convolutionForwardWrapper(C.cudnnHandle_t(handle.handle), &alphaValue, C.cudnnTensorDescriptor_t(inputDesc.handle), inputData.Ptr(), C.cudnnFilterDescriptor_t(filterDesc.handle), filterData.Ptr(), C.cudnnConvolutionDescriptor_t(convDesc.handle), workspacePtr, workspaceSize, &betaValue, C.cudnnTensorDescriptor_t(outputDesc.handle), outputData.Ptr()); status != C.CUDNN_STATUS_SUCCESS { return cudnnError("cudnnConvolutionForward", status) }
	return nil
}

func nativePoolingForward(handle *DNNHandle, poolingDesc *PoolingDescriptor, alpha float32, inputDesc *TensorDescriptor, inputData *memory.Memory, beta float32, outputDesc *TensorDescriptor, outputData *memory.Memory) error {
	alphaValue := C.float(alpha)
	betaValue := C.float(beta)
	if status := C.poolingForwardWrapper(C.cudnnHandle_t(handle.handle), C.cudnnPoolingDescriptor_t(poolingDesc.handle), &alphaValue, C.cudnnTensorDescriptor_t(inputDesc.handle), inputData.Ptr(), &betaValue, C.cudnnTensorDescriptor_t(outputDesc.handle), outputData.Ptr()); status != C.CUDNN_STATUS_SUCCESS { return cudnnError("cudnnPoolingForward", status) }
	return nil
}

func nativeActivationForward(handle *DNNHandle, activationDesc *ActivationDescriptor, alpha float32, inputDesc *TensorDescriptor, inputData *memory.Memory, beta float32, outputDesc *TensorDescriptor, outputData *memory.Memory) error {
	alphaValue := C.float(alpha)
	betaValue := C.float(beta)
	if status := C.activationForwardWrapper(C.cudnnHandle_t(handle.handle), C.cudnnActivationDescriptor_t(activationDesc.handle), &alphaValue, C.cudnnTensorDescriptor_t(inputDesc.handle), inputData.Ptr(), &betaValue, C.cudnnTensorDescriptor_t(outputDesc.handle), outputData.Ptr()); status != C.CUDNN_STATUS_SUCCESS { return cudnnError("cudnnActivationForward", status) }
	return nil
}

func nativeBatchNormForwardInference(handle *DNNHandle, mode DNNBatchNormMode, alpha, beta float32, inputDesc *TensorDescriptor, input *memory.Memory, outputDesc *TensorDescriptor, output *memory.Memory, bnScaleBiasDesc *TensorDescriptor, bnScale, bnBias, estimatedMean, estimatedVariance *memory.Memory, epsilon float64) error {
	nativeMode, err := nativeBatchNormMode(mode)
	if err != nil { return err }
	alphaValue := C.float(alpha)
	betaValue := C.float(beta)
	if status := C.batchNormForwardInferenceWrapper(C.cudnnHandle_t(handle.handle), C.int(nativeMode), &alphaValue, &betaValue, C.cudnnTensorDescriptor_t(inputDesc.handle), input.Ptr(), C.cudnnTensorDescriptor_t(outputDesc.handle), output.Ptr(), C.cudnnTensorDescriptor_t(bnScaleBiasDesc.handle), bnScale.Ptr(), bnBias.Ptr(), estimatedMean.Ptr(), estimatedVariance.Ptr(), C.double(epsilon)); status != C.CUDNN_STATUS_SUCCESS { return cudnnError("cudnnBatchNormalizationForwardInference", status) }
	return nil
}

func nativeDNNDataType(dataType DNNDataType) (C.int, error) { switch dataType { case DNNDataFloat: return C.CUDNN_DATA_FLOAT, nil; case DNNDataDouble: return C.CUDNN_DATA_DOUBLE, nil; case DNNDataHalf: return C.CUDNN_DATA_HALF, nil; case DNNDataInt8: return C.CUDNN_DATA_INT8, nil; case DNNDataInt32: return C.CUDNN_DATA_INT32, nil; case DNNDataUint8: return C.CUDNN_DATA_UINT8, nil; case DNNDataInt64: return C.CUDNN_DATA_INT64, nil; default: return 0, fmt.Errorf("unsupported cuDNN data type %v", dataType) } }
func nativeTensorFormat(format DNNTensorFormat) (C.int, error) { switch format { case DNNTensorNHWC: return C.CUDNN_TENSOR_NHWC, nil; case DNNTensorNCHW: return C.CUDNN_TENSOR_NCHW, nil; case DNNTensorNCHWVectC: return C.CUDNN_TENSOR_NCHW_VECT_C, nil; default: return 0, fmt.Errorf("unsupported cuDNN tensor format %v", format) } }
func nativeConvolutionMode(mode DNNConvolutionMode) (C.int, error) { switch mode { case DNNConvolution: return C.CUDNN_CONVOLUTION, nil; case DNNCrossCorrelation: return C.CUDNN_CROSS_CORRELATION, nil; default: return 0, fmt.Errorf("unsupported cuDNN convolution mode %v", mode) } }
func nativePoolingMode(mode DNNPoolingMode) (C.int, error) { switch mode { case DNNPoolingMax: return C.CUDNN_POOLING_MAX, nil; case DNNPoolingAverageCountIncludePadding: return C.CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING, nil; case DNNPoolingAverageCountExcludePadding: return C.CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING, nil; case DNNPoolingMaxDeterministic: return C.CUDNN_POOLING_MAX_DETERMINISTIC, nil; default: return 0, fmt.Errorf("unsupported cuDNN pooling mode %v", mode) } }
func nativeActivationMode(mode DNNActivationMode) (C.int, error) { switch mode { case DNNActivationSigmoid: return C.CUDNN_ACTIVATION_SIGMOID, nil; case DNNActivationRelu: return C.CUDNN_ACTIVATION_RELU, nil; case DNNActivationTanh: return C.CUDNN_ACTIVATION_TANH, nil; case DNNActivationClippedRelu: return C.CUDNN_ACTIVATION_CLIPPED_RELU, nil; case DNNActivationElu: return C.CUDNN_ACTIVATION_ELU, nil; case DNNActivationIdentity: return C.CUDNN_ACTIVATION_IDENTITY, nil; case DNNActivationSwish: return C.CUDNN_ACTIVATION_SWISH, nil; default: return 0, fmt.Errorf("unsupported cuDNN activation mode %v", mode) } }
func nativeNanPropagation(mode DNNNanPropagation) (C.int, error) { switch mode { case DNNNotPropagateNaN: return C.CUDNN_NOT_PROPAGATE_NAN, nil; case DNNPropagateNaN: return C.CUDNN_PROPAGATE_NAN, nil; default: return 0, fmt.Errorf("unsupported cuDNN NaN propagation mode %v", mode) } }
func nativeBatchNormMode(mode DNNBatchNormMode) (C.int, error) { switch mode { case DNNBatchNormPerActivation: return C.CUDNN_BATCHNORM_PER_ACTIVATION, nil; case DNNBatchNormSpatial: return C.CUDNN_BATCHNORM_SPATIAL, nil; default: return 0, fmt.Errorf("unsupported cuDNN batch norm mode %v", mode) } }
func cudnnError(operation string, status C.cudnnStatus_t) error { return fmt.Errorf("%s failed: %s (%d)", operation, C.GoString(C.cudnnGetErrorString(status)), int(status)) }