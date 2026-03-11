// Package libraries provides CUDA runtime library bindings for GoCUDA
// This implements cuDNN functionality for Deep Neural Networks
package libraries

import (
	"fmt"
	"math"
	"time"

	cuda "github.com/stitch1968/gocuda"
	"github.com/stitch1968/gocuda/memory"
)

// cuDNN - Deep Neural Network Library

// DNNHandle represents a cuDNN library handle
type DNNHandle struct {
	handle    uintptr
	destroyed bool
	native    bool
}

// TensorDescriptor describes the layout of a tensor
type TensorDescriptor struct {
	handle     uintptr
	dataType   DNNDataType
	format     DNNTensorFormat
	dimensions []int
	strides    []int
	destroyed  bool
	native     bool
}

// FilterDescriptor describes a convolution filter
type FilterDescriptor struct {
	handle     uintptr
	dataType   DNNDataType
	format     DNNTensorFormat
	dimensions []int // [outputChannels, inputChannels, height, width]
	destroyed  bool
	native     bool
}

// ConvolutionDescriptor describes a convolution operation
type ConvolutionDescriptor struct {
	handle    uintptr
	padH      int
	padW      int
	strideH   int
	strideW   int
	dilationH int
	dilationW int
	mode      DNNConvolutionMode
	dataType  DNNDataType
	destroyed bool
	native    bool
}

// PoolingDescriptor describes a pooling operation
type PoolingDescriptor struct {
	handle    uintptr
	mode      DNNPoolingMode
	windowH   int
	windowW   int
	padH      int
	padW      int
	strideH   int
	strideW   int
	destroyed bool
	native    bool
}

// ActivationDescriptor describes an activation function
type ActivationDescriptor struct {
	handle    uintptr
	mode      DNNActivationMode
	nanOpt    DNNNanPropagation
	coeff     float64
	destroyed bool
	native    bool
}

// BatchNormDescriptor describes batch normalization
type BatchNormDescriptor struct {
	handle    uintptr
	mode      DNNBatchNormMode
	destroyed bool
	native    bool
}

// DNNDataType represents cuDNN data types
type DNNDataType int

const (
	DNNDataFloat DNNDataType = iota
	DNNDataDouble
	DNNDataHalf
	DNNDataInt8
	DNNDataInt32
	DNNDataInt8x4
	DNNDataUint8
	DNNDataUint8x4
	DNNDataInt8x32
	DNNDataBFloat16
	DNNDataInt64
)

// DNNTensorFormat represents tensor memory layouts
type DNNTensorFormat int

const (
	DNNTensorNHWC      DNNTensorFormat = iota // batch, height, width, channels
	DNNTensorNCHW                             // batch, channels, height, width
	DNNTensorNCHWVectC                        // vectorized channels
	DNNTensorNHWCVectC                        // vectorized channels
)

// DNNConvolutionMode represents convolution modes
type DNNConvolutionMode int

const (
	DNNConvolution DNNConvolutionMode = iota
	DNNCrossCorrelation
)

// DNNPoolingMode represents pooling modes
type DNNPoolingMode int

const (
	DNNPoolingMax DNNPoolingMode = iota
	DNNPoolingAverageCountIncludePadding
	DNNPoolingAverageCountExcludePadding
	DNNPoolingMaxDeterministic
)

// DNNActivationMode represents activation function types
type DNNActivationMode int

const (
	DNNActivationSigmoid DNNActivationMode = iota
	DNNActivationRelu
	DNNActivationTanh
	DNNActivationClippedRelu
	DNNActivationElu
	DNNActivationIdentity
	DNNActivationSwish
)

// DNNNanPropagation represents NaN propagation modes
type DNNNanPropagation int

const (
	DNNNotPropagateNaN DNNNanPropagation = iota
	DNNPropagateNaN
)

// DNNBatchNormMode represents batch normalization modes
type DNNBatchNormMode int

const (
	DNNBatchNormPerActivation DNNBatchNormMode = iota
	DNNBatchNormSpatial
)

// CreateDNNHandle creates a cuDNN library handle
func CreateDNNHandle() (*DNNHandle, error) {
	if err := ensureCudaReady(); err != nil {
		return nil, err
	}
	if cuda.ShouldUseCuda() && cudnnNativeAvailable() {
		return createNativeDNNHandle()
	}

	handle := &DNNHandle{
		handle: uintptr(time.Now().UnixNano()),
	}

	return handle, nil
}

// DestroyHandle destroys the cuDNN handle
func (h *DNNHandle) DestroyHandle() error {
	if h.destroyed {
		return fmt.Errorf("handle already destroyed")
	}
	h.destroyed = true
	if h.native {
		return destroyNativeDNNHandle(h)
	}
	return nil
}

// Tensor Descriptor Operations

// CreateTensorDescriptor creates a tensor descriptor
func CreateTensorDescriptor() (*TensorDescriptor, error) {
	if err := ensureCudaReady(); err != nil {
		return nil, err
	}
	if cuda.ShouldUseCuda() && cudnnNativeAvailable() {
		return createNativeTensorDescriptor()
	}

	desc := &TensorDescriptor{
		handle: uintptr(time.Now().UnixNano()),
	}

	return desc, nil
}

// SetTensorNdDescriptor sets the tensor descriptor for N-dimensional tensors
func (desc *TensorDescriptor) SetTensorNdDescriptor(dataType DNNDataType, dimensions []int, strides []int) error {
	if desc.destroyed {
		return fmt.Errorf("descriptor has been destroyed")
	}
	if len(dimensions) != len(strides) {
		return fmt.Errorf("dimensions and strides must have the same length")
	}

	desc.dataType = dataType
	desc.dimensions = make([]int, len(dimensions))
	desc.strides = make([]int, len(strides))
	copy(desc.dimensions, dimensions)
	copy(desc.strides, strides)
	if desc.native {
		return setNativeTensorNdDescriptor(desc, dataType, dimensions, strides)
	}

	return nil
}

// SetTensor4dDescriptor sets the tensor descriptor for 4D tensors
func (desc *TensorDescriptor) SetTensor4dDescriptor(format DNNTensorFormat, dataType DNNDataType, n, c, h, w int) error {
	if desc.destroyed {
		return fmt.Errorf("descriptor has been destroyed")
	}

	desc.format = format
	desc.dataType = dataType
	desc.dimensions = []int{n, c, h, w}

	// Calculate strides based on format
	if format == DNNTensorNCHW {
		desc.strides = []int{c * h * w, h * w, w, 1}
	} else if format == DNNTensorNHWC {
		desc.strides = []int{h * w * c, 1, w * c, c}
	}
	if desc.native {
		return setNativeTensor4dDescriptor(desc, format, dataType, n, c, h, w)
	}

	return nil
}

// DestroyTensorDescriptor destroys a tensor descriptor
func (desc *TensorDescriptor) DestroyTensorDescriptor() error {
	if desc.destroyed {
		return fmt.Errorf("descriptor already destroyed")
	}
	desc.destroyed = true
	if desc.native {
		return destroyNativeTensorDescriptor(desc)
	}
	return nil
}

// Filter Descriptor Operations

// CreateFilterDescriptor creates a filter descriptor
func CreateFilterDescriptor() (*FilterDescriptor, error) {
	if err := ensureCudaReady(); err != nil {
		return nil, err
	}
	if cuda.ShouldUseCuda() && cudnnNativeAvailable() {
		return createNativeFilterDescriptor()
	}

	desc := &FilterDescriptor{
		handle: uintptr(time.Now().UnixNano()),
	}

	return desc, nil
}

// SetFilterNdDescriptor sets the filter descriptor for N-dimensional filters
func (desc *FilterDescriptor) SetFilterNdDescriptor(dataType DNNDataType, format DNNTensorFormat, dimensions []int) error {
	if desc.destroyed {
		return fmt.Errorf("descriptor has been destroyed")
	}

	desc.dataType = dataType
	desc.format = format
	desc.dimensions = make([]int, len(dimensions))
	copy(desc.dimensions, dimensions)
	if desc.native {
		return setNativeFilterNdDescriptor(desc, dataType, format, dimensions)
	}

	return nil
}

// SetFilter4dDescriptor sets the filter descriptor for 4D filters
func (desc *FilterDescriptor) SetFilter4dDescriptor(dataType DNNDataType, format DNNTensorFormat, k, c, h, w int) error {
	if desc.destroyed {
		return fmt.Errorf("descriptor has been destroyed")
	}

	desc.dataType = dataType
	desc.format = format
	desc.dimensions = []int{k, c, h, w}
	if desc.native {
		return setNativeFilter4dDescriptor(desc, dataType, format, k, c, h, w)
	}

	return nil
}

// DestroyFilterDescriptor destroys a filter descriptor
func (desc *FilterDescriptor) DestroyFilterDescriptor() error {
	if desc.destroyed {
		return fmt.Errorf("descriptor already destroyed")
	}
	desc.destroyed = true
	if desc.native {
		return destroyNativeFilterDescriptor(desc)
	}
	return nil
}

// Convolution Operations

// CreateConvolutionDescriptor creates a convolution descriptor
func CreateConvolutionDescriptor() (*ConvolutionDescriptor, error) {
	if err := ensureCudaReady(); err != nil {
		return nil, err
	}
	if cuda.ShouldUseCuda() && cudnnNativeAvailable() {
		return createNativeConvolutionDescriptor()
	}

	desc := &ConvolutionDescriptor{
		handle:    uintptr(time.Now().UnixNano()),
		dilationH: 1,
		dilationW: 1,
	}

	return desc, nil
}

// SetConvolution2dDescriptor sets the convolution descriptor for 2D convolutions
func (desc *ConvolutionDescriptor) SetConvolution2dDescriptor(padH, padW, strideH, strideW, dilationH, dilationW int, mode DNNConvolutionMode, dataType DNNDataType) error {
	if desc.destroyed {
		return fmt.Errorf("descriptor has been destroyed")
	}

	desc.padH = padH
	desc.padW = padW
	desc.strideH = strideH
	desc.strideW = strideW
	desc.dilationH = dilationH
	desc.dilationW = dilationW
	desc.mode = mode
	desc.dataType = dataType
	if desc.native {
		return setNativeConvolution2dDescriptor(desc, padH, padW, strideH, strideW, dilationH, dilationW, mode, dataType)
	}

	return nil
}

// GetConvolution2dForwardOutputDim calculates output dimensions for convolution
func (desc *ConvolutionDescriptor) GetConvolution2dForwardOutputDim(inputDesc *TensorDescriptor, filterDesc *FilterDescriptor) (n, c, h, w int, err error) {
	if desc.destroyed || inputDesc.destroyed || filterDesc.destroyed {
		return 0, 0, 0, 0, fmt.Errorf("one or more descriptors have been destroyed")
	}

	// Extract dimensions
	if len(inputDesc.dimensions) < 4 || len(filterDesc.dimensions) < 4 {
		return 0, 0, 0, 0, fmt.Errorf("invalid tensor dimensions")
	}
	if desc.native {
		return nativeConvolutionForwardOutputDim(desc, inputDesc, filterDesc)
	}

	inputN, _, inputH, inputW := inputDesc.dimensions[0], inputDesc.dimensions[1], inputDesc.dimensions[2], inputDesc.dimensions[3]
	filterK, _, filterH, filterW := filterDesc.dimensions[0], filterDesc.dimensions[1], filterDesc.dimensions[2], filterDesc.dimensions[3]

	// Calculate output dimensions
	outputH := (inputH+2*desc.padH-desc.dilationH*(filterH-1)-1)/desc.strideH + 1
	outputW := (inputW+2*desc.padW-desc.dilationW*(filterW-1)-1)/desc.strideW + 1

	return inputN, filterK, outputH, outputW, nil
}

// ConvolutionForward performs forward convolution
func (h *DNNHandle) ConvolutionForward(alpha float32, inputDesc *TensorDescriptor, inputData *memory.Memory,
	filterDesc *FilterDescriptor, filterData *memory.Memory, convDesc *ConvolutionDescriptor,
	beta float32, outputDesc *TensorDescriptor, outputData *memory.Memory) error {

	if h.destroyed {
		return fmt.Errorf("handle has been destroyed")
	}

	// Validate descriptors
	if inputDesc.destroyed || filterDesc.destroyed || convDesc.destroyed || outputDesc.destroyed {
		return fmt.Errorf("one or more descriptors have been destroyed")
	}
	if h.native {
		return nativeConvolutionForward(h, alpha, inputDesc, inputData, filterDesc, filterData, convDesc, beta, outputDesc, outputData)
	}

	return h.performConvolution(inputData, filterData, outputData, inputDesc, filterDesc, outputDesc, convDesc, alpha, beta)
}

// DestroyConvolutionDescriptor destroys a convolution descriptor
func (desc *ConvolutionDescriptor) DestroyConvolutionDescriptor() error {
	if desc.destroyed {
		return fmt.Errorf("descriptor already destroyed")
	}
	desc.destroyed = true
	if desc.native {
		return destroyNativeConvolutionDescriptor(desc)
	}
	return nil
}

// Pooling Operations

// CreatePoolingDescriptor creates a pooling descriptor
func CreatePoolingDescriptor() (*PoolingDescriptor, error) {
	if err := ensureCudaReady(); err != nil {
		return nil, err
	}
	if cuda.ShouldUseCuda() && cudnnNativeAvailable() {
		return createNativePoolingDescriptor()
	}

	desc := &PoolingDescriptor{
		handle: uintptr(time.Now().UnixNano()),
	}

	return desc, nil
}

// SetPooling2dDescriptor sets the pooling descriptor for 2D pooling
func (desc *PoolingDescriptor) SetPooling2dDescriptor(mode DNNPoolingMode, windowH, windowW, padH, padW, strideH, strideW int) error {
	if desc.destroyed {
		return fmt.Errorf("descriptor has been destroyed")
	}

	desc.mode = mode
	desc.windowH = windowH
	desc.windowW = windowW
	desc.padH = padH
	desc.padW = padW
	desc.strideH = strideH
	desc.strideW = strideW
	if desc.native {
		return setNativePooling2dDescriptor(desc, mode, windowH, windowW, padH, padW, strideH, strideW)
	}

	return nil
}

// PoolingForward performs forward pooling
func (h *DNNHandle) PoolingForward(poolingDesc *PoolingDescriptor, alpha float32, inputDesc *TensorDescriptor, inputData *memory.Memory,
	beta float32, outputDesc *TensorDescriptor, outputData *memory.Memory) error {

	if h.destroyed || poolingDesc.destroyed || inputDesc.destroyed || outputDesc.destroyed {
		return fmt.Errorf("one or more descriptors have been destroyed")
	}
	if h.native {
		return nativePoolingForward(h, poolingDesc, alpha, inputDesc, inputData, beta, outputDesc, outputData)
	}

	return h.performPooling(inputData, outputData, inputDesc, outputDesc, poolingDesc, alpha, beta)
}

// DestroyPoolingDescriptor destroys a pooling descriptor
func (desc *PoolingDescriptor) DestroyPoolingDescriptor() error {
	if desc.destroyed {
		return fmt.Errorf("descriptor already destroyed")
	}
	desc.destroyed = true
	if desc.native {
		return destroyNativePoolingDescriptor(desc)
	}
	return nil
}

// Activation Operations

// CreateActivationDescriptor creates an activation descriptor
func CreateActivationDescriptor() (*ActivationDescriptor, error) {
	if err := ensureCudaReady(); err != nil {
		return nil, err
	}
	if cuda.ShouldUseCuda() && cudnnNativeAvailable() {
		return createNativeActivationDescriptor()
	}

	desc := &ActivationDescriptor{
		handle: uintptr(time.Now().UnixNano()),
	}

	return desc, nil
}

// SetActivationDescriptor sets the activation descriptor
func (desc *ActivationDescriptor) SetActivationDescriptor(mode DNNActivationMode, nanOpt DNNNanPropagation, coeff float64) error {
	if desc.destroyed {
		return fmt.Errorf("descriptor has been destroyed")
	}

	desc.mode = mode
	desc.nanOpt = nanOpt
	desc.coeff = coeff
	if desc.native {
		return setNativeActivationDescriptor(desc, mode, nanOpt, coeff)
	}

	return nil
}

// ActivationForward performs forward activation
func (h *DNNHandle) ActivationForward(activationDesc *ActivationDescriptor, alpha float32, inputDesc *TensorDescriptor, inputData *memory.Memory,
	beta float32, outputDesc *TensorDescriptor, outputData *memory.Memory) error {

	if h.destroyed || activationDesc.destroyed || inputDesc.destroyed || outputDesc.destroyed {
		return fmt.Errorf("one or more descriptors have been destroyed")
	}
	if h.native {
		return nativeActivationForward(h, activationDesc, alpha, inputDesc, inputData, beta, outputDesc, outputData)
	}

	return h.performActivation(inputData, outputData, inputDesc, outputDesc, activationDesc, alpha, beta)
}

// DestroyActivationDescriptor destroys an activation descriptor
func (desc *ActivationDescriptor) DestroyActivationDescriptor() error {
	if desc.destroyed {
		return fmt.Errorf("descriptor already destroyed")
	}
	desc.destroyed = true
	if desc.native {
		return destroyNativeActivationDescriptor(desc)
	}
	return nil
}

// Batch Normalization Operations

// CreateBatchNormDescriptor creates a batch normalization descriptor
func CreateBatchNormDescriptor() (*BatchNormDescriptor, error) {
	if err := ensureCudaReady(); err != nil {
		return nil, err
	}

	desc := &BatchNormDescriptor{
		handle: uintptr(time.Now().UnixNano()),
		native: cuda.ShouldUseCuda() && cudnnNativeAvailable(),
	}

	return desc, nil
}

// BatchNormalizationForwardInference performs batch normalization inference
func (h *DNNHandle) BatchNormalizationForwardInference(mode DNNBatchNormMode, alpha, beta float32,
	inputDesc *TensorDescriptor, input *memory.Memory, outputDesc *TensorDescriptor, output *memory.Memory,
	bnScaleBiasDesc *TensorDescriptor, bnScale, bnBias, estimatedMean, estimatedVariance *memory.Memory, epsilon float64) error {

	if h.destroyed {
		return fmt.Errorf("handle has been destroyed")
	}
	if h.native {
		return nativeBatchNormForwardInference(h, mode, alpha, beta, inputDesc, input, outputDesc, output, bnScaleBiasDesc, bnScale, bnBias, estimatedMean, estimatedVariance, epsilon)
	}

	return h.performBatchNorm(input, output, bnScale, bnBias, estimatedMean, estimatedVariance, inputDesc, outputDesc, bnScaleBiasDesc, mode, alpha, beta, epsilon)
}

// DestroyBatchNormDescriptor destroys a batch normalization descriptor
func (desc *BatchNormDescriptor) DestroyBatchNormDescriptor() error {
	if desc.destroyed {
		return fmt.Errorf("descriptor already destroyed")
	}
	desc.destroyed = true
	return nil
}

// Helper functions for computation simulation

func (h *DNNHandle) performConvolution(input, filter, output *memory.Memory, inputDesc *TensorDescriptor, filterDesc *FilterDescriptor, outputDesc *TensorDescriptor, convDesc *ConvolutionDescriptor, alpha, beta float32) error {
	if input == nil || filter == nil || output == nil {
		return fmt.Errorf("input, filter, and output cannot be nil")
	}

	if inputDesc.dataType != DNNDataFloat || filterDesc.dataType != DNNDataFloat || outputDesc.dataType != DNNDataFloat {
		return fmt.Errorf("deterministic cuDNN convolution currently supports DNNDataFloat only")
	}
	inVals, err := readMathFloat32Memory(input, tensorElementCount(inputDesc))
	if err != nil {
		return err
	}
	filterVals, err := readMathFloat32Memory(filter, filterElementCount(filterDesc))
	if err != nil {
		return err
	}
	outVals, err := readMathFloat32Memory(output, tensorElementCount(outputDesc))
	if err != nil {
		return err
	}
	outputN, outputC, outputH, outputW := outputDesc.dimensions[0], outputDesc.dimensions[1], outputDesc.dimensions[2], outputDesc.dimensions[3]
	inputN, inputC, inputH, inputW := inputDesc.dimensions[0], inputDesc.dimensions[1], inputDesc.dimensions[2], inputDesc.dimensions[3]
	filterK, _, filterH, filterW := filterDesc.dimensions[0], filterDesc.dimensions[1], filterDesc.dimensions[2], filterDesc.dimensions[3]
	for n := range outputN {
		for k := range outputC {
			for oh := range outputH {
				for ow := range outputW {
					sum := float32(0)
					for c := range inputC {
						for fh := range filterH {
							for fw := range filterW {
								ih := oh*convDesc.strideH - convDesc.padH + fh*maxInt(convDesc.dilationH, 1)
								iw := ow*convDesc.strideW - convDesc.padW + fw*maxInt(convDesc.dilationW, 1)
								if ih < 0 || ih >= inputH || iw < 0 || iw >= inputW {
									continue
								}
								inputIndex := tensor4DIndex(inputDesc, n, c, ih, iw)
								filterIndex := ((k*inputC+c)*filterH+fh)*filterW + fw
								sum += inVals[inputIndex] * filterVals[filterIndex]
							}
						}
					}
					outputIndex := tensor4DIndex(outputDesc, n, k, oh, ow)
					outVals[outputIndex] = alpha*sum + beta*outVals[outputIndex]
				}
			}
		}
	}
	_ = inputN
	_ = filterK
	return writeMathFloat32Memory(output, outVals)
}

func (h *DNNHandle) performPooling(input, output *memory.Memory, inputDesc, outputDesc *TensorDescriptor, poolDesc *PoolingDescriptor, alpha, beta float32) error {
	if input == nil || output == nil {
		return fmt.Errorf("input and output cannot be nil")
	}

	if inputDesc.dataType != DNNDataFloat || outputDesc.dataType != DNNDataFloat {
		return fmt.Errorf("deterministic cuDNN pooling currently supports DNNDataFloat only")
	}
	inVals, err := readMathFloat32Memory(input, tensorElementCount(inputDesc))
	if err != nil {
		return err
	}
	outVals, err := readMathFloat32Memory(output, tensorElementCount(outputDesc))
	if err != nil {
		return err
	}
	for n := 0; n < outputDesc.dimensions[0]; n++ {
		for c := 0; c < outputDesc.dimensions[1]; c++ {
			for oh := 0; oh < outputDesc.dimensions[2]; oh++ {
				for ow := 0; ow < outputDesc.dimensions[3]; ow++ {
					accum := float32(0)
					count := 0
					maxValue := float32(math.Inf(-1))
					for wh := 0; wh < poolDesc.windowH; wh++ {
						for ww := 0; ww < poolDesc.windowW; ww++ {
							ih := oh*poolDesc.strideH - poolDesc.padH + wh
							iw := ow*poolDesc.strideW - poolDesc.padW + ww
							inside := ih >= 0 && ih < inputDesc.dimensions[2] && iw >= 0 && iw < inputDesc.dimensions[3]
							if !inside && poolDesc.mode == DNNPoolingAverageCountExcludePadding {
								continue
							}
							value := float32(0)
							if inside {
								value = inVals[tensor4DIndex(inputDesc, n, c, ih, iw)]
							}
							if poolDesc.mode == DNNPoolingMax || poolDesc.mode == DNNPoolingMaxDeterministic {
								if value > maxValue {
									maxValue = value
								}
							} else {
								accum += value
								count++
							}
						}
					}
					result := maxValue
					if poolDesc.mode != DNNPoolingMax && poolDesc.mode != DNNPoolingMaxDeterministic {
						if count > 0 {
							result = accum / float32(count)
						} else {
							result = 0
						}
					}
					outIndex := tensor4DIndex(outputDesc, n, c, oh, ow)
					outVals[outIndex] = alpha*result + beta*outVals[outIndex]
				}
			}
		}
	}
	return writeMathFloat32Memory(output, outVals)
}

func (h *DNNHandle) performActivation(input, output *memory.Memory, inputDesc, outputDesc *TensorDescriptor, activDesc *ActivationDescriptor, alpha, beta float32) error {
	if input == nil || output == nil {
		return fmt.Errorf("input and output cannot be nil")
	}

	if inputDesc.dataType != DNNDataFloat || outputDesc.dataType != DNNDataFloat {
		return fmt.Errorf("deterministic cuDNN activation currently supports DNNDataFloat only")
	}
	outputSize := tensorElementCount(outputDesc)
	outputData, err := readMathFloat32Memory(output, outputSize)
	if err != nil {
		return err
	}
	inputData, err := readMathFloat32Memory(input, tensorElementCount(inputDesc))
	if err != nil {
		return err
	}

	// Apply activation function
	for i := 0; i < len(outputData) && i < len(inputData); i++ {
		var activated float32
		switch activDesc.mode {
		case DNNActivationRelu:
			activated = float32(math.Max(0, float64(inputData[i])))
		case DNNActivationSigmoid:
			activated = float32(1.0 / (1.0 + math.Exp(-float64(inputData[i]))))
		case DNNActivationTanh:
			activated = float32(math.Tanh(float64(inputData[i])))
		case DNNActivationIdentity:
			activated = inputData[i]
		case DNNActivationClippedRelu:
			clipped := math.Min(float64(activDesc.coeff), math.Max(0, float64(inputData[i])))
			activated = float32(clipped)
		case DNNActivationElu:
			if inputData[i] >= 0 {
				activated = inputData[i]
			} else {
				activated = float32(activDesc.coeff * (math.Exp(float64(inputData[i])) - 1))
			}
		default:
			activated = inputData[i]
		}
		outputData[i] = alpha*activated + beta*outputData[i]
	}

	return writeMathFloat32Memory(output, outputData)
}

func (h *DNNHandle) performBatchNorm(input, output, scale, bias, mean, variance *memory.Memory, inputDesc, outputDesc, bnScaleBiasDesc *TensorDescriptor, mode DNNBatchNormMode, alpha, beta float32, epsilon float64) error {
	if input == nil || output == nil {
		return fmt.Errorf("input and output cannot be nil")
	}

	if inputDesc.dataType != DNNDataFloat || outputDesc.dataType != DNNDataFloat {
		return fmt.Errorf("deterministic cuDNN batch norm currently supports DNNDataFloat only")
	}
	outputSize := tensorElementCount(outputDesc)
	outputData, err := readMathFloat32Memory(output, outputSize)
	if err != nil {
		return err
	}
	inputData, err := readMathFloat32Memory(input, tensorElementCount(inputDesc))
	if err != nil {
		return err
	}
	scaleData, err := readMathFloat32Memory(scale, tensorElementCount(bnScaleBiasDesc))
	if err != nil {
		return err
	}
	biasData, err := readMathFloat32Memory(bias, tensorElementCount(bnScaleBiasDesc))
	if err != nil {
		return err
	}
	meanData, err := readMathFloat32Memory(mean, tensorElementCount(bnScaleBiasDesc))
	if err != nil {
		return err
	}
	varData, err := readMathFloat32Memory(variance, tensorElementCount(bnScaleBiasDesc))
	if err != nil {
		return err
	}
	channels := inputDesc.dimensions[1]
	for n := 0; n < inputDesc.dimensions[0]; n++ {
		for c := range channels {
			for hIndex := 0; hIndex < inputDesc.dimensions[2]; hIndex++ {
				for wIndex := 0; wIndex < inputDesc.dimensions[3]; wIndex++ {
					index := tensor4DIndex(inputDesc, n, c, hIndex, wIndex)
					paramIndex := c
					if mode == DNNBatchNormPerActivation {
						paramIndex = (c*inputDesc.dimensions[2]+hIndex)*inputDesc.dimensions[3] + wIndex
					}
					normalized := (inputData[index] - meanData[paramIndex]) / float32(math.Sqrt(float64(varData[paramIndex])+epsilon))
					scaled := normalized*scaleData[paramIndex] + biasData[paramIndex]
					outputData[index] = alpha*scaled + beta*outputData[index]
				}
			}
		}
	}
	return writeMathFloat32Memory(output, outputData)
}

func tensorElementCount(desc *TensorDescriptor) int {
	total := 1
	for _, dim := range desc.dimensions {
		total *= dim
	}
	return total
}

func filterElementCount(desc *FilterDescriptor) int {
	total := 1
	for _, dim := range desc.dimensions {
		total *= dim
	}
	return total
}

func tensor4DIndex(desc *TensorDescriptor, n, c, h, w int) int {
	if desc.format == DNNTensorNHWC {
		return ((n*desc.dimensions[2]+h)*desc.dimensions[3]+w)*desc.dimensions[1] + c
	}
	return ((n*desc.dimensions[1]+c)*desc.dimensions[2]+h)*desc.dimensions[3] + w
}
