// Package libraries provides CUDA runtime library bindings for GoCUDA
// This implements cuDNN functionality for Deep Neural Networks
package libraries

import (
	"errors"
	"fmt"
	"math"
	"runtime"
	"slices"
	"sync"
	"time"
	"unsafe"

	cuda "github.com/stitch1968/gocuda"
	"github.com/stitch1968/gocuda/internal"
	"github.com/stitch1968/gocuda/memory"
	"github.com/stitch1968/gocuda/streams"
)

// cuDNN - Deep Neural Network Library

// DNNHandle represents a cuDNN library handle
type DNNHandle struct {
	handle             uintptr
	destroyed          bool
	native             bool
	deviceID           int
	mu                 sync.Mutex
	pendingBuffers     []*memory.Retention
	pendingDescriptors []*dnnDescriptorState
	pendingWorkspaces  []*memory.Memory
	nativeStream       *streams.NativeStream
}

// TensorDescriptor describes the layout of a tensor
type TensorDescriptor struct {
	dnnDescriptorState
	handle     uintptr
	dataType   DNNDataType
	format     DNNTensorFormat
	dimensions []int
	strides    []int
	native     bool
}

// FilterDescriptor describes a convolution filter
type FilterDescriptor struct {
	dnnDescriptorState
	handle     uintptr
	dataType   DNNDataType
	format     DNNTensorFormat
	dimensions []int // [outputChannels, inputChannels, height, width]
	native     bool
}

// ConvolutionDescriptor describes a convolution operation
type ConvolutionDescriptor struct {
	dnnDescriptorState
	handle    uintptr
	padH      int
	padW      int
	strideH   int
	strideW   int
	dilationH int
	dilationW int
	mode      DNNConvolutionMode
	dataType  DNNDataType
	native    bool
}

// PoolingDescriptor describes a pooling operation
type PoolingDescriptor struct {
	dnnDescriptorState
	handle  uintptr
	mode    DNNPoolingMode
	windowH int
	windowW int
	padH    int
	padW    int
	strideH int
	strideW int
	native  bool
}

// ActivationDescriptor describes an activation function
type ActivationDescriptor struct {
	dnnDescriptorState
	handle uintptr
	mode   DNNActivationMode
	nanOpt DNNNanPropagation
	coeff  float64
	native bool
}

// BatchNormDescriptor describes batch normalization
type BatchNormDescriptor struct {
	dnnDescriptorState
	handle uintptr
	mode   DNNBatchNormMode
	native bool
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

func validDNNDataType(dataType DNNDataType) bool {
	return dataType >= DNNDataFloat && dataType <= DNNDataInt64
}

func validDNNTensorFormat(format DNNTensorFormat) bool {
	return format >= DNNTensorNHWC && format <= DNNTensorNHWCVectC
}

func validateDNN4DShape(dimensions ...int) error {
	count := uint64(1)
	for i, dimension := range dimensions {
		if dimension <= 0 || int64(dimension) > math.MaxInt32 || uint64(dimension) > uint64(math.MaxInt32)/count {
			return fmt.Errorf("invalid 4D dimension or tensor size at index %d", i)
		}
		count *= uint64(dimension)
	}
	return nil
}

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
		runtime.LockOSThread()
		defer runtime.UnlockOSThread()
		deviceID, err := internal.CurrentDevice()
		if err != nil {
			return nil, err
		}
		return createNativeDNNHandleOnDevice(deviceID)
	}

	handle := &DNNHandle{
		handle: uintptr(time.Now().UnixNano()),
	}

	return handle, nil
}

// CreateNativeDNNHandleOnDevice creates a device-bound native-only cuDNN handle.
func CreateNativeDNNHandleOnDevice(deviceID int) (*DNNHandle, error) {
	if err := ensureCudaReady(); err != nil {
		return nil, err
	}
	if !cuda.ShouldUseCuda() || !cudnnNativeAvailable() {
		return nil, streams.ErrNativeCUDAUnavailable
	}
	if deviceID < 0 || deviceID >= cuda.GetCudaDeviceCount() {
		return nil, fmt.Errorf("device ID %d out of range", deviceID)
	}
	return createNativeDNNHandleOnDevice(deviceID)
}

func createNativeDNNHandleOnDevice(deviceID int) (*DNNHandle, error) {
	var handle *DNNHandle
	err := internal.RunOnDevice(deviceID, func() error {
		var createErr error
		handle, createErr = createNativeDNNHandle()
		return createErr
	})
	if err != nil {
		if handle != nil {
			err = errors.Join(err, internal.RunOnDevice(deviceID, func() error { return destroyNativeDNNHandle(handle) }))
		}
		return nil, err
	}
	handle.deviceID = deviceID
	return handle, nil
}

// Backend reports whether this handle uses native cuDNN or Go simulation.
func (h *DNNHandle) Backend() LibraryImplementationMode {
	if h == nil {
		return ""
	}
	if h.native {
		return ImplementationModeNativeCUDA
	}
	return ImplementationModeValidatedGo
}

// DestroyHandle destroys the cuDNN handle
func (h *DNNHandle) DestroyHandle() error {
	if h == nil {
		return fmt.Errorf("nil cuDNN handle")
	}
	h.mu.Lock()
	defer h.mu.Unlock()
	if h.destroyed {
		return fmt.Errorf("handle already destroyed")
	}
	if h.native {
		if err := h.synchronizeLocked(); err != nil {
			return err
		}
		if err := internal.RunOnDevice(h.deviceID, func() error {
			if h.nativeStream != nil {
				if err := setNativeDNNStream(h, nil); err != nil {
					return err
				}
				h.nativeStream = nil
			}
			return destroyNativeDNNHandle(h)
		}); err != nil {
			return err
		}
	}
	h.destroyed = true
	return nil
}

// SetStream attaches a native CUDA stream, or selects the default stream when
// stream is nil. Existing work is synchronized before changing streams.
// cuDNN operations remain synchronous.
func (h *DNNHandle) SetStream(stream *streams.NativeStream) error {
	if h == nil {
		return fmt.Errorf("nil cuDNN handle")
	}
	h.mu.Lock()
	defer h.mu.Unlock()
	if h.destroyed {
		return fmt.Errorf("cuDNN handle has been destroyed")
	}
	if !h.native {
		if stream != nil {
			return fmt.Errorf("native stream attachment requires a CUDA cuDNN handle")
		}
		return nil
	}
	if stream != nil && stream.DeviceID() != h.deviceID {
		return fmt.Errorf("stream device %d does not match cuDNN handle device %d", stream.DeviceID(), h.deviceID)
	}
	if err := h.synchronizeLocked(); err != nil {
		return err
	}
	setStream := func(handle unsafe.Pointer) error {
		if err := setNativeDNNStream(h, handle); err != nil {
			return err
		}
		h.nativeStream = stream
		return nil
	}
	if stream != nil {
		return stream.WithNativeHandle(setStream)
	}
	return internal.RunOnDevice(h.deviceID, func() error { return setStream(nil) })
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
	if desc == nil {
		return fmt.Errorf("nil tensor descriptor")
	}
	if err := desc.lockMutable(); err != nil {
		return err
	}
	defer desc.mu.Unlock()
	if !validDNNDataType(dataType) {
		return fmt.Errorf("invalid tensor data type")
	}
	if len(dimensions) == 0 || len(dimensions) != len(strides) || int64(len(dimensions)) > math.MaxInt32 {
		return fmt.Errorf("dimensions and strides must have the same nonzero length")
	}
	for i := range dimensions {
		if dimensions[i] <= 0 || int64(dimensions[i]) > math.MaxInt32 ||
			int64(strides[i]) < math.MinInt32 || int64(strides[i]) > math.MaxInt32 {
			return fmt.Errorf("tensor dimension or stride at index %d is invalid", i)
		}
	}
	if desc.native {
		if err := setNativeTensorNdDescriptor(desc, dataType, dimensions, strides); err != nil {
			return err
		}
	}
	desc.dataType = dataType
	desc.dimensions = make([]int, len(dimensions))
	desc.strides = make([]int, len(strides))
	copy(desc.dimensions, dimensions)
	copy(desc.strides, strides)
	return nil
}

// SetTensor4dDescriptor sets the tensor descriptor for 4D tensors
func (desc *TensorDescriptor) SetTensor4dDescriptor(format DNNTensorFormat, dataType DNNDataType, n, c, h, w int) error {
	if desc == nil {
		return fmt.Errorf("nil tensor descriptor")
	}
	if err := desc.lockMutable(); err != nil {
		return err
	}
	defer desc.mu.Unlock()
	if !validDNNTensorFormat(format) || !validDNNDataType(dataType) {
		return fmt.Errorf("invalid tensor format or data type")
	}
	if err := validateDNN4DShape(n, c, h, w); err != nil {
		return err
	}
	if desc.native {
		if err := setNativeTensor4dDescriptor(desc, format, dataType, n, c, h, w); err != nil {
			return err
		}
	}
	var strides []int
	if format == DNNTensorNCHW {
		strides = []int{c * h * w, h * w, w, 1}
	} else if format == DNNTensorNHWC {
		strides = []int{h * w * c, 1, w * c, c}
	}
	desc.format = format
	desc.dataType = dataType
	desc.dimensions = []int{n, c, h, w}
	desc.strides = strides
	return nil
}

// DestroyTensorDescriptor destroys a tensor descriptor
func (desc *TensorDescriptor) DestroyTensorDescriptor() error {
	if desc == nil {
		return fmt.Errorf("nil tensor descriptor")
	}
	if err := desc.lockMutable(); err != nil {
		return err
	}
	defer desc.mu.Unlock()
	if desc.native {
		if err := destroyNativeTensorDescriptor(desc); err != nil {
			return err
		}
	}
	desc.destroyed = true
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
	if desc == nil {
		return fmt.Errorf("nil filter descriptor")
	}
	if err := desc.lockMutable(); err != nil {
		return err
	}
	defer desc.mu.Unlock()
	if !validDNNDataType(dataType) || !validDNNTensorFormat(format) {
		return fmt.Errorf("invalid filter format or data type")
	}
	if len(dimensions) == 0 || int64(len(dimensions)) > math.MaxInt32 {
		return fmt.Errorf("filter dimensions cannot be empty")
	}
	for i, dimension := range dimensions {
		if dimension <= 0 || int64(dimension) > math.MaxInt32 {
			return fmt.Errorf("filter dimension at index %d is invalid", i)
		}
	}
	if desc.native {
		if err := setNativeFilterNdDescriptor(desc, dataType, format, dimensions); err != nil {
			return err
		}
	}
	desc.dataType = dataType
	desc.format = format
	desc.dimensions = make([]int, len(dimensions))
	copy(desc.dimensions, dimensions)
	return nil
}

// SetFilter4dDescriptor sets the filter descriptor for 4D filters
func (desc *FilterDescriptor) SetFilter4dDescriptor(dataType DNNDataType, format DNNTensorFormat, k, c, h, w int) error {
	if desc == nil {
		return fmt.Errorf("nil filter descriptor")
	}
	if err := desc.lockMutable(); err != nil {
		return err
	}
	defer desc.mu.Unlock()
	if !validDNNTensorFormat(format) || !validDNNDataType(dataType) {
		return fmt.Errorf("invalid filter format or data type")
	}
	for i, dimension := range []int{k, c, h, w} {
		if dimension <= 0 || int64(dimension) > math.MaxInt32 {
			return fmt.Errorf("invalid filter dimension at index %d", i)
		}
	}
	if desc.native {
		if err := setNativeFilter4dDescriptor(desc, dataType, format, k, c, h, w); err != nil {
			return err
		}
	}
	desc.dataType = dataType
	desc.format = format
	desc.dimensions = []int{k, c, h, w}
	return nil
}

// DestroyFilterDescriptor destroys a filter descriptor
func (desc *FilterDescriptor) DestroyFilterDescriptor() error {
	if desc == nil {
		return fmt.Errorf("nil filter descriptor")
	}
	if err := desc.lockMutable(); err != nil {
		return err
	}
	defer desc.mu.Unlock()
	if desc.native {
		if err := destroyNativeFilterDescriptor(desc); err != nil {
			return err
		}
	}
	desc.destroyed = true
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
	if desc == nil {
		return fmt.Errorf("nil convolution descriptor")
	}
	if err := desc.lockMutable(); err != nil {
		return err
	}
	defer desc.mu.Unlock()
	if padH < 0 || padW < 0 || strideH <= 0 || strideW <= 0 || dilationH <= 0 || dilationW <= 0 ||
		int64(padH) > math.MaxInt32 || int64(padW) > math.MaxInt32 ||
		int64(strideH) > math.MaxInt32 || int64(strideW) > math.MaxInt32 ||
		int64(dilationH) > math.MaxInt32 || int64(dilationW) > math.MaxInt32 ||
		(mode != DNNConvolution && mode != DNNCrossCorrelation) || !validDNNDataType(dataType) {
		return fmt.Errorf("invalid convolution padding, stride, dilation, mode, or compute type")
	}
	if desc.native {
		if err := setNativeConvolution2dDescriptor(desc, padH, padW, strideH, strideW, dilationH, dilationW, mode, dataType); err != nil {
			return err
		}
	}
	desc.padH = padH
	desc.padW = padW
	desc.strideH = strideH
	desc.strideW = strideW
	desc.dilationH = dilationH
	desc.dilationW = dilationW
	desc.mode = mode
	desc.dataType = dataType
	return nil
}

// GetConvolution2dForwardOutputDim calculates output dimensions for convolution
func (desc *ConvolutionDescriptor) GetConvolution2dForwardOutputDim(inputDesc *TensorDescriptor, filterDesc *FilterDescriptor) (n, c, h, w int, err error) {
	if desc == nil || inputDesc == nil || filterDesc == nil {
		return 0, 0, 0, 0, fmt.Errorf("descriptors cannot be nil")
	}
	held, err := retainDNNDescriptors(&desc.dnnDescriptorState, &inputDesc.dnnDescriptorState, &filterDesc.dnnDescriptorState)
	if err != nil {
		return 0, 0, 0, 0, err
	}
	defer releaseDNNDescriptors(held)
	if len(inputDesc.dimensions) != 4 || len(filterDesc.dimensions) != 4 ||
		desc.padH < 0 || desc.padW < 0 || desc.strideH <= 0 || desc.strideW <= 0 || desc.dilationH <= 0 || desc.dilationW <= 0 ||
		int64(desc.padH) > math.MaxInt32 || int64(desc.padW) > math.MaxInt32 ||
		int64(desc.strideH) > math.MaxInt32 || int64(desc.strideW) > math.MaxInt32 ||
		int64(desc.dilationH) > math.MaxInt32 || int64(desc.dilationW) > math.MaxInt32 {
		return 0, 0, 0, 0, fmt.Errorf("invalid tensor dimensions")
	}
	for i := range 4 {
		if inputDesc.dimensions[i] <= 0 || filterDesc.dimensions[i] <= 0 ||
			int64(inputDesc.dimensions[i]) > math.MaxInt32 || int64(filterDesc.dimensions[i]) > math.MaxInt32 {
			return 0, 0, 0, 0, fmt.Errorf("invalid tensor or filter dimension at index %d", i)
		}
	}
	if inputDesc.dimensions[1] != filterDesc.dimensions[1] {
		return 0, 0, 0, 0, fmt.Errorf("input and filter channel counts do not match")
	}
	paddedH := int64(inputDesc.dimensions[2]) + 2*int64(desc.padH)
	paddedW := int64(inputDesc.dimensions[3]) + 2*int64(desc.padW)
	filterH := (int64(filterDesc.dimensions[2])-1)*int64(desc.dilationH) + 1
	filterW := (int64(filterDesc.dimensions[3])-1)*int64(desc.dilationW) + 1
	if paddedH < filterH || paddedW < filterW {
		return 0, 0, 0, 0, fmt.Errorf("convolution filter exceeds padded input")
	}
	outputH := (paddedH-filterH)/int64(desc.strideH) + 1
	outputW := (paddedW-filterW)/int64(desc.strideW) + 1
	if outputH > math.MaxInt32 || outputW > math.MaxInt32 {
		return 0, 0, 0, 0, fmt.Errorf("convolution output dimensions exceed 32-bit range")
	}
	if desc.native {
		if !inputDesc.native || !filterDesc.native {
			return 0, 0, 0, 0, fmt.Errorf("native convolution requires native input and filter descriptors")
		}
		return nativeConvolutionForwardOutputDim(desc, inputDesc, filterDesc)
	}
	return inputDesc.dimensions[0], filterDesc.dimensions[0], int(outputH), int(outputW), nil
}

// ConvolutionForward performs forward convolution
func (h *DNNHandle) ConvolutionForward(alpha float32, inputDesc *TensorDescriptor, inputData *memory.Memory,
	filterDesc *FilterDescriptor, filterData *memory.Memory, convDesc *ConvolutionDescriptor,
	beta float32, outputDesc *TensorDescriptor, outputData *memory.Memory) error {

	if h == nil || inputDesc == nil || filterDesc == nil || convDesc == nil || outputDesc == nil ||
		inputData == nil || filterData == nil || outputData == nil {
		return fmt.Errorf("handle, descriptors, and buffers cannot be nil")
	}
	states := []*dnnDescriptorState{&inputDesc.dnnDescriptorState, &filterDesc.dnnDescriptorState,
		&convDesc.dnnDescriptorState, &outputDesc.dnnDescriptorState}
	return h.execute(states, []*memory.Memory{inputData, filterData, outputData}, func() (*memory.Memory, error) {
		n, c, height, width, err := convDesc.GetConvolution2dForwardOutputDim(inputDesc, filterDesc)
		if err != nil {
			return nil, err
		}
		if len(outputDesc.dimensions) != 4 || outputDesc.dimensions[0] != n || outputDesc.dimensions[1] != c ||
			outputDesc.dimensions[2] != height || outputDesc.dimensions[3] != width {
			return nil, fmt.Errorf("convolution output dimensions do not match the computed shape")
		}
		if h.native {
			if !inputDesc.native || !filterDesc.native || !convDesc.native || !outputDesc.native {
				return nil, fmt.Errorf("native convolution requires native descriptors")
			}
			if err := validateDNNScalingTypes(inputDesc.dataType, filterDesc.dataType, convDesc.dataType, outputDesc.dataType); err != nil {
				return nil, err
			}
			if err := validateDNNTensorBuffer(inputDesc, inputData); err != nil {
				return nil, fmt.Errorf("convolution input: %w", err)
			}
			if err := validateDNNFilterBuffer(filterDesc, filterData); err != nil {
				return nil, fmt.Errorf("convolution filter: %w", err)
			}
			if err := validateDNNTensorBuffer(outputDesc, outputData); err != nil {
				return nil, fmt.Errorf("convolution output: %w", err)
			}
			return nativeConvolutionForward(h, alpha, inputDesc, inputData, filterDesc, filterData, convDesc, beta, outputDesc, outputData)
		}
		return nil, h.performConvolution(inputData, filterData, outputData, inputDesc, filterDesc, outputDesc, convDesc, alpha, beta)
	})
}

// DestroyConvolutionDescriptor destroys a convolution descriptor
func (desc *ConvolutionDescriptor) DestroyConvolutionDescriptor() error {
	if desc == nil {
		return fmt.Errorf("nil convolution descriptor")
	}
	if err := desc.lockMutable(); err != nil {
		return err
	}
	defer desc.mu.Unlock()
	if desc.native {
		if err := destroyNativeConvolutionDescriptor(desc); err != nil {
			return err
		}
	}
	desc.destroyed = true
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
	if desc == nil {
		return fmt.Errorf("nil pooling descriptor")
	}
	if err := desc.lockMutable(); err != nil {
		return err
	}
	defer desc.mu.Unlock()
	if err := validateDNNPoolingConfig(mode, windowH, windowW, padH, padW, strideH, strideW); err != nil {
		return err
	}
	if desc.native {
		if err := setNativePooling2dDescriptor(desc, mode, windowH, windowW, padH, padW, strideH, strideW); err != nil {
			return err
		}
	}
	desc.mode = mode
	desc.windowH = windowH
	desc.windowW = windowW
	desc.padH = padH
	desc.padW = padW
	desc.strideH = strideH
	desc.strideW = strideW
	return nil
}

func validateDNNPoolingConfig(mode DNNPoolingMode, windowH, windowW, padH, padW, strideH, strideW int) error {
	switch mode {
	case DNNPoolingMax, DNNPoolingMaxDeterministic, DNNPoolingAverageCountIncludePadding, DNNPoolingAverageCountExcludePadding:
	default:
		return fmt.Errorf("unsupported cuDNN pooling mode %d", mode)
	}
	for _, value := range []int{windowH, windowW, strideH, strideW} {
		if value <= 0 || int64(value) > math.MaxInt32 {
			return fmt.Errorf("pooling windows and strides must be positive 32-bit integers")
		}
	}
	for _, value := range []int{padH, padW} {
		if value < 0 || int64(value) > math.MaxInt32 {
			return fmt.Errorf("pooling padding must be nonnegative 32-bit integers")
		}
	}
	return nil
}

// PoolingForward performs forward pooling
func (h *DNNHandle) PoolingForward(poolingDesc *PoolingDescriptor, alpha float32, inputDesc *TensorDescriptor, inputData *memory.Memory,
	beta float32, outputDesc *TensorDescriptor, outputData *memory.Memory) error {

	if h == nil || poolingDesc == nil || inputDesc == nil || outputDesc == nil {
		return fmt.Errorf("handle and descriptors cannot be nil")
	}
	states := []*dnnDescriptorState{&poolingDesc.dnnDescriptorState, &inputDesc.dnnDescriptorState, &outputDesc.dnnDescriptorState}
	return h.execute(states, []*memory.Memory{inputData, outputData}, func() (*memory.Memory, error) {
		if h.native {
			if !poolingDesc.native || !inputDesc.native || !outputDesc.native {
				return nil, fmt.Errorf("native pooling requires native descriptors")
			}
			if inputDesc.dataType != outputDesc.dataType {
				return nil, fmt.Errorf("native pooling input and output types must match")
			}
			if err := validateDNNTensorBuffer(inputDesc, inputData); err != nil {
				return nil, fmt.Errorf("pooling input: %w", err)
			}
			if err := validateDNNTensorBuffer(outputDesc, outputData); err != nil {
				return nil, fmt.Errorf("pooling output: %w", err)
			}
			return nil, nativePoolingForward(h, poolingDesc, alpha, inputDesc, inputData, beta, outputDesc, outputData)
		}
		return nil, h.performPooling(inputData, outputData, inputDesc, outputDesc, poolingDesc, alpha, beta)
	})
}

// DestroyPoolingDescriptor destroys a pooling descriptor
func (desc *PoolingDescriptor) DestroyPoolingDescriptor() error {
	if desc == nil {
		return fmt.Errorf("nil pooling descriptor")
	}
	if err := desc.lockMutable(); err != nil {
		return err
	}
	defer desc.mu.Unlock()
	if desc.native {
		if err := destroyNativePoolingDescriptor(desc); err != nil {
			return err
		}
	}
	desc.destroyed = true
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
	if desc == nil {
		return fmt.Errorf("nil activation descriptor")
	}
	if err := desc.lockMutable(); err != nil {
		return err
	}
	defer desc.mu.Unlock()
	if desc.native {
		if err := setNativeActivationDescriptor(desc, mode, nanOpt, coeff); err != nil {
			return err
		}
	}
	desc.mode = mode
	desc.nanOpt = nanOpt
	desc.coeff = coeff
	return nil
}

// ActivationForward performs forward activation
func (h *DNNHandle) ActivationForward(activationDesc *ActivationDescriptor, alpha float32, inputDesc *TensorDescriptor, inputData *memory.Memory,
	beta float32, outputDesc *TensorDescriptor, outputData *memory.Memory) error {

	if h == nil || activationDesc == nil || inputDesc == nil || outputDesc == nil {
		return fmt.Errorf("handle and descriptors cannot be nil")
	}
	states := []*dnnDescriptorState{&activationDesc.dnnDescriptorState, &inputDesc.dnnDescriptorState, &outputDesc.dnnDescriptorState}
	return h.execute(states, []*memory.Memory{inputData, outputData}, func() (*memory.Memory, error) {
		if h.native {
			if !activationDesc.native || !inputDesc.native || !outputDesc.native {
				return nil, fmt.Errorf("native activation requires native descriptors")
			}
			if inputDesc.dataType != outputDesc.dataType {
				return nil, fmt.Errorf("native activation input and output types must match")
			}
			if inputData.Ptr() == outputData.Ptr() &&
				(!slices.Equal(inputDesc.dimensions, outputDesc.dimensions) || !slices.Equal(inputDesc.strides, outputDesc.strides)) {
				return nil, fmt.Errorf("in-place native activation requires matching tensor dimensions and strides")
			}
			if err := validateDNNTensorBuffer(inputDesc, inputData); err != nil {
				return nil, fmt.Errorf("activation input: %w", err)
			}
			if err := validateDNNTensorBuffer(outputDesc, outputData); err != nil {
				return nil, fmt.Errorf("activation output: %w", err)
			}
			return nil, nativeActivationForward(h, activationDesc, alpha, inputDesc, inputData, beta, outputDesc, outputData)
		}
		return nil, h.performActivation(inputData, outputData, inputDesc, outputDesc, activationDesc, alpha, beta)
	})
}

// DestroyActivationDescriptor destroys an activation descriptor
func (desc *ActivationDescriptor) DestroyActivationDescriptor() error {
	if desc == nil {
		return fmt.Errorf("nil activation descriptor")
	}
	if err := desc.lockMutable(); err != nil {
		return err
	}
	defer desc.mu.Unlock()
	if desc.native {
		if err := destroyNativeActivationDescriptor(desc); err != nil {
			return err
		}
	}
	desc.destroyed = true
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

	if h == nil || inputDesc == nil || outputDesc == nil || bnScaleBiasDesc == nil {
		return fmt.Errorf("handle and descriptors cannot be nil")
	}
	states := []*dnnDescriptorState{&inputDesc.dnnDescriptorState, &outputDesc.dnnDescriptorState, &bnScaleBiasDesc.dnnDescriptorState}
	buffers := []*memory.Memory{input, output, bnScale, bnBias, estimatedMean, estimatedVariance}
	return h.execute(states, buffers, func() (*memory.Memory, error) {
		if h.native {
			if !inputDesc.native || !outputDesc.native || !bnScaleBiasDesc.native {
				return nil, fmt.Errorf("native batch normalization requires native descriptors")
			}
			if err := validateDNNScalingTypes(inputDesc.dataType, outputDesc.dataType, bnScaleBiasDesc.dataType); err != nil {
				return nil, err
			}
			for i, entry := range []struct {
				desc   *TensorDescriptor
				buffer *memory.Memory
			}{{inputDesc, input}, {outputDesc, output}, {bnScaleBiasDesc, bnScale}, {bnScaleBiasDesc, bnBias},
				{bnScaleBiasDesc, estimatedMean}, {bnScaleBiasDesc, estimatedVariance}} {
				if err := validateDNNTensorBuffer(entry.desc, entry.buffer); err != nil {
					return nil, fmt.Errorf("batch normalization buffer %d: %w", i, err)
				}
			}
			return nil, nativeBatchNormForwardInference(h, mode, alpha, beta, inputDesc, input, outputDesc, output, bnScaleBiasDesc, bnScale, bnBias, estimatedMean, estimatedVariance, epsilon)
		}
		return nil, h.performBatchNorm(input, output, bnScale, bnBias, estimatedMean, estimatedVariance, inputDesc, outputDesc, bnScaleBiasDesc, mode, alpha, beta, epsilon)
	})
}

// DestroyBatchNormDescriptor destroys a batch normalization descriptor
func (desc *BatchNormDescriptor) DestroyBatchNormDescriptor() error {
	if desc == nil {
		return fmt.Errorf("nil batch-normalization descriptor")
	}
	if err := desc.lockMutable(); err != nil {
		return err
	}
	defer desc.mu.Unlock()
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
	if convDesc.dataType != DNNDataFloat {
		return fmt.Errorf("simulation convolution requires DNNDataFloat compute type")
	}
	if err := validatePackedTensor4D(inputDesc); err != nil {
		return fmt.Errorf("convolution input layout: %w", err)
	}
	if err := validatePackedTensor4D(outputDesc); err != nil {
		return fmt.Errorf("convolution output layout: %w", err)
	}
	if filterDesc.format != DNNTensorNCHW {
		return fmt.Errorf("simulation convolution requires NCHW filter layout")
	}
	maxElements := uint64(^uint(0)>>1) / 4
	filterCount := uint64(1)
	for _, dimension := range filterDesc.dimensions {
		if dimension <= 0 || uint64(dimension) > maxElements/filterCount {
			return fmt.Errorf("invalid convolution filter size")
		}
		filterCount *= uint64(dimension)
	}
	inVals, err := readMathFloat32Memory(input, tensorElementCount(inputDesc))
	if err != nil {
		return err
	}
	filterVals, err := readMathFloat32Memory(filter, int(filterCount))
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
								filterRow, filterCol := fh, fw
								if convDesc.mode == DNNConvolution {
									filterRow, filterCol = filterH-1-fh, filterW-1-fw
								}
								filterIndex := ((k*inputC+c)*filterH+filterRow)*filterW + filterCol
								sum += inVals[inputIndex] * filterVals[filterIndex]
							}
						}
					}
					outputIndex := tensor4DIndex(outputDesc, n, k, oh, ow)
					if beta == 0 {
						outVals[outputIndex] = alpha * sum
					} else {
						outVals[outputIndex] = alpha*sum + beta*outVals[outputIndex]
					}
				}
			}
		}
	}
	_ = inputN
	_ = filterK
	return writeMathFloat32Memory(output, outVals)
}

func validatePackedTensor4D(desc *TensorDescriptor) error {
	if len(desc.dimensions) != 4 {
		return fmt.Errorf("expected a four-dimensional tensor")
	}
	if _, err := tensorFloat32StorageSize(desc); err != nil {
		return err
	}
	c, h, w := desc.dimensions[1], desc.dimensions[2], desc.dimensions[3]
	var expected []int
	switch desc.format {
	case DNNTensorNCHW:
		expected = []int{c * h * w, h * w, w, 1}
	case DNNTensorNHWC:
		expected = []int{h * w * c, 1, w * c, c}
	default:
		return fmt.Errorf("unsupported tensor format")
	}
	for i := range expected {
		if desc.dimensions[i] > 1 && desc.strides[i] != expected[i] {
			return fmt.Errorf("simulation convolution requires packed tensor strides")
		}
	}
	return nil
}

func (h *DNNHandle) performPooling(input, output *memory.Memory, inputDesc, outputDesc *TensorDescriptor, poolDesc *PoolingDescriptor, alpha, beta float32) error {
	if input == nil || output == nil {
		return fmt.Errorf("input and output cannot be nil")
	}

	if inputDesc.dataType != DNNDataFloat || outputDesc.dataType != DNNDataFloat {
		return fmt.Errorf("deterministic cuDNN pooling currently supports DNNDataFloat only")
	}
	if err := validateDNNPoolingConfig(poolDesc.mode, poolDesc.windowH, poolDesc.windowW, poolDesc.padH, poolDesc.padW, poolDesc.strideH, poolDesc.strideW); err != nil {
		return err
	}
	if poolDesc.padH >= poolDesc.windowH || poolDesc.padW >= poolDesc.windowW {
		return fmt.Errorf("simulation pooling requires padding smaller than the window")
	}
	if len(inputDesc.dimensions) != 4 || len(outputDesc.dimensions) != 4 {
		return fmt.Errorf("simulation pooling requires four-dimensional tensors")
	}
	if inputDesc.dimensions[0] != outputDesc.dimensions[0] || inputDesc.dimensions[1] != outputDesc.dimensions[1] {
		return fmt.Errorf("pooling input and output batch and channel dimensions must match")
	}
	for i, axis := range []struct{ window, pad, stride int }{
		{poolDesc.windowH, poolDesc.padH, poolDesc.strideH},
		{poolDesc.windowW, poolDesc.padW, poolDesc.strideW},
	} {
		numerator := int64(inputDesc.dimensions[i+2]) + 2*int64(axis.pad) - int64(axis.window)
		if numerator < 0 || int64(outputDesc.dimensions[i+2]) != numerator/int64(axis.stride)+1 {
			return fmt.Errorf("pooling output dimensions do not match the computed shape")
		}
	}
	inSize, err := tensorFloat32StorageSize(inputDesc)
	if err != nil {
		return fmt.Errorf("pooling input layout: %w", err)
	}
	outSize, err := tensorFloat32StorageSize(outputDesc)
	if err != nil {
		return fmt.Errorf("pooling output layout: %w", err)
	}
	inVals, err := readMathFloat32Memory(input, inSize)
	if err != nil {
		return err
	}
	outVals, err := readMathFloat32Memory(output, outSize)
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
							ih := int64(oh)*int64(poolDesc.strideH) - int64(poolDesc.padH) + int64(wh)
							iw := int64(ow)*int64(poolDesc.strideW) - int64(poolDesc.padW) + int64(ww)
							inside := ih >= 0 && ih < int64(inputDesc.dimensions[2]) && iw >= 0 && iw < int64(inputDesc.dimensions[3])
							if !inside && poolDesc.mode != DNNPoolingAverageCountIncludePadding {
								continue
							}
							value := float32(0)
							if inside {
								index := n*inputDesc.strides[0] + c*inputDesc.strides[1] + int(ih)*inputDesc.strides[2] + int(iw)*inputDesc.strides[3]
								value = inVals[index]
							}
							if poolDesc.mode == DNNPoolingMax || poolDesc.mode == DNNPoolingMaxDeterministic {
								if value > maxValue || math.IsNaN(float64(value)) {
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
					outIndex := n*outputDesc.strides[0] + c*outputDesc.strides[1] + oh*outputDesc.strides[2] + ow*outputDesc.strides[3]
					if beta == 0 {
						outVals[outIndex] = alpha * result
					} else {
						outVals[outIndex] = alpha*result + beta*outVals[outIndex]
					}
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
	switch activDesc.mode {
	case DNNActivationRelu, DNNActivationSigmoid, DNNActivationTanh, DNNActivationIdentity,
		DNNActivationClippedRelu, DNNActivationElu:
	default:
		return fmt.Errorf("activation mode %d is not supported in simulation", activDesc.mode)
	}
	if len(inputDesc.dimensions) != len(outputDesc.dimensions) {
		return fmt.Errorf("activation input and output dimensions must match")
	}
	for i := range inputDesc.dimensions {
		if inputDesc.dimensions[i] != outputDesc.dimensions[i] {
			return fmt.Errorf("activation input and output dimensions must match")
		}
	}
	inputSize, err := tensorFloat32StorageSize(inputDesc)
	if err != nil {
		return fmt.Errorf("activation input layout: %w", err)
	}
	outputSize, err := tensorFloat32StorageSize(outputDesc)
	if err != nil {
		return fmt.Errorf("activation output layout: %w", err)
	}
	outputData, err := readMathFloat32Memory(output, outputSize)
	if err != nil {
		return err
	}
	inputData, err := readMathFloat32Memory(input, inputSize)
	if err != nil {
		return err
	}

	indices := make([]int, len(inputDesc.dimensions))
	for {
		inputIndex, outputIndex := 0, 0
		for i, index := range indices {
			inputIndex += index * inputDesc.strides[i]
			outputIndex += index * outputDesc.strides[i]
		}
		value := inputData[inputIndex]
		var activated float32
		switch activDesc.mode {
		case DNNActivationRelu:
			activated = float32(math.Max(0, float64(value)))
		case DNNActivationSigmoid:
			activated = float32(1.0 / (1.0 + math.Exp(-float64(value))))
		case DNNActivationTanh:
			activated = float32(math.Tanh(float64(value)))
		case DNNActivationIdentity:
			activated = value
		case DNNActivationClippedRelu:
			clipped := math.Min(float64(activDesc.coeff), math.Max(0, float64(value)))
			activated = float32(clipped)
		case DNNActivationElu:
			if value >= 0 {
				activated = value
			} else {
				activated = float32(activDesc.coeff * (math.Exp(float64(value)) - 1))
			}
		}
		if beta == 0 {
			outputData[outputIndex] = alpha * activated
		} else {
			outputData[outputIndex] = alpha*activated + beta*outputData[outputIndex]
		}
		position := len(indices) - 1
		for position >= 0 {
			indices[position]++
			if indices[position] < inputDesc.dimensions[position] {
				break
			}
			indices[position] = 0
			position--
		}
		if position < 0 {
			break
		}
	}

	return writeMathFloat32Memory(output, outputData)
}

func tensorFloat32StorageSize(desc *TensorDescriptor) (int, error) {
	if desc == nil || len(desc.dimensions) == 0 || len(desc.dimensions) != len(desc.strides) {
		return 0, fmt.Errorf("tensor dimensions and strides must have the same nonzero length")
	}
	maxElements := uint64(^uint(0)>>1) / 4
	span, count := uint64(1), uint64(1)
	for i, dimension := range desc.dimensions {
		stride := desc.strides[i]
		if dimension <= 0 || stride <= 0 {
			return 0, fmt.Errorf("tensor dimensions and strides must be positive")
		}
		if uint64(dimension-1) > (maxElements-span)/uint64(stride) || uint64(dimension) > maxElements/count {
			return 0, fmt.Errorf("tensor layout exceeds supported size")
		}
		span += uint64(dimension-1) * uint64(stride)
		count *= uint64(dimension)
	}
	return int(span), nil
}

func (h *DNNHandle) performBatchNorm(input, output, scale, bias, mean, variance *memory.Memory, inputDesc, outputDesc, bnScaleBiasDesc *TensorDescriptor, mode DNNBatchNormMode, alpha, beta float32, epsilon float64) error {
	if input == nil || output == nil {
		return fmt.Errorf("input and output cannot be nil")
	}

	if inputDesc.dataType != DNNDataFloat || outputDesc.dataType != DNNDataFloat || bnScaleBiasDesc.dataType != DNNDataFloat {
		return fmt.Errorf("deterministic cuDNN batch norm currently supports DNNDataFloat only")
	}
	if len(inputDesc.dimensions) != 4 || len(outputDesc.dimensions) != 4 || len(bnScaleBiasDesc.dimensions) != 4 {
		return fmt.Errorf("simulation batch normalization requires four-dimensional tensors")
	}
	if !slices.Equal(inputDesc.dimensions, outputDesc.dimensions) {
		return fmt.Errorf("batch normalization input and output dimensions must match")
	}
	expectedParams := []int{1, inputDesc.dimensions[1], 1, 1}
	switch mode {
	case DNNBatchNormSpatial:
	case DNNBatchNormPerActivation:
		expectedParams[2], expectedParams[3] = inputDesc.dimensions[2], inputDesc.dimensions[3]
	default:
		return fmt.Errorf("unsupported batch normalization mode %d", mode)
	}
	if !slices.Equal(bnScaleBiasDesc.dimensions, expectedParams) {
		return fmt.Errorf("batch normalization parameter dimensions do not match the mode")
	}
	inputSize, err := tensorFloat32StorageSize(inputDesc)
	if err != nil {
		return fmt.Errorf("batch normalization input layout: %w", err)
	}
	outputSize, err := tensorFloat32StorageSize(outputDesc)
	if err != nil {
		return fmt.Errorf("batch normalization output layout: %w", err)
	}
	paramSize, err := tensorFloat32StorageSize(bnScaleBiasDesc)
	if err != nil {
		return fmt.Errorf("batch normalization parameter layout: %w", err)
	}
	outputData, err := readMathFloat32Memory(output, outputSize)
	if err != nil {
		return err
	}
	inputData, err := readMathFloat32Memory(input, inputSize)
	if err != nil {
		return err
	}
	scaleData, err := readMathFloat32Memory(scale, paramSize)
	if err != nil {
		return err
	}
	biasData, err := readMathFloat32Memory(bias, paramSize)
	if err != nil {
		return err
	}
	meanData, err := readMathFloat32Memory(mean, paramSize)
	if err != nil {
		return err
	}
	varData, err := readMathFloat32Memory(variance, paramSize)
	if err != nil {
		return err
	}
	channels := inputDesc.dimensions[1]
	for n := 0; n < inputDesc.dimensions[0]; n++ {
		for c := range channels {
			for hIndex := 0; hIndex < inputDesc.dimensions[2]; hIndex++ {
				for wIndex := 0; wIndex < inputDesc.dimensions[3]; wIndex++ {
					inputIndex := n*inputDesc.strides[0] + c*inputDesc.strides[1] + hIndex*inputDesc.strides[2] + wIndex*inputDesc.strides[3]
					outputIndex := n*outputDesc.strides[0] + c*outputDesc.strides[1] + hIndex*outputDesc.strides[2] + wIndex*outputDesc.strides[3]
					paramIndex := c * bnScaleBiasDesc.strides[1]
					if mode == DNNBatchNormPerActivation {
						paramIndex += hIndex*bnScaleBiasDesc.strides[2] + wIndex*bnScaleBiasDesc.strides[3]
					}
					normalized := (inputData[inputIndex] - meanData[paramIndex]) / float32(math.Sqrt(float64(varData[paramIndex])+epsilon))
					scaled := normalized*scaleData[paramIndex] + biasData[paramIndex]
					if beta == 0 {
						outputData[outputIndex] = alpha * scaled
					} else {
						outputData[outputIndex] = alpha*scaled + beta*outputData[outputIndex]
					}
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
