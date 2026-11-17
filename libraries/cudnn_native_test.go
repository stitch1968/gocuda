//go:build cuda

package libraries

import (
	"reflect"
	"testing"

	cuda "github.com/stitch1968/gocuda"
	"github.com/stitch1968/gocuda/memory"
)

func TestCuDNNNativeNDimDescriptors(t *testing.T) {
	if err := cuda.Initialize(); err != nil {
		t.Fatal(err)
	}
	if !cuda.ShouldUseCuda() || !cudnnNativeAvailable() {
		t.Skip("native cuDNN unavailable")
	}
	input, err := CreateTensorDescriptor()
	if err != nil {
		t.Fatal(err)
	}
	defer input.DestroyTensorDescriptor()
	filter, err := CreateFilterDescriptor()
	if err != nil {
		t.Fatal(err)
	}
	defer filter.DestroyFilterDescriptor()
	conv, err := CreateConvolutionDescriptor()
	if err != nil {
		t.Fatal(err)
	}
	defer conv.DestroyConvolutionDescriptor()
	if err := input.SetTensorNdDescriptor(DNNDataFloat, []int{2, 3, 7, 9}, []int{189, 63, 9, 1}); err != nil {
		t.Fatal(err)
	}
	if err := filter.SetFilterNdDescriptor(DNNDataFloat, DNNTensorNCHW, []int{4, 3, 3, 2}); err != nil {
		t.Fatal(err)
	}
	if err := filter.SetFilterNdDescriptor(DNNDataFloat, DNNTensorNCHW, []int{1, 1, 1, 1, 1, 1, 1, 1, 1}); err == nil {
		t.Fatal("expected cuDNN to reject an unsupported filter rank")
	}
	if !reflect.DeepEqual(filter.dimensions, []int{4, 3, 3, 2}) {
		t.Fatalf("failed native filter update changed metadata: %v", filter.dimensions)
	}
	if err := conv.SetConvolution2dDescriptor(1, 0, 2, 2, 1, 1, DNNCrossCorrelation, DNNDataFloat); err != nil {
		t.Fatal(err)
	}
	n, c, h, w, err := conv.GetConvolution2dForwardOutputDim(input, filter)
	if err != nil || n != 2 || c != 4 || h != 4 || w != 4 {
		t.Fatalf("native N-dimensional output shape = %d,%d,%d,%d, error = %v", n, c, h, w, err)
	}
}

func TestCuDNNNativeNDimTensorStrides(t *testing.T) {
	if err := cuda.Initialize(); err != nil {
		t.Fatal(err)
	}
	if !cuda.ShouldUseCuda() || !cudnnNativeAvailable() {
		t.Skip("native cuDNN unavailable")
	}
	handle, err := CreateDNNHandle()
	if err != nil {
		t.Fatal(err)
	}
	defer handle.DestroyHandle()
	tensor, err := CreateTensorDescriptor()
	if err != nil {
		t.Fatal(err)
	}
	defer tensor.DestroyTensorDescriptor()
	activation, err := CreateActivationDescriptor()
	if err != nil {
		t.Fatal(err)
	}
	defer activation.DestroyActivationDescriptor()
	if err := tensor.SetTensorNdDescriptor(DNNDataFloat, []int{1, 1, 2, 2}, []int{8, 8, 4, 1}); err != nil {
		t.Fatal(err)
	}
	if err := tensor.SetTensorNdDescriptor(DNNDataFloat, []int{2}, []int{1}); err == nil {
		t.Fatal("expected cuDNN to reject an unsupported tensor rank")
	}
	if !reflect.DeepEqual(tensor.dimensions, []int{1, 1, 2, 2}) || !reflect.DeepEqual(tensor.strides, []int{8, 8, 4, 1}) {
		t.Fatalf("failed native tensor update changed metadata: %v/%v", tensor.dimensions, tensor.strides)
	}
	if err := activation.SetActivationDescriptor(DNNActivationRelu, DNNPropagateNaN, 0); err != nil {
		t.Fatal(err)
	}
	input, err := memory.Alloc(32)
	if err != nil {
		t.Fatal(err)
	}
	defer input.Free()
	output, err := memory.Alloc(32)
	if err != nil {
		t.Fatal(err)
	}
	defer output.Free()
	if err := writeMathFloat32Memory(input, []float32{1, -2, 99, 99, -3, 4, 99, 99}); err != nil {
		t.Fatal(err)
	}
	if err := writeMathFloat32Memory(output, []float32{-7, -7, -7, -7, -7, -7, -7, -7}); err != nil {
		t.Fatal(err)
	}
	if err := handle.ActivationForward(activation, 1, tensor, input, 0, tensor, output); err != nil {
		t.Fatal(err)
	}
	got, err := readMathFloat32Memory(output, 8)
	want := []float32{1, 0, -7, -7, 0, 4, -7, -7}
	if err != nil || !reflect.DeepEqual(got, want) {
		t.Fatalf("native strided activation = %v, error = %v, want %v", got, err, want)
	}
}

func TestCuDNNNativeFiveDimensionalActivation(t *testing.T) {
	if err := cuda.Initialize(); err != nil {
		t.Fatal(err)
	}
	if !cuda.ShouldUseCuda() || !cudnnNativeAvailable() {
		t.Skip("native cuDNN unavailable")
	}
	handle, err := CreateDNNHandle()
	if err != nil {
		t.Fatal(err)
	}
	defer handle.DestroyHandle()
	inputDesc, err := CreateTensorDescriptor()
	if err != nil {
		t.Fatal(err)
	}
	defer inputDesc.DestroyTensorDescriptor()
	outputDesc, err := CreateTensorDescriptor()
	if err != nil {
		t.Fatal(err)
	}
	defer outputDesc.DestroyTensorDescriptor()
	activation, err := CreateActivationDescriptor()
	if err != nil {
		t.Fatal(err)
	}
	defer activation.DestroyActivationDescriptor()
	dims := []int{1, 1, 2, 2, 2}
	if err := inputDesc.SetTensorNdDescriptor(DNNDataFloat, dims, []int{20, 20, 10, 3, 1}); err != nil {
		t.Fatal(err)
	}
	if err := outputDesc.SetTensorNdDescriptor(DNNDataFloat, dims, []int{20, 20, 10, 4, 1}); err != nil {
		t.Fatal(err)
	}
	if err := activation.SetActivationDescriptor(DNNActivationRelu, DNNPropagateNaN, 0); err != nil {
		t.Fatal(err)
	}
	inputValues := make([]float32, 20)
	for i, index := range []int{0, 1, 3, 4, 10, 11, 13, 14} {
		inputValues[index] = []float32{-1, 2, -3, 4, -5, 6, -7, 8}[i]
	}
	outputValues := make([]float32, 20)
	want := make([]float32, 20)
	for i := range outputValues {
		outputValues[i], want[i] = -7, -7
	}
	for i, index := range []int{0, 1, 4, 5, 10, 11, 14, 15} {
		want[index] = []float32{7, 11, 7, 15, 7, 19, 7, 23}[i]
	}
	input := solverTestBuffer(t, inputValues)
	output := solverTestBuffer(t, outputValues)
	if err := handle.ActivationForward(activation, 2, inputDesc, input, -1, outputDesc, output); err != nil {
		t.Fatal(err)
	}
	got, err := readMathFloat32Memory(output, len(want))
	if err != nil || !reflect.DeepEqual(got, want) {
		t.Fatalf("native 5D activation = %v, error = %v, want %v", got, err, want)
	}
}

func TestCuDNNNativeBackendSelection(t *testing.T) {
	if err := cuda.Initialize(); err != nil {
		t.Fatalf("Initialize failed: %v", err)
	}
	if !cuda.ShouldUseCuda() {
		t.Skip("CUDA runtime not available")
	}
	if !cudnnNativeAvailable() {
		t.Skip("cuDNN native backend not available")
	}

	handle, err := CreateDNNHandle()
	if err != nil {
		t.Fatalf("CreateDNNHandle failed: %v", err)
	}
	defer handle.DestroyHandle()

	if !handle.native {
		t.Fatal("expected native cuDNN handle in CUDA build")
	}

	inputDesc, err := CreateTensorDescriptor()
	if err != nil {
		t.Fatalf("CreateTensorDescriptor failed: %v", err)
	}
	defer inputDesc.DestroyTensorDescriptor()
	if !inputDesc.native {
		t.Fatal("expected native tensor descriptor")
	}

	filterDesc, err := CreateFilterDescriptor()
	if err != nil {
		t.Fatalf("CreateFilterDescriptor failed: %v", err)
	}
	defer filterDesc.DestroyFilterDescriptor()
	if !filterDesc.native {
		t.Fatal("expected native filter descriptor")
	}

	convDesc, err := CreateConvolutionDescriptor()
	if err != nil {
		t.Fatalf("CreateConvolutionDescriptor failed: %v", err)
	}
	defer convDesc.DestroyConvolutionDescriptor()
	if !convDesc.native {
		t.Fatal("expected native convolution descriptor")
	}

	poolDesc, err := CreatePoolingDescriptor()
	if err != nil {
		t.Fatalf("CreatePoolingDescriptor failed: %v", err)
	}
	defer poolDesc.DestroyPoolingDescriptor()
	if !poolDesc.native {
		t.Fatal("expected native pooling descriptor")
	}

	activationDesc, err := CreateActivationDescriptor()
	if err != nil {
		t.Fatalf("CreateActivationDescriptor failed: %v", err)
	}
	defer activationDesc.DestroyActivationDescriptor()
	if !activationDesc.native {
		t.Fatal("expected native activation descriptor")
	}
}

func TestCuDNNNativeConvolutionForward(t *testing.T) {
	if err := cuda.Initialize(); err != nil {
		t.Fatalf("Initialize failed: %v", err)
	}
	if !cuda.ShouldUseCuda() {
		t.Skip("CUDA runtime not available")
	}
	if !cudnnNativeAvailable() {
		t.Skip("cuDNN native backend not available")
	}

	handle, err := CreateDNNHandle()
	if err != nil {
		t.Fatalf("CreateDNNHandle failed: %v", err)
	}
	defer handle.DestroyHandle()

	inputDesc, err := CreateTensorDescriptor()
	if err != nil {
		t.Fatalf("CreateTensorDescriptor failed: %v", err)
	}
	defer inputDesc.DestroyTensorDescriptor()

	filterDesc, err := CreateFilterDescriptor()
	if err != nil {
		t.Fatalf("CreateFilterDescriptor failed: %v", err)
	}
	defer filterDesc.DestroyFilterDescriptor()

	convDesc, err := CreateConvolutionDescriptor()
	if err != nil {
		t.Fatalf("CreateConvolutionDescriptor failed: %v", err)
	}
	defer convDesc.DestroyConvolutionDescriptor()

	outputDesc, err := CreateTensorDescriptor()
	if err != nil {
		t.Fatalf("CreateTensorDescriptor output failed: %v", err)
	}
	defer outputDesc.DestroyTensorDescriptor()

	if err := inputDesc.SetTensor4dDescriptor(DNNTensorNCHW, DNNDataFloat, 1, 1, 3, 3); err != nil {
		t.Fatalf("SetTensor4dDescriptor input failed: %v", err)
	}
	if err := filterDesc.SetFilter4dDescriptor(DNNDataFloat, DNNTensorNCHW, 1, 1, 2, 2); err != nil {
		t.Fatalf("SetFilter4dDescriptor failed: %v", err)
	}
	if err := convDesc.SetConvolution2dDescriptor(0, 0, 1, 1, 1, 1, DNNCrossCorrelation, DNNDataFloat); err != nil {
		t.Fatalf("SetConvolution2dDescriptor failed: %v", err)
	}
	outN, outC, outH, outW, err := convDesc.GetConvolution2dForwardOutputDim(inputDesc, filterDesc)
	if err != nil {
		t.Fatalf("GetConvolution2dForwardOutputDim failed: %v", err)
	}
	if err := outputDesc.SetTensor4dDescriptor(DNNTensorNCHW, DNNDataFloat, outN, outC, outH, outW); err != nil {
		t.Fatalf("SetTensor4dDescriptor output failed: %v", err)
	}

	input, err := memory.Alloc(9 * 4)
	if err != nil {
		t.Fatalf("Alloc input failed: %v", err)
	}
	defer input.Free()
	filter, err := memory.Alloc(4 * 4)
	if err != nil {
		t.Fatalf("Alloc filter failed: %v", err)
	}
	defer filter.Free()
	output, err := memory.Alloc(4 * 4)
	if err != nil {
		t.Fatalf("Alloc output failed: %v", err)
	}
	defer output.Free()

	if err := writeMathFloat32Memory(input, []float32{1, 2, 3, 4, 5, 6, 7, 8, 9}); err != nil {
		t.Fatalf("write input failed: %v", err)
	}
	if err := writeMathFloat32Memory(filter, []float32{1, 0, 0, 1}); err != nil {
		t.Fatalf("write filter failed: %v", err)
	}
	if err := writeMathFloat32Memory(output, []float32{0, 0, 0, 0}); err != nil {
		t.Fatalf("write output failed: %v", err)
	}

	if err := handle.ConvolutionForward(1, inputDesc, input, filterDesc, filter, convDesc, 0, outputDesc, output); err != nil {
		t.Fatalf("ConvolutionForward failed: %v", err)
	}

	values, err := readMathFloat32Memory(output, 4)
	if err != nil {
		t.Fatalf("read output failed: %v", err)
	}
	assertFloat32Slice(t, values, []float32{6, 8, 12, 14}, 1e-5)
}
