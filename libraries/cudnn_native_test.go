//go:build cuda

package libraries

import (
	"testing"

	cuda "github.com/stitch1968/gocuda"
	"github.com/stitch1968/gocuda/memory"
)

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
