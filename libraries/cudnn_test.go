package libraries

import (
	"math"
	"testing"

	"github.com/stitch1968/gocuda/memory"
)

func TestCuDNNConvolutionPoolingActivationAndBatchNorm(t *testing.T) {
	handle, err := CreateDNNHandle()
	if err != nil {
		t.Fatalf("CreateDNNHandle failed: %v", err)
	}
	defer handle.DestroyHandle()

	inputDesc, _ := CreateTensorDescriptor()
	defer inputDesc.DestroyTensorDescriptor()
	filterDesc, _ := CreateFilterDescriptor()
	defer filterDesc.DestroyFilterDescriptor()
	convDesc, _ := CreateConvolutionDescriptor()
	defer convDesc.DestroyConvolutionDescriptor()
	outputDesc, _ := CreateTensorDescriptor()
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

	input, _ := memory.Alloc(9 * 4)
	filter, _ := memory.Alloc(4 * 4)
	convOut, _ := memory.Alloc(4 * 4)
	defer input.Free()
	defer filter.Free()
	defer convOut.Free()
	writeMathFloat32Memory(input, []float32{1, 2, 3, 4, 5, 6, 7, 8, 9})
	writeMathFloat32Memory(filter, []float32{1, 0, 0, 1})
	writeMathFloat32Memory(convOut, []float32{0, 0, 0, 0})

	if err := handle.ConvolutionForward(1, inputDesc, input, filterDesc, filter, convDesc, 0, outputDesc, convOut); err != nil {
		t.Fatalf("ConvolutionForward failed: %v", err)
	}
	convValues, _ := readMathFloat32Memory(convOut, 4)
	assertFloat32Slice(t, convValues, []float32{6, 8, 12, 14}, 1e-5)

	poolDesc, _ := CreatePoolingDescriptor()
	defer poolDesc.DestroyPoolingDescriptor()
	poolOutDesc, _ := CreateTensorDescriptor()
	defer poolOutDesc.DestroyTensorDescriptor()
	if err := poolDesc.SetPooling2dDescriptor(DNNPoolingMax, 2, 2, 0, 0, 1, 1); err != nil {
		t.Fatalf("SetPooling2dDescriptor failed: %v", err)
	}
	if err := poolOutDesc.SetTensor4dDescriptor(DNNTensorNCHW, DNNDataFloat, 1, 1, 2, 2); err != nil {
		t.Fatalf("SetTensor4dDescriptor pool output failed: %v", err)
	}
	poolInput, _ := memory.Alloc(9 * 4)
	poolOutput, _ := memory.Alloc(4 * 4)
	defer poolInput.Free()
	defer poolOutput.Free()
	writeMathFloat32Memory(poolInput, []float32{1, 3, 2, 4, 6, 5, 7, 9, 8})
	writeMathFloat32Memory(poolOutput, []float32{0, 0, 0, 0})
	if err := handle.PoolingForward(poolDesc, 1, inputDesc, poolInput, 0, poolOutDesc, poolOutput); err != nil {
		t.Fatalf("PoolingForward failed: %v", err)
	}
	poolValues, _ := readMathFloat32Memory(poolOutput, 4)
	assertFloat32Slice(t, poolValues, []float32{6, 6, 9, 9}, 1e-5)

	activationDesc, _ := CreateActivationDescriptor()
	defer activationDesc.DestroyActivationDescriptor()
	if err := activationDesc.SetActivationDescriptor(DNNActivationRelu, DNNPropagateNaN, 0); err != nil {
		t.Fatalf("SetActivationDescriptor failed: %v", err)
	}
	activationInput, _ := memory.Alloc(4 * 4)
	activationOutput, _ := memory.Alloc(4 * 4)
	defer activationInput.Free()
	defer activationOutput.Free()
	writeMathFloat32Memory(activationInput, []float32{-2, 1.5, -0.5, 4})
	writeMathFloat32Memory(activationOutput, []float32{0, 0, 0, 0})
	actDesc, _ := CreateTensorDescriptor()
	defer actDesc.DestroyTensorDescriptor()
	if err := actDesc.SetTensor4dDescriptor(DNNTensorNCHW, DNNDataFloat, 1, 1, 2, 2); err != nil {
		t.Fatalf("SetTensor4dDescriptor act failed: %v", err)
	}
	if err := handle.ActivationForward(activationDesc, 1, actDesc, activationInput, 0, actDesc, activationOutput); err != nil {
		t.Fatalf("ActivationForward failed: %v", err)
	}
	activationValues, _ := readMathFloat32Memory(activationOutput, 4)
	assertFloat32Slice(t, activationValues, []float32{0, 1.5, 0, 4}, 1e-5)

	bnInputDesc, _ := CreateTensorDescriptor()
	defer bnInputDesc.DestroyTensorDescriptor()
	bnParamDesc, _ := CreateTensorDescriptor()
	defer bnParamDesc.DestroyTensorDescriptor()
	if err := bnInputDesc.SetTensor4dDescriptor(DNNTensorNCHW, DNNDataFloat, 1, 2, 1, 1); err != nil {
		t.Fatalf("SetTensor4dDescriptor bn input failed: %v", err)
	}
	if err := bnParamDesc.SetTensor4dDescriptor(DNNTensorNCHW, DNNDataFloat, 1, 2, 1, 1); err != nil {
		t.Fatalf("SetTensor4dDescriptor bn params failed: %v", err)
	}
	bnInput, _ := memory.Alloc(2 * 4)
	bnOutput, _ := memory.Alloc(2 * 4)
	bnScale, _ := memory.Alloc(2 * 4)
	bnBias, _ := memory.Alloc(2 * 4)
	bnMean, _ := memory.Alloc(2 * 4)
	bnVariance, _ := memory.Alloc(2 * 4)
	defer bnInput.Free()
	defer bnOutput.Free()
	defer bnScale.Free()
	defer bnBias.Free()
	defer bnMean.Free()
	defer bnVariance.Free()
	writeMathFloat32Memory(bnInput, []float32{3, 10})
	writeMathFloat32Memory(bnOutput, []float32{0, 0})
	writeMathFloat32Memory(bnScale, []float32{2, 0.5})
	writeMathFloat32Memory(bnBias, []float32{1, -1})
	writeMathFloat32Memory(bnMean, []float32{1, 6})
	writeMathFloat32Memory(bnVariance, []float32{4, 16})
	if err := handle.BatchNormalizationForwardInference(DNNBatchNormSpatial, 1, 0, bnInputDesc, bnInput, bnInputDesc, bnOutput, bnParamDesc, bnScale, bnBias, bnMean, bnVariance, 0); err != nil {
		t.Fatalf("BatchNormalizationForwardInference failed: %v", err)
	}
	bnValues, _ := readMathFloat32Memory(bnOutput, 2)
	assertFloat32Slice(t, bnValues, []float32{3, -0.5}, 1e-5)
}

func assertFloat32Slice(t *testing.T, got, want []float32, tolerance float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("unexpected length: got %d want %d", len(got), len(want))
	}
	for index := range got {
		if math.Abs(float64(got[index]-want[index])) > tolerance {
			t.Fatalf("unexpected value[%d]: got %v want %v", index, got[index], want[index])
		}
	}
}
