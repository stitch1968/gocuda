package libraries

import (
	"math"
	"reflect"
	"strconv"
	"strings"
	"testing"

	"github.com/stitch1968/gocuda/memory"
)

func TestCuDNNNDimDescriptorValidation(t *testing.T) {
	tensor, err := CreateTensorDescriptor()
	if err != nil {
		t.Fatal(err)
	}
	defer tensor.DestroyTensorDescriptor()
	filter, err := CreateFilterDescriptor()
	if err != nil {
		t.Fatal(err)
	}
	defer filter.DestroyFilterDescriptor()
	dimensions := []int{1, 2, 3, 4}
	strides := []int{24, 12, 4, 1}
	filterDimensions := []int{2, 2, 3, 4}
	if err := tensor.SetTensorNdDescriptor(DNNDataFloat, dimensions, strides); err != nil {
		t.Fatal(err)
	}
	if err := filter.SetFilterNdDescriptor(DNNDataFloat, DNNTensorNCHW, filterDimensions); err != nil {
		t.Fatal(err)
	}
	dimensions[0], strides[0], filterDimensions[0] = 9, 9, 9
	wantDims, wantStrides, wantFilter := []int{1, 2, 3, 4}, []int{24, 12, 4, 1}, []int{2, 2, 3, 4}
	for _, tc := range []struct {
		name string
		run  func() error
	}{
		{"empty tensor", func() error { return tensor.SetTensorNdDescriptor(DNNDataFloat, nil, nil) }},
		{"mismatched tensor", func() error { return tensor.SetTensorNdDescriptor(DNNDataFloat, []int{1}, nil) }},
		{"invalid tensor data type", func() error { return tensor.SetTensorNdDescriptor(DNNDataType(-1), wantDims, wantStrides) }},
		{"zero tensor dimension", func() error { return tensor.SetTensorNdDescriptor(DNNDataFloat, []int{1, 0, 3, 4}, wantStrides) }},
		{"negative tensor dimension", func() error { return tensor.SetTensorNdDescriptor(DNNDataFloat, []int{1, -2, 3, 4}, wantStrides) }},
		{"empty filter", func() error { return filter.SetFilterNdDescriptor(DNNDataFloat, DNNTensorNCHW, nil) }},
		{"invalid filter format", func() error { return filter.SetFilterNdDescriptor(DNNDataFloat, DNNTensorFormat(-1), wantFilter) }},
		{"zero filter dimension", func() error { return filter.SetFilterNdDescriptor(DNNDataFloat, DNNTensorNCHW, []int{2, 0, 3, 4}) }},
		{"negative filter dimension", func() error { return filter.SetFilterNdDescriptor(DNNDataFloat, DNNTensorNCHW, []int{2, -2, 3, 4}) }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if err := tc.run(); err == nil {
				t.Fatal("expected invalid descriptor to fail")
			}
		})
	}
	if strconv.IntSize > 32 {
		tooLarge := int64(math.MaxInt32) + 1
		if err := tensor.SetTensorNdDescriptor(DNNDataFloat, []int{1, 2, 3, int(tooLarge)}, wantStrides); err == nil {
			t.Fatal("expected out-of-range tensor dimension to fail")
		}
		if err := tensor.SetTensorNdDescriptor(DNNDataFloat, wantDims, []int{int(tooLarge), 12, 4, 1}); err == nil {
			t.Fatal("expected out-of-range tensor stride to fail")
		}
		if err := filter.SetFilterNdDescriptor(DNNDataFloat, DNNTensorNCHW, []int{2, 2, 3, int(tooLarge)}); err == nil {
			t.Fatal("expected out-of-range filter dimension to fail")
		}
	}
	if !reflect.DeepEqual(tensor.dimensions, wantDims) || !reflect.DeepEqual(tensor.strides, wantStrides) ||
		!reflect.DeepEqual(filter.dimensions, wantFilter) {
		t.Fatalf("failed descriptor update changed state: tensor %v/%v, filter %v", tensor.dimensions, tensor.strides, filter.dimensions)
	}
}

func TestCuDNN4DDescriptorValidation(t *testing.T) {
	tensor, err := CreateTensorDescriptor()
	if err != nil {
		t.Fatal(err)
	}
	defer tensor.DestroyTensorDescriptor()
	filter, err := CreateFilterDescriptor()
	if err != nil {
		t.Fatal(err)
	}
	defer filter.DestroyFilterDescriptor()
	if err := tensor.SetTensor4dDescriptor(DNNTensorNCHW, DNNDataFloat, 1, 2, 3, 4); err != nil {
		t.Fatal(err)
	}
	if err := filter.SetFilter4dDescriptor(DNNDataFloat, DNNTensorNCHW, 3, 2, 2, 2); err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct {
		name string
		run  func() error
	}{
		{"zero tensor dimension", func() error { return tensor.SetTensor4dDescriptor(DNNTensorNCHW, DNNDataFloat, 1, 0, 3, 4) }},
		{"negative tensor dimension", func() error { return tensor.SetTensor4dDescriptor(DNNTensorNCHW, DNNDataFloat, 1, 2, -3, 4) }},
		{"tensor size overflow", func() error { return tensor.SetTensor4dDescriptor(DNNTensorNCHW, DNNDataFloat, 2, 1, 1, math.MaxInt32) }},
		{"invalid tensor format", func() error { return tensor.SetTensor4dDescriptor(DNNTensorFormat(-1), DNNDataFloat, 1, 2, 3, 4) }},
		{"invalid tensor data type", func() error { return tensor.SetTensor4dDescriptor(DNNTensorNCHW, DNNDataType(-1), 1, 2, 3, 4) }},
		{"zero filter dimension", func() error {
			return filter.SetFilter4dDescriptor(DNNDataFloat, DNNTensorNCHW, 3, 2, 0, 2)
		}},
		{"negative filter dimension", func() error { return filter.SetFilter4dDescriptor(DNNDataFloat, DNNTensorNCHW, 3, 2, -1, 2) }},
		{"invalid filter format", func() error { return filter.SetFilter4dDescriptor(DNNDataFloat, DNNTensorFormat(-1), 3, 2, 2, 2) }},
		{"invalid filter data type", func() error { return filter.SetFilter4dDescriptor(DNNDataType(-1), DNNTensorNCHW, 3, 2, 2, 2) }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if err := tc.run(); err == nil {
				t.Fatal("expected invalid 4D descriptor to fail")
			}
		})
	}
	if strconv.IntSize > 32 {
		tooLarge := int64(math.MaxInt32) + 1
		if err := tensor.SetTensor4dDescriptor(DNNTensorNCHW, DNNDataFloat, 1, 1, 1, int(tooLarge)); err == nil {
			t.Fatal("expected out-of-range tensor dimension to fail")
		}
		if err := filter.SetFilter4dDescriptor(DNNDataFloat, DNNTensorNCHW, 1, 1, 1, int(tooLarge)); err == nil {
			t.Fatal("expected out-of-range filter dimension to fail")
		}
	}
	if tensor.native {
		if err := tensor.SetTensor4dDescriptor(DNNTensorNHWCVectC, DNNDataFloat, 1, 2, 3, 4); err == nil {
			t.Fatal("expected unsupported native tensor format to fail")
		}
	}
	if filter.native {
		if err := filter.SetFilter4dDescriptor(DNNDataFloat, DNNTensorNHWCVectC, 3, 2, 2, 2); err == nil {
			t.Fatal("expected unsupported native filter format to fail")
		}
	}
	if tensor.format != DNNTensorNCHW || tensor.dataType != DNNDataFloat ||
		!reflect.DeepEqual(tensor.dimensions, []int{1, 2, 3, 4}) || !reflect.DeepEqual(tensor.strides, []int{24, 12, 4, 1}) ||
		filter.format != DNNTensorNCHW || filter.dataType != DNNDataFloat || !reflect.DeepEqual(filter.dimensions, []int{3, 2, 2, 2}) {
		t.Fatalf("failed 4D update changed descriptor state: tensor %v/%v, filter %v", tensor.dimensions, tensor.strides, filter.dimensions)
	}
}

func TestCuDNNSimulationNDimActivationStrides(t *testing.T) {
	handle := &DNNHandle{}
	inputDesc := &TensorDescriptor{}
	outputDesc := &TensorDescriptor{}
	activation := &ActivationDescriptor{}
	dims := []int{1, 1, 2, 2}
	if err := inputDesc.SetTensorNdDescriptor(DNNDataFloat, dims, []int{8, 8, 4, 1}); err != nil {
		t.Fatal(err)
	}
	if err := outputDesc.SetTensorNdDescriptor(DNNDataFloat, dims, []int{9, 9, 5, 1}); err != nil {
		t.Fatal(err)
	}
	if err := activation.SetActivationDescriptor(DNNActivationRelu, DNNPropagateNaN, 0); err != nil {
		t.Fatal(err)
	}
	input := solverTestBuffer(t, []float32{1, -2, 99, 99, -3, 4, 99, 99})
	output := solverTestBuffer(t, []float32{float32(math.NaN()), float32(math.NaN()), -7, -7, -7, float32(math.NaN()), float32(math.NaN()), -7, -7})
	if err := handle.ActivationForward(activation, 1, inputDesc, input, 0, outputDesc, output); err != nil {
		t.Fatal(err)
	}
	got, err := readMathFloat32Memory(output, 9)
	want := []float32{1, 0, -7, -7, -7, 0, 4, -7, -7}
	if err != nil || !reflect.DeepEqual(got, want) {
		t.Fatalf("simulation strided activation = %v, error = %v, want %v", got, err, want)
	}
	if err := handle.ActivationForward(activation, 1, inputDesc, input, 0, inputDesc, input); err != nil {
		t.Fatal(err)
	}
	got, err = readMathFloat32Memory(input, 8)
	want = []float32{1, 0, 99, 99, 0, 4, 99, 99}
	if err != nil || !reflect.DeepEqual(got, want) {
		t.Fatalf("simulation in-place activation = %v, error = %v, want %v", got, err, want)
	}
}

func TestCuDNNSimulationFiveDimensionalActivation(t *testing.T) {
	handle := &DNNHandle{}
	inputDesc := &TensorDescriptor{}
	outputDesc := &TensorDescriptor{}
	activation := &ActivationDescriptor{}
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
		t.Fatalf("simulation 5D activation = %v, error = %v, want %v", got, err, want)
	}
}

func TestCuDNNSimulationNDimActivationRejectsInvalidLayout(t *testing.T) {
	handle := &DNNHandle{}
	inputDesc := &TensorDescriptor{}
	outputDesc := &TensorDescriptor{}
	activation := &ActivationDescriptor{}
	if err := inputDesc.SetTensorNdDescriptor(DNNDataFloat, []int{1, 1, 2, 2}, []int{8, 8, 4, 1}); err != nil {
		t.Fatal(err)
	}
	if err := outputDesc.SetTensorNdDescriptor(DNNDataFloat, []int{1, 1, 2, 1}, []int{2, 2, 1, 1}); err != nil {
		t.Fatal(err)
	}
	input := solverTestBuffer(t, []float32{1, 2, 0, 0, 3, 4})
	output := solverTestBuffer(t, []float32{7, 8, 9, 10, 11, 12})
	if err := handle.ActivationForward(activation, 1, inputDesc, input, 0, outputDesc, output); err == nil {
		t.Fatal("expected mismatched tensor shapes to fail")
	}
	if err := outputDesc.SetTensorNdDescriptor(DNNDataFloat, []int{1, 1, 2, 2}, []int{8, 8, -4, 1}); err != nil {
		t.Fatal(err)
	}
	if err := handle.ActivationForward(activation, 1, inputDesc, input, 0, outputDesc, output); err == nil {
		t.Fatal("expected unsupported negative strides to fail")
	}
	if err := outputDesc.SetTensorNdDescriptor(DNNDataFloat, []int{1, 1, 2, 2}, []int{8, 8, 4, 1}); err != nil {
		t.Fatal(err)
	}
	short := solverTestBuffer(t, []float32{1, 2, 3, 4, 5})
	if err := handle.ActivationForward(activation, 1, inputDesc, short, 0, outputDesc, output); err == nil {
		t.Fatal("expected insufficient input storage to fail")
	}
	if err := inputDesc.SetTensorNdDescriptor(DNNDataFloat, []int{1, 1, math.MaxInt32, 2}, []int{1, 1, math.MaxInt32, 1}); err != nil {
		t.Fatal(err)
	}
	if err := outputDesc.SetTensorNdDescriptor(DNNDataFloat, []int{1, 1, math.MaxInt32, 2}, []int{1, 1, math.MaxInt32, 1}); err != nil {
		t.Fatal(err)
	}
	if err := handle.ActivationForward(activation, 1, inputDesc, input, 0, outputDesc, output); err == nil {
		t.Fatal("expected overflowing tensor layout to fail")
	}
	got, err := readMathFloat32Memory(output, 6)
	if err != nil || !reflect.DeepEqual(got, []float32{7, 8, 9, 10, 11, 12}) {
		t.Fatalf("invalid activation changed output: %v, error = %v", got, err)
	}
}

func TestCuDNNSimulationActivationRejectsUnsupportedModes(t *testing.T) {
	handle := &DNNHandle{}
	desc := &TensorDescriptor{}
	activation := &ActivationDescriptor{}
	if err := desc.SetTensor4dDescriptor(DNNTensorNCHW, DNNDataFloat, 1, 1, 1, 2); err != nil {
		t.Fatal(err)
	}
	input := solverTestBuffer(t, []float32{-1, 2})
	output := solverTestBuffer(t, []float32{7, 8})
	for _, mode := range []DNNActivationMode{DNNActivationSwish, DNNActivationMode(-1)} {
		if err := activation.SetActivationDescriptor(mode, DNNPropagateNaN, 0); err != nil {
			t.Fatal(err)
		}
		if err := handle.ActivationForward(activation, 1, desc, input, 0, desc, output); err == nil {
			t.Fatalf("expected unsupported activation mode %d to fail", mode)
		}
	}
	got, err := readMathFloat32Memory(output, 2)
	if err != nil || !reflect.DeepEqual(got, []float32{7, 8}) {
		t.Fatalf("unsupported activation changed output: %v, error = %v", got, err)
	}
}

func TestCuDNNConvolutionDescriptorValidation(t *testing.T) {
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
	if err := input.SetTensor4dDescriptor(DNNTensorNCHW, DNNDataFloat, 1, 1, 5, 5); err != nil {
		t.Fatal(err)
	}
	if err := filter.SetFilter4dDescriptor(DNNDataFloat, DNNTensorNCHW, 1, 1, 3, 3); err != nil {
		t.Fatal(err)
	}
	if err := conv.SetConvolution2dDescriptor(1, 1, 2, 2, 1, 1, DNNCrossCorrelation, DNNDataFloat); err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct {
		name string
		run  func() error
	}{
		{"negative padding", func() error {
			return conv.SetConvolution2dDescriptor(-1, 1, 2, 2, 1, 1, DNNCrossCorrelation, DNNDataFloat)
		}},
		{"zero stride", func() error {
			return conv.SetConvolution2dDescriptor(1, 1, 0, 2, 1, 1, DNNCrossCorrelation, DNNDataFloat)
		}},
		{"negative stride", func() error {
			return conv.SetConvolution2dDescriptor(1, 1, 2, -1, 1, 1, DNNCrossCorrelation, DNNDataFloat)
		}},
		{"zero dilation", func() error {
			return conv.SetConvolution2dDescriptor(1, 1, 2, 2, 0, 1, DNNCrossCorrelation, DNNDataFloat)
		}},
		{"invalid mode", func() error {
			return conv.SetConvolution2dDescriptor(1, 1, 2, 2, 1, 1, DNNConvolutionMode(-1), DNNDataFloat)
		}},
		{"invalid data type", func() error {
			return conv.SetConvolution2dDescriptor(1, 1, 2, 2, 1, 1, DNNCrossCorrelation, DNNDataType(-1))
		}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if err := tc.run(); err == nil {
				t.Fatal("expected invalid convolution descriptor to fail")
			}
		})
	}
	if strconv.IntSize > 32 {
		tooLarge := int64(math.MaxInt32) + 1
		if err := conv.SetConvolution2dDescriptor(int(tooLarge), 1, 2, 2, 1, 1, DNNCrossCorrelation, DNNDataFloat); err == nil {
			t.Fatal("expected out-of-range padding to fail")
		}
	}
	n, c, h, w, err := conv.GetConvolution2dForwardOutputDim(input, filter)
	if err != nil || n != 1 || c != 1 || h != 3 || w != 3 {
		t.Fatalf("failed updates changed convolution shape: %d,%d,%d,%d, error = %v", n, c, h, w, err)
	}
	if err := input.SetTensor4dDescriptor(DNNTensorNCHW, DNNDataFloat, 1, 2, 5, 5); err != nil {
		t.Fatal(err)
	}
	if _, _, _, _, err := conv.GetConvolution2dForwardOutputDim(input, filter); err == nil {
		t.Fatal("expected mismatched channel counts to fail")
	}
	if err := input.SetTensor4dDescriptor(DNNTensorNCHW, DNNDataFloat, 1, 1, 2, 2); err != nil {
		t.Fatal(err)
	}
	if err := filter.SetFilter4dDescriptor(DNNDataFloat, DNNTensorNCHW, 1, 1, 4, 4); err != nil {
		t.Fatal(err)
	}
	if err := conv.SetConvolution2dDescriptor(0, 0, 2, 2, 1, 1, DNNCrossCorrelation, DNNDataFloat); err != nil {
		t.Fatal(err)
	}
	if _, _, _, _, err := conv.GetConvolution2dForwardOutputDim(input, filter); err == nil {
		t.Fatal("expected filter larger than padded input to fail")
	}
}

func TestCuDNNSimulationConvolutionOutputValidation(t *testing.T) {
	input := &TensorDescriptor{dimensions: []int{1, 1, 1, 1}}
	filter := &FilterDescriptor{dimensions: []int{1, 1, 1, 1}}
	unconfigured := &ConvolutionDescriptor{dilationH: 1, dilationW: 1}
	if _, _, _, _, err := unconfigured.GetConvolution2dForwardOutputDim(input, filter); err == nil {
		t.Fatal("expected unconfigured convolution to fail")
	}
	overflow := &ConvolutionDescriptor{padH: math.MaxInt32, strideH: 1, strideW: 1, dilationH: 1, dilationW: 1}
	if _, _, _, _, err := overflow.GetConvolution2dForwardOutputDim(input, filter); err == nil {
		t.Fatal("expected overflowing convolution output dimension to fail")
	}
}

func TestCuDNNConvolutionModes(t *testing.T) {
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
	filterDesc, err := CreateFilterDescriptor()
	if err != nil {
		t.Fatal(err)
	}
	defer filterDesc.DestroyFilterDescriptor()
	conv, err := CreateConvolutionDescriptor()
	if err != nil {
		t.Fatal(err)
	}
	defer conv.DestroyConvolutionDescriptor()
	if err := inputDesc.SetTensor4dDescriptor(DNNTensorNCHW, DNNDataFloat, 1, 1, 3, 3); err != nil {
		t.Fatal(err)
	}
	if err := outputDesc.SetTensor4dDescriptor(DNNTensorNCHW, DNNDataFloat, 1, 1, 2, 2); err != nil {
		t.Fatal(err)
	}
	if err := filterDesc.SetFilter4dDescriptor(DNNDataFloat, DNNTensorNCHW, 1, 1, 2, 2); err != nil {
		t.Fatal(err)
	}
	input := solverTestBuffer(t, []float32{1, 2, 3, 4, 5, 6, 7, 8, 9})
	filter := solverTestBuffer(t, []float32{1, 2, 3, 4})
	for _, tc := range []struct {
		name string
		mode DNNConvolutionMode
		want []float32
	}{
		{"cross-correlation", DNNCrossCorrelation, []float32{37, 47, 67, 77}},
		{"convolution", DNNConvolution, []float32{23, 33, 53, 63}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			output := solverTestBuffer(t, []float32{float32(math.NaN()), float32(math.NaN()), float32(math.NaN()), float32(math.NaN())})
			if err := conv.SetConvolution2dDescriptor(0, 0, 1, 1, 1, 1, tc.mode, DNNDataFloat); err != nil {
				t.Fatal(err)
			}
			if err := handle.ConvolutionForward(1, inputDesc, input, filterDesc, filter, conv, 0, outputDesc, output); err != nil {
				t.Fatal(err)
			}
			got, err := readMathFloat32Memory(output, 4)
			if err != nil || !reflect.DeepEqual(got, tc.want) {
				t.Fatalf("convolution output = %v, error = %v, want %v", got, err, tc.want)
			}
		})
	}
}

func TestCuDNNConvolutionRejectsInvalidOutput(t *testing.T) {
	handle := &DNNHandle{}
	inputDesc := &TensorDescriptor{}
	outputDesc := &TensorDescriptor{}
	filterDesc := &FilterDescriptor{}
	conv := &ConvolutionDescriptor{}
	if err := inputDesc.SetTensor4dDescriptor(DNNTensorNCHW, DNNDataFloat, 1, 1, 3, 3); err != nil {
		t.Fatal(err)
	}
	if err := outputDesc.SetTensor4dDescriptor(DNNTensorNCHW, DNNDataFloat, 1, 1, 1, 1); err != nil {
		t.Fatal(err)
	}
	if err := filterDesc.SetFilter4dDescriptor(DNNDataFloat, DNNTensorNCHW, 1, 1, 2, 2); err != nil {
		t.Fatal(err)
	}
	if err := conv.SetConvolution2dDescriptor(0, 0, 1, 1, 1, 1, DNNCrossCorrelation, DNNDataFloat); err != nil {
		t.Fatal(err)
	}
	input := solverTestBuffer(t, []float32{1, 2, 3, 4, 5, 6, 7, 8, 9})
	filter := solverTestBuffer(t, []float32{1, 2, 3, 4})
	output := solverTestBuffer(t, []float32{9})
	if err := handle.ConvolutionForward(1, inputDesc, input, filterDesc, filter, conv, 0, outputDesc, output); err == nil {
		t.Fatal("expected invalid convolution output dimensions to fail")
	}
	got, err := readMathFloat32Memory(output, 1)
	if err != nil || len(got) != 1 || got[0] != 9 {
		t.Fatalf("invalid convolution changed output: %v, error = %v", got, err)
	}
	if err := handle.ConvolutionForward(1, nil, input, filterDesc, filter, conv, 0, outputDesc, output); err == nil {
		t.Fatal("expected nil convolution descriptor to fail")
	}
}

func TestCuDNNConvolutionConvenience(t *testing.T) {
	input := solverTestBuffer(t, []float32{1, 2, 3, 4, 5, 6, 7, 8, 9})
	filter := solverTestBuffer(t, []float32{1, 2, 3, 4})
	output := solverTestBuffer(t, []float32{9, 9, 9, 9})
	inputDims, filterDims, outputDims := []int{1, 1, 3, 3}, []int{1, 1, 2, 2}, []int{1, 1, 2, 2}
	if err := ConvolutionForward(input, filter, output, inputDims, filterDims, outputDims, 0, 0, 0, 1); err == nil {
		t.Fatal("expected invalid convolution stride to fail")
	}
	if err := ConvolutionForward(input, filter, output, []int{1, 1}, filterDims, outputDims, 0, 0, 1, 1); err == nil {
		t.Fatal("expected invalid convolution dimensions to fail")
	}
	got, err := readMathFloat32Memory(output, 4)
	if err != nil || !reflect.DeepEqual(got, []float32{9, 9, 9, 9}) {
		t.Fatalf("invalid convolution changed output: %v, error = %v", got, err)
	}
	if err := ConvolutionForward(input, filter, output, inputDims, filterDims, outputDims, 0, 0, 1, 1); err != nil {
		t.Fatal(err)
	}
	got, err = readMathFloat32Memory(output, 4)
	if err != nil || !reflect.DeepEqual(got, []float32{23, 33, 53, 63}) {
		t.Fatalf("convenience convolution = %v, error = %v", got, err)
	}
}

func TestCuDNNActivationConvenience(t *testing.T) {
	input := solverTestBuffer(t, []float32{-1, 2})
	output := solverTestBuffer(t, []float32{9, 9})
	if err := ApplyActivation(input, output, []int{1, 2}, DNNActivationRelu); err == nil || !strings.Contains(err.Error(), "dimensions must have length 4") {
		t.Fatalf("invalid activation rank error = %v", err)
	}
	if err := ApplyActivation(input, output, []int{1, 1, 1, 0}, DNNActivationRelu); err == nil || !strings.Contains(err.Error(), "invalid 4D dimension") {
		t.Fatalf("invalid activation shape error = %v", err)
	}
	got, err := readMathFloat32Memory(output, 2)
	if err != nil || !reflect.DeepEqual(got, []float32{9, 9}) {
		t.Fatalf("invalid activation changed output: %v, error = %v", got, err)
	}
	if err := ApplyActivation(input, output, []int{1, 1, 1, 2}, DNNActivationRelu); err != nil {
		t.Fatal(err)
	}
	got, err = readMathFloat32Memory(output, 2)
	if err != nil || !reflect.DeepEqual(got, []float32{0, 2}) {
		t.Fatalf("convenience activation = %v, error = %v", got, err)
	}
}

func TestCuDNNPoolingPaddingAndStrides(t *testing.T) {
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
	pool, err := CreatePoolingDescriptor()
	if err != nil {
		t.Fatal(err)
	}
	defer pool.DestroyPoolingDescriptor()
	if err := inputDesc.SetTensorNdDescriptor(DNNDataFloat, []int{1, 1, 2, 2}, []int{6, 6, 3, 1}); err != nil {
		t.Fatal(err)
	}
	if err := outputDesc.SetTensorNdDescriptor(DNNDataFloat, []int{1, 1, 3, 3}, []int{15, 15, 5, 1}); err != nil {
		t.Fatal(err)
	}
	input := solverTestBuffer(t, []float32{-1, -2, 99, -3, -4})
	output := solverTestBuffer(t, make([]float32, 13))
	cases := []struct {
		name string
		mode DNNPoolingMode
		want []float32
	}{
		{"max", DNNPoolingMax, []float32{-1, -1, -2, -1, -1, -2, -3, -3, -4}},
		{"average include padding", DNNPoolingAverageCountIncludePadding, []float32{-.25, -.75, -.5, -1, -2.5, -1.5, -.75, -1.75, -1}},
		{"average exclude padding", DNNPoolingAverageCountExcludePadding, []float32{-1, -1.5, -2, -2, -2.5, -3, -3, -3.5, -4}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if err := pool.SetPooling2dDescriptor(tc.mode, 2, 2, 1, 1, 1, 1); err != nil {
				t.Fatal(err)
			}
			initial := []float32{float32(math.NaN()), float32(math.NaN()), float32(math.NaN()), 123, 123,
				float32(math.NaN()), float32(math.NaN()), float32(math.NaN()), 123, 123,
				float32(math.NaN()), float32(math.NaN()), float32(math.NaN())}
			if err := writeMathFloat32Memory(output, initial); err != nil {
				t.Fatal(err)
			}
			if err := handle.PoolingForward(pool, 1, inputDesc, input, 0, outputDesc, output); err != nil {
				t.Fatal(err)
			}
			got, err := readMathFloat32Memory(output, len(initial))
			if err != nil {
				t.Fatal(err)
			}
			want := []float32{tc.want[0], tc.want[1], tc.want[2], 123, 123,
				tc.want[3], tc.want[4], tc.want[5], 123, 123,
				tc.want[6], tc.want[7], tc.want[8]}
			if !reflect.DeepEqual(got, want) {
				t.Fatalf("pooling output = %v, want %v", got, want)
			}
		})
	}
}

func TestCuDNNPoolingValidationAndNaN(t *testing.T) {
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
	pool, err := CreatePoolingDescriptor()
	if err != nil {
		t.Fatal(err)
	}
	defer pool.DestroyPoolingDescriptor()
	if err := inputDesc.SetTensor4dDescriptor(DNNTensorNCHW, DNNDataFloat, 1, 1, 2, 2); err != nil {
		t.Fatal(err)
	}
	if err := outputDesc.SetTensor4dDescriptor(DNNTensorNCHW, DNNDataFloat, 1, 1, 1, 1); err != nil {
		t.Fatal(err)
	}
	if err := pool.SetPooling2dDescriptor(DNNPoolingMax, 2, 2, 0, 0, 1, 1); err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct {
		mode                         DNNPoolingMode
		windowH, windowW, padH, padW int
		strideH, strideW             int
	}{
		{DNNPoolingMax, 0, 2, 0, 0, 1, 1},
		{DNNPoolingMax, 2, 2, -1, 0, 1, 1},
		{DNNPoolingMax, 2, 2, 0, 0, 0, 1},
		{DNNPoolingMode(-1), 2, 2, 0, 0, 1, 1},
	} {
		if err := pool.SetPooling2dDescriptor(tc.mode, tc.windowH, tc.windowW, tc.padH, tc.padW, tc.strideH, tc.strideW); err == nil {
			t.Fatalf("invalid pooling configuration accepted: %+v", tc)
		}
	}
	input := solverTestBuffer(t, []float32{1, float32(math.NaN()), 2, 3})
	output := solverTestBuffer(t, []float32{7})
	if err := handle.PoolingForward(pool, 1, inputDesc, input, 0, outputDesc, output); err != nil {
		t.Fatal(err)
	}
	got, err := readMathFloat32Memory(output, 1)
	if err != nil || !math.IsNaN(float64(got[0])) {
		t.Fatalf("max pooling must propagate NaN: %v, error = %v", got, err)
	}
}

func TestCuDNNSimulationPoolingRejectsInvalidShapes(t *testing.T) {
	handle := &DNNHandle{}
	inputDesc := &TensorDescriptor{}
	outputDesc := &TensorDescriptor{}
	pool := &PoolingDescriptor{}
	if err := pool.SetPooling2dDescriptor(DNNPoolingMax, 2, 2, 0, 0, 1, 1); err != nil {
		t.Fatal(err)
	}
	if err := inputDesc.SetTensorNdDescriptor(DNNDataFloat, []int{1, 2, 2}, []int{4, 2, 1}); err != nil {
		t.Fatal(err)
	}
	if err := outputDesc.SetTensor4dDescriptor(DNNTensorNCHW, DNNDataFloat, 1, 1, 2, 2); err != nil {
		t.Fatal(err)
	}
	input := solverTestBuffer(t, []float32{1, 2, 3, 4})
	output := solverTestBuffer(t, []float32{7, 7, 7, 7})
	if err := handle.PoolingForward(pool, 1, inputDesc, input, 0, outputDesc, output); err == nil {
		t.Fatal("expected invalid input rank to fail")
	}
	if err := inputDesc.SetTensor4dDescriptor(DNNTensorNCHW, DNNDataFloat, 1, 1, 2, 2); err != nil {
		t.Fatal(err)
	}
	if err := handle.PoolingForward(pool, 1, inputDesc, input, 0, outputDesc, output); err == nil {
		t.Fatal("expected invalid output shape to fail")
	}
	if err := outputDesc.SetTensor4dDescriptor(DNNTensorNCHW, DNNDataFloat, 1, 2, 1, 1); err != nil {
		t.Fatal(err)
	}
	if err := handle.PoolingForward(pool, 1, inputDesc, input, 0, outputDesc, output); err == nil {
		t.Fatal("expected mismatched channel counts to fail")
	}
	if err := pool.SetPooling2dDescriptor(DNNPoolingMax, 2, 2, 2, 2, 1, 1); err != nil {
		t.Fatal(err)
	}
	if err := outputDesc.SetTensor4dDescriptor(DNNTensorNCHW, DNNDataFloat, 1, 1, 5, 5); err != nil {
		t.Fatal(err)
	}
	largeOutput := solverTestBuffer(t, make([]float32, 25))
	if err := handle.PoolingForward(pool, 1, inputDesc, input, 0, outputDesc, largeOutput); err == nil || !strings.Contains(err.Error(), "padding smaller than the window") {
		t.Fatalf("unsupported padding error = %v", err)
	}
	got, err := readMathFloat32Memory(output, 4)
	if err != nil || !reflect.DeepEqual(got, []float32{7, 7, 7, 7}) {
		t.Fatalf("rejected pooling changed output: %v, error = %v", got, err)
	}
}

func TestCuDNNSimulationBatchNormStrides(t *testing.T) {
	handle := &DNNHandle{}
	inputDesc, outputDesc, params := &TensorDescriptor{}, &TensorDescriptor{}, &TensorDescriptor{}
	if err := inputDesc.SetTensorNdDescriptor(DNNDataFloat, []int{1, 2, 1, 2}, []int{10, 5, 3, 1}); err != nil {
		t.Fatal(err)
	}
	if err := outputDesc.SetTensorNdDescriptor(DNNDataFloat, []int{1, 2, 1, 2}, []int{12, 6, 4, 1}); err != nil {
		t.Fatal(err)
	}
	input := solverTestBuffer(t, []float32{1, 3, 99, 99, 99, 4, 8})
	output := solverTestBuffer(t, []float32{10, 10, 99, 99, 99, 99, 10, 10})
	if err := params.SetTensorNdDescriptor(DNNDataFloat, []int{1, 2, 1, 1}, []int{4, 2, 1, 1}); err != nil {
		t.Fatal(err)
	}
	scale := solverTestBuffer(t, []float32{2, 99, .5})
	bias := solverTestBuffer(t, []float32{1, 99, -1})
	mean := solverTestBuffer(t, []float32{1, 99, 6})
	variance := solverTestBuffer(t, []float32{4, 99, 16})
	if err := handle.BatchNormalizationForwardInference(DNNBatchNormSpatial, 2, 3, inputDesc, input,
		outputDesc, output, params, scale, bias, mean, variance, 0); err != nil {
		t.Fatal(err)
	}
	got, err := readMathFloat32Memory(output, 8)
	want := []float32{32, 36, 99, 99, 99, 99, 27.5, 28.5}
	if err != nil || !reflect.DeepEqual(got, want) {
		t.Fatalf("spatial batch normalization = %v, error = %v, want %v", got, err, want)
	}

	if err := params.SetTensorNdDescriptor(DNNDataFloat, []int{1, 2, 1, 2}, []int{8, 4, 2, 1}); err != nil {
		t.Fatal(err)
	}
	scale = solverTestBuffer(t, []float32{1, 2, 99, 99, 3, 4})
	bias = solverTestBuffer(t, []float32{0, 0, 99, 99, 0, 0})
	mean = solverTestBuffer(t, []float32{0, 0, 99, 99, 0, 0})
	variance = solverTestBuffer(t, []float32{1, 1, 99, 99, 1, 1})
	initial := []float32{float32(math.NaN()), float32(math.NaN()), 99, 99, 99, 99, float32(math.NaN()), float32(math.NaN())}
	if err := writeMathFloat32Memory(output, initial); err != nil {
		t.Fatal(err)
	}
	if err := handle.BatchNormalizationForwardInference(DNNBatchNormPerActivation, 1, 0, inputDesc, input,
		outputDesc, output, params, scale, bias, mean, variance, 0); err != nil {
		t.Fatal(err)
	}
	got, err = readMathFloat32Memory(output, 8)
	want = []float32{1, 6, 99, 99, 99, 99, 12, 32}
	if err != nil || !reflect.DeepEqual(got, want) {
		t.Fatalf("per-activation batch normalization = %v, error = %v, want %v", got, err, want)
	}
}

func TestCuDNNSimulationBatchNormRejectsInvalidShapes(t *testing.T) {
	handle := &DNNHandle{}
	inputDesc, outputDesc, params := &TensorDescriptor{}, &TensorDescriptor{}, &TensorDescriptor{}
	if err := inputDesc.SetTensor4dDescriptor(DNNTensorNCHW, DNNDataFloat, 1, 2, 1, 2); err != nil {
		t.Fatal(err)
	}
	if err := outputDesc.SetTensor4dDescriptor(DNNTensorNCHW, DNNDataFloat, 1, 1, 1, 2); err != nil {
		t.Fatal(err)
	}
	if err := params.SetTensor4dDescriptor(DNNTensorNCHW, DNNDataFloat, 1, 2, 1, 1); err != nil {
		t.Fatal(err)
	}
	input := solverTestBuffer(t, []float32{1, 2, 3, 4})
	output := solverTestBuffer(t, []float32{7, 7, 7, 7})
	param := solverTestBuffer(t, []float32{1, 1})
	run := func(mode DNNBatchNormMode) error {
		return handle.BatchNormalizationForwardInference(mode, 1, 0, inputDesc, input, outputDesc, output,
			params, param, param, param, param, 0)
	}
	if err := run(DNNBatchNormSpatial); err == nil {
		t.Fatal("expected mismatched output dimensions to fail")
	}
	if err := outputDesc.SetTensor4dDescriptor(DNNTensorNCHW, DNNDataFloat, 1, 2, 1, 2); err != nil {
		t.Fatal(err)
	}
	if err := run(DNNBatchNormPerActivation); err == nil {
		t.Fatal("expected invalid parameter dimensions to fail")
	}
	if err := run(DNNBatchNormMode(-1)); err == nil {
		t.Fatal("expected invalid batch normalization mode to fail")
	}
	if err := params.SetTensor4dDescriptor(DNNTensorNCHW, DNNDataHalf, 1, 2, 1, 1); err != nil {
		t.Fatal(err)
	}
	if err := run(DNNBatchNormSpatial); err == nil {
		t.Fatal("expected invalid simulation parameter type to fail")
	}
	got, err := readMathFloat32Memory(output, 4)
	if err != nil || !reflect.DeepEqual(got, []float32{7, 7, 7, 7}) {
		t.Fatalf("rejected batch normalization changed output: %v, error = %v", got, err)
	}
}

func TestCuDNNBatchNormPerActivation(t *testing.T) {
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
	params, err := CreateTensorDescriptor()
	if err != nil {
		t.Fatal(err)
	}
	defer params.DestroyTensorDescriptor()
	if err := tensor.SetTensor4dDescriptor(DNNTensorNCHW, DNNDataFloat, 1, 2, 1, 2); err != nil {
		t.Fatal(err)
	}
	if err := params.SetTensor4dDescriptor(DNNTensorNCHW, DNNDataFloat, 1, 2, 1, 2); err != nil {
		t.Fatal(err)
	}
	input := solverTestBuffer(t, []float32{1, 3, 4, 8})
	output := solverTestBuffer(t, []float32{float32(math.NaN()), float32(math.NaN()), float32(math.NaN()), float32(math.NaN())})
	scale := solverTestBuffer(t, []float32{1, 2, 3, 4})
	zero := solverTestBuffer(t, []float32{0, 0, 0, 0})
	variance := solverTestBuffer(t, []float32{1, 1, 1, 1})
	const epsilon = 1e-5
	if err := handle.BatchNormalizationForwardInference(DNNBatchNormPerActivation, 1, 0, tensor, input, tensor, output,
		params, scale, zero, zero, variance, epsilon); err != nil {
		t.Fatal(err)
	}
	got, err := readMathFloat32Memory(output, 4)
	if err != nil {
		t.Fatal(err)
	}
	factor := float32(1 / math.Sqrt(1+epsilon))
	assertFloat32Slice(t, got, []float32{factor, 6 * factor, 12 * factor, 32 * factor}, 1e-5)
}

func TestCuDNNSimulationConvolutionRejectsStridedLayouts(t *testing.T) {
	handle := &DNNHandle{}
	inputDesc := &TensorDescriptor{}
	outputDesc := &TensorDescriptor{}
	filterDesc := &FilterDescriptor{}
	conv := &ConvolutionDescriptor{}
	if err := inputDesc.SetTensorNdDescriptor(DNNDataFloat, []int{1, 1, 3, 3}, []int{12, 12, 4, 1}); err != nil {
		t.Fatal(err)
	}
	if err := outputDesc.SetTensor4dDescriptor(DNNTensorNCHW, DNNDataFloat, 1, 1, 2, 2); err != nil {
		t.Fatal(err)
	}
	if err := filterDesc.SetFilter4dDescriptor(DNNDataFloat, DNNTensorNCHW, 1, 1, 2, 2); err != nil {
		t.Fatal(err)
	}
	if err := conv.SetConvolution2dDescriptor(0, 0, 1, 1, 1, 1, DNNCrossCorrelation, DNNDataFloat); err != nil {
		t.Fatal(err)
	}
	input := solverTestBuffer(t, make([]float32, 12))
	filter := solverTestBuffer(t, []float32{1, 2, 3, 4})
	output := solverTestBuffer(t, []float32{1, 2, 3, 4})
	if err := handle.ConvolutionForward(1, inputDesc, input, filterDesc, filter, conv, 0, outputDesc, output); err == nil {
		t.Fatal("expected unsupported strided convolution layout to fail")
	}
	if err := inputDesc.SetTensor4dDescriptor(DNNTensorNCHW, DNNDataFloat, 1, 1, 3, 3); err != nil {
		t.Fatal(err)
	}
	if err := filterDesc.SetFilter4dDescriptor(DNNDataFloat, DNNTensorNHWC, 1, 1, 2, 2); err != nil {
		t.Fatal(err)
	}
	if err := handle.ConvolutionForward(1, inputDesc, input, filterDesc, filter, conv, 0, outputDesc, output); err == nil {
		t.Fatal("expected unsupported convolution filter layout to fail")
	}
	got, err := readMathFloat32Memory(output, 4)
	if err != nil || !reflect.DeepEqual(got, []float32{1, 2, 3, 4}) {
		t.Fatalf("rejected convolution changed output: %v, error = %v", got, err)
	}
}

func TestCuDNNSimulationConvolutionNHWC(t *testing.T) {
	handle := &DNNHandle{}
	inputDesc := &TensorDescriptor{}
	outputDesc := &TensorDescriptor{}
	filterDesc := &FilterDescriptor{}
	conv := &ConvolutionDescriptor{}
	if err := inputDesc.SetTensor4dDescriptor(DNNTensorNHWC, DNNDataFloat, 1, 2, 2, 2); err != nil {
		t.Fatal(err)
	}
	if err := outputDesc.SetTensor4dDescriptor(DNNTensorNHWC, DNNDataFloat, 1, 1, 2, 2); err != nil {
		t.Fatal(err)
	}
	if err := filterDesc.SetFilter4dDescriptor(DNNDataFloat, DNNTensorNCHW, 1, 2, 1, 1); err != nil {
		t.Fatal(err)
	}
	if err := conv.SetConvolution2dDescriptor(0, 0, 1, 1, 1, 1, DNNCrossCorrelation, DNNDataFloat); err != nil {
		t.Fatal(err)
	}
	input := solverTestBuffer(t, []float32{1, 10, 2, 20, 3, 30, 4, 40})
	filter := solverTestBuffer(t, []float32{1, 2})
	output := solverTestBuffer(t, []float32{1, 1, 1, 1})
	if err := handle.ConvolutionForward(1, inputDesc, input, filterDesc, filter, conv, 1, outputDesc, output); err != nil {
		t.Fatal(err)
	}
	got, err := readMathFloat32Memory(output, 4)
	if err != nil || !reflect.DeepEqual(got, []float32{22, 43, 64, 85}) {
		t.Fatalf("NHWC convolution = %v, error = %v", got, err)
	}
}

func TestCuDNNSimulationConvolutionSingletonStrides(t *testing.T) {
	handle := &DNNHandle{}
	inputDesc := &TensorDescriptor{}
	outputDesc := &TensorDescriptor{}
	filterDesc := &FilterDescriptor{}
	conv := &ConvolutionDescriptor{}
	if err := inputDesc.SetTensorNdDescriptor(DNNDataFloat, []int{1, 1, 3, 3}, []int{100, 77, 3, 1}); err != nil {
		t.Fatal(err)
	}
	if err := outputDesc.SetTensorNdDescriptor(DNNDataFloat, []int{1, 1, 2, 2}, []int{100, 55, 2, 1}); err != nil {
		t.Fatal(err)
	}
	if err := filterDesc.SetFilter4dDescriptor(DNNDataFloat, DNNTensorNCHW, 1, 1, 2, 2); err != nil {
		t.Fatal(err)
	}
	if err := conv.SetConvolution2dDescriptor(0, 0, 1, 1, 1, 1, DNNCrossCorrelation, DNNDataFloat); err != nil {
		t.Fatal(err)
	}
	input := solverTestBuffer(t, []float32{1, 2, 3, 4, 5, 6, 7, 8, 9})
	filter := solverTestBuffer(t, []float32{1, 2, 3, 4})
	output := solverTestBuffer(t, []float32{0, 0, 0, 0})
	if err := handle.ConvolutionForward(1, inputDesc, input, filterDesc, filter, conv, 0, outputDesc, output); err != nil {
		t.Fatal(err)
	}
	got, err := readMathFloat32Memory(output, 4)
	if err != nil || !reflect.DeepEqual(got, []float32{37, 47, 67, 77}) {
		t.Fatalf("singleton-stride convolution = %v, error = %v", got, err)
	}
}

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
		if math.IsNaN(float64(got[index])) || math.Abs(float64(got[index]-want[index])) > tolerance {
			t.Fatalf("unexpected value[%d]: got %v want %v", index, got[index], want[index])
		}
	}
}
