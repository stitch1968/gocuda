//go:build cuda

package libraries

import (
	"errors"
	"fmt"
	"math"
	"reflect"
	"sync"
	"testing"

	cuda "github.com/stitch1968/gocuda"
	"github.com/stitch1968/gocuda/kernels"
	"github.com/stitch1968/gocuda/memory"
	"github.com/stitch1968/gocuda/streams"
)

func dnnTestTensor(t *testing.T, dataType DNNDataType, n, c, h, w int) *TensorDescriptor {
	t.Helper()
	desc, err := CreateTensorDescriptor()
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := desc.DestroyTensorDescriptor(); err != nil {
			t.Error(err)
		}
	})
	if err := desc.SetTensor4dDescriptor(DNNTensorNCHW, dataType, n, c, h, w); err != nil {
		t.Fatal(err)
	}
	return desc
}

func dnnTestDeviceBuffer(t *testing.T, deviceID int, size int64) *memory.Memory {
	t.Helper()
	buffer, err := memory.AllocDeviceOnlyOnDevice(size, deviceID)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := buffer.Free(); err != nil {
			t.Error(err)
		}
	})
	return buffer
}

func dnnTestDeviceBuffer64(t *testing.T, deviceID int, values []float64) *memory.Memory {
	t.Helper()
	buffer := dnnTestDeviceBuffer(t, deviceID, int64(len(values)*8))
	if err := writeMathFloat64Memory(buffer, values); err != nil {
		t.Fatal(err)
	}
	return buffer
}

func dnnTestPinnedBuffer(t *testing.T, values []float32) *memory.Memory {
	t.Helper()
	buffer, err := memory.AllocWithTypeAndStream(nil, int64(len(values)*4), memory.TypePinned)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := buffer.Free(); err != nil {
			t.Error(err)
		}
	})
	view, err := memory.View[float32](buffer, len(values))
	if err != nil {
		t.Fatal(err)
	}
	copy(view, values)
	return buffer
}

func dnnTestFloat64Values(t *testing.T, buffer *memory.Memory, want []float64, tolerance float64) {
	t.Helper()
	got, err := readMathFloat64Memory(buffer, len(want))
	if err != nil {
		t.Fatal(err)
	}
	for i := range want {
		if math.IsNaN(got[i]) || math.Abs(got[i]-want[i]) > tolerance {
			t.Fatalf("output[%d] = %v, want %v", i, got[i], want[i])
		}
	}
}

func TestCuDNNNativeDoublePrecisionOnEachDevice(t *testing.T) {
	if err := cuda.Initialize(); err != nil {
		t.Fatal(err)
	}
	if !cuda.ShouldUseCuda() || !cudnnNativeAvailable() {
		t.Skip("native cuDNN unavailable")
	}
	devices, err := cuda.GetDevices()
	if err != nil {
		t.Fatal(err)
	}
	for deviceID := range devices {
		t.Run(devices[deviceID].Name, func(t *testing.T) {
			handle, err := CreateNativeDNNHandleOnDevice(deviceID)
			if err != nil {
				t.Fatal(err)
			}
			t.Cleanup(func() {
				if !handle.destroyed {
					if err := handle.DestroyHandle(); err != nil {
						t.Error(err)
					}
				}
			})
			if handle.Backend() != ImplementationModeNativeCUDA || handle.deviceID != deviceID {
				t.Fatal("cuDNN handle did not bind the requested native device")
			}
			stream, err := streams.NewNativeStream(deviceID, streams.StreamNonBlocking)
			if err != nil {
				t.Fatal(err)
			}
			t.Cleanup(func() {
				if err := stream.Close(); err != nil {
					t.Error(err)
				}
			})
			tensor := dnnTestTensor(t, DNNDataDouble, 1, 1, 2, 2)
			poolOutputDesc := dnnTestTensor(t, DNNDataDouble, 1, 1, 1, 1)
			bnDesc := dnnTestTensor(t, DNNDataDouble, 1, 2, 1, 1)
			activation, err := CreateActivationDescriptor()
			if err != nil {
				t.Fatal(err)
			}
			t.Cleanup(func() {
				if err := activation.DestroyActivationDescriptor(); err != nil {
					t.Error(err)
				}
			})
			if err := activation.SetActivationDescriptor(DNNActivationRelu, DNNPropagateNaN, 0); err != nil {
				t.Fatal(err)
			}
			if err := activation.SetActivationDescriptor(DNNActivationMode(-1), DNNPropagateNaN, 0); err == nil || activation.mode != DNNActivationRelu {
				t.Fatalf("failed activation update changed descriptor: error = %v, mode = %d", err, activation.mode)
			}
			pool, err := CreatePoolingDescriptor()
			if err != nil {
				t.Fatal(err)
			}
			t.Cleanup(func() {
				if err := pool.DestroyPoolingDescriptor(); err != nil {
					t.Error(err)
				}
			})
			if err := pool.SetPooling2dDescriptor(DNNPoolingMax, 2, 2, 0, 0, 1, 1); err != nil {
				t.Fatal(err)
			}
			if err := pool.SetPooling2dDescriptor(DNNPoolingMode(-1), 2, 2, 0, 0, 1, 1); err == nil || pool.mode != DNNPoolingMax {
				t.Fatalf("failed pooling update changed descriptor: error = %v, mode = %d", err, pool.mode)
			}
			filter, err := CreateFilterDescriptor()
			if err != nil {
				t.Fatal(err)
			}
			t.Cleanup(func() {
				if err := filter.DestroyFilterDescriptor(); err != nil {
					t.Error(err)
				}
			})
			if err := filter.SetFilter4dDescriptor(DNNDataDouble, DNNTensorNCHW, 1, 1, 1, 1); err != nil {
				t.Fatal(err)
			}
			conv, err := CreateConvolutionDescriptor()
			if err != nil {
				t.Fatal(err)
			}
			t.Cleanup(func() {
				if err := conv.DestroyConvolutionDescriptor(); err != nil {
					t.Error(err)
				}
			})
			if err := conv.SetConvolution2dDescriptor(0, 0, 1, 1, 1, 1, DNNCrossCorrelation, DNNDataDouble); err != nil {
				t.Fatal(err)
			}
			input := dnnTestDeviceBuffer64(t, deviceID, []float64{-1, 2, -3, 4})
			output := dnnTestDeviceBuffer64(t, deviceID, []float64{1, 1, 1, 1})
			poolOutput := dnnTestDeviceBuffer64(t, deviceID, []float64{1})
			filterData := dnnTestDeviceBuffer64(t, deviceID, []float64{2})
			bnInput := dnnTestDeviceBuffer64(t, deviceID, []float64{3, 10})
			bnOutput := dnnTestDeviceBuffer64(t, deviceID, []float64{1, 1})
			scale := dnnTestDeviceBuffer64(t, deviceID, []float64{2, 0.5})
			bias := dnnTestDeviceBuffer64(t, deviceID, []float64{1, -1})
			mean := dnnTestDeviceBuffer64(t, deviceID, []float64{1, 6})
			variance := dnnTestDeviceBuffer64(t, deviceID, []float64{4, 16})
			t.Cleanup(func() {
				if !handle.destroyed {
					if err := handle.DestroyHandle(); err != nil {
						t.Error(err)
					}
				}
			})
			if err := handle.SetStream(stream); err != nil {
				t.Fatal(err)
			}
			if err := handle.ActivationForward(activation, 2, tensor, input, 3, tensor, output); err != nil {
				t.Fatal(err)
			}
			dnnTestFloat64Values(t, output, []float64{3, 7, 3, 11}, 1e-12)
			if err := handle.ActivationForward(activation, 1, tensor, input, 0, tensor, input); err != nil {
				t.Fatal(err)
			}
			dnnTestFloat64Values(t, input, []float64{0, 2, 0, 4}, 1e-12)
			if err := handle.PoolingForward(pool, 2, tensor, input, 3, poolOutputDesc, poolOutput); err != nil {
				t.Fatal(err)
			}
			dnnTestFloat64Values(t, poolOutput, []float64{11}, 1e-12)
			if err := writeMathFloat64Memory(input, []float64{1, 2, 3, 4}); err != nil {
				t.Fatal(err)
			}
			if err := writeMathFloat64Memory(output, []float64{1, 1, 1, 1}); err != nil {
				t.Fatal(err)
			}
			if err := handle.ConvolutionForward(0.5, tensor, input, filter, filterData, conv, 2, tensor, output); err != nil {
				t.Fatal(err)
			}
			dnnTestFloat64Values(t, output, []float64{3, 4, 5, 6}, 1e-12)
			const epsilon = 1e-5
			if err := handle.BatchNormalizationForwardInference(DNNBatchNormSpatial, 2, 3, bnDesc, bnInput,
				bnDesc, bnOutput, bnDesc, scale, bias, mean, variance, epsilon); err != nil {
				t.Fatal(err)
			}
			want := []float64{
				3 + 2*(2*(3-1)/math.Sqrt(4+epsilon)+1),
				3 + 2*(0.5*(10-6)/math.Sqrt(16+epsilon)-1),
			}
			dnnTestFloat64Values(t, bnOutput, want, 1e-10)
			if err := handle.Synchronize(); err != nil {
				t.Fatal(err)
			}
			if len(devices) > 1 {
				other := dnnTestDeviceBuffer64(t, (deviceID+1)%len(devices), []float64{1, 2, 3, 4})
				if err := handle.ActivationForward(activation, 1, tensor, other, 0, tensor, output); err == nil {
					t.Fatal("expected cross-device cuDNN input to fail")
				}
				dnnTestFloat64Values(t, output, []float64{3, 4, 5, 6}, 1e-12)
				otherStream, err := streams.NewNativeStream((deviceID+1)%len(devices), streams.StreamNonBlocking)
				if err != nil {
					t.Fatal(err)
				}
				t.Cleanup(func() {
					if err := otherStream.Close(); err != nil {
						t.Error(err)
					}
				})
				if err := handle.SetStream(otherStream); err == nil {
					t.Fatal("expected cross-device cuDNN stream to fail")
				}
				if handle.nativeStream != stream {
					t.Fatal("failed stream change must preserve the attached stream")
				}
			}
		})
	}
}

func TestCuDNNNativeRejectsInvalidBuffers(t *testing.T) {
	if err := cuda.Initialize(); err != nil {
		t.Fatal(err)
	}
	if !cuda.ShouldUseCuda() || !cudnnNativeAvailable() {
		t.Skip("native cuDNN unavailable")
	}
	handle, err := CreateNativeDNNHandleOnDevice(0)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if !handle.destroyed {
			if err := handle.DestroyHandle(); err != nil {
				t.Error(err)
			}
		}
	})
	tensor := dnnTestTensor(t, DNNDataFloat, 1, 1, 2, 2)
	doubleTensor := dnnTestTensor(t, DNNDataDouble, 1, 1, 2, 2)
	strided, err := CreateTensorDescriptor()
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := strided.DestroyTensorDescriptor(); err != nil {
			t.Error(err)
		}
	})
	if err := strided.SetTensorNdDescriptor(DNNDataFloat, []int{1, 1, 2, 2}, []int{8, 8, 4, 1}); err != nil {
		t.Fatal(err)
	}
	param := dnnTestTensor(t, DNNDataFloat, 1, 1, 1, 1)
	activation, err := CreateActivationDescriptor()
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := activation.DestroyActivationDescriptor(); err != nil {
			t.Error(err)
		}
	})
	if err := activation.SetActivationDescriptor(DNNActivationRelu, DNNPropagateNaN, 0); err != nil {
		t.Fatal(err)
	}
	pool, err := CreatePoolingDescriptor()
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := pool.DestroyPoolingDescriptor(); err != nil {
			t.Error(err)
		}
	})
	if err := pool.SetPooling2dDescriptor(DNNPoolingMax, 1, 1, 0, 0, 1, 1); err != nil {
		t.Fatal(err)
	}
	filter, err := CreateFilterDescriptor()
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := filter.DestroyFilterDescriptor(); err != nil {
			t.Error(err)
		}
	})
	if err := filter.SetFilter4dDescriptor(DNNDataFloat, DNNTensorNCHW, 1, 1, 1, 1); err != nil {
		t.Fatal(err)
	}
	conv, err := CreateConvolutionDescriptor()
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := conv.DestroyConvolutionDescriptor(); err != nil {
			t.Error(err)
		}
	})
	if err := conv.SetConvolution2dDescriptor(0, 0, 1, 1, 1, 1, DNNCrossCorrelation, DNNDataFloat); err != nil {
		t.Fatal(err)
	}
	input := dnnTestDeviceBuffer(t, 0, 16)
	output := dnnTestDeviceBuffer(t, 0, 16)
	doubleOutput := dnnTestDeviceBuffer64(t, 0, []float64{1, 2, 3, 4})
	short := dnnTestDeviceBuffer(t, 0, 2)
	aliased := dnnTestDeviceBuffer(t, 0, 32)
	paramBuffer := dnnTestDeviceBuffer(t, 0, 4)
	host, err := memory.AllocWithTypeAndStream(nil, 16, memory.TypePinned)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := host.Free(); err != nil {
			t.Error(err)
		}
	})
	if err := writeMathFloat32Memory(input, []float32{1, 2, 3, 4}); err != nil {
		t.Fatal(err)
	}
	if err := writeMathFloat32Memory(output, []float32{7, 8, 9, 10}); err != nil {
		t.Fatal(err)
	}
	if err := writeMathFloat32Memory(paramBuffer, []float32{1}); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if !handle.destroyed {
			if err := handle.DestroyHandle(); err != nil {
				t.Error(err)
			}
		}
	})
	for _, tc := range []struct {
		name string
		run  func() error
	}{
		{"nil input", func() error { return handle.ActivationForward(activation, 1, tensor, nil, 0, tensor, output) }},
		{"short activation input", func() error { return handle.ActivationForward(activation, 1, tensor, short, 0, tensor, output) }},
		{"short activation output", func() error { return handle.ActivationForward(activation, 1, tensor, input, 0, tensor, short) }},
		{"pinned input", func() error { return handle.ActivationForward(activation, 1, tensor, host, 0, tensor, output) }},
		{"non-native descriptor", func() error {
			return handle.ActivationForward(activation, 1, &TensorDescriptor{}, input, 0, tensor, output)
		}},
		{"mixed activation precision", func() error {
			return handle.ActivationForward(activation, 1, tensor, input, 0, doubleTensor, doubleOutput)
		}},
		{"in-place mismatched strides", func() error {
			return handle.ActivationForward(activation, 1, tensor, aliased, 0, strided, aliased)
		}},
		{"short pooling input", func() error { return handle.PoolingForward(pool, 1, tensor, short, 0, tensor, output) }},
		{"short convolution filter", func() error {
			return handle.ConvolutionForward(1, tensor, input, filter, short, conv, 0, tensor, output)
		}},
		{"short batch-norm scale", func() error {
			return handle.BatchNormalizationForwardInference(DNNBatchNormSpatial, 1, 0, tensor, input, tensor, output,
				param, short, paramBuffer, paramBuffer, paramBuffer, 1e-5)
		}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if err := tc.run(); err == nil {
				t.Fatal("expected invalid native buffer to fail")
			}
		})
	}
	got, err := readMathFloat32Memory(output, 4)
	if err != nil || !reflect.DeepEqual(got, []float32{7, 8, 9, 10}) {
		t.Fatalf("invalid cuDNN operation changed output: %v, error = %v", got, err)
	}
}

func TestCuDNNNativeHalfActivation(t *testing.T) {
	if err := cuda.Initialize(); err != nil {
		t.Fatal(err)
	}
	if !cuda.ShouldUseCuda() || !cudnnNativeAvailable() {
		t.Skip("native cuDNN unavailable")
	}
	testCuDNNNative16Activation(t, DNNDataHalf, []uint16{0xc000, 0x3e00, 0xb800, 0x4400},
		0x3c00, []uint16{0x3c00, 0x4400, 0x3c00, 0x4880})
}

func TestCuDNNNativeBFloat16Activation(t *testing.T) {
	if err := cuda.Initialize(); err != nil {
		t.Fatal(err)
	}
	if !cuda.ShouldUseCuda() || !cudnnNativeAvailable() {
		t.Skip("native cuDNN unavailable")
	}
	devices, err := cuda.GetDevices()
	if err != nil {
		t.Fatal(err)
	}
	if devices[0].Properties.Major < 8 {
		t.Skip("bfloat16 requires compute capability 8.0 or newer")
	}
	testCuDNNNative16Activation(t, DNNDataBFloat16, []uint16{0xc000, 0x3fc0, 0xbf00, 0x4080},
		0x3f80, []uint16{0x3f80, 0x4080, 0x3f80, 0x4110})
}

func testCuDNNNative16Activation(t *testing.T, dataType DNNDataType, inputValues []uint16, one uint16, want []uint16) {
	t.Helper()
	handle, err := CreateNativeDNNHandleOnDevice(0)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if !handle.destroyed {
			if err := handle.DestroyHandle(); err != nil {
				t.Error(err)
			}
		}
	})
	tensor := dnnTestTensor(t, dataType, 1, 1, 1, 4)
	activation, err := CreateActivationDescriptor()
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := activation.DestroyActivationDescriptor(); err != nil {
			t.Error(err)
		}
	})
	if err := activation.SetActivationDescriptor(DNNActivationRelu, DNNPropagateNaN, 0); err != nil {
		t.Fatal(err)
	}
	input := dnnTestDeviceBuffer(t, 0, 8)
	output := dnnTestDeviceBuffer(t, 0, 8)
	short := dnnTestDeviceBuffer(t, 0, 6)
	t.Cleanup(func() {
		if !handle.destroyed {
			if err := handle.DestroyHandle(); err != nil {
				t.Error(err)
			}
		}
	})
	if err := writeMathTypedMemory(input, inputValues); err != nil {
		t.Fatal(err)
	}
	if err := writeMathTypedMemory(output, []uint16{one, one, one, one}); err != nil {
		t.Fatal(err)
	}
	if err := handle.ActivationForward(activation, 2, tensor, input, 1, tensor, output); err != nil {
		t.Fatal(err)
	}
	got, err := readMathTypedMemory[uint16](output, 4)
	if err != nil || !reflect.DeepEqual(got, want) {
		t.Fatalf("16-bit activation = %v, error = %v, want %v", got, err, want)
	}
	if err := handle.ActivationForward(activation, 1, tensor, short, 0, tensor, output); err == nil {
		t.Fatal("expected a short 16-bit input to fail")
	}
	got, err = readMathTypedMemory[uint16](output, 4)
	if err != nil || !reflect.DeepEqual(got, want) {
		t.Fatalf("rejected 16-bit activation changed output: %v, error = %v", got, err)
	}
}

func TestCuDNNNativeCompletesBeforeReturn(t *testing.T) {
	if err := cuda.Initialize(); err != nil {
		t.Fatal(err)
	}
	if !cuda.ShouldUseCuda() || !cudnnNativeAvailable() {
		t.Skip("native cuDNN unavailable")
	}
	devices, err := cuda.GetDevices()
	if err != nil {
		t.Fatal(err)
	}
	if devices[0].Properties.Major < 7 {
		t.Skip("the delay kernel requires compute capability 7.0 or newer")
	}
	gate, err := streams.NewNativeStream(0, streams.StreamDefault)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := gate.Close(); err != nil {
			t.Error(err)
		}
	})
	event, err := streams.NewNativeEvent(0, streams.NativeEventDisableTiming)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := event.Close(); err != nil {
			t.Error(err)
		}
	})
	module, err := kernels.LoadPTX(0, nativeTestDelayPTX)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := module.Close(); err != nil {
			t.Error(err)
		}
	})
	delay, err := module.Function("wait_ticks")
	if err != nil {
		t.Fatal(err)
	}
	handle, err := CreateNativeDNNHandleOnDevice(0)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if !handle.destroyed {
			if err := handle.DestroyHandle(); err != nil {
				t.Error(err)
			}
		}
	})
	desc := dnnTestTensor(t, DNNDataFloat, 1, 1, 1, 2)
	activation, err := CreateActivationDescriptor()
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := activation.DestroyActivationDescriptor(); err != nil {
			t.Error(err)
		}
	})
	if err := activation.SetActivationDescriptor(DNNActivationRelu, DNNPropagateNaN, 0); err != nil {
		t.Fatal(err)
	}
	input := dnnTestDeviceBuffer(t, 0, 8)
	output := dnnTestDeviceBuffer(t, 0, 8)
	t.Cleanup(func() {
		if !handle.destroyed {
			if err := handle.DestroyHandle(); err != nil {
				t.Error(err)
			}
		}
	})
	if err := writeMathFloat32Memory(input, []float32{-1, 2}); err != nil {
		t.Fatal(err)
	}
	if err := handle.ActivationForward(activation, 1, desc, input, 0, desc, output); err != nil {
		t.Fatal(err)
	}
	if err := delay.Launch(gate, kernels.Dim3{X: 1, Y: 1, Z: 1}, kernels.Dim3{X: 1, Y: 1, Z: 1}, 0, kernels.Uint64Arg(200_000_000)); err != nil {
		t.Fatal(err)
	}
	if err := event.Record(gate); err != nil {
		t.Fatal(err)
	}
	complete, err := event.Query()
	if err != nil {
		t.Fatal(err)
	}
	if complete {
		t.Skip("delay completed before default-stream synchronization could be tested")
	}
	if err := handle.ActivationForward(activation, 1, desc, input, 0, desc, output); err != nil {
		t.Fatal(err)
	}
	complete, err = event.Query()
	if err != nil || !complete {
		t.Fatalf("producer event complete = %v, error = %v", complete, err)
	}
	got, err := readMathFloat32Memory(output, 2)
	if err != nil || !reflect.DeepEqual(got, []float32{0, 2}) {
		t.Fatalf("synchronous activation = %v, error = %v", got, err)
	}
	if err := activation.SetActivationDescriptor(DNNActivationRelu, DNNPropagateNaN, 0); err != nil {
		t.Fatalf("completed operation must release descriptor: %v", err)
	}
	if err := input.Free(); err != nil {
		t.Fatalf("completed operation must release input: %v", err)
	}
	if err := output.Free(); err != nil {
		t.Fatalf("completed operation must release output: %v", err)
	}
}

func TestCuDNNNativeStreamOrderingOnEachDevice(t *testing.T) {
	if err := cuda.Initialize(); err != nil {
		t.Fatal(err)
	}
	if !cuda.ShouldUseCuda() || !cudnnNativeAvailable() {
		t.Skip("native cuDNN unavailable")
	}
	devices, err := cuda.GetDevices()
	if err != nil {
		t.Fatal(err)
	}
	for deviceID := range devices {
		t.Run(devices[deviceID].Name, func(t *testing.T) {
			if devices[deviceID].Properties.Major < 7 {
				t.Skip("the delay kernel requires compute capability 7.0 or newer")
			}
			stream, err := streams.NewNativeStream(deviceID, streams.StreamNonBlocking)
			if err != nil {
				t.Fatal(err)
			}
			t.Cleanup(func() {
				if err := stream.Close(); err != nil {
					t.Error(err)
				}
			})
			gate, err := streams.NewNativeStream(deviceID, streams.StreamNonBlocking)
			if err != nil {
				t.Fatal(err)
			}
			t.Cleanup(func() {
				if err := gate.Close(); err != nil {
					t.Error(err)
				}
			})
			event, err := streams.NewNativeEvent(deviceID, streams.NativeEventDisableTiming)
			if err != nil {
				t.Fatal(err)
			}
			t.Cleanup(func() {
				if err := event.Close(); err != nil {
					t.Error(err)
				}
			})
			module, err := kernels.LoadPTX(deviceID, nativeTestDelayPTX)
			if err != nil {
				t.Fatal(err)
			}
			t.Cleanup(func() {
				if err := module.Close(); err != nil {
					t.Error(err)
				}
			})
			delay, err := module.Function("wait_ticks")
			if err != nil {
				t.Fatal(err)
			}
			handle, err := CreateNativeDNNHandleOnDevice(deviceID)
			if err != nil {
				t.Fatal(err)
			}
			t.Cleanup(func() {
				if !handle.destroyed {
					if err := handle.DestroyHandle(); err != nil {
						t.Error(err)
					}
				}
			})
			desc := dnnTestTensor(t, DNNDataFloat, 1, 1, 1, 2)
			activation, err := CreateActivationDescriptor()
			if err != nil {
				t.Fatal(err)
			}
			t.Cleanup(func() {
				if err := activation.DestroyActivationDescriptor(); err != nil {
					t.Error(err)
				}
			})
			if err := activation.SetActivationDescriptor(DNNActivationRelu, DNNPropagateNaN, 0); err != nil {
				t.Fatal(err)
			}
			input := dnnTestDeviceBuffer(t, deviceID, 8)
			output := dnnTestDeviceBuffer(t, deviceID, 8)
			host := dnnTestPinnedBuffer(t, []float32{-1, 2})
			host2 := dnnTestPinnedBuffer(t, []float32{3, -4})
			t.Cleanup(func() {
				if !handle.destroyed {
					if err := handle.DestroyHandle(); err != nil {
						t.Error(err)
					}
				}
			})
			if err := handle.SetStream(stream); err != nil {
				t.Fatal(err)
			}
			closed, err := streams.NewNativeStream(deviceID, streams.StreamNonBlocking)
			if err != nil {
				t.Fatal(err)
			}
			t.Cleanup(func() {
				if err := closed.Close(); err != nil {
					t.Error(err)
				}
			})
			if err := closed.Close(); err != nil {
				t.Fatal(err)
			}
			if err := handle.SetStream(closed); !errors.Is(err, streams.ErrNativeStreamClosed) {
				t.Fatalf("attaching a closed stream error = %v", err)
			}
			if handle.nativeStream != stream {
				t.Fatal("failed stream change must preserve the attached stream")
			}
			if err := writeMathFloat32Memory(input, []float32{0, 0}); err != nil {
				t.Fatal(err)
			}
			if err := handle.ActivationForward(activation, 1, desc, input, 0, desc, output); err != nil {
				t.Fatal(err)
			}
			if err := delay.Launch(gate, kernels.Dim3{X: 1, Y: 1, Z: 1}, kernels.Dim3{X: 1, Y: 1, Z: 1}, 0, kernels.Uint64Arg(200_000_000)); err != nil {
				t.Fatal(err)
			}
			if err := event.Record(gate); err != nil {
				t.Fatal(err)
			}
			if err := stream.WaitEvent(event); err != nil {
				t.Fatal(err)
			}
			if err := stream.CopyAsync(input, host, 8); err != nil {
				t.Fatal(err)
			}
			complete, err := event.Query()
			if err != nil {
				t.Fatal(err)
			}
			if complete {
				t.Skip("delay completed before stream ordering could be tested")
			}
			if err := handle.ActivationForward(activation, 1, desc, input, 0, desc, output); err != nil {
				t.Fatal(err)
			}
			complete, err = event.Query()
			if err != nil || !complete {
				t.Fatalf("producer event complete = %v, error = %v", complete, err)
			}
			got, err := readMathFloat32Memory(output, 2)
			if err != nil || !reflect.DeepEqual(got, []float32{0, 2}) {
				t.Fatalf("stream activation = %v, error = %v", got, err)
			}
			if err := host.Free(); err == nil {
				t.Fatal("stream must retain queued copy buffers until synchronization")
			}
			if err := handle.Synchronize(); err != nil {
				t.Fatal(err)
			}
			if handle.nativeStream != stream {
				t.Fatal("synchronization must leave the stream attached")
			}
			if err := host.Free(); err != nil {
				t.Fatalf("handle synchronization must release copy buffers: %v", err)
			}
			next, err := streams.NewNativeStream(deviceID, streams.StreamNonBlocking)
			if err != nil {
				t.Fatal(err)
			}
			t.Cleanup(func() {
				if err := next.Close(); err != nil {
					t.Error(err)
				}
			})
			if err := stream.CopyAsync(input, host2, 8); err != nil {
				t.Fatal(err)
			}
			if err := handle.SetStream(next); err != nil {
				t.Fatal(err)
			}
			if err := host2.Free(); err != nil {
				t.Fatalf("stream change must release copy buffers: %v", err)
			}
			if err := handle.ActivationForward(activation, 1, desc, input, 0, desc, output); err != nil {
				t.Fatal(err)
			}
			got, err = readMathFloat32Memory(output, 2)
			if err != nil || !reflect.DeepEqual(got, []float32{3, 0}) {
				t.Fatalf("switched-stream activation = %v, error = %v", got, err)
			}
			if err := next.Close(); err != nil {
				t.Fatal(err)
			}
			if err := handle.ActivationForward(activation, 1, desc, input, 0, desc, output); !errors.Is(err, streams.ErrNativeStreamClosed) {
				t.Fatalf("operation on closed stream error = %v", err)
			}
			if err := handle.Synchronize(); err != nil {
				t.Fatalf("synchronizing an already closed stream: %v", err)
			}
			if err := handle.SetStream(nil); err != nil {
				t.Fatal(err)
			}
			if err := handle.ActivationForward(activation, 1, desc, input, 0, desc, output); err != nil {
				t.Fatal(err)
			}
			if err := input.Free(); err != nil {
				t.Fatalf("closed-stream failure must not retain input: %v", err)
			}
			if err := output.Free(); err != nil {
				t.Fatalf("closed-stream failure must not retain output: %v", err)
			}
		})
	}
}

func TestCuDNNNativeConcurrentExecutionOnEachDevice(t *testing.T) {
	if err := cuda.Initialize(); err != nil {
		t.Fatal(err)
	}
	if !cuda.ShouldUseCuda() || !cudnnNativeAvailable() {
		t.Skip("native cuDNN unavailable")
	}
	devices, err := cuda.GetDevices()
	if err != nil {
		t.Fatal(err)
	}
	for deviceID := range devices {
		t.Run(devices[deviceID].Name, func(t *testing.T) {
			handle, err := CreateNativeDNNHandleOnDevice(deviceID)
			if err != nil {
				t.Fatal(err)
			}
			t.Cleanup(func() {
				if !handle.destroyed {
					if err := handle.DestroyHandle(); err != nil {
						t.Error(err)
					}
				}
			})
			desc := dnnTestTensor(t, DNNDataFloat, 1, 1, 1, 2)
			activation, err := CreateActivationDescriptor()
			if err != nil {
				t.Fatal(err)
			}
			t.Cleanup(func() {
				if err := activation.DestroyActivationDescriptor(); err != nil {
					t.Error(err)
				}
			})
			if err := activation.SetActivationDescriptor(DNNActivationRelu, DNNPropagateNaN, 0); err != nil {
				t.Fatal(err)
			}
			const workers = 4
			inputs, outputs := make([]*memory.Memory, workers), make([]*memory.Memory, workers)
			for i := range workers {
				inputs[i] = dnnTestDeviceBuffer(t, deviceID, 8)
				outputs[i] = dnnTestDeviceBuffer(t, deviceID, 8)
				if err := writeMathFloat32Memory(inputs[i], []float32{-1, float32(i + 1)}); err != nil {
					t.Fatal(err)
				}
			}
			t.Cleanup(func() {
				if !handle.destroyed {
					if err := handle.DestroyHandle(); err != nil {
						t.Error(err)
					}
				}
			})
			var wg sync.WaitGroup
			errs := make(chan error, workers+1)
			for i := range workers {
				wg.Add(1)
				go func(index int) {
					defer wg.Done()
					for range 4 {
						if err := handle.ActivationForward(activation, 1, desc, inputs[index], 0, desc, outputs[index]); err != nil {
							errs <- err
							return
						}
					}
					got, err := readMathFloat32Memory(outputs[index], 2)
					if err != nil {
						errs <- err
						return
					}
					if got[0] != 0 || got[1] != float32(index+1) {
						errs <- fmt.Errorf("worker %d got %v", index, got)
					}
				}(i)
			}
			wg.Add(1)
			go func() {
				defer wg.Done()
				for range 16 {
					err := activation.SetActivationDescriptor(DNNActivationRelu, DNNPropagateNaN, 0)
					if err != nil && !errors.Is(err, ErrDNNDescriptorInUse) {
						errs <- err
						return
					}
				}
			}()
			wg.Wait()
			close(errs)
			for err := range errs {
				t.Error(err)
			}
		})
	}
}

func TestCuDNNNativePendingCleanupRetry(t *testing.T) {
	if err := cuda.Initialize(); err != nil {
		t.Fatal(err)
	}
	if !cuda.ShouldUseCuda() || !cudnnNativeAvailable() {
		t.Skip("native cuDNN unavailable")
	}
	handle, err := CreateNativeDNNHandleOnDevice(0)
	if err != nil {
		t.Fatal(err)
	}
	defer handle.DestroyHandle()
	stream, err := streams.NewNativeStream(0, streams.StreamNonBlocking)
	if err != nil {
		t.Fatal(err)
	}
	defer stream.Close()
	if err := handle.SetStream(stream); err != nil {
		t.Fatal(err)
	}
	tensor := dnnTestTensor(t, DNNDataFloat, 1, 1, 1, 1)
	buffer := dnnTestDeviceBuffer(t, 0, 4)
	workspace := dnnTestDeviceBuffer(t, 0, 4)
	descriptors, err := retainDNNDescriptors(&tensor.dnnDescriptorState)
	if err != nil {
		t.Fatal(err)
	}
	bufferHeld, err := buffer.Retain()
	if err != nil {
		t.Fatal(err)
	}
	workspaceHeld, err := workspace.Retain()
	if err != nil {
		t.Fatal(err)
	}
	defer workspaceHeld.Release()
	handle.pendingDescriptors = append(handle.pendingDescriptors, descriptors...)
	handle.pendingBuffers = append(handle.pendingBuffers, bufferHeld)
	handle.pendingWorkspaces = append(handle.pendingWorkspaces, workspace)
	if err := tensor.DestroyTensorDescriptor(); !errors.Is(err, ErrDNNDescriptorInUse) {
		t.Fatalf("pending descriptor destroy error = %v", err)
	}
	if err := buffer.Free(); err == nil {
		t.Fatal("pending input buffer must remain retained")
	}
	if err := handle.Synchronize(); err == nil {
		t.Fatal("expected retained workspace cleanup to fail")
	}
	if err := tensor.SetTensor4dDescriptor(DNNTensorNCHW, DNNDataFloat, 1, 1, 1, 1); err != nil {
		t.Fatalf("completed stream must release descriptor: %v", err)
	}
	if err := buffer.Free(); err != nil {
		t.Fatalf("completed stream must release buffer: %v", err)
	}
	workspaceHeld.Release()
	if err := handle.Synchronize(); err != nil {
		t.Fatal(err)
	}
	if workspace.Ptr() != nil {
		t.Fatal("workspace should be freed after cleanup retry")
	}
	if err := stream.Close(); err != nil {
		t.Fatal(err)
	}
	if err := handle.DestroyHandle(); err != nil {
		t.Fatalf("destroy with an already closed stream: %v", err)
	}
}
