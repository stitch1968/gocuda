//go:build cuda

package libraries

import (
	"fmt"
	"math"
	"math/cmplx"
	"testing"

	cuda "github.com/stitch1968/gocuda"
	"github.com/stitch1968/gocuda/memory"
	"github.com/stitch1968/gocuda/streams"
)

func TestFFTDoubleNativeStreamOnEachDevice(t *testing.T) {
	if err := cuda.Initialize(); err != nil {
		t.Fatal(err)
	}
	if !cuda.ShouldUseCuda() {
		t.Skip("CUDA hardware unavailable")
	}
	devices, err := cuda.GetDevices()
	if err != nil {
		t.Fatal(err)
	}
	caller, err := cuda.NewContext(0)
	if err != nil {
		t.Fatal(err)
	}
	for deviceID := range devices {
		t.Run(fmt.Sprintf("%d_%s", deviceID, devices[deviceID].Name), func(t *testing.T) {
			if err := caller.Run(func() error {
				testFFTDoubleNativeStream(t, deviceID, len(devices))
				return nil
			}); err != nil {
				t.Fatal(err)
			}
		})
	}
}

func testFFTDoubleNativeStream(t *testing.T, deviceID, deviceCount int) {
	t.Helper()
	stream, err := streams.NewNativeStream(deviceID, streams.StreamNonBlocking)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := stream.Close(); err != nil {
			t.Error(err)
		}
	})
	ctx, err := CreateNativeFFTContextOnDevice(deviceID)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := ctx.DestroyContext(); err != nil {
			t.Error(err)
		}
	})
	plans := make([]*FFTPlan, 3)
	for i, kind := range []FFTType{FFTTypeD2Z, FFTTypeZ2D, FFTTypeZ2Z} {
		plans[i], err = ctx.CreatePlan1D(5, kind, 1)
		if err != nil {
			t.Fatal(err)
		}
		if err := plans[i].SetStream(stream); err != nil {
			t.Fatal(err)
		}
	}
	many, err := ctx.CreatePlanMany([]int{2, 3}, FFTTypeZ2Z, 2)
	if err != nil {
		t.Fatal(err)
	}
	if err := many.SetStream(stream); err != nil {
		t.Fatal(err)
	}
	allocate := func(size int64, device int) *memory.Memory {
		t.Helper()
		buffer, err := memory.AllocDeviceOnlyOnDevice(size, device)
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
	input, spectrum, output := allocate(40, deviceID), allocate(48, deviceID), allocate(40, deviceID)
	complexInput, complexOutput := allocate(80, deviceID), allocate(80, deviceID)
	batchedInput, batchedOutput := allocate(192, deviceID), allocate(192, deviceID)
	var wrongDevice *memory.Memory
	if deviceCount > 1 {
		wrongDevice = allocate(80, (deviceID+1)%deviceCount)
	}
	t.Cleanup(func() {
		if err := stream.Synchronize(); err != nil {
			t.Error(err)
		}
	})
	values := []float64{1 + math.Ldexp(1, -32), -2, 3, 4, 5}
	if err := writeMathTypedMemory(input, values); err != nil {
		t.Fatal(err)
	}
	complexValues := []Complex128{{Real: values[0], Imag: .25}, {}, {}, {}, {}}
	if err := writeMathTypedMemory(complexInput, complexValues); err != nil {
		t.Fatal(err)
	}
	batchedValues := make([]Complex128, 12)
	batchedValues[0] = Complex128{Real: values[0]}
	batchedValues[6] = Complex128{Real: 2}
	if err := writeMathTypedMemory(batchedInput, batchedValues); err != nil {
		t.Fatal(err)
	}
	if wrongDevice != nil {
		if err := ctx.ExecZ2Z(plans[2], complexInput, wrongDevice, FFTForward); err == nil {
			t.Fatal("expected cross-device output to fail")
		}
	}
	if err := ctx.ExecD2Z(plans[0], input, spectrum); err != nil {
		t.Fatal(err)
	}
	if err := ctx.ExecZ2D(plans[1], spectrum, output); err != nil {
		t.Fatal(err)
	}
	if err := ctx.ExecZ2Z(plans[2], complexInput, complexOutput, FFTForward); err != nil {
		t.Fatal(err)
	}
	if err := ctx.ExecZ2Z(many, batchedInput, batchedOutput, FFTForward); err != nil {
		t.Fatal(err)
	}
	for _, buffer := range []*memory.Memory{input, spectrum, output, complexInput, complexOutput, batchedInput, batchedOutput} {
		if err := buffer.Free(); err == nil {
			t.Fatal("expected queued FFT buffers to remain retained")
		}
	}
	if err := stream.Synchronize(); err != nil {
		t.Fatal(err)
	}
	got, err := readMathTypedMemory[float64](output, len(values))
	if err != nil {
		t.Fatal(err)
	}
	for i, value := range got {
		if diff := math.Abs(value - 5*values[i]); math.IsNaN(diff) || diff > 1e-11 {
			t.Fatalf("round trip[%d] = %v, want %v", i, value, 5*values[i])
		}
	}
	transformed, err := readMathTypedMemory[Complex128](complexOutput, len(complexValues))
	if err != nil {
		t.Fatal(err)
	}
	for i, value := range transformed {
		if diff := cmplx.Abs(complex(value.Real, value.Imag) - complex(values[0], .25)); math.IsNaN(diff) || diff > 1e-12 {
			t.Fatalf("complex output[%d] = %+v, want %+v", i, value, complexValues[0])
		}
	}
	batchedResult, err := readMathTypedMemory[Complex128](batchedOutput, len(batchedValues))
	if err != nil {
		t.Fatal(err)
	}
	for i, value := range batchedResult {
		want := batchedValues[0].Real
		if i >= 6 {
			want = 2
		}
		if diff := cmplx.Abs(complex(value.Real-want, value.Imag)); math.IsNaN(diff) || diff > 1e-12 {
			t.Fatalf("batched output[%d] = %+v, want %v", i, value, want)
		}
	}
}
