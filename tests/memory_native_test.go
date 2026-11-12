//go:build cuda

package tests

import (
	"testing"
	"unsafe"

	cuda "github.com/stitch1968/gocuda"
	"github.com/stitch1968/gocuda/internal"
	"github.com/stitch1968/gocuda/libraries"
	"github.com/stitch1968/gocuda/memory"
)

func TestDeviceOnlyNativeAllocationAndFFT(t *testing.T) {
	if !cuda.ShouldUseCuda() {
		t.Skip("CUDA hardware unavailable")
	}
	input, err := memory.AllocDeviceOnlyOnDevice(32, 0)
	if err != nil {
		t.Fatal(err)
	}
	defer input.Free()
	output, err := memory.AllocDeviceOnlyOnDevice(32, 0)
	if err != nil {
		t.Fatal(err)
	}
	defer output.Free()
	isDevice, err := internal.CudaPointerIsDevice(input.Ptr())
	if err != nil {
		t.Fatal(err)
	}
	if !isDevice {
		t.Fatal("expected cudaMalloc device-only allocation")
	}
	managed, err := memory.Alloc(32)
	if err != nil {
		t.Fatal(err)
	}
	defer managed.Free()
	isDevice, err = internal.CudaPointerIsDevice(managed.Ptr())
	if err != nil {
		t.Fatal(err)
	}
	if isDevice {
		t.Fatal("legacy allocation must remain managed")
	}

	values := []libraries.Complex64{{Real: 1}, {Real: 2}, {Real: 3}, {Real: 4}}
	inputBytes := unsafe.Slice((*byte)(unsafe.Pointer(&values[0])), 32)
	if err := memory.CopyHostToDevice(input, inputBytes); err != nil {
		t.Fatal(err)
	}
	ctx, err := libraries.CreateFFTContext()
	if err != nil {
		t.Fatal(err)
	}
	defer ctx.DestroyContext()
	plan, err := ctx.CreatePlan1D(4, libraries.FFTTypeC2C, 1)
	if err != nil {
		t.Fatal(err)
	}
	if err := ctx.ExecC2C(plan, input, output, libraries.FFTForward); err != nil {
		t.Fatal(err)
	}
	result := make([]libraries.Complex64, 4)
	resultBytes := unsafe.Slice((*byte)(unsafe.Pointer(&result[0])), 32)
	if err := memory.CopyDeviceToHost(resultBytes, output); err != nil {
		t.Fatal(err)
	}
	want := []libraries.Complex64{{Real: 10}, {Real: -2, Imag: 2}, {Real: -2}, {Real: -2, Imag: -2}}
	for i := range result {
		if result[i] != want[i] {
			t.Fatalf("output[%d] = %+v, want %+v", i, result[i], want[i])
		}
	}
}
