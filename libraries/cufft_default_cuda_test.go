//go:build cuda

package libraries

import (
	"testing"

	cuda "github.com/stitch1968/gocuda"
	"github.com/stitch1968/gocuda/memory"
)

func TestNativeFFTDefaultStreamCompletesBeforeReturn(t *testing.T) {
	if err := cuda.Initialize(); err != nil {
		t.Fatal(err)
	}
	if !cuda.ShouldUseCuda() {
		t.Skip("CUDA hardware unavailable")
	}
	ctx, err := CreateNativeFFTContextOnDevice(0)
	if err != nil {
		t.Fatal(err)
	}
	defer ctx.DestroyContext()
	plan, err := ctx.CreatePlan1D(4, FFTTypeC2C, 1)
	if err != nil {
		t.Fatal(err)
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
	if err := writeMathTypedMemory(input, []Complex64{{Real: 1}, {Real: 2}, {Real: 3}, {Real: 4}}); err != nil {
		t.Fatal(err)
	}
	if err := ctx.ExecC2C(plan, input, output, FFTForward); err != nil {
		t.Fatal(err)
	}
	if len(plan.defaultRetentions) != 0 {
		t.Fatal("default stream execution must release retentions before returning")
	}
	result, err := readMathTypedMemory[Complex64](output, 4)
	if err != nil {
		t.Fatal(err)
	}
	want := []Complex64{{Real: 10}, {Real: -2, Imag: 2}, {Real: -2}, {Real: -2, Imag: -2}}
	for i := range want {
		if result[i] != want[i] {
			t.Fatalf("output[%d] = %+v, want %+v", i, result[i], want[i])
		}
	}
	if err := input.Free(); err != nil {
		t.Fatalf("free input after execution: %v", err)
	}
	if err := output.Free(); err != nil {
		t.Fatalf("free output after execution: %v", err)
	}
}
