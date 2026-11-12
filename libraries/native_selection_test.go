package libraries

import (
	"errors"
	"testing"

	cuda "github.com/stitch1968/gocuda"
	"github.com/stitch1968/gocuda/streams"
)

func TestLibraryBackendReporting(t *testing.T) {
	ctx, err := CreateFFTContext()
	if err != nil {
		t.Fatal(err)
	}
	defer ctx.DestroyContext()
	generator, err := CreateRandomGenerator(RngTypeXorwow)
	if err != nil {
		t.Fatal(err)
	}
	defer generator.Destroy()
	solver, err := CreateSolverContext()
	if err != nil {
		t.Fatal(err)
	}
	defer solver.DestroyContext()
	expected := ImplementationModeValidatedGo
	if cuda.ShouldUseCuda() {
		expected = ImplementationModeNativeCUDA
	}
	if got := ctx.Backend(); got != expected {
		t.Fatalf("FFT backend = %q, want %q", got, expected)
	}
	if got := generator.Backend(); got != expected {
		t.Fatalf("random backend = %q, want %q", got, expected)
	}
	if got := solver.Backend(); got != expected {
		t.Fatalf("solver backend = %q, want %q", got, expected)
	}
	if !cuda.ShouldUseCuda() {
		if _, err := CreateNativeFFTContextOnDevice(0); !errors.Is(err, streams.ErrNativeCUDAUnavailable) {
			t.Fatalf("native FFT error = %v, want CUDA unavailable", err)
		}
		if _, err := CreateNativeRandomGeneratorOnDevice(0, RngTypeXorwow); !errors.Is(err, streams.ErrNativeCUDAUnavailable) {
			t.Fatalf("native random error = %v, want CUDA unavailable", err)
		}
		if _, err := CreateNativeSolverContextOnDevice(0); !errors.Is(err, streams.ErrNativeCUDAUnavailable) {
			t.Fatalf("native solver error = %v, want CUDA unavailable", err)
		}
	} else {
		if _, err := CreateNativeFFTContextOnDevice(-1); err == nil {
			t.Fatal("negative FFT device must fail")
		}
		if _, err := CreateNativeRandomGeneratorOnDevice(-1, RngTypeXorwow); err == nil {
			t.Fatal("negative random device must fail")
		}
		if _, err := CreateNativeSolverContextOnDevice(-1); err == nil {
			t.Fatal("negative solver device must fail")
		}
	}
}
