package libraries

import (
	"errors"
	"testing"

	cuda "github.com/stitch1968/gocuda"
	"github.com/stitch1968/gocuda/streams"
)

func TestCuDSSNativeHandlePolicy(t *testing.T) {
	handle, err := CreateDSSHandle(DSSConfig{MatrixFormat: DSSMatrixFormatCSR})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := handle.Destroy(); err != nil {
			t.Error(err)
		}
	})
	want := ImplementationModeValidatedGo
	if cuda.ShouldUseCuda() && cudssNativeAvailable() {
		want = ImplementationModeNativeCUDA
	}
	if got := handle.Backend(); got != want {
		t.Fatalf("backend = %q, want %q", got, want)
	}
	strict := &DSSHandle{native: true, strictNative: true}
	if err := strict.Analyze(&DSSMatrix{}); !errors.Is(err, ErrDSSNativeOperationUnsupported) {
		t.Fatalf("strict fallback error = %v", err)
	}
	if !cuda.ShouldUseCuda() || !cudssNativeAvailable() {
		if _, err := CreateNativeDSSHandleOnDevice(DSSConfig{}, 0); !errors.Is(err, streams.ErrNativeCUDAUnavailable) {
			t.Fatalf("native constructor error = %v", err)
		}
	} else {
		if _, err := CreateNativeDSSHandleOnDevice(DSSConfig{}, -1); err == nil {
			t.Fatal("expected invalid device ID to fail")
		}
		for _, config := range []DSSConfig{
			{MatrixFormat: DSSMatrixFormatCOO},
			{Factorization: DSSFactorizationQR},
			{Factorization: DSSFactorizationLDLT},
			{Factorization: DSSFactorizationCholesky},
			{Symmetry: true},
			{Ordering: DSSOrderingAMD},
			{Refinement: DSSRefinementSingle},
			{PivotType: DSSPivotPartial},
			{PivotThreshold: 0.1},
		} {
			if _, err := CreateNativeDSSHandleOnDevice(config, 0); !errors.Is(err, ErrDSSNativeOperationUnsupported) {
				t.Fatalf("unsupported native configuration %+v error = %v", config, err)
			}
		}
	}
}
