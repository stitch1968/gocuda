package libraries

import (
	"errors"
	"testing"

	cuda "github.com/stitch1968/gocuda"
	"github.com/stitch1968/gocuda/streams"
)

func TestSparseBackendReporting(t *testing.T) {
	ctx, err := CreateSparseContext()
	if err != nil {
		t.Fatal(err)
	}
	defer ctx.DestroyContext()
	want := ImplementationModeValidatedGo
	if cuda.ShouldUseCuda() {
		want = ImplementationModeNativeCUDA
	}
	if got, err := ctx.SpMVBackend(MatrixFormatCSR); err != nil || got != want {
		t.Fatalf("CSR backend = %q, error = %v, want %q", got, err, want)
	}
	if got, err := ctx.SpMVBackend(MatrixFormatCOO); err != nil || got != want {
		t.Fatalf("COO backend = %q, error = %v, want %q", got, err, want)
	}
	if _, err := ctx.SpMVBackend(MatrixFormatCSC); err == nil {
		t.Fatal("expected unsupported sparse format to fail")
	}
	if got, err := ctx.SpSVBackend(MatrixFormatCSR); err != nil || got != want {
		t.Fatalf("CSR triangular solve backend = %q, error = %v, want %q", got, err, want)
	}
	if got, err := ctx.SpSVBackend(MatrixFormatCOO); err != nil || got != ImplementationModeValidatedGo {
		t.Fatalf("COO triangular solve backend = %q, error = %v", got, err)
	}
	if got, err := ctx.SpGEMMBackend(MatrixFormatCSR, MatrixFormatCSR); err != nil || got != want {
		t.Fatalf("CSR product backend = %q, error = %v, want %q", got, err, want)
	}
	if got, err := ctx.SpGEMMBackend(MatrixFormatCOO, MatrixFormatCSR); err != nil || got != ImplementationModeValidatedGo {
		t.Fatalf("mixed product backend = %q, error = %v", got, err)
	}
	for _, format := range []MatrixFormat{MatrixFormatCSR, MatrixFormatCOO} {
		if got, err := ctx.SparseToDenseBackend(format); err != nil || got != want {
			t.Fatalf("sparse-to-dense backend = %q, error = %v, want %q", got, err, want)
		}
	}
	if got, err := ctx.DenseToSparseBackend(MatrixFormatCSR); err != nil || got != want {
		t.Fatalf("dense-to-CSR backend = %q, error = %v, want %q", got, err, want)
	}
	if got, err := ctx.DenseToSparseBackend(MatrixFormatCOO); err != nil || got != ImplementationModeValidatedGo {
		t.Fatalf("dense-to-COO backend = %q, error = %v", got, err)
	}
	if _, err := ctx.DenseToSparseBackend(MatrixFormatCSC); err == nil {
		t.Fatal("expected unsupported dense-to-sparse format to fail")
	}
	if !cuda.ShouldUseCuda() {
		if _, err := CreateNativeSparseContextOnDevice(0); !errors.Is(err, streams.ErrNativeCUDAUnavailable) {
			t.Fatalf("native sparse error = %v, want CUDA unavailable", err)
		}
	} else if _, err := CreateNativeSparseContextOnDevice(-1); err == nil {
		t.Fatal("expected invalid native sparse device to fail")
	}
}
