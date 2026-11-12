//go:build !cuda

package tests

import (
	"errors"
	"testing"

	"github.com/stitch1968/gocuda/kernels"
	"github.com/stitch1968/gocuda/libraries"
	"github.com/stitch1968/gocuda/streams"
)

func TestNativeOpsRequiresCUDA(t *testing.T) {
	if _, err := kernels.NewNativeOps(0); !errors.Is(err, streams.ErrNativeCUDAUnavailable) {
		t.Fatalf("expected native CUDA unavailable, got %v", err)
	}

}

func TestNativeBLASRequiresStream(t *testing.T) {
	if _, err := libraries.NewNativeBLAS(nil); err == nil {
		t.Fatal("expected nil native stream to fail")
	}
}
