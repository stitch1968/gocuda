//go:build !cuda

package tests

import (
	"errors"
	"testing"

	"github.com/stitch1968/gocuda/kernels"
	"github.com/stitch1968/gocuda/streams"
)

func TestNativeKernelRequiresCUDA(t *testing.T) {
	if _, err := kernels.LoadPTX(0, ".version 7.0"); !errors.Is(err, streams.ErrNativeCUDAUnavailable) {
		t.Fatalf("expected native CUDA unavailable, got %v", err)
	}
}
