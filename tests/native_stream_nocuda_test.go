//go:build !cuda

package tests

import (
	"errors"
	"testing"

	"github.com/stitch1968/gocuda/streams"
)

func TestNativeStreamRequiresCUDA(t *testing.T) {
	if _, err := streams.NewNativeStream(0, streams.StreamDefault); !errors.Is(err, streams.ErrNativeCUDAUnavailable) {
		t.Fatalf("expected native CUDA unavailable, got %v", err)
	}
	if _, err := streams.NewNativeEvent(0, streams.NativeEventDefault); !errors.Is(err, streams.ErrNativeCUDAUnavailable) {
		t.Fatalf("expected native CUDA unavailable, got %v", err)
	}
}
