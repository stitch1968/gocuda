//go:build cuda

package tests

import (
	"errors"
	"testing"
	"unsafe"

	cuda "github.com/stitch1968/gocuda"
	"github.com/stitch1968/gocuda/memory"
	"github.com/stitch1968/gocuda/streams"
)

func TestNativeStreamRetainsAsyncCopyBuffers(t *testing.T) {
	if err := cuda.Initialize(); err != nil {
		t.Fatal(err)
	}
	if !cuda.ShouldUseCuda() {
		t.Skip("CUDA hardware unavailable")
	}
	stream, err := streams.NewNativeStream(0, streams.StreamNonBlocking)
	if err != nil {
		t.Fatal(err)
	}
	defer stream.Close()
	host, err := memory.AllocWithTypeAndStream(nil, 64, memory.TypePinned)
	if err != nil {
		t.Fatal(err)
	}
	defer host.Free()
	device, err := memory.AllocDeviceOnlyOnDevice(64, 0)
	if err != nil {
		t.Fatal(err)
	}
	defer device.Free()
	defer func() {
		if err := stream.Synchronize(); err != nil {
			t.Error(err)
		}
	}()
	if err := stream.CopyAsync(device, host, 64); err != nil {
		t.Fatal(err)
	}
	if err := host.Free(); err == nil {
		t.Fatal("expected pinned input to remain retained")
	}
	if err := device.Free(); err == nil {
		t.Fatal("expected device output to remain retained")
	}
	if err := stream.Synchronize(); err != nil {
		t.Fatal(err)
	}
	expected := errors.New("submission failed")
	if err := stream.EnqueueWithBuffers([]*memory.Memory{device}, func(unsafe.Pointer) error { return expected }); !errors.Is(err, expected) {
		t.Fatalf("submission error = %v, want %v", err, expected)
	}
	if err := device.Free(); err == nil {
		t.Fatal("expected buffer retention even after submission error")
	}
	if err := stream.Synchronize(); err != nil {
		t.Fatal(err)
	}
	if err := host.Free(); err != nil {
		t.Fatalf("free input after synchronization: %v", err)
	}
	if err := device.Free(); err != nil {
		t.Fatalf("free output after synchronization: %v", err)
	}
}
