package tests

import (
	"testing"

	"github.com/stitch1968/gocuda/memory"
	"github.com/stitch1968/gocuda/performance"
)

func TestMultiGPUP2PCopy(t *testing.T) {
	multiGPU, err := performance.NewMultiGPU()
	if err != nil {
		t.Fatalf("Failed to create multi-GPU manager: %v", err)
	}
	if err := multiGPU.EnableP2P(); err != nil {
		t.Fatalf("Failed to enable P2P: %v", err)
	}

	src, err := memory.AllocOnDevice(4, 0)
	if err != nil {
		t.Fatalf("Failed to allocate source memory: %v", err)
	}
	defer src.Free()

	dst, err := memory.AllocOnDevice(4, 1)
	if err != nil {
		t.Fatalf("Failed to allocate destination memory: %v", err)
	}
	defer dst.Free()

	if err := memory.CopyHostToDevice(src, []byte{9, 8, 7, 6}); err != nil {
		t.Fatalf("Failed to seed source memory: %v", err)
	}

	if err := multiGPU.P2PCopy(0, 1, src, dst, 4); err != nil {
		t.Fatalf("P2P copy failed: %v", err)
	}

	host := make([]byte, 4)
	if err := memory.CopyDeviceToHost(host, dst); err != nil {
		t.Fatalf("Failed to copy destination back to host: %v", err)
	}

	want := []byte{9, 8, 7, 6}
	for i := range want {
		if host[i] != want[i] {
			t.Fatalf("unexpected copied byte at %d: got %d want %d", i, host[i], want[i])
		}
	}
}
