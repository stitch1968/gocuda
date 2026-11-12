package tests

import (
	"bytes"
	"sync"
	"testing"
	"unsafe"

	cuda "github.com/stitch1968/gocuda"
	"github.com/stitch1968/gocuda/memory"
)

func TestMemoryRetentionPreventsFree(t *testing.T) {
	mem, err := memory.Alloc(16)
	if err != nil {
		t.Fatal(err)
	}
	retention, err := mem.Retain()
	if err != nil {
		t.Fatal(err)
	}
	var workers sync.WaitGroup
	for range 8 {
		workers.Add(1)
		go func() {
			defer workers.Done()
			for range 100 {
				other, err := mem.Retain()
				if err != nil {
					t.Errorf("retain failed: %v", err)
					return
				}
				if err := mem.Free(); err == nil {
					t.Error("free succeeded while allocation was retained")
				}
				other.Release()
			}
		}()
	}
	workers.Wait()
	if err := mem.Free(); err == nil {
		t.Fatal("free succeeded while allocation was retained")
	}
	copyOfRetention := *retention
	retention.Release()
	copyOfRetention.Release()
	retention.Release()
	if err := mem.Free(); err != nil {
		t.Fatal(err)
	}
	if _, err := mem.Retain(); err == nil {
		t.Fatal("expected retaining freed memory to fail")
	}
}

func TestMemoryTypesAndTransfers(t *testing.T) {
	types := []memory.Type{memory.TypeDevice, memory.TypeHost, memory.TypePinned, memory.TypeUnified, memory.TypeDeviceOnly}
	for _, sourceType := range types {
		for _, destinationType := range types {
			source, err := memory.AllocWithTypeAndStream(nil, 16, sourceType)
			if err != nil {
				t.Fatal(err)
			}
			destination, err := memory.AllocWithTypeAndStream(nil, 16, destinationType)
			if err != nil {
				t.Fatal(err)
			}
			if uintptr(source.Ptr())%uintptr(source.GetAlignment()) != 0 {
				t.Fatalf("type %d returned incorrectly aligned memory", sourceType)
			}
			expected := []byte{1, 2, 3, 4, 5}
			if err := memory.CopyHostToDevice(source, expected); err != nil {
				t.Fatalf("type %d host copy: %v", sourceType, err)
			}
			if err := memory.CopyDeviceToDevice(destination, source); err != nil {
				t.Fatalf("type %d -> %d copy: %v", sourceType, destinationType, err)
			}
			got := make([]byte, len(expected))
			if err := memory.CopyDeviceToHost(got, destination); err != nil {
				t.Fatalf("type %d host copy: %v", destinationType, err)
			}
			if !bytes.Equal(got, expected) {
				t.Fatalf("type %d -> %d: got %v, want %v", sourceType, destinationType, got, expected)
			}
			if err := source.Free(); err != nil {
				t.Fatal(err)
			}
			if err := destination.Free(); err != nil {
				t.Fatal(err)
			}
		}
	}
}

func TestDeviceOnlyRequiresTransfers(t *testing.T) {
	mem, err := cuda.Alloc(16).DeviceOnly().Allocate()
	if err != nil {
		t.Fatal(err)
	}
	defer mem.Free()
	if mem.GetType() != memory.TypeDeviceOnly || mem.Data() != nil {
		t.Fatal("device-only allocation must not expose host data")
	}
	if _, err := memory.View[byte](mem, 1); err == nil {
		t.Fatal("expected host view of device-only memory to fail")
	}
	if _, err := memory.View[byte](mem, 0); err == nil {
		t.Fatal("expected even a zero-length device-only view to fail")
	}
	if err := cuda.ExecuteVectorAdd(mem, mem, mem, 1); err == nil {
		t.Fatal("host-based kernel must reject device-only memory")
	}
	if _, err := memory.AllocDeviceOnlyOnDevice(16, -1); err == nil {
		t.Fatal("expected invalid device ID to fail")
	}
	if _, err := cuda.NewContext(-1); err == nil {
		t.Fatal("expected invalid context device ID to fail")
	}
}

func TestFreedAllocationsRejectTransfers(t *testing.T) {
	mem, err := memory.Alloc(16)
	if err != nil {
		t.Fatal(err)
	}
	if err := mem.Free(); err != nil {
		t.Fatal(err)
	}
	if err := memory.CopyHostToDevice(mem, []byte{1}); err == nil {
		t.Fatal("expected copy to freed allocation to fail")
	}
	if err := memory.CopyDeviceToHost(make([]byte, 1), mem); err == nil {
		t.Fatal("expected copy from freed allocation to fail")
	}
	if _, err := memory.View[byte](mem, 1); err == nil {
		t.Fatal("expected view of freed allocation to fail")
	}
}

func TestMemoryTypeValidation(t *testing.T) {
	for _, kind := range []memory.Type{memory.TypeShared, memory.TypeConstant, memory.TypeTexture, -1} {
		if _, err := memory.AllocWithTypeAndStream(nil, 16, kind); err == nil {
			t.Fatalf("expected type %d to be rejected", kind)
		}
	}
	mem, err := memory.Alloc(16)
	if err != nil {
		t.Fatal(err)
	}
	defer mem.Free()
	if _, err := memory.View[uint64](mem, int(^uint(0)>>1)/int(unsafe.Sizeof(uint64(0)))+1); err == nil {
		t.Fatal("expected overflowing view to fail")
	}
}
