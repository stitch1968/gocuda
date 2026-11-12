package tests

import (
	"testing"

	"github.com/stitch1968/gocuda/memory"
)

func TestMemoryPoolOwnership(t *testing.T) {
	initial := memory.GetManager().GetAllocationCount()
	pool := memory.NewPool()
	var allocations []*memory.Memory
	t.Cleanup(func() {
		if err := pool.Close(); err != nil {
			t.Error(err)
		}
		for _, allocation := range allocations {
			if err := allocation.Free(); err != nil {
				t.Error(err)
			}
		}
	})
	mem, err := pool.Get(32)
	if err != nil {
		t.Fatal(err)
	}
	allocations = append(allocations, mem)
	retention, err := mem.Retain()
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(retention.Release)
	if err := pool.Put(mem); err == nil {
		t.Fatal("expected retained memory to be rejected by pool")
	}
	retention.Release()
	if err := pool.Put(mem); err != nil {
		t.Fatal(err)
	}
	if err := pool.Put(mem); err == nil {
		t.Fatal("expected duplicate pool insertion to fail")
	}
	if err := mem.Free(); err == nil {
		t.Fatal("expected pooled allocation to reject Free")
	}
	if _, err := mem.Retain(); err == nil {
		t.Fatal("expected pooled allocation to reject Retain")
	}
	if mem.Ptr() != nil {
		t.Fatal("pooled memory must not expose a pointer to its former owner")
	}
	if _, err := memory.View[byte](mem, 1); err == nil {
		t.Fatal("expected pooled allocation to reject views")
	}
	if err := memory.CopyHostToDevice(mem, []byte{1}); err == nil {
		t.Fatal("expected pooled allocation to reject transfers")
	}
	reused, err := pool.Get(32)
	if err != nil {
		t.Fatal(err)
	}
	if reused != mem || reused.Ptr() == nil {
		t.Fatal("expected pool to return the cached live allocation")
	}
	checkedOut, err := pool.Get(16)
	if err != nil {
		t.Fatal(err)
	}
	allocations = append(allocations, checkedOut)
	if err := pool.Put(reused); err != nil {
		t.Fatal(err)
	}
	if err := pool.Close(); err != nil {
		t.Fatal(err)
	}
	if mem.Ptr() != nil {
		t.Fatal("expected pool close to free cached allocations")
	}
	if checkedOut.Ptr() == nil {
		t.Fatal("pool close must not free checked-out allocations")
	}
	if _, err := pool.Get(16); err == nil {
		t.Fatal("expected a closed pool to reject Get")
	}
	if err := pool.Put(checkedOut); err == nil {
		t.Fatal("expected a closed pool to reject Put")
	}
	if err := checkedOut.Free(); err != nil {
		t.Fatal(err)
	}
	if got := memory.GetManager().GetAllocationCount(); got != initial {
		t.Fatalf("allocation count = %d, want %d", got, initial)
	}
}
