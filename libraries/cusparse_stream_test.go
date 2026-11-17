package libraries

import (
	"testing"

	"github.com/stitch1968/gocuda/streams"
)

func TestSparseContextStreamSimulationPolicy(t *testing.T) {
	ctx := &SparseContext{}
	if err := ctx.SetStream(nil); err != nil {
		t.Fatal(err)
	}
	if err := ctx.Synchronize(); err != nil {
		t.Fatal(err)
	}
	if err := ctx.SetStream(&streams.NativeStream{}); err == nil {
		t.Fatal("simulation must reject native streams")
	}
	if err := ctx.DestroyContext(); err != nil {
		t.Fatal(err)
	}
	if err := ctx.SetStream(nil); err == nil {
		t.Fatal("destroyed context must reject stream changes")
	}
	if err := ctx.Synchronize(); err == nil {
		t.Fatal("destroyed context must reject synchronization")
	}
	var missing *SparseContext
	if err := missing.SetStream(nil); err == nil {
		t.Fatal("nil context must reject stream changes")
	}
	if err := missing.Synchronize(); err == nil {
		t.Fatal("nil context must reject synchronization")
	}
}
