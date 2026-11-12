package libraries

import (
	"testing"

	"github.com/stitch1968/gocuda/streams"
)

func TestSimulatedRandomGeneratorRejectsNativeStream(t *testing.T) {
	rg := &RandomGenerator{}
	if err := rg.SetStream(nil); err != nil {
		t.Fatal(err)
	}
	if err := rg.SetStream(&streams.NativeStream{}); err == nil {
		t.Fatal("simulation must reject native stream attachment")
	}
}
