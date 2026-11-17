//go:build cuda

package libraries

import (
	"testing"

	cuda "github.com/stitch1968/gocuda"
	"github.com/stitch1968/gocuda/kernels"
	"github.com/stitch1968/gocuda/streams"
)

func TestCuDSSNativePhaseWaitsForDefaultStream(t *testing.T) {
	if err := cuda.Initialize(); err != nil {
		t.Fatal(err)
	}
	if !cuda.ShouldUseCuda() || !cudssNativeAvailable() {
		t.Skip("native cuDSS unavailable")
	}
	devices, err := cuda.GetDevices()
	if err != nil {
		t.Fatal(err)
	}
	if devices[0].Properties.Major < 7 {
		t.Skip("the delay kernel requires compute capability 7.0 or newer")
	}
	gate, err := streams.NewNativeStream(0, streams.StreamDefault)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := gate.Close(); err != nil {
			t.Error(err)
		}
	})
	event, err := streams.NewNativeEvent(0, streams.NativeEventDisableTiming)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := event.Close(); err != nil {
			t.Error(err)
		}
	})
	module, err := kernels.LoadPTX(0, nativeTestDelayPTX)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := module.Close(); err != nil {
			t.Error(err)
		}
	})
	delay, err := module.Function("wait_ticks")
	if err != nil {
		t.Fatal(err)
	}
	handle, err := CreateNativeDSSHandleOnDevice(DSSConfig{}, 0)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := handle.Destroy(); err != nil {
			t.Error(err)
		}
	})
	if err := delay.Launch(gate, kernels.Dim3{X: 1, Y: 1, Z: 1}, kernels.Dim3{X: 1, Y: 1, Z: 1}, 0, kernels.Uint64Arg(200_000_000)); err != nil {
		t.Fatal(err)
	}
	if err := event.Record(gate); err != nil {
		t.Fatal(err)
	}
	complete, err := event.Query()
	if err != nil {
		t.Fatal(err)
	}
	if complete {
		t.Skip("delay completed before default-stream synchronization could be tested")
	}
	handle.mu.Lock()
	pending, err := handle.runNativePhaseLocked(func() error { return nil })
	handle.mu.Unlock()
	if err != nil || pending {
		t.Fatalf("native phase pending = %v, error = %v", pending, err)
	}
	complete, err = event.Query()
	if err != nil || !complete {
		t.Fatalf("producer event complete = %v, error = %v", complete, err)
	}
}
