//go:build cuda

package tests

import (
	"testing"

	cuda "github.com/stitch1968/gocuda"
	"github.com/stitch1968/gocuda/libraries"
	"github.com/stitch1968/gocuda/memory"
	"github.com/stitch1968/gocuda/streams"
)

func TestNativeFFTStreamChangeAndDestroySynchronize(t *testing.T) {
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
	t.Cleanup(func() {
		if err := stream.Close(); err != nil {
			t.Error(err)
		}
	})
	ctx, err := libraries.CreateFFTContext()
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := ctx.DestroyContext(); err != nil {
			t.Error(err)
		}
	})
	input, err := memory.AllocDeviceOnlyOnDevice(32, 0)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := input.Free(); err != nil {
			t.Error(err)
		}
	})
	first, err := memory.AllocDeviceOnlyOnDevice(32, 0)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := first.Free(); err != nil {
			t.Error(err)
		}
	})
	second, err := memory.AllocDeviceOnlyOnDevice(32, 0)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := second.Free(); err != nil {
			t.Error(err)
		}
	})
	host, err := memory.AllocWithTypeAndStream(nil, 32, memory.TypePinned)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := host.Free(); err != nil {
			t.Error(err)
		}
	})
	t.Cleanup(func() {
		if err := stream.Synchronize(); err != nil {
			t.Error(err)
		}
	})
	plan, err := ctx.CreatePlan1D(4, libraries.FFTTypeC2C, 1)
	if err != nil {
		t.Fatal(err)
	}
	if err := plan.SetStream(stream); err != nil {
		t.Fatal(err)
	}
	values, err := memory.View[libraries.Complex64](host, 4)
	if err != nil {
		t.Fatal(err)
	}
	copy(values, []libraries.Complex64{{Real: 1}, {Real: 2}, {Real: 3}, {Real: 4}})
	if err := stream.CopyAsync(input, host, 32); err != nil {
		t.Fatal(err)
	}
	if err := ctx.ExecC2C(plan, input, first, libraries.FFTForward); err != nil {
		t.Fatal(err)
	}
	if err := first.Free(); err == nil {
		t.Fatal("expected FFT output to remain retained")
	}
	if err := plan.SetStream(nil); err != nil {
		t.Fatal(err)
	}
	if err := first.Free(); err != nil {
		t.Fatalf("stream change did not release FFT output: %v", err)
	}
	if err := plan.SetStream(stream); err != nil {
		t.Fatal(err)
	}
	if err := ctx.ExecC2C(plan, input, second, libraries.FFTForward); err != nil {
		t.Fatal(err)
	}
	if err := plan.DestroyPlan(); err != nil {
		t.Fatal(err)
	}
	if err := second.Free(); err != nil {
		t.Fatalf("plan destruction did not release FFT output: %v", err)
	}
}
