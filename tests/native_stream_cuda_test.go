//go:build cuda

package tests

import (
	"math"
	"testing"

	cuda "github.com/stitch1968/gocuda"
	"github.com/stitch1968/gocuda/libraries"
	"github.com/stitch1968/gocuda/memory"
	"github.com/stitch1968/gocuda/streams"
)

func TestNativeStreamFFTAndEventOrdering(t *testing.T) {
	if err := cuda.Initialize(); err != nil {
		t.Fatal(err)
	}
	if !cuda.ShouldUseCuda() {
		t.Skip("CUDA hardware unavailable")
	}
	producer, err := streams.NewNativeStream(0, streams.StreamNonBlocking)
	if err != nil {
		t.Fatal(err)
	}
	defer producer.Close()
	consumer, err := streams.NewNativeStream(0, streams.StreamNonBlocking)
	if err != nil {
		t.Fatal(err)
	}
	defer consumer.Close()
	start, err := streams.NewNativeEvent(0, streams.NativeEventDefault)
	if err != nil {
		t.Fatal(err)
	}
	defer start.Close()
	ready, err := streams.NewNativeEvent(0, streams.NativeEventDefault)
	if err != nil {
		t.Fatal(err)
	}
	defer ready.Close()
	done, err := streams.NewNativeEvent(0, streams.NativeEventDefault)
	if err != nil {
		t.Fatal(err)
	}
	defer done.Close()

	hostInput, err := memory.AllocWithTypeAndStream(nil, 32, memory.TypePinned)
	if err != nil {
		t.Fatal(err)
	}
	defer hostInput.Free()
	hostOutput, err := memory.AllocWithTypeAndStream(nil, 32, memory.TypePinned)
	if err != nil {
		t.Fatal(err)
	}
	defer hostOutput.Free()
	input, err := memory.AllocDeviceOnlyOnDevice(32, 0)
	if err != nil {
		t.Fatal(err)
	}
	defer input.Free()
	output, err := memory.AllocDeviceOnlyOnDevice(32, 0)
	if err != nil {
		t.Fatal(err)
	}
	defer output.Free()

	values, err := memory.View[libraries.Complex64](hostInput, 4)
	if err != nil {
		t.Fatal(err)
	}
	copy(values, []libraries.Complex64{{Real: 1}, {Real: 2}, {Real: 3}, {Real: 4}})
	ctx, err := libraries.CreateFFTContext()
	if err != nil {
		t.Fatal(err)
	}
	defer ctx.DestroyContext()
	plan, err := ctx.CreatePlan1D(4, libraries.FFTTypeC2C, 1)
	if err != nil {
		t.Fatal(err)
	}
	if err := plan.SetStream(producer); err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := producer.Synchronize(); err != nil {
			t.Error(err)
		}
		if err := consumer.Synchronize(); err != nil {
			t.Error(err)
		}
	}()
	if err := start.Record(producer); err != nil {
		t.Fatal(err)
	}
	if err := producer.CopyAsync(input, hostInput, 32); err != nil {
		t.Fatal(err)
	}
	if err := ctx.ExecC2C(plan, input, output, libraries.FFTForward); err != nil {
		t.Fatal(err)
	}
	if err := output.Free(); err == nil {
		t.Fatal("expected cuFFT output to remain retained")
	}
	if err := ready.Record(producer); err != nil {
		t.Fatal(err)
	}
	if err := consumer.WaitEvent(ready); err != nil {
		t.Fatal(err)
	}
	if err := consumer.CopyAsync(hostOutput, output, 32); err != nil {
		t.Fatal(err)
	}
	if err := done.Record(consumer); err != nil {
		t.Fatal(err)
	}
	if err := done.Synchronize(); err != nil {
		t.Fatal(err)
	}
	complete, err := ready.Query()
	if err != nil || !complete {
		t.Fatalf("producer event complete = %v, error = %v", complete, err)
	}
	elapsed, err := start.ElapsedTime(ready)
	if err != nil || elapsed < 0 {
		t.Fatalf("elapsed = %v, error = %v", elapsed, err)
	}
	result, err := memory.View[libraries.Complex64](hostOutput, 4)
	if err != nil {
		t.Fatal(err)
	}
	want := []libraries.Complex64{{Real: 10}, {Real: -2, Imag: 2}, {Real: -2}, {Real: -2, Imag: -2}}
	for i := range result {
		if math.Abs(float64(result[i].Real-want[i].Real)) > 1e-4 || math.Abs(float64(result[i].Imag-want[i].Imag)) > 1e-4 {
			t.Fatalf("result[%d] = %+v, want %+v", i, result[i], want[i])
		}
	}
}

func TestNativeStreamValidationAndClose(t *testing.T) {
	if err := cuda.Initialize(); err != nil {
		t.Fatal(err)
	}
	if !cuda.ShouldUseCuda() {
		t.Skip("CUDA hardware unavailable")
	}
	if _, err := streams.NewNativeStream(-1, streams.StreamDefault); err == nil {
		t.Fatal("expected invalid device error")
	}
	if _, err := streams.NewNativeStream(0, streams.StreamFlags(99)); err == nil {
		t.Fatal("expected invalid stream flags error")
	}
	stream, err := streams.NewNativeStream(0, streams.StreamDefault)
	if err != nil {
		t.Fatal(err)
	}
	host, err := memory.AllocWithTypeAndStream(nil, 16, memory.TypeHost)
	if err != nil {
		t.Fatal(err)
	}
	defer host.Free()
	device, err := memory.AllocDeviceOnlyOnDevice(16, 0)
	if err != nil {
		t.Fatal(err)
	}
	defer device.Free()
	if err := stream.CopyAsync(device, host, 16); err == nil {
		t.Fatal("expected pageable host memory to be rejected")
	}
	if err := stream.CopyAsync(device, device, 17); err == nil {
		t.Fatal("expected oversized copy to be rejected")
	}
	event, err := streams.NewNativeEvent(0, streams.NativeEventDefault)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := event.Query(); err == nil {
		t.Fatal("expected unrecorded event to fail")
	}
	if err := stream.WaitEvent(event); err == nil {
		t.Fatal("expected wait on unrecorded event to fail")
	}
	if err := event.Close(); err != nil {
		t.Fatal(err)
	}
	withoutTiming, err := streams.NewNativeEvent(0, streams.NativeEventDisableTiming)
	if err != nil {
		t.Fatal(err)
	}
	if err := withoutTiming.Record(stream); err != nil {
		t.Fatal(err)
	}
	if err := withoutTiming.Synchronize(); err != nil {
		t.Fatal(err)
	}
	if _, err := withoutTiming.ElapsedTime(withoutTiming); err == nil {
		t.Fatal("expected timing-disabled event to reject elapsed time")
	}
	if err := withoutTiming.Close(); err != nil {
		t.Fatal(err)
	}
	if err := stream.Close(); err != nil {
		t.Fatal(err)
	}
	if err := stream.Synchronize(); err == nil {
		t.Fatal("expected closed stream to fail")
	}
}
