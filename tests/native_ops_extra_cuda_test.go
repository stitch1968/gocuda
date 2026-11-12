//go:build cuda

package tests

import (
	"testing"

	cuda "github.com/stitch1968/gocuda"
	"github.com/stitch1968/gocuda/kernels"
	"github.com/stitch1968/gocuda/memory"
	"github.com/stitch1968/gocuda/streams"
)

func TestNativeOpsSumAndConvolution(t *testing.T) {
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
	ops, err := kernels.NewNativeOps(0)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := ops.Close(); err != nil {
			t.Error(err)
		}
	})
	alloc := func(size int, kind memory.Type) *memory.Memory {
		t.Helper()
		var buffer *memory.Memory
		var err error
		if kind == memory.TypeDeviceOnly {
			buffer, err = memory.AllocDeviceOnlyOnDevice(int64(size*4), 0)
		} else {
			buffer, err = memory.AllocWithTypeAndStream(nil, int64(size*4), kind)
		}
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() {
			if err := buffer.Free(); err != nil {
				t.Error(err)
			}
		})
		return buffer
	}
	inputHost := alloc(300, memory.TypePinned)
	input := alloc(300, memory.TypeDeviceOnly)
	sumHost := alloc(1, memory.TypePinned)
	sum := alloc(1, memory.TypeDeviceOnly)
	convHost := alloc(12, memory.TypePinned)
	filterHost := alloc(6, memory.TypePinned)
	convOutHost := alloc(12, memory.TypePinned)
	convInput := alloc(12, memory.TypeDeviceOnly)
	filter := alloc(6, memory.TypeDeviceOnly)
	convOut := alloc(12, memory.TypeDeviceOnly)
	t.Cleanup(func() {
		if err := stream.Synchronize(); err != nil {
			t.Error(err)
		}
	})

	inputValues, err := memory.View[float32](inputHost, 300)
	if err != nil {
		t.Fatal(err)
	}
	var expectedSum float32
	for i := range inputValues {
		inputValues[i] = float32(i - 150)
		expectedSum += inputValues[i]
	}
	if err := stream.CopyAsync(input, inputHost, 300*4); err != nil {
		t.Fatal(err)
	}
	if err := ops.Sum(stream, input, input, 300); err == nil {
		t.Fatal("expected sum alias to fail")
	}
	if err := ops.Sum(stream, input, sum, 300); err != nil {
		t.Fatal(err)
	}
	if err := stream.CopyAsync(sumHost, sum, 4); err != nil {
		t.Fatal(err)
	}

	convValues, err := memory.View[float32](convHost, 12)
	if err != nil {
		t.Fatal(err)
	}
	filterValues, err := memory.View[float32](filterHost, 6)
	if err != nil {
		t.Fatal(err)
	}
	for i := range convValues {
		convValues[i] = float32(i + 1)
	}
	copy(filterValues, []float32{1, -1, 2, 0, 1, -2})
	if err := stream.CopyAsync(convInput, convHost, 12*4); err != nil {
		t.Fatal(err)
	}
	if err := stream.CopyAsync(filter, filterHost, 6*4); err != nil {
		t.Fatal(err)
	}
	if err := ops.Convolution2D(stream, convInput, filter, convInput, 4, 3, 3, 2, 4, 3); err == nil {
		t.Fatal("expected convolution alias to fail")
	}
	if err := ops.Convolution2D(stream, convInput, filter, convOut, 4, 3, 3, 2, 4, 3); err != nil {
		t.Fatal(err)
	}
	if err := stream.CopyAsync(convOutHost, convOut, 12*4); err != nil {
		t.Fatal(err)
	}
	if err := stream.Synchronize(); err != nil {
		t.Fatal(err)
	}
	sumValues, err := memory.View[float32](sumHost, 1)
	if err != nil {
		t.Fatal(err)
	}
	if sumValues[0] != expectedSum {
		t.Fatalf("sum = %v, want %v", sumValues[0], expectedSum)
	}
	result, err := memory.View[float32](convOutHost, 12)
	if err != nil {
		t.Fatal(err)
	}
	for y := range 3 {
		for x := range 4 {
			var want float32
			for fy := range 2 {
				for fx := range 3 {
					if y+fy < 3 && x+fx < 4 {
						want += convValues[(y+fy)*4+x+fx] * filterValues[fy*3+fx]
					}
				}
			}
			if got := result[y*4+x]; got != want {
				t.Fatalf("convolution[%d,%d] = %v, want %v", y, x, got, want)
			}
		}
	}
}
