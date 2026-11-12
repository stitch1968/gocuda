//go:build cuda

package tests

import (
	"math"
	"testing"

	cuda "github.com/stitch1968/gocuda"
	"github.com/stitch1968/gocuda/kernels"
	"github.com/stitch1968/gocuda/memory"
	"github.com/stitch1968/gocuda/streams"
)

func TestNativeGraphMemcpyAndMemset(t *testing.T) {
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
	module, err := kernels.LoadPTX(0, vectorAddPTX)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := module.Close(); err != nil {
			t.Error(err)
		}
	})
	function, err := module.Function("vector_add")
	if err != nil {
		t.Fatal(err)
	}
	graph, err := kernels.NewNativeGraph(0)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := graph.Close(); err != nil {
			t.Error(err)
		}
	})
	alloc := func(kind memory.Type) *memory.Memory {
		t.Helper()
		var buffer *memory.Memory
		var err error
		if kind == memory.TypeDeviceOnly {
			buffer, err = memory.AllocDeviceOnlyOnDevice(32, 0)
		} else {
			buffer, err = memory.AllocWithTypeAndStream(nil, 32, kind)
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
	hostInput, hostOutput := alloc(memory.TypePinned), alloc(memory.TypePinned)
	input, fill, output := alloc(memory.TypeDeviceOnly), alloc(memory.TypeUnified), alloc(memory.TypeDeviceOnly)
	mirror := alloc(memory.TypeUnified)
	t.Cleanup(func() {
		if err := stream.Synchronize(); err != nil {
			t.Error(err)
		}
		if err := graph.Close(); err != nil {
			t.Error(err)
		}
	})
	if _, err := graph.AddMemsetNode(hostInput, 0, 4, 8, nil); err == nil {
		t.Fatal("expected pinned host memset to be rejected")
	}
	if _, err := graph.AddMemsetNode(fill, 0, 3, 8, nil); err == nil {
		t.Fatal("expected invalid memset element size to fail")
	}
	if _, err := graph.AddMemcpyNode(input, hostInput, 33, nil); err == nil {
		t.Fatal("expected oversized graph copy to fail")
	}
	if _, err := graph.AddMemcpyNode(hostOutput, hostInput, 32, nil); err == nil {
		t.Fatal("expected host-to-host graph copy to be rejected")
	}
	upload, err := graph.AddMemcpyNode(input, hostInput, 32, nil)
	if err != nil {
		t.Fatal(err)
	}
	copyToManaged, err := graph.AddMemcpyNode(mirror, input, 32, []*kernels.NativeGraphNode{upload})
	if err != nil {
		t.Fatal(err)
	}
	memset, err := graph.AddMemsetNode(fill, math.Float32bits(5), 4, 8, nil)
	if err != nil {
		t.Fatal(err)
	}
	grid, block := kernels.Dim3{X: 1, Y: 1, Z: 1}, kernels.Dim3{X: 32, Y: 1, Z: 1}
	compute, err := graph.AddKernelNode(function, []*kernels.NativeGraphNode{copyToManaged, memset}, grid, block, 0,
		kernels.BufferArg(mirror, 0), kernels.BufferArg(fill, 0), kernels.BufferArg(output, 0), kernels.Uint32Arg(8))
	if err != nil {
		t.Fatal(err)
	}
	download, err := graph.AddMemcpyNode(hostOutput, output, 32, []*kernels.NativeGraphNode{compute})
	if err != nil {
		t.Fatal(err)
	}
	for _, edge := range [][2]*kernels.NativeGraphNode{{copyToManaged, upload}, {compute, copyToManaged}, {compute, memset}, {download, compute}} {
		if exists, err := edge[0].DependsOn(edge[1]); err != nil || !exists {
			t.Fatalf("expected graph dependency, got %v, error %v", exists, err)
		}
	}
	if err := graph.Instantiate(); err != nil {
		t.Fatal(err)
	}
	inputValues, err := memory.View[float32](hostInput, 8)
	if err != nil {
		t.Fatal(err)
	}
	outputValues, err := memory.View[float32](hostOutput, 8)
	if err != nil {
		t.Fatal(err)
	}
	for iteration := range 3 {
		for i := range inputValues {
			inputValues[i] = float32(i + iteration)
		}
		if err := graph.Launch(stream); err != nil {
			t.Fatal(err)
		}
		if err := stream.Synchronize(); err != nil {
			t.Fatal(err)
		}
		for i, value := range outputValues {
			if want := inputValues[i] + 5; value != want {
				t.Fatalf("iteration %d output[%d] = %v, want %v", iteration, i, value, want)
			}
		}
	}
	for i := range inputValues {
		inputValues[i] = float32(i + 10)
	}
	if err := graph.Launch(stream); err != nil {
		t.Fatal(err)
	}
	if err := graph.Close(); err != nil {
		t.Fatal(err)
	}
	for i, value := range outputValues {
		if want := inputValues[i] + 5; value != want {
			t.Fatalf("close did not wait for graph output[%d] = %v, want %v", i, value, want)
		}
		if err := input.Free(); err != nil {
			t.Fatalf("graph close did not release its buffer: %v", err)
		}
	}
}
