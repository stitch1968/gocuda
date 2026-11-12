//go:build cuda

package tests

import (
	"testing"

	cuda "github.com/stitch1968/gocuda"
	"github.com/stitch1968/gocuda/kernels"
	"github.com/stitch1968/gocuda/memory"
	"github.com/stitch1968/gocuda/streams"
)

func TestNativeGraphDependenciesAndReplay(t *testing.T) {
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
	defer stream.Close()
	module, err := kernels.LoadPTX(0, vectorAddPTX)
	if err != nil {
		t.Fatal(err)
	}
	defer module.Close()
	function, err := module.Function("vector_add")
	if err != nil {
		t.Fatal(err)
	}
	graph, err := kernels.NewNativeGraph(0)
	if err != nil {
		t.Fatal(err)
	}
	defer graph.Close()
	otherGraph, err := kernels.NewNativeGraph(0)
	if err != nil {
		t.Fatal(err)
	}
	defer otherGraph.Close()
	hostA, err := memory.AllocWithTypeAndStream(nil, 32, memory.TypePinned)
	if err != nil {
		t.Fatal(err)
	}
	defer hostA.Free()
	hostB, err := memory.AllocWithTypeAndStream(nil, 32, memory.TypePinned)
	if err != nil {
		t.Fatal(err)
	}
	defer hostB.Free()
	hostOut, err := memory.AllocWithTypeAndStream(nil, 32, memory.TypePinned)
	if err != nil {
		t.Fatal(err)
	}
	defer hostOut.Free()
	a, err := memory.AllocDeviceOnlyOnDevice(32, 0)
	if err != nil {
		t.Fatal(err)
	}
	defer a.Free()
	b, err := memory.AllocDeviceOnlyOnDevice(32, 0)
	if err != nil {
		t.Fatal(err)
	}
	defer b.Free()
	intermediate, err := memory.AllocDeviceOnlyOnDevice(32, 0)
	if err != nil {
		t.Fatal(err)
	}
	defer intermediate.Free()
	output, err := memory.AllocDeviceOnlyOnDevice(32, 0)
	if err != nil {
		t.Fatal(err)
	}
	defer output.Free()
	defer func() {
		if err := stream.Synchronize(); err != nil {
			t.Error(err)
		}
		if err := graph.Close(); err != nil {
			t.Error(err)
		}
		if err := otherGraph.Close(); err != nil {
			t.Error(err)
		}
	}()
	grid, block := kernels.Dim3{X: 1, Y: 1, Z: 1}, kernels.Dim3{X: 32, Y: 1, Z: 1}
	orphan, err := memory.AllocDeviceOnlyOnDevice(32, 0)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := graph.AddKernelNode(function, nil, grid, block, 0, kernels.BufferArg(orphan, 0)); err == nil {
		t.Fatal("expected invalid graph kernel arguments to fail")
	}
	if err := orphan.Free(); err != nil {
		t.Fatalf("failed graph node must not retain its buffer: %v", err)
	}
	first, err := graph.AddKernelNode(function, nil, grid, block, 0,
		kernels.BufferArg(a, 0), kernels.BufferArg(b, 0), kernels.BufferArg(intermediate, 0), kernels.Uint32Arg(8))
	if err != nil {
		t.Fatal(err)
	}
	if err := module.Close(); err == nil {
		t.Fatal("expected graph to retain the module")
	}
	args := []kernels.NativeArg{kernels.BufferArg(intermediate, 0), kernels.BufferArg(b, 0), kernels.BufferArg(output, 0), kernels.Uint32Arg(8)}
	if _, err := graph.AddKernelNode(function, []*kernels.NativeGraphNode{first, first}, grid, block, 0, args...); err == nil {
		t.Fatal("expected duplicate dependencies to fail")
	}
	if _, err := otherGraph.AddKernelNode(function, []*kernels.NativeGraphNode{first}, grid, block, 0, args...); err == nil {
		t.Fatal("expected cross-graph dependency to fail")
	}
	second, err := graph.AddKernelNode(function, []*kernels.NativeGraphNode{first}, grid, block, 0, args...)
	if err != nil {
		t.Fatal(err)
	}
	if depends, err := second.DependsOn(first); err != nil || !depends {
		t.Fatalf("second node depends on first = %v, error = %v", depends, err)
	}
	if depends, err := first.DependsOn(second); err != nil || depends {
		t.Fatalf("first node depends on second = %v, error = %v", depends, err)
	}
	if err := graph.Launch(stream); err == nil {
		t.Fatal("expected launch before instantiation to fail")
	}
	if err := graph.Instantiate(); err != nil {
		t.Fatal(err)
	}
	if _, err := graph.AddKernelNode(function, nil, grid, block, 0, args...); err == nil {
		t.Fatal("expected graph mutation after instantiation to fail")
	}
	aValues, err := memory.View[float32](hostA, 8)
	if err != nil {
		t.Fatal(err)
	}
	bValues, err := memory.View[float32](hostB, 8)
	if err != nil {
		t.Fatal(err)
	}
	result, err := memory.View[float32](hostOut, 8)
	if err != nil {
		t.Fatal(err)
	}
	for iteration := range 3 {
		for i := range 8 {
			aValues[i] = float32(i + iteration)
			bValues[i] = float32(2*i + iteration)
		}
		if err := stream.CopyAsync(a, hostA, 32); err != nil {
			t.Fatal(err)
		}
		if err := stream.CopyAsync(b, hostB, 32); err != nil {
			t.Fatal(err)
		}
		if err := graph.Launch(stream); err != nil {
			t.Fatal(err)
		}
		if err := stream.CopyAsync(hostOut, output, 32); err != nil {
			t.Fatal(err)
		}
		if err := stream.Synchronize(); err != nil {
			t.Fatal(err)
		}
		for i, value := range result {
			want := aValues[i] + 2*bValues[i]
			if value != want {
				t.Fatalf("iteration %d result[%d] = %v, want %v", iteration, i, value, want)
			}
		}
	}
	if err := a.Free(); err == nil {
		t.Fatal("expected graph-retained buffer to reject Free")
	}
	if err := graph.Close(); err != nil {
		t.Fatal(err)
	}
	if err := a.Free(); err != nil {
		t.Fatalf("buffer should be releasable after graph close: %v", err)
	}
	if err := module.Close(); err != nil {
		t.Fatal(err)
	}
}
