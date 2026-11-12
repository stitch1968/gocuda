package tests

import (
	"context"
	"errors"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	cuda "github.com/stitch1968/gocuda"
	"github.com/stitch1968/gocuda/streams"
)

func TestHostCallbackErrorsSurfaceOnSynchronize(t *testing.T) {
	ctx, err := cuda.NewContext(0)
	if err != nil {
		t.Fatal(err)
	}
	stream, err := ctx.NewStream()
	if err != nil {
		t.Fatal(err)
	}
	defer stream.Close()
	expected := errors.New("callback failed")
	if err := cuda.GoWithStream(stream, func(context.Context, ...any) error {
		return expected
	}); err != nil {
		t.Fatal(err)
	}
	if err := stream.Synchronize(); !errors.Is(err, expected) {
		t.Fatalf("synchronization error = %v, want %v", err, expected)
	}
	if err := stream.Synchronize(); err != nil {
		t.Fatalf("error should have been consumed: %v", err)
	}
}

func TestHostEventsMeasureQueuedWork(t *testing.T) {
	stream, err := streams.CreateStream()
	if err != nil {
		t.Fatal(err)
	}
	defer streams.GetManager().DestroyStream(stream)
	elapsed, err := streams.MeasureKernelTime(stream, func() {
		time.Sleep(20 * time.Millisecond)
	})
	if err != nil {
		t.Fatal(err)
	}
	if elapsed < 15*time.Millisecond {
		t.Fatalf("host event elapsed time = %v, want at least 15ms", elapsed)
	}
}

func TestGraphExecutesDependenciesAndReportsErrors(t *testing.T) {
	ctx, err := cuda.NewContext(0)
	if err != nil {
		t.Fatal(err)
	}
	stream, err := ctx.NewStream()
	if err != nil {
		t.Fatal(err)
	}
	defer stream.Close()
	graph, err := cuda.CreateGraph()
	if err != nil {
		t.Fatal(err)
	}
	var order []int
	first := &cuda.SimpleKernel{Func: func(args ...any) error {
		order = append(order, args[0].(int))
		return nil
	}}
	node, err := graph.AddKernelNode(first, nil, 1)
	if err != nil {
		t.Fatal(err)
	}
	expected := errors.New("graph kernel failed")
	second := &cuda.SimpleKernel{Func: func(args ...any) error {
		order = append(order, args[0].(int))
		return expected
	}}
	if _, err := graph.AddKernelNode(second, []*cuda.GraphNode{node}, 2); err != nil {
		t.Fatal(err)
	}
	if err := graph.Launch(stream); err != nil {
		t.Fatal(err)
	}
	if err := stream.Synchronize(); !errors.Is(err, expected) {
		t.Fatalf("graph error = %v, want %v", err, expected)
	}
	if len(order) != 2 || order[0] != 1 || order[1] != 2 {
		t.Fatalf("unexpected graph execution order: %v", order)
	}
	other, err := cuda.CreateGraph()
	if err != nil {
		t.Fatal(err)
	}
	if _, err := other.AddKernelNode(first, []*cuda.GraphNode{node}); err == nil {
		t.Fatal("expected cross-graph dependency to fail")
	}
}

func TestStreamCloseConcurrentWithSubmission(t *testing.T) {
	ctx, err := cuda.NewContext(0)
	if err != nil {
		t.Fatal(err)
	}
	stream, err := ctx.NewStream()
	if err != nil {
		t.Fatal(err)
	}
	var submitted, executed atomic.Int64
	start := make(chan struct{})
	var wg sync.WaitGroup
	for range 8 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			<-start
			for range 100 {
				err := cuda.GoWithStream(stream, func(context.Context, ...any) error {
					executed.Add(1)
					return nil
				})
				if err != nil {
					return
				}
				submitted.Add(1)
			}
		}()
	}
	close(start)
	if err := stream.Close(); err != nil {
		t.Fatal(err)
	}
	wg.Wait()
	if submitted.Load() != executed.Load() {
		t.Fatalf("submitted %d callbacks but executed %d", submitted.Load(), executed.Load())
	}
	if err := cuda.GoWithStream(stream, func(context.Context, ...any) error { return nil }); err == nil {
		t.Fatal("expected submission to a closed stream to fail")
	}
}
