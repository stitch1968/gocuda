package libraries

import (
	"fmt"
	"sync"
	"testing"

	"github.com/stitch1968/gocuda/memory"
)

func TestSolverContextConcurrentExecution(t *testing.T) {
	ctx, err := CreateSolverContext()
	if err != nil {
		t.Fatal(err)
	}
	const workers = 4
	buffers := make([]*memory.Memory, workers)
	for i := range buffers {
		buffers[i] = solverTestBuffer(t, []float32{1, 0, 0, 1})
	}
	var wg sync.WaitGroup
	errs := make(chan error, workers)
	for i := range workers {
		wg.Add(1)
		go func(index int) {
			defer wg.Done()
			for range 4 {
				if err := ctx.CholeskyFactorization(buffers[index], 2); err != nil {
					errs <- err
					return
				}
			}
			values, err := readMathFloat32Memory(buffers[index], 4)
			if err != nil {
				errs <- err
				return
			}
			if values[0] != 1 || values[3] != 1 {
				errs <- fmt.Errorf("worker %d got %v", index, values)
			}
		}(i)
	}
	wg.Wait()
	close(errs)
	for err := range errs {
		t.Error(err)
	}
	if err := ctx.DestroyContext(); err != nil {
		t.Fatal(err)
	}
	if err := ctx.CholeskyFactorization(buffers[0], 2); err == nil {
		t.Fatal("expected destroyed context to reject execution")
	}
}
