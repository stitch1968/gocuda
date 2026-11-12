package libraries

import (
	"fmt"
	"sync"
	"testing"

	"github.com/stitch1968/gocuda/memory"
)

func TestSparseContextConcurrentSpMV(t *testing.T) {
	ctx, err := CreateSparseContext()
	if err != nil {
		t.Fatal(err)
	}
	const workers = 4
	matrices := make([]*SparseMatrix, workers)
	inputs := make([]*memory.Memory, workers)
	outputs := make([]*memory.Memory, workers)
	for i := range workers {
		matrices[i] = sparseTestCSR(t, ctx, 2, 2, []int32{0, 2, 3}, []int32{0, 1, 1}, []float32{2, 1, 3})
		inputs[i] = solverTestBuffer(t, []float32{float32(i + 1), 2})
		outputs[i] = solverTestBuffer(t, []float32{0, 0})
	}
	var wg sync.WaitGroup
	errs := make(chan error, workers)
	for i := range workers {
		wg.Add(1)
		go func(index int) {
			defer wg.Done()
			for range 4 {
				if err := ctx.SpMV(1, matrices[index], inputs[index], 0, outputs[index]); err != nil {
					errs <- err
					return
				}
			}
			values, err := readMathFloat32Memory(outputs[index], 2)
			if err != nil {
				errs <- err
				return
			}
			if values[0] != float32(2*(index+1)+2) || values[1] != 6 {
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
	if err := ctx.SpMV(1, matrices[0], inputs[0], 0, outputs[0]); err == nil {
		t.Fatal("expected destroyed sparse context to reject execution")
	}
}
