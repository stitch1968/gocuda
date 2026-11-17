package libraries

import (
	"fmt"
	"sync"
	"testing"

	"github.com/stitch1968/gocuda/memory"
)

func TestSparseContextConcurrentSpSVTranspose(t *testing.T) {
	ctx, err := CreateSparseContext()
	if err != nil {
		t.Fatal(err)
	}
	defer ctx.DestroyContext()
	matrix, err := ctx.CreateCOOMatrix(2, 2, []int32{1, 0, 1}, []int32{1, 0, 0}, []float32{3, 2, 1})
	if err != nil {
		t.Fatal(err)
	}
	defer matrix.Destroy()
	const workers = 4
	inputs := make([]*memory.Memory, workers)
	outputs := make([]*memory.Memory, workers)
	for i := range workers {
		rhs := []float32{2, 7}
		if i%2 != 0 {
			rhs = []float32{4, 6}
		}
		inputs[i] = solverTestBuffer(t, rhs)
		outputs[i] = solverTestBuffer(t, []float32{0, 0})
	}
	var wg sync.WaitGroup
	errs := make(chan error, workers)
	for i := range workers {
		wg.Add(1)
		go func(index int) {
			defer wg.Done()
			op := BLASOpN
			if index%2 != 0 {
				op = BLASOpT
			}
			for range 3 {
				if err := ctx.SpSVEx(matrix, inputs[index], outputs[index], op); err != nil {
					errs <- err
					return
				}
			}
			got, err := readMathFloat32Memory(outputs[index], 2)
			if err != nil {
				errs <- err
				return
			}
			if got[0] != 1 || got[1] != 2 {
				errs <- fmt.Errorf("worker %d got transpose solve %v", index, got)
			}
		}(i)
	}
	wg.Wait()
	close(errs)
	for err := range errs {
		t.Error(err)
	}
}

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
				op := BLASOpN
				if index%2 != 0 {
					op = BLASOpT
				}
				if err := ctx.SpMVEx(1, matrices[index], inputs[index], 0, outputs[index], op); err != nil {
					errs <- err
					return
				}
			}
			values, err := readMathFloat32Memory(outputs[index], 2)
			if err != nil {
				errs <- err
				return
			}
			want := []float32{float32(2*(index+1) + 2), 6}
			if index%2 != 0 {
				want = []float32{float32(2 * (index + 1)), float32(index + 7)}
			}
			if values[0] != want[0] || values[1] != want[1] {
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
