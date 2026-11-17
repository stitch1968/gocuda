package libraries

import (
	"fmt"
	"sync"
	"testing"

	"github.com/stitch1968/gocuda/memory"
)

func TestSparseContextConcurrentDenseMMBatched(t *testing.T) {
	ctx, err := CreateSparseContext()
	if err != nil {
		t.Fatal(err)
	}
	defer ctx.DestroyContext()
	matrix, err := ctx.CreateCOOMatrix(2, 2, []int32{1, 0, 0}, []int32{1, 1, 0}, []float32{3, 1, 2})
	if err != nil {
		t.Fatal(err)
	}
	defer matrix.Destroy()
	input := solverTestBuffer(t, []float32{1, 2, 3, 4, 2, 0, 1, -1})
	const workers = 4
	outputs := make([]*memory.Memory, workers)
	for i := range outputs {
		outputs[i] = solverTestBuffer(t, make([]float32, 8))
	}
	var wg sync.WaitGroup
	errs := make(chan error, workers)
	for _, output := range outputs {
		wg.Add(1)
		go func(output *memory.Memory) {
			defer wg.Done()
			for range 3 {
				if err := ctx.SpMMDenseBatched(1, matrix, input, 2, 2, 2, BLASOpN, BLASOpN, 2, 4, 4, 1, output); err != nil {
					errs <- err
					return
				}
			}
			got, err := readMathFloat32Memory(output, 8)
			if err != nil {
				errs <- err
				return
			}
			want := []float32{15, 24, 27, 36, 15, -3, 9, -9}
			for i := range want {
				if got[i] != want[i] {
					errs <- fmt.Errorf("incorrect concurrent batch output: %v", got)
					return
				}
			}
		}(output)
	}
	wg.Wait()
	close(errs)
	for err := range errs {
		t.Error(err)
	}
}

func TestSparseContextConcurrentDenseMM(t *testing.T) {
	ctx, err := CreateSparseContext()
	if err != nil {
		t.Fatal(err)
	}
	defer ctx.DestroyContext()
	matrix, err := ctx.CreateCOOMatrix(2, 2, []int32{1, 0, 0}, []int32{1, 1, 0}, []float32{3, 1, 2})
	if err != nil {
		t.Fatal(err)
	}
	defer matrix.Destroy()
	input := solverTestBuffer(t, []float32{1, 2, 3, 4})
	const workers = 4
	outputs := make([]*memory.Memory, workers)
	for i := range outputs {
		outputs[i] = solverTestBuffer(t, []float32{0, 0, 0, 0})
	}
	var wg sync.WaitGroup
	errs := make(chan error, workers)
	for _, output := range outputs {
		wg.Add(1)
		go func(output *memory.Memory) {
			defer wg.Done()
			for range 3 {
				if err := ctx.SpMMDense(1, matrix, input, 2, 1, output); err != nil {
					errs <- err
					return
				}
			}
			got, err := readMathFloat32Memory(output, 4)
			if err != nil {
				errs <- err
				return
			}
			if got[0] != 15 || got[1] != 24 || got[2] != 27 || got[3] != 36 {
				errs <- fmt.Errorf("incorrect concurrent sparse-dense output: %v", got)
			}
		}(output)
	}
	wg.Wait()
	close(errs)
	for err := range errs {
		t.Error(err)
	}
}

func TestSparseContextConcurrentScaledProduct(t *testing.T) {
	ctx, err := CreateSparseContext()
	if err != nil {
		t.Fatal(err)
	}
	defer ctx.DestroyContext()
	matrix, err := ctx.CreateCOOMatrix(2, 2, []int32{1, 0, 0}, []int32{1, 1, 0}, []float32{3, 1, 2})
	if err != nil {
		t.Fatal(err)
	}
	defer matrix.Destroy()
	const workers = 4
	outputs := make([]*SparseMatrix, workers)
	for i := range outputs {
		outputs[i], err = ctx.CreateCOOMatrix(2, 2, []int32{0, 0, 1, 1}, []int32{0, 1, 0, 1}, []float32{1, 2, 3, 4})
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() {
			if err := outputs[i].Destroy(); err != nil {
				t.Error(err)
			}
		})
	}
	var wg sync.WaitGroup
	errs := make(chan error, workers)
	for _, output := range outputs {
		wg.Add(1)
		go func(output *SparseMatrix) {
			defer wg.Done()
			for range 3 {
				if err := ctx.SpMM(1, matrix, matrix, 1, output); err != nil {
					errs <- err
					return
				}
			}
			rows, err := readSparseRows(output)
			if err != nil {
				errs <- err
				return
			}
			if rows[0][0] != 13 || rows[0][1] != 17 || rows[1][0] != 3 || rows[1][1] != 31 {
				errs <- fmt.Errorf("incorrect concurrent scaled product: %v", rows)
			}
		}(output)
	}
	wg.Wait()
	close(errs)
	for err := range errs {
		t.Error(err)
	}
}

func TestSparseContextConcurrentSpGEMM(t *testing.T) {
	ctx, err := CreateSparseContext()
	if err != nil {
		t.Fatal(err)
	}

	defer ctx.DestroyContext()
	a := sparseTestCSR(t, ctx, 2, 2, []int32{0, 2, 3}, []int32{0, 1, 1}, []float32{2, 1, 3})
	coo, err := ctx.CreateCOOMatrix(2, 2, []int32{1, 0, 0}, []int32{1, 1, 0}, []float32{3, 1, 2})
	if err != nil {
		t.Fatal(err)
	}
	defer coo.Destroy()
	const workers = 4
	var wg sync.WaitGroup
	errs := make(chan error, workers)
	for worker := range workers {
		wg.Add(1)
		go func(worker int) {
			defer wg.Done()
			left, right := a, a
			if worker&1 != 0 {
				left = coo
			}
			if worker&2 != 0 {
				right = coo
			}
			for range 3 {
				product, err := ctx.SpGEMM(left, right)
				if err != nil {
					errs <- err
					return
				}
				rows, readErr := readSparseRows(product)
				destroyErr := product.Destroy()
				if readErr != nil {
					errs <- readErr
					return
				}
				if destroyErr != nil {
					errs <- destroyErr
					return
				}
				if rows[0][0] != 4 || rows[0][1] != 5 || rows[1][1] != 9 {
					errs <- fmt.Errorf("incorrect sparse product: %v", rows)
					return
				}
			}
		}(worker)
	}
	wg.Wait()
	close(errs)
	for err := range errs {
		t.Error(err)
	}
}

func TestSparseContextConcurrentConversions(t *testing.T) {
	ctx, err := CreateSparseContext()
	if err != nil {
		t.Fatal(err)
	}
	defer ctx.DestroyContext()
	input := solverTestBuffer(t, []float32{0, 2, 3, 0, 4, 0})
	const workers = 4
	var wg sync.WaitGroup
	errs := make(chan error, workers)
	for worker := range workers {
		wg.Add(1)
		go func(worker int) {
			defer wg.Done()
			format := MatrixFormatCSR
			if worker%2 != 0 {
				format = MatrixFormatCOO
			}
			for range 3 {
				matrix, err := ctx.DenseToSparse(input, 2, 3, format)
				if err != nil {
					errs <- err
					return
				}
				dense, err := ctx.SparseToDense(matrix)
				destroyErr := matrix.Destroy()
				if err != nil || destroyErr != nil {
					var freeErr error
					if dense != nil {
						freeErr = dense.Free()
					}
					errs <- fmt.Errorf("convert or destroy sparse matrix and free result: %v / %v / %v", err, destroyErr, freeErr)
					return
				}
				values, readErr := readMathFloat32Memory(dense, 6)
				freeErr := dense.Free()
				if readErr != nil || freeErr != nil {
					errs <- fmt.Errorf("read or free dense matrix: %v / %v", readErr, freeErr)
					return
				}
				if values[0] != 0 || values[1] != 2 || values[2] != 3 ||
					values[3] != 0 || values[4] != 4 || values[5] != 0 {
					errs <- fmt.Errorf("incorrect dense conversion: %v", values)
					return
				}
			}
		}(worker)
	}
	wg.Wait()
	close(errs)
	for err := range errs {
		t.Error(err)
	}
}
