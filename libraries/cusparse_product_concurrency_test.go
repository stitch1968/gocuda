package libraries

import (
	"fmt"
	"sync"
	"testing"
)

func TestSparseContextConcurrentSpGEMM(t *testing.T) {
	ctx, err := CreateSparseContext()
	if err != nil {
		t.Fatal(err)
	}

	defer ctx.DestroyContext()
	a := sparseTestCSR(t, ctx, 2, 2, []int32{0, 2, 3}, []int32{0, 1, 1}, []float32{2, 1, 3})
	const workers = 4
	var wg sync.WaitGroup
	errs := make(chan error, workers)
	for range workers {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for range 3 {
				product, err := ctx.SpGEMM(a, a)
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
		}()
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
	for range workers {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for range 3 {
				matrix, err := ctx.DenseToSparse(input, 2, 3, MatrixFormatCSR)
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
		}()
	}
	wg.Wait()
	close(errs)
	for err := range errs {
		t.Error(err)
	}
}
