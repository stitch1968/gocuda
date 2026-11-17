package libraries

import (
	"fmt"
	"math"
	"strings"
	"sync"
	"testing"

	"github.com/stitch1968/gocuda/memory"
)

func TestCuDSSConcurrentSolveAndDestroy(t *testing.T) {
	allocate := func(size int64) *memory.Memory {
		t.Helper()
		buffer, err := memory.Alloc(size)
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
	rows, columns, values := allocate(12), allocate(8), allocate(8)
	if err := writeInt32Memory(rows, []int32{0, 1, 2}, 3); err != nil {
		t.Fatal(err)
	}
	if err := writeInt32Memory(columns, []int32{0, 1}, 2); err != nil {
		t.Fatal(err)
	}
	if err := writeMathFloat32Memory(values, []float32{2, 4}); err != nil {
		t.Fatal(err)
	}
	const workers = 4
	var rhs, output [workers]*memory.Memory
	for i := range workers {
		rhs[i], output[i] = allocate(8), allocate(8)
		if err := writeMathFloat32Memory(rhs[i], []float32{float32(2 * (i + 1)), float32(4 * (i + 1))}); err != nil {
			t.Fatal(err)
		}
		if err := writeMathFloat32Memory(output[i], []float32{0, 0}); err != nil {
			t.Fatal(err)
		}
	}
	matrix, err := CreateDSSMatrix(2, 2, rows, columns, values, DSSMatrixFormatCSR, false)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := matrix.Destroy(); err != nil {
			t.Error(err)
		}
	})
	handle, err := CreateDSSHandle(DSSConfig{MatrixFormat: DSSMatrixFormatCSR})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := handle.Destroy(); err != nil {
			t.Error(err)
		}
	})
	if err := handle.Analyze(matrix); err != nil {
		t.Fatal(err)
	}
	if err := handle.Factor(matrix); err != nil {
		t.Fatal(err)
	}
	run := func(raceDestroy bool) {
		start := make(chan struct{})
		errs := make(chan error, workers+1)
		var wg sync.WaitGroup
		for i := range workers {
			wg.Add(1)
			go func(index int) {
				defer wg.Done()
				<-start
				for range 3 {
					_, err := handle.Solve(rhs[index], output[index], 1)
					if err != nil {
						if !raceDestroy || !strings.Contains(err.Error(), "matrix must be factored") {
							errs <- err
						}
						return
					}
					got, err := readMathFloat32Memory(output[index], 2)
					if err != nil {
						errs <- err
						return
					}
					want := float64(index + 1)
					if math.Abs(float64(got[0])-want) > 1e-5 || math.Abs(float64(got[1])-want) > 1e-5 {
						errs <- fmt.Errorf("worker %d got %v, want [%v %v]", index, got, want, want)
						return
					}
				}
			}(i)
		}
		if raceDestroy {
			wg.Add(1)
			go func() {
				defer wg.Done()
				<-start
				if err := handle.Destroy(); err != nil {
					errs <- err
				}
			}()
		}
		close(start)
		wg.Wait()
		close(errs)
		for err := range errs {
			t.Error(err)
		}
	}
	run(false)
	run(true)
}
