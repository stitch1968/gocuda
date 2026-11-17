package libraries

import (
	"errors"
	"sync"
	"testing"

	"github.com/stitch1968/gocuda/memory"
)

func TestCuDSSMatrixRetentionAndOwnership(t *testing.T) {
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
	matrix, err := CreateDSSMatrix(2, 2, rows, columns, values, DSSMatrixFormatCSR, false)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := matrix.Destroy(); err != nil {
			t.Error(err)
		}
	})
	first, err := CreateDSSHandle(DSSConfig{MatrixFormat: DSSMatrixFormatCSR})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := first.Destroy(); err != nil {
			t.Error(err)
		}
	})
	second, err := CreateDSSHandle(DSSConfig{MatrixFormat: DSSMatrixFormatCSR})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := second.Destroy(); err != nil {
			t.Error(err)
		}
	})
	for _, buffer := range []*memory.Memory{rows, columns, values} {
		if err := buffer.Free(); err == nil {
			t.Fatal("matrix must retain its input buffers")
		}
	}
	if err := first.Analyze(matrix); err != nil {
		t.Fatal(err)
	}
	if err := matrix.Destroy(); !errors.Is(err, ErrDSSMatrixInUse) {
		t.Fatalf("destroy attached matrix error = %v", err)
	}
	if err := second.Analyze(matrix); !errors.Is(err, ErrDSSMatrixInUse) {
		t.Fatalf("shared matrix error = %v", err)
	}
	var wg sync.WaitGroup
	errs := make(chan error, 2)
	wg.Add(2)
	go func() {
		defer wg.Done()
		for range 4 {
			if err := first.Analyze(matrix); err != nil {
				errs <- err
				return
			}
		}
	}()
	go func() {
		defer wg.Done()
		for range 4 {
			if err := matrix.Destroy(); !errors.Is(err, ErrDSSMatrixInUse) {
				errs <- errors.New("matrix destruction was not blocked while attached")
				return
			}
		}
	}()
	wg.Wait()
	close(errs)
	for err := range errs {
		t.Error(err)
	}
	if err := first.Destroy(); err != nil {
		t.Fatal(err)
	}
	if err := second.Analyze(matrix); err != nil {
		t.Fatalf("matrix must be reusable after handle destruction: %v", err)
	}
	if err := second.Destroy(); err != nil {
		t.Fatal(err)
	}
	if err := matrix.Destroy(); err != nil {
		t.Fatal(err)
	}
	for _, buffer := range []*memory.Memory{rows, columns, values} {
		if err := buffer.Free(); err != nil {
			t.Fatalf("destroy must release matrix buffers: %v", err)
		}
	}
}

func TestCuDSSPendingCleanupRetry(t *testing.T) {
	buffer, err := memory.Alloc(8)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := buffer.Free(); err != nil {
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
	retained, err := retainLibraryBuffers(buffer)
	if err != nil {
		t.Fatal(err)
	}
	want := errors.New("injected cleanup failure")
	fail := true
	handle.pendingOperations = append(handle.pendingOperations, dssPendingOperation{
		retentions: retained,
		cleanup: func() error {
			if fail {
				return want
			}
			return nil
		},
	})
	if err := handle.Synchronize(); !errors.Is(err, want) {
		t.Fatalf("synchronization error = %v", err)
	}
	if err := buffer.Free(); err == nil {
		t.Fatal("failed cleanup must retain its buffers")
	}
	if err := handle.Destroy(); !errors.Is(err, want) {
		t.Fatalf("destruction error = %v", err)
	}
	fail = false
	if err := handle.Synchronize(); err != nil {
		t.Fatal(err)
	}
	if err := buffer.Free(); err != nil {
		t.Fatalf("successful cleanup must release its buffers: %v", err)
	}
}

func TestCuDSSConvenienceReleasesBuffers(t *testing.T) {
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
	rows, columns, values, rhs, output := allocate(12), allocate(8), allocate(8), allocate(8), allocate(8)
	if err := writeInt32Memory(rows, []int32{0, 1, 2}, 3); err != nil {
		t.Fatal(err)
	}
	if err := writeInt32Memory(columns, []int32{0, 1}, 2); err != nil {
		t.Fatal(err)
	}
	if err := writeMathFloat32Memory(values, []float32{2, 4}); err != nil {
		t.Fatal(err)
	}
	if err := writeMathFloat32Memory(rhs, []float32{2, 4}); err != nil {
		t.Fatal(err)
	}
	if err := writeMathFloat32Memory(output, []float32{0, 0}); err != nil {
		t.Fatal(err)
	}
	if _, err := SolveDirect(2, 2, rows, columns, values, rhs, output); err != nil {
		t.Fatal(err)
	}
	for _, buffer := range []*memory.Memory{rows, columns, values, rhs, output} {
		if err := buffer.Free(); err != nil {
			t.Fatalf("convenience solve leaked a buffer retention: %v", err)
		}
	}
	badRows, badColumns, badValues, badRHS, badOutput := allocate(12), allocate(8), allocate(8), allocate(8), allocate(8)
	if err := writeInt32Memory(badRows, []int32{0, 1, 2}, 3); err != nil {
		t.Fatal(err)
	}
	if err := writeInt32Memory(badColumns, []int32{0, 2}, 2); err != nil {
		t.Fatal(err)
	}
	if err := writeMathFloat32Memory(badValues, []float32{2, 4}); err != nil {
		t.Fatal(err)
	}
	if _, err := SolveDirect(2, 2, badRows, badColumns, badValues, badRHS, badOutput); err == nil {
		t.Fatal("expected invalid CSR to fail")
	}
	for _, buffer := range []*memory.Memory{badRows, badColumns, badValues, badRHS, badOutput} {
		if err := buffer.Free(); err != nil {
			t.Fatalf("failed convenience solve leaked a buffer retention: %v", err)
		}
	}
}
