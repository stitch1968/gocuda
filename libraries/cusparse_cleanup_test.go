package libraries

import (
	"errors"
	"testing"

	"github.com/stitch1968/gocuda/memory"
)

func TestSparsePendingCleanupRetainsResourcesUntilRetry(t *testing.T) {
	ctx := &SparseContext{}
	matrix, err := ctx.CreateCSRMatrix(1, 1, []int32{0, 1}, []int32{0}, []float32{1})
	if err != nil {
		t.Fatal(err)
	}
	defer matrix.Destroy()
	workspace, err := memory.Alloc(16)
	if err != nil {
		t.Fatal(err)
	}
	defer workspace.Free()
	buffer, err := memory.Alloc(4)
	if err != nil {
		t.Fatal(err)
	}
	defer buffer.Free()
	retention, err := buffer.Retain()
	if err != nil {
		t.Fatal(err)
	}
	defer retention.Release()
	expected := errors.New("cleanup failed")
	calls := 0
	ctx.pending = []*memory.Retention{retention}
	ctx.pendingMatrices = []*SparseMatrix{matrix}
	ctx.pendingWorkspaces = []*memory.Memory{workspace}
	ctx.pendingCleanups = []func() error{func() error {
		calls++
		if calls == 1 {
			return expected
		}
		return nil
	}}
	if err := ctx.releasePending(); !errors.Is(err, expected) {
		t.Fatalf("cleanup error = %v, want %v", err, expected)
	}
	if err := buffer.Free(); err == nil {
		t.Fatal("buffer must remain retained after descriptor cleanup failure")
	}
	if matrix.Values().Ptr() == nil || workspace.Ptr() == nil {
		t.Fatal("matrix and workspace must remain live after cleanup failure")
	}
	if err := ctx.releasePending(); err != nil {
		t.Fatal(err)
	}
	if calls != 2 || matrix.Values().Ptr() != nil || workspace.Ptr() != nil {
		t.Fatal("retry must complete descriptor, matrix and workspace cleanup")
	}
	if err := buffer.Free(); err != nil {
		t.Fatalf("buffer should be released after cleanup: %v", err)
	}
}
