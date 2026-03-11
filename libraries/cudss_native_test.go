//go:build cuda

package libraries

import (
	"math"
	"testing"

	cuda "github.com/stitch1968/gocuda"
	"github.com/stitch1968/gocuda/memory"
)

func TestCuDSSNativeBackendSelection(t *testing.T) {
	if err := cuda.Initialize(); err != nil {
		t.Fatalf("Initialize failed: %v", err)
	}
	if !cuda.ShouldUseCuda() {
		t.Skip("CUDA runtime not available")
	}
	if !cudssNativeAvailable() {
		t.Skip("cuDSS native backend not available")
	}

	handle, err := CreateDSSHandle(DSSConfig{MatrixFormat: DSSMatrixFormatCSR, Factorization: DSSFactorizationLU})
	if err != nil {
		t.Fatalf("CreateDSSHandle failed: %v", err)
	}
	defer handle.Destroy()
	if !handle.native {
		t.Fatal("expected native cuDSS handle in CUDA build")
	}
}

func TestCuDSSNativeSolve(t *testing.T) {
	if err := cuda.Initialize(); err != nil {
		t.Fatalf("Initialize failed: %v", err)
	}
	if !cuda.ShouldUseCuda() {
		t.Skip("CUDA runtime not available")
	}
	if !cudssNativeAvailable() {
		t.Skip("cuDSS native backend not available")
	}

	rowPtr, _ := memory.Alloc(3 * 4)
	colInd, _ := memory.Alloc(4 * 4)
	values, _ := memory.Alloc(4 * 4)
	b, _ := memory.Alloc(2 * 4)
	x, _ := memory.Alloc(2 * 4)
	defer rowPtr.Free()
	defer colInd.Free()
	defer values.Free()
	defer b.Free()
	defer x.Free()

	_ = writeInt32Memory(rowPtr, []int32{0, 2, 4}, 3)
	_ = writeInt32Memory(colInd, []int32{0, 1, 0, 1}, 4)
	_ = writeMathFloat32Memory(values, []float32{4, 1, 2, 3})
	_ = writeMathFloat32Memory(b, []float32{1, 2})

	matrix, err := CreateDSSMatrix(2, 4, rowPtr, colInd, values, DSSMatrixFormatCSR, false)
	if err != nil {
		t.Fatalf("CreateDSSMatrix failed: %v", err)
	}
	defer matrix.Destroy()

	handle, err := CreateDSSHandle(DSSConfig{MatrixFormat: DSSMatrixFormatCSR, Factorization: DSSFactorizationLU})
	if err != nil {
		t.Fatalf("CreateDSSHandle failed: %v", err)
	}
	defer handle.Destroy()

	if err := handle.Analyze(matrix); err != nil {
		t.Fatalf("Analyze failed: %v", err)
	}
	if err := handle.Factor(matrix); err != nil {
		t.Fatalf("Factor failed: %v", err)
	}
	info, err := handle.Solve(b, x, 1)
	if err != nil {
		t.Fatalf("Solve failed: %v", err)
	}
	if info.Residual > 1e-3 {
		t.Fatalf("unexpected residual: %v", info.Residual)
	}
	xValues, err := readMathFloat32Memory(x, 2)
	if err != nil {
		t.Fatalf("read solution failed: %v", err)
	}
	if math.Abs(float64(xValues[0]-0.1)) > 1e-2 || math.Abs(float64(xValues[1]-0.6)) > 1e-2 {
		t.Fatalf("unexpected solution: %v", xValues)
	}
}
