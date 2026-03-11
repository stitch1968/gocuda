//go:build cuda

package libraries

import (
	"math"
	"testing"

	cuda "github.com/stitch1968/gocuda"
	"github.com/stitch1968/gocuda/memory"
)

func TestCUTLASSNativeBackendSelection(t *testing.T) {
	if err := cuda.Initialize(); err != nil {
		t.Fatalf("Initialize failed: %v", err)
	}
	if !cuda.ShouldUseCuda() {
		t.Skip("CUDA runtime not available")
	}
	if !cutlassNativeAvailable() {
		t.Skip("CUTLASS native backend not available")
	}

	handle, err := CreateCutlassGemm(CutlassGemmDesc{M: 2, N: 2, K: 2, DataType: CutlassFloat32, LayoutA: CutlassRowMajor, LayoutB: CutlassRowMajor, LayoutC: CutlassRowMajor, Alpha: 1, Beta: 0})
	if err != nil {
		t.Fatalf("CreateCutlassGemm failed: %v", err)
	}
	defer handle.Destroy()
	if !handle.native {
		t.Fatal("expected native CUTLASS GEMM handle in CUDA build")
	}
}

func TestCUTLASSNativeGemm(t *testing.T) {
	if err := cuda.Initialize(); err != nil {
		t.Fatalf("Initialize failed: %v", err)
	}
	if !cuda.ShouldUseCuda() {
		t.Skip("CUDA runtime not available")
	}
	if !cutlassNativeAvailable() {
		t.Skip("CUTLASS native backend not available")
	}

	handle, err := CreateCutlassGemm(CutlassGemmDesc{M: 2, N: 2, K: 3, DataType: CutlassFloat32, LayoutA: CutlassRowMajor, LayoutB: CutlassRowMajor, LayoutC: CutlassRowMajor, Alpha: 1, Beta: 0})
	if err != nil {
		t.Fatalf("CreateCutlassGemm failed: %v", err)
	}
	defer handle.Destroy()

	a, _ := memory.Alloc(2 * 3 * 4)
	b, _ := memory.Alloc(3 * 2 * 4)
	c, _ := memory.Alloc(2 * 2 * 4)
	defer a.Free()
	defer b.Free()
	defer c.Free()
	_ = writeMathFloat32Memory(a, []float32{1, 2, 3, 4, 5, 6})
	_ = writeMathFloat32Memory(b, []float32{7, 8, 9, 10, 11, 12})
	_ = writeMathFloat32Memory(c, []float32{0, 0, 0, 0})

	if err := handle.CutlassGemm(a, b, c); err != nil {
		t.Fatalf("CutlassGemm failed: %v", err)
	}
	values, err := readMathFloat32Memory(c, 4)
	if err != nil {
		t.Fatalf("read output failed: %v", err)
	}
	expected := []float32{58, 64, 139, 154}
	for index, value := range values {
		if math.Abs(float64(value-expected[index])) > 1e-4 {
			t.Fatalf("unexpected gemm[%d]: got %v want %v", index, value, expected[index])
		}
	}
}

func TestCUTLASSNativeRank2k(t *testing.T) {
	if err := cuda.Initialize(); err != nil {
		t.Fatalf("Initialize failed: %v", err)
	}
	if !cuda.ShouldUseCuda() {
		t.Skip("CUDA runtime not available")
	}
	if !cutlassNativeAvailable() {
		t.Skip("CUTLASS native backend not available")
	}

	a, _ := memory.Alloc(2 * 2 * 4)
	b, _ := memory.Alloc(2 * 2 * 4)
	c, _ := memory.Alloc(2 * 2 * 4)
	defer a.Free()
	defer b.Free()
	defer c.Free()
	_ = writeMathFloat32Memory(a, []float32{1, 2, 3, 4})
	_ = writeMathFloat32Memory(b, []float32{5, 6, 7, 8})
	_ = writeMathFloat32Memory(c, []float32{0, 0, 0, 0})

	if err := CutlassRank2k(a, b, c, 2, 2, 1, 0); err != nil {
		t.Fatalf("CutlassRank2k failed: %v", err)
	}
	values, err := readMathFloat32Memory(c, 4)
	if err != nil {
		t.Fatalf("read output failed: %v", err)
	}
	expected := []float32{20, 44, 44, 64}
	for index, value := range values {
		if math.Abs(float64(value-expected[index])) > 1e-4 {
			t.Fatalf("unexpected rank2k[%d]: got %v want %v", index, value, expected[index])
		}
	}
}

func TestCUTLASSNativeTrmm(t *testing.T) {
	if err := cuda.Initialize(); err != nil {
		t.Fatalf("Initialize failed: %v", err)
	}
	if !cuda.ShouldUseCuda() {
		t.Skip("CUDA runtime not available")
	}
	if !cutlassNativeAvailable() {
		t.Skip("CUTLASS native backend not available")
	}

	a, _ := memory.Alloc(2 * 2 * 4)
	b, _ := memory.Alloc(2 * 2 * 4)
	defer a.Free()
	defer b.Free()
	_ = writeMathFloat32Memory(a, []float32{1, 2, 0, 3})
	_ = writeMathFloat32Memory(b, []float32{4, 5, 6, 7})

	if err := CutlassTrmm(a, b, 2, 2, "Left", "Upper", "NoTrans", "NonUnit", 1); err != nil {
		t.Fatalf("CutlassTrmm failed: %v", err)
	}
	values, err := readMathFloat32Memory(b, 4)
	if err != nil {
		t.Fatalf("read output failed: %v", err)
	}
	expected := []float32{16, 19, 18, 21}
	for index, value := range values {
		if math.Abs(float64(value-expected[index])) > 1e-4 {
			t.Fatalf("unexpected trmm[%d]: got %v want %v", index, value, expected[index])
		}
	}
}
