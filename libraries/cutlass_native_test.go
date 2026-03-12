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
	expected := []float32{34, 62, 62, 106}
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

func TestCUTLASSNativeSpmm(t *testing.T) {
	if err := cuda.Initialize(); err != nil {
		t.Fatalf("Initialize failed: %v", err)
	}
	if !cuda.ShouldUseCuda() {
		t.Skip("CUDA runtime not available")
	}
	if !cutlassNativeAvailable() {
		t.Skip("CUTLASS native backend not available")
	}

	spA, _ := memory.Alloc(4 * 4)
	denseB, _ := memory.Alloc(4 * 4)
	denseC, _ := memory.Alloc(4 * 4)
	defer spA.Free()
	defer denseB.Free()
	defer denseC.Free()
	_ = writeMathFloat32Memory(spA, []float32{1, 0, 0, 2})
	_ = writeMathFloat32Memory(denseB, []float32{3, 4, 5, 6})
	_ = writeMathFloat32Memory(denseC, []float32{0, 0, 0, 0})

	if err := CutlassSpmm(spA, denseB, denseC, 2, 2, 2, 0.5); err != nil {
		t.Fatalf("CutlassSpmm failed: %v", err)
	}
	values, err := readMathFloat32Memory(denseC, 4)
	if err != nil {
		t.Fatalf("read output failed: %v", err)
	}
	expected := []float32{3, 4, 10, 12}
	for index, value := range values {
		if math.Abs(float64(value-expected[index])) > 1e-4 {
			t.Fatalf("unexpected spmm[%d]: got %v want %v", index, value, expected[index])
		}
	}
}

func TestCUTLASSNativeConvModes(t *testing.T) {
	if err := cuda.Initialize(); err != nil {
		t.Fatalf("Initialize failed: %v", err)
	}
	if !cuda.ShouldUseCuda() {
		t.Skip("CUDA runtime not available")
	}
	if !cutlassNativeAvailable() {
		t.Skip("CUTLASS native backend not available")
	}

	forward, err := CreateCutlassConv(CutlassConvDesc{N: 1, H: 3, W: 3, C: 1, K: 1, R: 2, S: 2, StrideH: 1, StrideW: 1, DilationH: 1, DilationW: 1, DataType: CutlassFloat32, Mode: CutlassConvForward})
	if err != nil {
		t.Fatalf("CreateCutlassConv forward failed: %v", err)
	}
	defer forward.Destroy()

	in, _ := memory.Alloc(3 * 3 * 4)
	filter, _ := memory.Alloc(2 * 2 * 4)
	out, _ := memory.Alloc(2 * 2 * 4)
	defer in.Free()
	defer filter.Free()
	defer out.Free()
	_ = writeMathFloat32Memory(in, []float32{1, 2, 3, 4, 5, 6, 7, 8, 9})
	_ = writeMathFloat32Memory(filter, []float32{1, 0, 0, 1})
	if err := forward.CutlassConv(in, filter, out); err != nil {
		t.Fatalf("CutlassConv forward failed: %v", err)
	}
	forwardValues, err := readMathFloat32Memory(out, 4)
	if err != nil {
		t.Fatalf("read forward output failed: %v", err)
	}
	forwardExpected := []float32{6, 8, 12, 14}
	for index, value := range forwardValues {
		if math.Abs(float64(value-forwardExpected[index])) > 1e-4 {
			t.Fatalf("unexpected conv forward[%d]: got %v want %v", index, value, forwardExpected[index])
		}
	}

	dgrad, err := CreateCutlassConv(CutlassConvDesc{N: 1, H: 3, W: 3, C: 1, K: 1, R: 2, S: 2, StrideH: 1, StrideW: 1, DilationH: 1, DilationW: 1, DataType: CutlassFloat32, Mode: CutlassConvDgrad})
	if err != nil {
		t.Fatalf("CreateCutlassConv dgrad failed: %v", err)
	}
	defer dgrad.Destroy()

	gradOut, _ := memory.Alloc(2 * 2 * 4)
	gradFilter, _ := memory.Alloc(2 * 2 * 4)
	gradIn, _ := memory.Alloc(3 * 3 * 4)
	defer gradOut.Free()
	defer gradFilter.Free()
	defer gradIn.Free()
	_ = writeMathFloat32Memory(gradOut, []float32{1, 2, 3, 4})
	_ = writeMathFloat32Memory(gradFilter, []float32{1, 0, 0, 1})
	if err := dgrad.CutlassConv(gradOut, gradFilter, gradIn); err != nil {
		t.Fatalf("CutlassConv dgrad failed: %v", err)
	}
	dgradValues, err := readMathFloat32Memory(gradIn, 9)
	if err != nil {
		t.Fatalf("read dgrad output failed: %v", err)
	}
	dgradExpected := []float32{1, 2, 0, 3, 5, 2, 0, 3, 4}
	for index, value := range dgradValues {
		if math.Abs(float64(value-dgradExpected[index])) > 1e-4 {
			t.Fatalf("unexpected conv dgrad[%d]: got %v want %v", index, value, dgradExpected[index])
		}
	}

	wgrad, err := CreateCutlassConv(CutlassConvDesc{N: 1, H: 3, W: 3, C: 1, K: 1, R: 2, S: 2, StrideH: 1, StrideW: 1, DilationH: 1, DilationW: 1, DataType: CutlassFloat32, Mode: CutlassConvWgrad})
	if err != nil {
		t.Fatalf("CreateCutlassConv wgrad failed: %v", err)
	}
	defer wgrad.Destroy()

	inputW, _ := memory.Alloc(3 * 3 * 4)
	gradOutW, _ := memory.Alloc(2 * 2 * 4)
	gradWeights, _ := memory.Alloc(2 * 2 * 4)
	defer inputW.Free()
	defer gradOutW.Free()
	defer gradWeights.Free()
	_ = writeMathFloat32Memory(inputW, []float32{1, 2, 3, 4, 5, 6, 7, 8, 9})
	_ = writeMathFloat32Memory(gradOutW, []float32{1, 2, 3, 4})
	if err := wgrad.CutlassConv(inputW, gradOutW, gradWeights); err != nil {
		t.Fatalf("CutlassConv wgrad failed: %v", err)
	}
	wgradValues, err := readMathFloat32Memory(gradWeights, 4)
	if err != nil {
		t.Fatalf("read wgrad output failed: %v", err)
	}
	wgradExpected := []float32{37, 47, 67, 77}
	for index, value := range wgradValues {
		if math.Abs(float64(value-wgradExpected[index])) > 1e-4 {
			t.Fatalf("unexpected conv wgrad[%d]: got %v want %v", index, value, wgradExpected[index])
		}
	}
}
