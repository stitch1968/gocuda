package libraries

import (
	"math"
	"testing"

	"github.com/stitch1968/gocuda/memory"
)

func TestCutlassGemmAndBatched(t *testing.T) {
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
	writeMathFloat32Memory(a, []float32{1, 2, 3, 4, 5, 6})
	writeMathFloat32Memory(b, []float32{7, 8, 9, 10, 11, 12})
	writeMathFloat32Memory(c, []float32{0, 0, 0, 0})

	if err := handle.CutlassGemm(a, b, c); err != nil {
		t.Fatalf("CutlassGemm failed: %v", err)
	}
	values, _ := readMathFloat32Memory(c, 4)
	expected := []float32{58, 64, 139, 154}
	for index, value := range values {
		if math.Abs(float64(value-expected[index])) > 1e-5 {
			t.Fatalf("unexpected gemm[%d]: got %v want %v", index, value, expected[index])
		}
	}

	c2, _ := memory.Alloc(2 * 2 * 4)
	defer c2.Free()
	writeMathFloat32Memory(c2, []float32{0, 0, 0, 0})
	if err := handle.CutlassGemmBatched([]*memory.Memory{a}, []*memory.Memory{b}, []*memory.Memory{c2}, 1); err != nil {
		t.Fatalf("CutlassGemmBatched failed: %v", err)
	}
}

func TestCutlassConvAndHelpers(t *testing.T) {
	conv, err := CreateCutlassConv(CutlassConvDesc{N: 1, H: 3, W: 3, C: 1, K: 1, R: 2, S: 2, StrideH: 1, StrideW: 1, DilationH: 1, DilationW: 1, DataType: CutlassFloat32, Mode: CutlassConvForward})
	if err != nil {
		t.Fatalf("CreateCutlassConv failed: %v", err)
	}
	defer conv.Destroy()

	in, _ := memory.Alloc(3 * 3 * 4)
	filter, _ := memory.Alloc(2 * 2 * 4)
	out, _ := memory.Alloc(2 * 2 * 4)
	defer in.Free()
	defer filter.Free()
	defer out.Free()
	writeMathFloat32Memory(in, []float32{1, 2, 3, 4, 5, 6, 7, 8, 9})
	writeMathFloat32Memory(filter, []float32{1, 0, 0, 1})
	if err := conv.CutlassConv(in, filter, out); err != nil {
		t.Fatalf("CutlassConv failed: %v", err)
	}
	convOut, _ := readMathFloat32Memory(out, 4)
	expectedConv := []float32{6, 8, 12, 14}
	for index, value := range convOut {
		if value != expectedConv[index] {
			t.Fatalf("unexpected conv[%d]: got %v want %v", index, value, expectedConv[index])
		}
	}

	spA, _ := memory.Alloc(4 * 4)
	denseB, _ := memory.Alloc(4 * 4)
	denseC, _ := memory.Alloc(4 * 4)
	defer spA.Free()
	defer denseB.Free()
	defer denseC.Free()
	writeMathFloat32Memory(spA, []float32{1, 0, 0, 2})
	writeMathFloat32Memory(denseB, []float32{3, 4, 5, 6})
	if err := CutlassSpmm(spA, denseB, denseC, 2, 2, 2, 0.5); err != nil {
		t.Fatalf("CutlassSpmm failed: %v", err)
	}

	rankA, _ := memory.Alloc(4 * 4)
	rankB, _ := memory.Alloc(4 * 4)
	rankC, _ := memory.Alloc(4 * 4)
	defer rankA.Free()
	defer rankB.Free()
	defer rankC.Free()
	writeMathFloat32Memory(rankA, []float32{1, 2, 3, 4})
	writeMathFloat32Memory(rankB, []float32{5, 6, 7, 8})
	writeMathFloat32Memory(rankC, []float32{0, 0, 0, 0})
	if err := CutlassRank2k(rankA, rankB, rankC, 2, 2, 1, 0); err != nil {
		t.Fatalf("CutlassRank2k failed: %v", err)
	}
	rankValues, err := readMathFloat32Memory(rankC, 4)
	if err != nil {
		t.Fatalf("read rank2k output failed: %v", err)
	}
	rankExpected := []float32{34, 62, 62, 106}
	for index, value := range rankValues {
		if math.Abs(float64(value-rankExpected[index])) > 1e-4 {
			t.Fatalf("unexpected rank2k[%d]: got %v want %v", index, value, rankExpected[index])
		}
	}

	triA, _ := memory.Alloc(4 * 4)
	triB, _ := memory.Alloc(4 * 4)
	defer triA.Free()
	defer triB.Free()
	writeMathFloat32Memory(triA, []float32{1, 2, 0, 3})
	writeMathFloat32Memory(triB, []float32{1, 0, 0, 1})
	if err := CutlassTrmm(triA, triB, 2, 2, "Left", "Upper", "NoTrans", "NonUnit", 1); err != nil {
		t.Fatalf("CutlassTrmm failed: %v", err)
	}
}
