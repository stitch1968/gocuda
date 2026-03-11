//go:build cuda

package libraries

import (
	"math"
	"testing"

	"github.com/stitch1968/gocuda/memory"
)

func TestCuTensorNativeBackendSelection(t *testing.T) {
	handle, err := CreateCuTensorHandle()
	if err != nil {
		t.Fatalf("CreateCuTensorHandle failed: %v", err)
	}
	defer handle.Destroy()

	if !handle.native {
		t.Fatalf("expected native cuTENSOR handle in cuda build")
	}
}

func TestCuTensorNativeContractionAndElementwise(t *testing.T) {
	handle, err := CreateCuTensorHandle()
	if err != nil {
		t.Fatalf("CreateCuTensorHandle failed: %v", err)
	}
	defer handle.Destroy()

	descA, _ := CreateCuTensorDescriptor(TensorFloat32, []int{2, 2}, TensorLayoutRowMajor)
	descB, _ := CreateCuTensorDescriptor(TensorFloat32, []int{2, 2}, TensorLayoutRowMajor)
	descC, _ := CreateCuTensorDescriptor(TensorFloat32, []int{2, 2}, TensorLayoutRowMajor)

	a, _ := memory.Alloc(4 * 4)
	b, _ := memory.Alloc(4 * 4)
	c, _ := memory.Alloc(4 * 4)
	defer a.Free()
	defer b.Free()
	defer c.Free()

	if err := writeMathFloat32Memory(a, []float32{1, 2, 3, 4}); err != nil {
		t.Fatalf("write a failed: %v", err)
	}
	if err := writeMathFloat32Memory(b, []float32{5, 6, 7, 8}); err != nil {
		t.Fatalf("write b failed: %v", err)
	}
	if err := writeMathFloat32Memory(c, []float32{1, 1, 1, 1}); err != nil {
		t.Fatalf("write c failed: %v", err)
	}

	if err := handle.TensorElementwiseAdd(1, a, descA, 1, b, descB, 1, c, descC); err != nil {
		t.Fatalf("TensorElementwiseAdd failed: %v", err)
	}
	added, err := readMathFloat32Memory(c, 4)
	if err != nil {
		t.Fatalf("read add result failed: %v", err)
	}
	expectedAdd := []float32{7, 9, 11, 13}
	for index, value := range added {
		if math.Abs(float64(value-expectedAdd[index])) > 1e-5 {
			t.Fatalf("unexpected add[%d]: got %v want %v", index, value, expectedAdd[index])
		}
	}

	descLeft, _ := CreateCuTensorDescriptor(TensorFloat64, []int{2, 3}, TensorLayoutRowMajor)
	descRight, _ := CreateCuTensorDescriptor(TensorFloat64, []int{3, 2}, TensorLayoutRowMajor)
	descOut, _ := CreateCuTensorDescriptor(TensorFloat64, []int{2, 2}, TensorLayoutRowMajor)

	left, _ := memory.Alloc(2 * 3 * 8)
	right, _ := memory.Alloc(3 * 2 * 8)
	out, _ := memory.Alloc(2 * 2 * 8)
	defer left.Free()
	defer right.Free()
	defer out.Free()

	writeMathFloat64Memory(left, []float64{1, 2, 3, 4, 5, 6})
	writeMathFloat64Memory(right, []float64{7, 8, 9, 10, 11, 12})
	writeMathFloat64Memory(out, []float64{0, 0, 0, 0})

	if err := handle.TensorContraction(1, left, descLeft, []int{0, 1}, right, descRight, []int{1, 2}, 0, out, descOut, []int{0, 2}, ContractionAlgoGEMM); err != nil {
		t.Fatalf("TensorContraction failed: %v", err)
	}
	values, err := readMathFloat64Memory(out, 4)
	if err != nil {
		t.Fatalf("read contraction result failed: %v", err)
	}
	expected := []float64{58, 64, 139, 154}
	for index, value := range values {
		if math.Abs(value-expected[index]) > 1e-9 {
			t.Fatalf("unexpected contraction[%d]: got %v want %v", index, value, expected[index])
		}
	}
}
