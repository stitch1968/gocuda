package libraries

import (
	"math"
	"testing"

	"github.com/stitch1968/gocuda/memory"
)

func TestCuTensorElementwiseReduceAndPermute(t *testing.T) {
	handle, err := CreateCuTensorHandle()
	if err != nil {
		t.Fatalf("CreateCuTensorHandle failed: %v", err)
	}
	defer handle.Destroy()

	desc, err := CreateCuTensorDescriptor(TensorFloat32, []int{2, 2}, TensorLayoutRowMajor)
	if err != nil {
		t.Fatalf("CreateCuTensorDescriptor failed: %v", err)
	}
	reduceDesc, err := CreateCuTensorDescriptor(TensorFloat32, []int{2}, TensorLayoutRowMajor)
	if err != nil {
		t.Fatalf("CreateCuTensorDescriptor reduce failed: %v", err)
	}
	permDesc, err := CreateCuTensorDescriptor(TensorFloat32, []int{2, 2}, TensorLayoutRowMajor)
	if err != nil {
		t.Fatalf("CreateCuTensorDescriptor perm failed: %v", err)
	}

	a, _ := memory.Alloc(4 * 4)
	b, _ := memory.Alloc(4 * 4)
	c, _ := memory.Alloc(4 * 4)
	reduced, _ := memory.Alloc(2 * 4)
	permuted, _ := memory.Alloc(4 * 4)
	defer a.Free()
	defer b.Free()
	defer c.Free()
	defer reduced.Free()
	defer permuted.Free()

	if err := writeMathFloat32Memory(a, []float32{1, 2, 3, 4}); err != nil {
		t.Fatalf("write a failed: %v", err)
	}
	if err := writeMathFloat32Memory(b, []float32{5, 6, 7, 8}); err != nil {
		t.Fatalf("write b failed: %v", err)
	}
	if err := writeMathFloat32Memory(c, []float32{0, 0, 0, 0}); err != nil {
		t.Fatalf("write c failed: %v", err)
	}

	if err := handle.TensorElementwiseAdd(1, a, desc, 1, b, desc, 0, c, desc); err != nil {
		t.Fatalf("TensorElementwiseAdd failed: %v", err)
	}
	added, err := readMathFloat32Memory(c, 4)
	if err != nil {
		t.Fatalf("read add output failed: %v", err)
	}
	expectedAdd := []float32{6, 8, 10, 12}
	for index, value := range added {
		if value != expectedAdd[index] {
			t.Fatalf("unexpected add[%d]: got %v want %v", index, value, expectedAdd[index])
		}
	}

	if err := handle.TensorReduce(1, c, desc, 0, reduced, reduceDesc, []int{1}, TensorReduceSum); err != nil {
		t.Fatalf("TensorReduce failed: %v", err)
	}
	reducedValues, err := readMathFloat32Memory(reduced, 2)
	if err != nil {
		t.Fatalf("read reduce output failed: %v", err)
	}
	if reducedValues[0] != 14 || reducedValues[1] != 22 {
		t.Fatalf("unexpected reduction result: got %v", reducedValues)
	}

	if err := handle.TensorPermute(1, c, desc, permuted, permDesc, []int{1, 0}); err != nil {
		t.Fatalf("TensorPermute failed: %v", err)
	}
	permutedValues, err := readMathFloat32Memory(permuted, 4)
	if err != nil {
		t.Fatalf("read permuted output failed: %v", err)
	}
	expectedPermute := []float32{6, 10, 8, 12}
	for index, value := range permutedValues {
		if value != expectedPermute[index] {
			t.Fatalf("unexpected permute[%d]: got %v want %v", index, value, expectedPermute[index])
		}
	}
}

func TestCuTensorContractionAndPlan(t *testing.T) {
	handle, err := CreateCuTensorHandle()
	if err != nil {
		t.Fatalf("CreateCuTensorHandle failed: %v", err)
	}
	defer handle.Destroy()

	descA, _ := CreateCuTensorDescriptor(TensorFloat64, []int{2, 3}, TensorLayoutRowMajor)
	descB, _ := CreateCuTensorDescriptor(TensorFloat64, []int{3, 2}, TensorLayoutRowMajor)
	descC, _ := CreateCuTensorDescriptor(TensorFloat64, []int{2, 2}, TensorLayoutRowMajor)

	a, _ := memory.Alloc(2 * 3 * 8)
	b, _ := memory.Alloc(3 * 2 * 8)
	c, _ := memory.Alloc(2 * 2 * 8)
	defer a.Free()
	defer b.Free()
	defer c.Free()

	writeMathFloat64Memory(a, []float64{1, 2, 3, 4, 5, 6})
	writeMathFloat64Memory(b, []float64{7, 8, 9, 10, 11, 12})
	writeMathFloat64Memory(c, []float64{0, 0, 0, 0})

	if err := handle.TensorContraction(1, a, descA, []int{0, 1}, b, descB, []int{1, 2}, 0, c, descC, []int{0, 2}, ContractionAlgoGEMM); err != nil {
		t.Fatalf("TensorContraction failed: %v", err)
	}
	values, err := readMathFloat64Memory(c, 4)
	if err != nil {
		t.Fatalf("read contraction output failed: %v", err)
	}
	expected := []float64{58, 64, 139, 154}
	for index, value := range values {
		if math.Abs(value-expected[index]) > 1e-9 {
			t.Fatalf("unexpected contraction[%d]: got %v want %v", index, value, expected[index])
		}
	}

	plan, err := handle.CreateContractionPlan(descA, []int{0, 1}, descB, []int{1, 2}, descC, []int{0, 2}, ContractionAlgoGEMM)
	if err != nil {
		t.Fatalf("CreateContractionPlan failed: %v", err)
	}
	writeMathFloat64Memory(c, []float64{1, 1, 1, 1})
	if err := handle.ExecuteContractionPlan(plan, 1, a, b, 1, c); err != nil {
		t.Fatalf("ExecuteContractionPlan failed: %v", err)
	}
	planned, err := readMathFloat64Memory(c, 4)
	if err != nil {
		t.Fatalf("read planned output failed: %v", err)
	}
	for index, value := range planned {
		if math.Abs(value-(expected[index]+1)) > 1e-9 {
			t.Fatalf("unexpected planned contraction[%d]: got %v want %v", index, value, expected[index]+1)
		}
	}
}
