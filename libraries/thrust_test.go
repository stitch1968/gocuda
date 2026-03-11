package libraries

import (
	"math"
	"testing"

	"github.com/stitch1968/gocuda/memory"
)

func TestThrustDeterministicAlgorithms(t *testing.T) {
	ctx, err := CreateThrustContext()
	if err != nil {
		t.Fatalf("CreateThrustContext failed: %v", err)
	}
	defer ctx.DestroyContext()

	data, err := memory.Alloc(5 * 4)
	if err != nil {
		t.Fatalf("alloc data failed: %v", err)
	}
	defer data.Free()
	if err := writeMathFloat32Memory(data, []float32{3, 1, 2, 2, 5}); err != nil {
		t.Fatalf("write data failed: %v", err)
	}

	if err := ctx.Sort(data, 5, PolicyDevice); err != nil {
		t.Fatalf("Sort failed: %v", err)
	}
	values, err := readMathFloat32Memory(data, 5)
	if err != nil {
		t.Fatalf("read sorted data failed: %v", err)
	}
	expectedSorted := []float32{1, 2, 2, 3, 5}
	for index, value := range values {
		if value != expectedSorted[index] {
			t.Fatalf("unexpected sorted value[%d]: got %v want %v", index, value, expectedSorted[index])
		}
	}

	newLen, err := ctx.Unique(data, 5, PolicyDevice)
	if err != nil {
		t.Fatalf("Unique failed: %v", err)
	}
	if newLen != 4 {
		t.Fatalf("unexpected unique length: got %d want 4", newLen)
	}
}

func TestThrustTransformScanAndReduce(t *testing.T) {
	ctx, err := CreateThrustContext()
	if err != nil {
		t.Fatalf("CreateThrustContext failed: %v", err)
	}
	defer ctx.DestroyContext()

	in, err := memory.Alloc(4 * 4)
	if err != nil {
		t.Fatalf("alloc in failed: %v", err)
	}
	defer in.Free()
	out, err := memory.Alloc(4 * 4)
	if err != nil {
		t.Fatalf("alloc out failed: %v", err)
	}
	defer out.Free()

	if err := writeMathFloat32Memory(in, []float32{1, 2, 3, 4}); err != nil {
		t.Fatalf("write in failed: %v", err)
	}
	if err := ctx.Transform(in, out, 4, "square", PolicyDevice); err != nil {
		t.Fatalf("Transform failed: %v", err)
	}
	transformed, err := readMathFloat32Memory(out, 4)
	if err != nil {
		t.Fatalf("read transform output failed: %v", err)
	}
	expectedTransform := []float32{1, 4, 9, 16}
	for index, value := range transformed {
		if value != expectedTransform[index] {
			t.Fatalf("unexpected transformed[%d]: got %v want %v", index, value, expectedTransform[index])
		}
	}

	if err := ctx.Scan(in, out, 4, PolicyDevice); err != nil {
		t.Fatalf("Scan failed: %v", err)
	}
	scanned, err := readMathFloat32Memory(out, 4)
	if err != nil {
		t.Fatalf("read scan output failed: %v", err)
	}
	expectedScan := []float32{1, 3, 6, 10}
	for index, value := range scanned {
		if value != expectedScan[index] {
			t.Fatalf("unexpected scan[%d]: got %v want %v", index, value, expectedScan[index])
		}
	}

	reduced, err := ctx.Reduce(in, 4, 10, PolicyDevice)
	if err != nil {
		t.Fatalf("Reduce failed: %v", err)
	}
	if reduced != 20 {
		t.Fatalf("unexpected reduce result: got %v want 20", reduced)
	}
}

func TestThrustSelectionAndSetOps(t *testing.T) {
	ctx, err := CreateThrustContext()
	if err != nil {
		t.Fatalf("CreateThrustContext failed: %v", err)
	}
	defer ctx.DestroyContext()

	data, err := memory.Alloc(6 * 4)
	if err != nil {
		t.Fatalf("alloc data failed: %v", err)
	}
	defer data.Free()
	copyDst, err := memory.Alloc(6 * 4)
	if err != nil {
		t.Fatalf("alloc copyDst failed: %v", err)
	}
	defer copyDst.Free()
	left, err := memory.Alloc(4 * 4)
	if err != nil {
		t.Fatalf("alloc left failed: %v", err)
	}
	defer left.Free()
	right, err := memory.Alloc(4 * 4)
	if err != nil {
		t.Fatalf("alloc right failed: %v", err)
	}
	defer right.Free()
	setOut, err := memory.Alloc(8 * 4)
	if err != nil {
		t.Fatalf("alloc setOut failed: %v", err)
	}
	defer setOut.Free()

	if err := writeMathFloat32Memory(data, []float32{-2, -1, 0, 1, 2, 3}); err != nil {
		t.Fatalf("write data failed: %v", err)
	}
	partitionIndex, err := ctx.Partition(data, 6, "positive", PolicyDevice)
	if err != nil {
		t.Fatalf("Partition failed: %v", err)
	}
	if partitionIndex != 3 {
		t.Fatalf("unexpected partition index: got %d want 3", partitionIndex)
	}

	count, err := ctx.CopyIf(data, copyDst, 6, "positive", PolicyDevice)
	if err != nil {
		t.Fatalf("CopyIf failed: %v", err)
	}
	if count != 3 {
		t.Fatalf("unexpected copy count: got %d want 3", count)
	}
	copied, err := readMathFloat32Memory(copyDst, 3)
	if err != nil {
		t.Fatalf("read copyIf output failed: %v", err)
	}
	for index, value := range copied {
		if value != float32(index+1) {
			t.Fatalf("unexpected copied[%d]: got %v want %d", index, value, index+1)
		}
	}

	if err := writeMathFloat32Memory(left, []float32{1, 2, 4, 4}); err != nil {
		t.Fatalf("write left failed: %v", err)
	}
	if err := writeMathFloat32Memory(right, []float32{2, 3, 4, 5}); err != nil {
		t.Fatalf("write right failed: %v", err)
	}
	unionCount, err := ctx.SetUnion(left, right, setOut, 4, 4, PolicyDevice)
	if err != nil {
		t.Fatalf("SetUnion failed: %v", err)
	}
	if unionCount != 5 {
		t.Fatalf("unexpected union count: got %d want 5", unionCount)
	}
	unionValues, err := readMathFloat32Memory(setOut, unionCount)
	if err != nil {
		t.Fatalf("read union output failed: %v", err)
	}
	expectedUnion := []float32{1, 2, 3, 4, 5}
	for index, value := range unionValues {
		if value != expectedUnion[index] {
			t.Fatalf("unexpected union[%d]: got %v want %v", index, value, expectedUnion[index])
		}
	}

	intersectionCount, err := ctx.SetIntersection(left, right, setOut, 4, 4, PolicyDevice)
	if err != nil {
		t.Fatalf("SetIntersection failed: %v", err)
	}
	if intersectionCount != 2 {
		t.Fatalf("unexpected intersection count: got %d want 2", intersectionCount)
	}
	intersectionValues, err := readMathFloat32Memory(setOut, intersectionCount)
	if err != nil {
		t.Fatalf("read intersection output failed: %v", err)
	}
	expectedIntersection := []float32{2, 4}
	for index, value := range intersectionValues {
		if math.Abs(float64(value-expectedIntersection[index])) > 1e-6 {
			t.Fatalf("unexpected intersection[%d]: got %v want %v", index, value, expectedIntersection[index])
		}
	}
}
