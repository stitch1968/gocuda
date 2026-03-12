//go:build cuda

package libraries

import (
	"testing"

	cuda "github.com/stitch1968/gocuda"
	"github.com/stitch1968/gocuda/memory"
)

func TestThrustNativeBackendSelection(t *testing.T) {
	if err := cuda.Initialize(); err != nil {
		t.Fatalf("Initialize failed: %v", err)
	}
	if !cuda.ShouldUseCuda() {
		t.Skip("CUDA runtime not available")
	}
	if !thrustNativeAvailable() {
		t.Skip("Thrust native backend not available")
	}

	ctx, err := CreateThrustContext()
	if err != nil {
		t.Fatalf("CreateThrustContext failed: %v", err)
	}
	defer ctx.DestroyContext()
	if !ctx.native {
		t.Fatal("expected native Thrust context in CUDA build")
	}
}

func TestThrustNativeCopyFillGenerate(t *testing.T) {
	if err := cuda.Initialize(); err != nil {
		t.Fatalf("Initialize failed: %v", err)
	}
	if !cuda.ShouldUseCuda() {
		t.Skip("CUDA runtime not available")
	}
	if !thrustNativeAvailable() {
		t.Skip("Thrust native backend not available")
	}

	ctx, err := CreateThrustContext()
	if err != nil {
		t.Fatalf("CreateThrustContext failed: %v", err)
	}
	defer ctx.DestroyContext()

	src, _ := memory.Alloc(4 * 4)
	dst, _ := memory.Alloc(4 * 4)
	defer src.Free()
	defer dst.Free()
	if err := writeMathFloat32Memory(src, []float32{1, 2, 3, 4}); err != nil {
		t.Fatalf("write src failed: %v", err)
	}
	if err := ctx.Copy(src, dst, 4, PolicyDevice); err != nil {
		t.Fatalf("Copy failed: %v", err)
	}
	values, err := readMathFloat32Memory(dst, 4)
	if err != nil {
		t.Fatalf("read dst failed: %v", err)
	}
	for index, value := range values {
		if value != float32(index+1) {
			t.Fatalf("unexpected copy[%d]: got %v want %d", index, value, index+1)
		}
	}

	if err := ctx.Fill(dst, 4, 0, PolicyDevice); err != nil {
		t.Fatalf("Fill failed: %v", err)
	}
	values, err = readMathFloat32Memory(dst, 4)
	if err != nil {
		t.Fatalf("read filled dst failed: %v", err)
	}
	for index, value := range values {
		if value != 0 {
			t.Fatalf("unexpected fill[%d]: got %v want 0", index, value)
		}
	}

	if err := ctx.Generate(dst, 4, "sequence", PolicyDevice); err != nil {
		t.Fatalf("Generate failed: %v", err)
	}
	values, err = readMathFloat32Memory(dst, 4)
	if err != nil {
		t.Fatalf("read generated dst failed: %v", err)
	}
	for index, value := range values {
		if value != float32(index) {
			t.Fatalf("unexpected sequence[%d]: got %v want %d", index, value, index)
		}
	}
}

func TestThrustNativeAlgorithmsExpanded(t *testing.T) {
	if err := cuda.Initialize(); err != nil {
		t.Fatalf("Initialize failed: %v", err)
	}
	if !cuda.ShouldUseCuda() {
		t.Skip("CUDA runtime not available")
	}
	if !thrustNativeAvailable() {
		t.Skip("Thrust native backend not available")
	}

	ctx, err := CreateThrustContext()
	if err != nil {
		t.Fatalf("CreateThrustContext failed: %v", err)
	}
	defer ctx.DestroyContext()

	data, _ := memory.Alloc(6 * 4)
	keys, _ := memory.Alloc(4 * 4)
	vals, _ := memory.Alloc(4 * 4)
	left, _ := memory.Alloc(4 * 4)
	right, _ := memory.Alloc(4 * 4)
	copyDst, _ := memory.Alloc(6 * 4)
	out, _ := memory.Alloc(6 * 4)
	setOut, _ := memory.Alloc(8 * 4)
	defer data.Free()
	defer keys.Free()
	defer vals.Free()
	defer left.Free()
	defer right.Free()
	defer copyDst.Free()
	defer out.Free()
	defer setOut.Free()

	if err := writeMathFloat32Memory(data, []float32{3, 1, 2, 2, 5}); err != nil {
		t.Fatalf("write data failed: %v", err)
	}
	if err := ctx.Sort(data, 5, PolicyDevice); err != nil {
		t.Fatalf("Sort failed: %v", err)
	}
	sorted, err := readMathFloat32Memory(data, 5)
	if err != nil {
		t.Fatalf("read sorted failed: %v", err)
	}
	expectedSorted := []float32{1, 2, 2, 3, 5}
	for index, value := range sorted {
		if value != expectedSorted[index] {
			t.Fatalf("unexpected sorted[%d]: got %v want %v", index, value, expectedSorted[index])
		}
	}

	if err := writeMathFloat32Memory(keys, []float32{3, 1, 2, 1}); err != nil {
		t.Fatalf("write keys failed: %v", err)
	}
	if err := writeMathFloat32Memory(vals, []float32{30, 10, 20, 11}); err != nil {
		t.Fatalf("write vals failed: %v", err)
	}
	if err := ctx.SortByKey(keys, vals, 4, PolicyDevice); err != nil {
		t.Fatalf("SortByKey failed: %v", err)
	}
	sortedKeys, err := readMathFloat32Memory(keys, 4)
	if err != nil {
		t.Fatalf("read sorted keys failed: %v", err)
	}
	sortedVals, err := readMathFloat32Memory(vals, 4)
	if err != nil {
		t.Fatalf("read sorted vals failed: %v", err)
	}
	expectedKeys := []float32{1, 1, 2, 3}
	expectedVals := []float32{10, 11, 20, 30}
	for index, value := range sortedKeys {
		if value != expectedKeys[index] {
			t.Fatalf("unexpected sortByKey keys[%d]: got %v want %v", index, value, expectedKeys[index])
		}
		if sortedVals[index] != expectedVals[index] {
			t.Fatalf("unexpected sortByKey vals[%d]: got %v want %v", index, sortedVals[index], expectedVals[index])
		}
	}

	if err := writeMathFloat32Memory(left, []float32{1, 2, 3, 4}); err != nil {
		t.Fatalf("write left failed: %v", err)
	}
	if err := writeMathFloat32Memory(right, []float32{10, 20, 30, 40}); err != nil {
		t.Fatalf("write right failed: %v", err)
	}
	if err := ctx.Transform(left, out, 4, "square", PolicyDevice); err != nil {
		t.Fatalf("Transform failed: %v", err)
	}
	transformed, err := readMathFloat32Memory(out, 4)
	if err != nil {
		t.Fatalf("read transformed failed: %v", err)
	}
	expectedTransform := []float32{1, 4, 9, 16}
	for index, value := range transformed {
		if value != expectedTransform[index] {
			t.Fatalf("unexpected transform[%d]: got %v want %v", index, value, expectedTransform[index])
		}
	}

	if err := ctx.TransformBinary(left, right, out, 4, "add", PolicyDevice); err != nil {
		t.Fatalf("TransformBinary failed: %v", err)
	}
	binaryOut, err := readMathFloat32Memory(out, 4)
	if err != nil {
		t.Fatalf("read transformBinary failed: %v", err)
	}
	expectedBinary := []float32{11, 22, 33, 44}
	for index, value := range binaryOut {
		if value != expectedBinary[index] {
			t.Fatalf("unexpected transformBinary[%d]: got %v want %v", index, value, expectedBinary[index])
		}
	}

	if err := ctx.Scan(left, out, 4, PolicyDevice); err != nil {
		t.Fatalf("Scan failed: %v", err)
	}
	scanned, err := readMathFloat32Memory(out, 4)
	if err != nil {
		t.Fatalf("read scan failed: %v", err)
	}
	expectedScan := []float32{1, 3, 6, 10}
	for index, value := range scanned {
		if value != expectedScan[index] {
			t.Fatalf("unexpected scan[%d]: got %v want %v", index, value, expectedScan[index])
		}
	}

	if err := ctx.ExclusiveScan(left, out, 4, 5, PolicyDevice); err != nil {
		t.Fatalf("ExclusiveScan failed: %v", err)
	}
	exclusive, err := readMathFloat32Memory(out, 4)
	if err != nil {
		t.Fatalf("read exclusive scan failed: %v", err)
	}
	expectedExclusive := []float32{5, 6, 8, 11}
	for index, value := range exclusive {
		if value != expectedExclusive[index] {
			t.Fatalf("unexpected exclusive[%d]: got %v want %v", index, value, expectedExclusive[index])
		}
	}

	reduced, err := ctx.Reduce(left, 4, 10, PolicyDevice)
	if err != nil {
		t.Fatalf("Reduce failed: %v", err)
	}
	if reduced != 20 {
		t.Fatalf("unexpected reduce: got %v want 20", reduced)
	}

	if err := writeMathFloat32Memory(data, []float32{-2, -1, 0, 1, 2}); err != nil {
		t.Fatalf("rewrite data failed: %v", err)
	}
	partitionIndex, err := ctx.Partition(data, 5, "positive", PolicyDevice)
	if err != nil {
		t.Fatalf("Partition failed: %v", err)
	}
	if partitionIndex != 2 {
		t.Fatalf("unexpected partition index: got %d want 2", partitionIndex)
	}
	partitioned, err := readMathFloat32Memory(data, 5)
	if err != nil {
		t.Fatalf("read partitioned failed: %v", err)
	}
	expectedPartition := []float32{1, 2, -2, -1, 0}
	for index, value := range partitioned {
		if value != expectedPartition[index] {
			t.Fatalf("unexpected partitioned[%d]: got %v want %v", index, value, expectedPartition[index])
		}
	}

	if err := writeMathFloat32Memory(data, []float32{-2, -1, 0, 1, 2, 3}); err != nil {
		t.Fatalf("write copyIf data failed: %v", err)
	}
	selected, err := ctx.CopyIf(data, copyDst, 6, "positive", PolicyDevice)
	if err != nil {
		t.Fatalf("CopyIf failed: %v", err)
	}
	if selected != 3 {
		t.Fatalf("unexpected copyIf count: got %d want 3", selected)
	}
	copied, err := readMathFloat32Memory(copyDst, selected)
	if err != nil {
		t.Fatalf("read copyIf failed: %v", err)
	}
	expectedCopyIf := []float32{1, 2, 3}
	for index, value := range copied {
		if value != expectedCopyIf[index] {
			t.Fatalf("unexpected copyIf[%d]: got %v want %v", index, value, expectedCopyIf[index])
		}
	}

	if err := writeMathFloat32Memory(left, []float32{1, 2, 4, 4}); err != nil {
		t.Fatalf("rewrite left failed: %v", err)
	}
	if err := writeMathFloat32Memory(right, []float32{2, 3, 4, 5}); err != nil {
		t.Fatalf("rewrite right failed: %v", err)
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
		t.Fatalf("read union failed: %v", err)
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
		t.Fatalf("read intersection failed: %v", err)
	}
	expectedIntersection := []float32{2, 4}
	for index, value := range intersectionValues {
		if value != expectedIntersection[index] {
			t.Fatalf("unexpected intersection[%d]: got %v want %v", index, value, expectedIntersection[index])
		}
	}

	if err := writeMathFloat32Memory(data, []float32{1, 2, 2, 3, 3, 5}); err != nil {
		t.Fatalf("rewrite unique data failed: %v", err)
	}
	uniqueCount, err := ctx.Unique(data, 6, PolicyDevice)
	if err != nil {
		t.Fatalf("Unique failed: %v", err)
	}
	if uniqueCount != 4 {
		t.Fatalf("unexpected unique count: got %d want 4", uniqueCount)
	}
	uniqueValues, err := readMathFloat32Memory(data, uniqueCount)
	if err != nil {
		t.Fatalf("read unique values failed: %v", err)
	}
	expectedUnique := []float32{1, 2, 3, 5}
	for index, value := range uniqueValues {
		if value != expectedUnique[index] {
			t.Fatalf("unexpected unique[%d]: got %v want %v", index, value, expectedUnique[index])
		}
	}

	if err := writeMathFloat32Memory(left, []float32{1, 3, 5, 7}); err != nil {
		t.Fatalf("rewrite merge left failed: %v", err)
	}
	if err := writeMathFloat32Memory(right, []float32{2, 4, 6, 8}); err != nil {
		t.Fatalf("rewrite merge right failed: %v", err)
	}
	if err := ctx.Merge(left, right, setOut, 4, 4, PolicyDevice); err != nil {
		t.Fatalf("Merge failed: %v", err)
	}
	mergedValues, err := readMathFloat32Memory(setOut, 8)
	if err != nil {
		t.Fatalf("read merge values failed: %v", err)
	}
	expectedMerge := []float32{1, 2, 3, 4, 5, 6, 7, 8}
	for index, value := range mergedValues {
		if value != expectedMerge[index] {
			t.Fatalf("unexpected merge[%d]: got %v want %v", index, value, expectedMerge[index])
		}
	}

	if err := writeMathFloat32Memory(data, []float32{9, -3, 4, 2, 7, -1}); err != nil {
		t.Fatalf("rewrite min/max data failed: %v", err)
	}
	findIndex, err := ctx.Find(data, 6, 2, PolicyDevice)
	if err != nil {
		t.Fatalf("Find failed: %v", err)
	}
	if findIndex != 3 {
		t.Fatalf("unexpected find index: got %d want 3", findIndex)
	}
	count, err := ctx.Count(data, 6, 7, PolicyDevice)
	if err != nil {
		t.Fatalf("Count failed: %v", err)
	}
	if count != 1 {
		t.Fatalf("unexpected count: got %d want 1", count)
	}
	minValue, minIndex, err := ctx.MinElement(data, 6, PolicyDevice)
	if err != nil {
		t.Fatalf("MinElement failed: %v", err)
	}
	if minValue != -3 || minIndex != 1 {
		t.Fatalf("unexpected min element: got (%v, %d) want (-3, 1)", minValue, minIndex)
	}
	maxValue, maxIndex, err := ctx.MaxElement(data, 6, PolicyDevice)
	if err != nil {
		t.Fatalf("MaxElement failed: %v", err)
	}
	if maxValue != 9 || maxIndex != 0 {
		t.Fatalf("unexpected max element: got (%v, %d) want (9, 0)", maxValue, maxIndex)
	}
}
