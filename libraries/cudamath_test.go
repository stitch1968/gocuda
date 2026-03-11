package libraries

import (
	"math"
	"testing"

	"github.com/stitch1968/gocuda/memory"
)

func TestMathContextVectorAddAndReduction(t *testing.T) {
	ctx, err := CreateMathContext(MathConfig{DataType: MathDataFloat32, VectorSize: 4})
	if err != nil {
		t.Fatalf("CreateMathContext failed: %v", err)
	}
	defer ctx.Destroy()

	a, err := memory.Alloc(4 * 4)
	if err != nil {
		t.Fatalf("alloc a failed: %v", err)
	}
	defer a.Free()
	b, err := memory.Alloc(4 * 4)
	if err != nil {
		t.Fatalf("alloc b failed: %v", err)
	}
	defer b.Free()
	out, err := memory.Alloc(4 * 4)
	if err != nil {
		t.Fatalf("alloc out failed: %v", err)
	}
	defer out.Free()

	if err := writeMathFloat32Memory(a, []float32{1, 2, 3, 4}); err != nil {
		t.Fatalf("write a failed: %v", err)
	}
	if err := writeMathFloat32Memory(b, []float32{5, 6, 7, 8}); err != nil {
		t.Fatalf("write b failed: %v", err)
	}

	if err := ctx.VectorAdd(a, b, out, 4); err != nil {
		t.Fatalf("VectorAdd failed: %v", err)
	}

	values, err := readMathFloat32Memory(out, 4)
	if err != nil {
		t.Fatalf("read out failed: %v", err)
	}
	expected := []float32{6, 8, 10, 12}
	for index, value := range values {
		if value != expected[index] {
			t.Fatalf("unexpected output[%d]: got %v want %v", index, value, expected[index])
		}
	}

	sum, err := ctx.VectorSum(out, 4)
	if err != nil {
		t.Fatalf("VectorSum failed: %v", err)
	}
	if sum != 36 {
		t.Fatalf("unexpected sum: got %v want 36", sum)
	}

	maxValue, maxIndex, err := ctx.VectorMax(out, 4)
	if err != nil {
		t.Fatalf("VectorMax failed: %v", err)
	}
	if maxValue != 12 || maxIndex != 3 {
		t.Fatalf("unexpected max result: got (%v, %d)", maxValue, maxIndex)
	}
}

func TestMathContextHighPrecisionUnaryOps(t *testing.T) {
	ctx, err := CreateMathContext(MathConfig{DataType: MathDataFloat64, VectorSize: 3, Precision: MathPrecisionIEEE})
	if err != nil {
		t.Fatalf("CreateMathContext failed: %v", err)
	}
	defer ctx.Destroy()

	in, err := memory.Alloc(3 * 8)
	if err != nil {
		t.Fatalf("alloc in failed: %v", err)
	}
	defer in.Free()
	out, err := memory.Alloc(3 * 8)
	if err != nil {
		t.Fatalf("alloc out failed: %v", err)
	}
	defer out.Free()

	if err := writeMathFloat64Memory(in, []float64{1, 4, 9}); err != nil {
		t.Fatalf("write in failed: %v", err)
	}
	if err := ctx.VectorSqrt(in, out, 3); err != nil {
		t.Fatalf("VectorSqrt failed: %v", err)
	}

	values, err := readMathFloat64Memory(out, 3)
	if err != nil {
		t.Fatalf("read out failed: %v", err)
	}
	expected := []float64{1, 2, 3}
	for index, value := range values {
		if math.Abs(value-expected[index]) > 1e-9 {
			t.Fatalf("unexpected sqrt[%d]: got %v want %v", index, value, expected[index])
		}
	}

	norm, err := ctx.VectorNorm(in, 3)
	if err != nil {
		t.Fatalf("VectorNorm failed: %v", err)
	}
	if math.Abs(norm-math.Sqrt(98)) > 1e-9 {
		t.Fatalf("unexpected norm: got %v want %v", norm, math.Sqrt(98))
	}
}

func TestMathContextComplexMagnitude(t *testing.T) {
	ctx, err := CreateMathContext(MathConfig{DataType: MathDataComplexFloat32, VectorSize: 2})
	if err != nil {
		t.Fatalf("CreateMathContext failed: %v", err)
	}
	defer ctx.Destroy()

	in, err := memory.Alloc(2 * 8)
	if err != nil {
		t.Fatalf("alloc in failed: %v", err)
	}
	defer in.Free()
	out, err := memory.Alloc(2 * 4)
	if err != nil {
		t.Fatalf("alloc out failed: %v", err)
	}
	defer out.Free()

	if err := writeMathComplex64Memory(in, []complex64{3 + 4i, 5 + 12i}); err != nil {
		t.Fatalf("write in failed: %v", err)
	}
	if err := ctx.VectorComplexAbs(in, out, 2); err != nil {
		t.Fatalf("VectorComplexAbs failed: %v", err)
	}

	values, err := readMathFloat32Memory(out, 2)
	if err != nil {
		t.Fatalf("read out failed: %v", err)
	}
	expected := []float32{5, 13}
	for index, value := range values {
		if math.Abs(float64(value-expected[index])) > 1e-5 {
			t.Fatalf("unexpected abs[%d]: got %v want %v", index, value, expected[index])
		}
	}
}
