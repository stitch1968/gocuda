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
