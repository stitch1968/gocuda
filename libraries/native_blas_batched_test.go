package libraries

import (
	"math"
	"testing"

	"github.com/stitch1968/gocuda/memory"
)

func TestNativeBLASBatchedValidation(t *testing.T) {
	allocate := func(size int64) *memory.Memory {
		t.Helper()
		buffer, err := memory.Alloc(size)
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() {
			if err := buffer.Free(); err != nil {
				t.Error(err)
			}
		})
		return buffer
	}
	a, b, c := allocate(32), allocate(32), allocate(32)
	small := allocate(16)
	blas := &NativeBLAS{deviceID: a.GetDeviceID()}
	validate := func(a, b, c *memory.Memory, strideA, strideB, strideC int64, batch int) error {
		return blas.validateGEMMBatched(a, b, c, 2, 2, 2, 2, 2, 2, BLASOpN, BLASOpN, strideA, strideB, strideC, batch, 4)
	}
	if err := validate(a, b, c, 1, 4, 4, 2); err != nil {
		t.Fatalf("overlapping input batches should be allowed: %v", err)
	}
	for _, tc := range []struct {
		name                      string
		a, b, c                   *memory.Memory
		strideA, strideB, strideC int64
		batch                     int
	}{
		{"zero input stride", a, b, c, 0, 4, 4, 2},
		{"negative input stride", a, b, c, 4, -1, 4, 2},
		{"overlapping outputs", a, b, c, 4, 4, 3, 2},
		{"zero output stride", a, b, c, 4, 4, 0, 2},
		{"short first batch", small, b, c, 4, 4, 4, 2},
		{"short output batch", a, b, small, 4, 4, 4, 2},
		{"overflowing stride", a, b, c, math.MaxInt64, 4, 4, 2},
		{"zero batches", a, b, c, 4, 4, 4, 0},
		{"output aliases input", a, b, a, 4, 4, 4, 2},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if err := validate(tc.a, tc.b, tc.c, tc.strideA, tc.strideB, tc.strideC, tc.batch); err == nil {
				t.Fatal("expected invalid batch to fail")
			}
		})
	}
	if uint64(^uint(0)) > math.MaxInt32 {
		tooMany := int64(math.MaxInt32) + 1
		if err := validate(a, b, c, 4, 4, 4, int(tooMany)); err == nil {
			t.Fatal("expected batch count exceeding cuBLAS int range to fail")
		}
	}
}
