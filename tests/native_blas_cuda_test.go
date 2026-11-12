//go:build cuda

package tests

import (
	"math"
	"testing"
	"unsafe"

	cuda "github.com/stitch1968/gocuda"
	"github.com/stitch1968/gocuda/libraries"
	"github.com/stitch1968/gocuda/memory"
	"github.com/stitch1968/gocuda/streams"
)

func TestNativeBLASTransposesAndStrides(t *testing.T) {
	if err := cuda.Initialize(); err != nil {
		t.Fatal(err)
	}
	if !cuda.ShouldUseCuda() {
		t.Skip("CUDA hardware unavailable")
	}
	stream, err := streams.NewNativeStream(0, streams.StreamNonBlocking)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := stream.Close(); err != nil {
			t.Error(err)
		}
	})
	blas, err := libraries.NewNativeBLAS(stream)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := blas.Close(); err != nil {
			t.Error(err)
		}
	})
	for _, opA := range []libraries.BLASOperation{libraries.BLASOpN, libraries.BLASOpT} {
		for _, opB := range []libraries.BLASOperation{libraries.BLASOpN, libraries.BLASOpT} {
			t.Run("float32", func(t *testing.T) { testNativeBLASStrides[float32](t, stream, blas, opA, opB) })
			t.Run("float64", func(t *testing.T) { testNativeBLASStrides[float64](t, stream, blas, opA, opB) })
		}
	}
}

func testNativeBLASStrides[T ~float32 | ~float64](t *testing.T, stream *streams.NativeStream, blas *libraries.NativeBLAS, opA, opB libraries.BLASOperation) {
	t.Helper()
	const m, n, k = 3, 4, 2
	aRows, aCols, bRows, bCols := m, k, k, n
	if opA == libraries.BLASOpT {
		aRows, aCols = k, m
	}
	if opB == libraries.BLASOpT {
		bRows, bCols = n, k
	}
	lda, ldb, ldc := aCols+2, bCols+1, n+2
	var zero T
	elementSize := int(unsafe.Sizeof(zero))
	alloc := func(count int, kind memory.Type) *memory.Memory {
		t.Helper()
		var buffer *memory.Memory
		var err error
		if kind == memory.TypeDeviceOnly {
			buffer, err = memory.AllocDeviceOnlyOnDevice(int64(count*elementSize), 0)
		} else {
			buffer, err = memory.AllocWithTypeAndStream(nil, int64(count*elementSize), kind)
		}
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
	hostA, hostB, hostC := alloc(aRows*lda, memory.TypePinned), alloc(bRows*ldb, memory.TypePinned), alloc(m*ldc, memory.TypePinned)
	a, b, c := alloc(aRows*lda, memory.TypeDeviceOnly), alloc(bRows*ldb, memory.TypeDeviceOnly), alloc(m*ldc, memory.TypeDeviceOnly)
	t.Cleanup(func() {
		if err := stream.Synchronize(); err != nil {
			t.Error(err)
		}
	})
	valuesA, err := memory.View[T](hostA, aRows*lda)
	if err != nil {
		t.Fatal(err)
	}
	valuesB, err := memory.View[T](hostB, bRows*ldb)
	if err != nil {
		t.Fatal(err)
	}
	valuesC, err := memory.View[T](hostC, m*ldc)
	if err != nil {
		t.Fatal(err)
	}
	for row := range m {
		for inner := range k {
			index := row*lda + inner
			if opA == libraries.BLASOpT {
				index = inner*lda + row
			}
			valuesA[index] = T(row*2 + inner + 1)
		}
	}
	for inner := range k {
		for col := range n {
			index := inner*ldb + col
			if opB == libraries.BLASOpT {
				index = col*ldb + inner
			}
			valuesB[index] = T(inner + col - 2)
		}
	}
	for i := range valuesC {
		valuesC[i] = -7
	}
	for _, copyOp := range []struct {
		dst, src *memory.Memory
		size     int
	}{
		{a, hostA, aRows * lda * elementSize},
		{b, hostB, bRows * ldb * elementSize},
		{c, hostC, m * ldc * elementSize},
	} {
		if err := stream.CopyAsync(copyOp.dst, copyOp.src, int64(copyOp.size)); err != nil {
			t.Fatal(err)
		}
	}
	if err := blas.SGEMMEx(a, b, c, m, n, k, 1, ldb, ldc, opA, opB, 1, 0); err == nil {
		t.Fatal("expected invalid A stride to fail")
	}
	if elementSize == 4 {
		err = blas.SGEMMEx(a, b, c, m, n, k, lda, ldb, ldc, opA, opB, 1.5, -0.5)
	} else {
		err = blas.DGEMMEx(a, b, c, m, n, k, lda, ldb, ldc, opA, opB, 1.5, -0.5)
	}
	if err != nil {
		t.Fatal(err)
	}
	if err := stream.CopyAsync(hostC, c, int64(m*ldc*elementSize)); err != nil {
		t.Fatal(err)
	}
	if err := stream.Synchronize(); err != nil {
		t.Fatal(err)
	}
	for row := range m {
		for col := range n {
			var sum float64
			for inner := range k {
				sum += float64(row*2+inner+1) * float64(inner+col-2)
			}
			want := 1.5*sum + 3.5
			if got := float64(valuesC[row*ldc+col]); math.Abs(got-want) > 1e-5 {
				t.Fatalf("opA=%d opB=%d output[%d,%d]=%v, want %v", opA, opB, row, col, got, want)
			}
		}
		for col := n; col < ldc; col++ {
			if got := valuesC[row*ldc+col]; got != -7 {
				t.Fatalf("output padding[%d,%d] was modified: %v", row, col, got)
			}
		}
	}
}
