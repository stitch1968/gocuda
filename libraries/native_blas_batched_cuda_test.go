//go:build cuda

package libraries

import (
	"fmt"
	"math"
	"testing"
	"unsafe"

	cuda "github.com/stitch1968/gocuda"
	"github.com/stitch1968/gocuda/memory"
	"github.com/stitch1968/gocuda/streams"
)

func TestNativeBLASStridedBatchedOnEachDevice(t *testing.T) {
	if err := cuda.Initialize(); err != nil {
		t.Fatal(err)
	}
	if !cuda.ShouldUseCuda() {
		t.Skip("CUDA hardware unavailable")
	}
	devices, err := cuda.GetDevices()
	if err != nil {
		t.Fatal(err)
	}
	caller, err := cuda.NewContext(0)
	if err != nil {
		t.Fatal(err)
	}
	for deviceID := range devices {
		t.Run(fmt.Sprintf("%d_%s", deviceID, devices[deviceID].Name), func(t *testing.T) {
			if err := caller.Run(func() error {
				testNativeBLASBatchesOnDevice(t, deviceID, len(devices))
				return nil
			}); err != nil {
				t.Fatal(err)
			}
		})
	}
}

func testNativeBLASBatchesOnDevice(t *testing.T, deviceID, deviceCount int) {
	t.Helper()
	stream, err := streams.NewNativeStream(deviceID, streams.StreamNonBlocking)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := stream.Close(); err != nil {
			t.Error(err)
		}
	})
	blas, err := NewNativeBLAS(stream)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := blas.Close(); err != nil {
			t.Error(err)
		}
	})
	for _, opA := range []BLASOperation{BLASOpN, BLASOpT} {
		for _, opB := range []BLASOperation{BLASOpN, BLASOpT} {
			t.Run(fmt.Sprintf("float32_%d_%d", opA, opB), func(t *testing.T) {
				testNativeBLASBatchedValues[float32](t, deviceID, stream, blas, opA, opB)
			})
			t.Run(fmt.Sprintf("float64_%d_%d", opA, opB), func(t *testing.T) {
				testNativeBLASBatchedValues[float64](t, deviceID, stream, blas, opA, opB)
			})
		}
	}
	t.Run("overlapping inputs", func(t *testing.T) {
		testNativeBLASBatchedOverlap(t, deviceID, stream, blas)
	})
	if deviceCount > 1 {
		t.Run("cross-device rejection", func(t *testing.T) {
			allocate := func(device int) *memory.Memory {
				t.Helper()
				buffer, err := memory.AllocDeviceOnlyOnDevice(4, device)
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
			foreign, local, output := allocate((deviceID+1)%deviceCount), allocate(deviceID), allocate(deviceID)
			t.Cleanup(func() {
				if err := stream.Synchronize(); err != nil {
					t.Error(err)
				}
			})
			if err := blas.SGEMMStridedBatched(foreign, local, output, 1, 1, 1, 1, 1, 1,
				BLASOpN, BLASOpN, 1, 1, 1, 1, 1, 0); err == nil {
				t.Fatal("expected cross-device input to fail")
			}
		})
	}
}

func testNativeBLASBatchedValues[T ~float32 | ~float64](t *testing.T, deviceID int, stream *streams.NativeStream, blas *NativeBLAS, opA, opB BLASOperation) {
	t.Helper()
	const m, n, k, batch = 3, 4, 2, 3
	aRows, aCols, bRows, bCols := m, k, k, n
	if opA == BLASOpT {
		aRows, aCols = k, m
	}
	if opB == BLASOpT {
		bRows, bCols = n, k
	}
	lda, ldb, ldc := aCols+2, bCols+1, n+2
	spanA, spanB, spanC := (aRows-1)*lda+aCols, (bRows-1)*ldb+bCols, (m-1)*ldc+n
	strideA, strideB, strideC := spanA+3, spanB+2, spanC+4
	aValues := make([]T, (batch-1)*strideA+spanA)
	bValues := make([]T, (batch-1)*strideB+spanB)
	cValues := make([]T, (batch-1)*strideC+spanC)
	for i := range cValues {
		cValues[i] = -7
	}
	for instance := range batch {
		for row := range m {
			for inner := range k {
				index := row*lda + inner
				if opA == BLASOpT {
					index = inner*lda + row
				}
				aValues[instance*strideA+index] = T(float64(row*2+inner+instance+1)/4 + math.Ldexp(float64(instance+1), -32))
			}
		}
		for inner := range k {
			for col := range n {
				index := inner*ldb + col
				if opB == BLASOpT {
					index = col*ldb + inner
				}
				bValues[instance*strideB+index] = T(float64(inner*3+col-instance-2) / 2)
			}
		}
	}
	var zero T
	elementSize := int64(unsafe.Sizeof(zero))
	allocate := func(count int) *memory.Memory {
		t.Helper()
		buffer, err := memory.AllocDeviceOnlyOnDevice(int64(count)*elementSize, deviceID)
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
	a, b, c := allocate(len(aValues)), allocate(len(bValues)), allocate(len(cValues))
	t.Cleanup(func() {
		if err := stream.Synchronize(); err != nil {
			t.Error(err)
		}
	})
	for _, entry := range []struct {
		mem    *memory.Memory
		values []T
	}{{a, aValues}, {b, bValues}, {c, cValues}} {
		if err := writeMathTypedMemory(entry.mem, entry.values); err != nil {
			t.Fatal(err)
		}
	}
	var err error
	alpha, beta := 1.5, -.5
	if elementSize == 4 {
		err = blas.SGEMMStridedBatched(a, b, c, m, n, k, lda, ldb, ldc, opA, opB,
			int64(strideA), int64(strideB), int64(strideC), batch, float32(alpha), float32(beta))
	} else {
		alpha += math.Ldexp(1, -30)
		beta += math.Ldexp(1, -32)
		err = blas.DGEMMStridedBatched(a, b, c, m, n, k, lda, ldb, ldc, opA, opB,
			int64(strideA), int64(strideB), int64(strideC), batch, alpha, beta)
	}
	if err != nil {
		t.Fatal(err)
	}
	for _, buffer := range []*memory.Memory{a, b, c} {
		if err := buffer.Free(); err == nil {
			t.Fatal("expected queued GEMM buffers to remain retained")
		}
	}
	if err := stream.Synchronize(); err != nil {
		t.Fatal(err)
	}
	got, err := readMathTypedMemory[T](c, len(cValues))
	if err != nil {
		t.Fatal(err)
	}
	want := make([]float64, len(cValues))
	for i := range want {
		want[i] = -7
	}
	for instance := range batch {
		for row := range m {
			for col := range n {
				sum := 0.0
				for inner := range k {
					aIndex, bIndex := row*lda+inner, inner*ldb+col
					if opA == BLASOpT {
						aIndex = inner*lda + row
					}
					if opB == BLASOpT {
						bIndex = col*ldb + inner
					}
					sum += float64(aValues[instance*strideA+aIndex]) * float64(bValues[instance*strideB+bIndex])
				}
				want[instance*strideC+row*ldc+col] = alpha*sum - 7*beta
			}
		}
	}
	tolerance := 1e-6
	if elementSize == 8 {
		tolerance = 1e-12
	}
	for i, value := range got {
		if diff := math.Abs(float64(value) - want[i]); math.IsNaN(diff) || diff > tolerance {
			t.Fatalf("output[%d] = %v, want %v", i, value, want[i])
		}
	}
}

func testNativeBLASBatchedOverlap(t *testing.T, deviceID int, stream *streams.NativeStream, blas *NativeBLAS) {
	t.Helper()
	allocate := func(values []float32, size int64) *memory.Memory {
		t.Helper()
		buffer, err := memory.AllocDeviceOnlyOnDevice(size, deviceID)
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() {
			if err := buffer.Free(); err != nil {
				t.Error(err)
			}
		})
		if values != nil {
			if err := writeMathFloat32Memory(buffer, values); err != nil {
				t.Fatal(err)
			}
		}
		return buffer
	}
	a := allocate([]float32{1, 2, 3, 4}, 16)
	b := allocate([]float32{2, 3, 4}, 12)
	c := allocate(nil, 24)
	t.Cleanup(func() {
		if err := stream.Synchronize(); err != nil {
			t.Error(err)
		}
	})
	if err := blas.SGEMMStridedBatched(a, b, c, 2, 1, 1, 1, 1, 1, BLASOpN, BLASOpN, 1, 1, 2, 3, 1, 0); err != nil {
		t.Fatal(err)
	}
	if err := stream.Synchronize(); err != nil {
		t.Fatal(err)
	}
	got, err := readMathFloat32Memory(c, 6)
	if err != nil {
		t.Fatal(err)
	}
	want := []float32{2, 4, 6, 9, 12, 16}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("output[%d] = %v, want %v", i, got[i], want[i])
		}
	}
}
