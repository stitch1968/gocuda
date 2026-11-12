//go:build cuda

package tests

import (
	"math"
	"testing"

	cuda "github.com/stitch1968/gocuda"
	"github.com/stitch1968/gocuda/kernels"
	"github.com/stitch1968/gocuda/libraries"
	"github.com/stitch1968/gocuda/memory"
	"github.com/stitch1968/gocuda/streams"
)

func TestNativeOpsVectorAndMatrix(t *testing.T) {
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
	ops, err := kernels.NewNativeOps(0)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := ops.Close(); err != nil {
			t.Error(err)
		}
	})
	allocate := func(size int, kind memory.Type) *memory.Memory {
		t.Helper()
		var buffer *memory.Memory
		var err error
		if kind == memory.TypeDeviceOnly {
			buffer, err = memory.AllocDeviceOnlyOnDevice(int64(size*4), 0)
		} else {
			buffer, err = memory.AllocWithTypeAndStream(nil, int64(size*4), kind)
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
	const count = 300
	hostA := allocate(count, memory.TypePinned)
	hostB := allocate(count, memory.TypePinned)
	hostOut := allocate(count, memory.TypePinned)
	a := allocate(count, memory.TypeDeviceOnly)
	b := allocate(count, memory.TypeDeviceOnly)
	output := allocate(count, memory.TypeDeviceOnly)
	t.Cleanup(func() {
		if err := stream.Synchronize(); err != nil {
			t.Error(err)
		}
	})
	aValues, err := memory.View[float32](hostA, count)
	if err != nil {
		t.Fatal(err)
	}
	bValues, err := memory.View[float32](hostB, count)
	if err != nil {
		t.Fatal(err)
	}
	for i := range count {
		aValues[i] = float32(i - 150)
		bValues[i] = float32((i % 7) - 3)
	}
	if err := stream.CopyAsync(a, hostA, count*4); err != nil {
		t.Fatal(err)
	}
	if err := stream.CopyAsync(b, hostB, count*4); err != nil {
		t.Fatal(err)
	}
	if err := ops.VectorAdd(stream, a, b, output, count); err != nil {
		t.Fatal(err)
	}
	if err := ops.SAXPY(stream, 2, a, output, count); err != nil {
		t.Fatal(err)
	}
	if err := stream.CopyAsync(hostOut, output, count*4); err != nil {
		t.Fatal(err)
	}
	if err := stream.Synchronize(); err != nil {
		t.Fatal(err)
	}
	values, err := memory.View[float32](hostOut, count)
	if err != nil {
		t.Fatal(err)
	}
	for i, value := range values {
		if want := 3*aValues[i] + bValues[i]; value != want {
			t.Fatalf("vector output[%d] = %v, want %v", i, value, want)
		}
	}
	if err := ops.VectorAdd(stream, a, b, output, count+1); err == nil {
		t.Fatal("expected undersized vector buffer to fail")
	}

	const m, n, k = 19, 21, 7
	matHostA := allocate(m*k, memory.TypePinned)
	matHostB := allocate(k*n, memory.TypePinned)
	matHostOut := allocate(m*n, memory.TypePinned)
	matA := allocate(m*k, memory.TypeDeviceOnly)
	matB := allocate(k*n, memory.TypeDeviceOnly)
	matOut := allocate(m*n, memory.TypeDeviceOnly)
	// Register after the matrix buffers so it runs before they are freed.
	t.Cleanup(func() {
		if err := stream.Synchronize(); err != nil {
			t.Error(err)
		}
	})
	ma, err := memory.View[float32](matHostA, m*k)
	if err != nil {
		t.Fatal(err)
	}
	mb, err := memory.View[float32](matHostB, k*n)
	if err != nil {
		t.Fatal(err)
	}
	for i := range ma {
		ma[i] = float32(i%5 - 2)
	}
	for i := range mb {
		mb[i] = float32(i%7 - 3)
	}
	if err := ops.MatrixMultiply(stream, matA, matB, matA, 1, 3, 3); err == nil {
		t.Fatal("expected aliased output to fail")
	}
	if err := stream.CopyAsync(matA, matHostA, m*k*4); err != nil {
		t.Fatal(err)
	}
	if err := stream.CopyAsync(matB, matHostB, k*n*4); err != nil {
		t.Fatal(err)
	}
	if err := ops.MatrixMultiply(stream, matA, matB, matOut, m, n, k); err != nil {
		t.Fatal(err)
	}
	if err := stream.CopyAsync(matHostOut, matOut, m*n*4); err != nil {
		t.Fatal(err)
	}
	if err := stream.Synchronize(); err != nil {
		t.Fatal(err)
	}
	result, err := memory.View[float32](matHostOut, m*n)
	if err != nil {
		t.Fatal(err)
	}
	for row := range m {
		for col := range n {
			var want float32
			for inner := range k {
				want += ma[row*k+inner] * mb[inner*n+col]
			}
			if got := result[row*n+col]; math.Abs(float64(got-want)) > 1e-5 {
				t.Fatalf("matrix output[%d,%d] = %v, want %v", row, col, got, want)
			}
		}
	}
	blas, err := libraries.NewNativeBLAS(stream)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := blas.Close(); err != nil {
			t.Error(err)
		}
	})
	t.Cleanup(func() {
		if err := stream.Synchronize(); err != nil {
			t.Error(err)
		}
	})
	if err := blas.SGEMM(matA, matB, matA, 1, 3, 3, 1, 0); err == nil {
		t.Fatal("expected cuBLAS to reject an aliased output")
	}
	if err := blas.SGEMM(matHostA, matB, matOut, m, n, k, 1, 0); err == nil {
		t.Fatal("expected cuBLAS to reject host memory")
	}
	for i := range result {
		result[i] = 4
	}
	if err := stream.CopyAsync(matOut, matHostOut, m*n*4); err != nil {
		t.Fatal(err)
	}
	if err := blas.SGEMM(matA, matB, matOut, m, n, k, 2, 0.5); err != nil {
		t.Fatal(err)
	}
	if err := matA.Free(); err == nil {
		t.Fatal("expected cuBLAS input to remain retained")
	}
	if err := stream.CopyAsync(matHostOut, matOut, m*n*4); err != nil {
		t.Fatal(err)
	}
	if err := blas.Close(); err != nil {
		t.Fatal(err)
	}
	for row := range m {
		for col := range n {
			var sum float32
			for inner := range k {
				sum += ma[row*k+inner] * mb[inner*n+col]
			}
			want := 2*sum + 2
			if got := result[row*n+col]; math.Abs(float64(got-want)) > 1e-5 {
				t.Fatalf("cuBLAS output[%d,%d] = %v, want %v", row, col, got, want)
			}
		}
	}
	if err := blas.SGEMM(matA, matB, matOut, m, n, k, 1, 0); err == nil {
		t.Fatal("expected closed cuBLAS handle to fail")
	}
}
