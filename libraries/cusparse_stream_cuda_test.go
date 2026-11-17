//go:build cuda

package libraries

import (
	"errors"
	"fmt"
	"reflect"
	"sync"
	"testing"

	cuda "github.com/stitch1968/gocuda"
	"github.com/stitch1968/gocuda/internal"
	"github.com/stitch1968/gocuda/kernels"
	"github.com/stitch1968/gocuda/memory"
	"github.com/stitch1968/gocuda/streams"
)

func sparseStreamDeviceBuffer(t *testing.T, size int64, deviceID int) *memory.Memory {
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
	return buffer
}

func sparseStreamPinnedBuffer(t *testing.T, values []float32) *memory.Memory {
	t.Helper()
	buffer, err := memory.AllocWithTypeAndStream(nil, int64(len(values)*4), memory.TypePinned)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := buffer.Free(); err != nil {
			t.Error(err)
		}
	})
	view, err := memory.View[float32](buffer, len(values))
	if err != nil {
		t.Fatal(err)
	}
	copy(view, values)
	return buffer
}

func sparseStreamCheckValues(t *testing.T, buffer *memory.Memory, want []float32) {
	t.Helper()
	got, err := readMathFloat32Memory(buffer, len(want))
	if err != nil || !reflect.DeepEqual(got, want) {
		t.Fatalf("stream output = %v, error = %v; want %v", got, err, want)
	}
}

func TestNativeSparseContextStreamFence(t *testing.T) {
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
	if devices[0].Properties.Major < 7 {
		t.Skip("the delay kernel requires compute capability 7.0 or newer")
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
	gate, err := streams.NewNativeStream(0, streams.StreamNonBlocking)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := gate.Close(); err != nil {
			t.Error(err)
		}
	})
	event, err := streams.NewNativeEvent(0, streams.NativeEventDisableTiming)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := event.Close(); err != nil {
			t.Error(err)
		}
	})
	module, err := kernels.LoadPTX(0, nativeTestDelayPTX)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := module.Close(); err != nil {
			t.Error(err)
		}
	})
	delay, err := module.Function("wait_ticks")
	if err != nil {
		t.Fatal(err)
	}
	ctx, err := CreateNativeSparseContextOnDevice(0)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := ctx.DestroyContext(); err != nil {
			t.Error(err)
		}
	})
	matrix, err := ctx.CreateCSRMatrix(2, 2, []int32{0, 2, 3}, []int32{0, 1, 1}, []float32{2, 1, 3})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := matrix.Destroy(); err != nil {
			t.Error(err)
		}
	})
	values := sparseStreamPinnedBuffer(t, []float32{4, 2, 6})
	input := sparseStreamDeviceBuffer(t, 8, 0)
	output := sparseStreamDeviceBuffer(t, 8, 0)
	t.Cleanup(func() {
		if err := ctx.DestroyContext(); err != nil {
			t.Error(err)
		}
	})
	if err := writeMathFloat32Memory(input, []float32{1, 2}); err != nil {
		t.Fatal(err)
	}
	if err := ctx.SetStream(stream); err != nil {
		t.Fatal(err)
	}
	closed, err := streams.NewNativeStream(0, streams.StreamNonBlocking)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := closed.Close(); err != nil {
			t.Error(err)
		}
	})
	if err := closed.Close(); err != nil {
		t.Fatal(err)
	}
	if err := ctx.SetStream(closed); !errors.Is(err, streams.ErrNativeStreamClosed) {
		t.Fatalf("attaching a closed stream error = %v", err)
	}
	if ctx.nativeStream != stream {
		t.Fatal("failed stream change must preserve the previous stream")
	}
	if err := delay.Launch(gate, kernels.Dim3{X: 1, Y: 1, Z: 1}, kernels.Dim3{X: 1, Y: 1, Z: 1}, 0, kernels.Uint64Arg(200_000_000)); err != nil {
		t.Fatal(err)
	}
	if err := event.Record(gate); err != nil {
		t.Fatal(err)
	}
	if err := stream.WaitEvent(event); err != nil {
		t.Fatal(err)
	}
	if err := stream.CopyAsync(matrix.values, values, 12); err != nil {
		t.Fatal(err)
	}
	complete, err := event.Query()
	if err != nil {
		t.Fatal(err)
	}
	if complete {
		t.Skip("delay completed before the stream fence could be tested")
	}
	if err := ctx.SpMV(1, matrix, input, 0, output); err != nil {
		t.Fatal(err)
	}
	complete, err = event.Query()
	if err != nil || !complete {
		t.Fatalf("producer event complete = %v, error = %v", complete, err)
	}
	sparseStreamCheckValues(t, output, []float32{8, 12})
}

func TestNativeSparseContextStreamCleanupRetry(t *testing.T) {
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
	defer stream.Close()
	ctx, err := CreateNativeSparseContextOnDevice(0)
	if err != nil {
		t.Fatal(err)
	}
	defer ctx.DestroyContext()
	buffer, err := memory.AllocDeviceOnlyOnDevice(4, 0)
	if err != nil {
		t.Fatal(err)
	}
	defer buffer.Free()
	if err := ctx.SetStream(stream); err != nil {
		t.Fatal(err)
	}
	held, err := buffer.Retain()
	if err != nil {
		t.Fatal(err)
	}
	defer held.Release()
	expected := errors.New("descriptor cleanup failed")
	calls := 0
	ctx.pending = append(ctx.pending, held)
	ctx.pendingCleanups = append(ctx.pendingCleanups, func() error {
		calls++
		if calls == 1 {
			return expected
		}
		return nil
	})
	if err := ctx.SetStream(nil); !errors.Is(err, expected) {
		t.Fatalf("stream change error = %v, want %v", err, expected)
	}
	if ctx.nativeStream != stream {
		t.Fatal("failed cleanup must leave the original stream attached")
	}
	if err := buffer.Free(); err == nil {
		t.Fatal("buffer must remain retained after cleanup failure")
	}
	if err := ctx.SetStream(nil); err != nil {
		t.Fatal(err)
	}
	if calls != 2 || ctx.nativeStream != nil {
		t.Fatal("cleanup retry must complete before detaching")
	}
	if err := buffer.Free(); err != nil {
		t.Fatalf("successful cleanup must release the buffer: %v", err)
	}
	if err := ctx.SetStream(stream); err != nil {
		t.Fatal(err)
	}
	matrix, err := ctx.CreateCSRMatrix(1, 1, []int32{0, 1}, []int32{0}, []float32{2})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := matrix.Destroy(); err != nil {
			t.Error(err)
		}
	})
	input := sparseStreamDeviceBuffer(t, 4, 0)
	output := sparseStreamDeviceBuffer(t, 4, 0)
	pending := sparseStreamDeviceBuffer(t, 4, 0)
	if err := writeMathFloat32Memory(input, []float32{3}); err != nil {
		t.Fatal(err)
	}
	if err := writeMathFloat32Memory(output, []float32{9}); err != nil {
		t.Fatal(err)
	}
	pendingHeld, err := pending.Retain()
	if err != nil {
		t.Fatal(err)
	}
	defer pendingHeld.Release()
	operationCalls := 0
	ctx.pending = append(ctx.pending, pendingHeld)
	ctx.pendingCleanups = append(ctx.pendingCleanups, func() error {
		operationCalls++
		if operationCalls == 1 {
			return expected
		}
		return nil
	})
	if err := ctx.SpMV(1, matrix, input, 0, output); !errors.Is(err, expected) {
		t.Fatalf("operation cleanup error = %v, want %v", err, expected)
	}
	sparseStreamCheckValues(t, output, []float32{9})
	if err := pending.Free(); err == nil {
		t.Fatal("buffer must remain retained after operation cleanup failure")
	}
	if err := ctx.SpMV(1, matrix, input, 0, output); err != nil {
		t.Fatal(err)
	}
	sparseStreamCheckValues(t, output, []float32{6})
	if operationCalls != 2 {
		t.Fatalf("operation cleanup attempts = %d, want 2", operationCalls)
	}
	if err := pending.Free(); err != nil {
		t.Fatalf("operation cleanup retry must release buffer: %v", err)
	}
	syncPending := sparseStreamDeviceBuffer(t, 4, 0)
	syncHeld, err := syncPending.Retain()
	if err != nil {
		t.Fatal(err)
	}
	defer syncHeld.Release()
	syncCalls := 0
	ctx.pending = append(ctx.pending, syncHeld)
	ctx.pendingCleanups = append(ctx.pendingCleanups, func() error {
		syncCalls++
		if syncCalls == 1 {
			return expected
		}
		return nil
	})
	if err := ctx.Synchronize(); !errors.Is(err, expected) {
		t.Fatalf("synchronization cleanup error = %v, want %v", err, expected)
	}
	if err := syncPending.Free(); err == nil {
		t.Fatal("buffer must remain retained after synchronization cleanup failure")
	}
	if err := ctx.Synchronize(); err != nil {
		t.Fatal(err)
	}
	if syncCalls != 2 || ctx.nativeStream != stream {
		t.Fatal("synchronization retry must release resources without detaching")
	}
	if err := syncPending.Free(); err != nil {
		t.Fatalf("synchronization retry must release buffer: %v", err)
	}
	if err := stream.Close(); err != nil {
		t.Fatal(err)
	}
	if err := ctx.DestroyContext(); err != nil {
		t.Fatalf("destroy with an already closed stream: %v", err)
	}
}

func TestNativeSparseContextStreamShared(t *testing.T) {
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
	contexts := make([]*SparseContext, 2)
	for i := range contexts {
		contexts[i], err = CreateNativeSparseContextOnDevice(0)
		if err != nil {
			t.Fatal(err)
		}
		ctx := contexts[i]
		t.Cleanup(func() {
			if err := ctx.DestroyContext(); err != nil {
				t.Error(err)
			}
		})
		if err := ctx.SetStream(stream); err != nil {
			t.Fatal(err)
		}
	}
	matrix, err := contexts[0].CreateCSRMatrix(2, 2, []int32{0, 2, 3}, []int32{0, 1, 1}, []float32{2, 1, 3})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := matrix.Destroy(); err != nil {
			t.Error(err)
		}
	})
	input := sparseStreamDeviceBuffer(t, 8, 0)
	if err := writeMathFloat32Memory(input, []float32{1, 2}); err != nil {
		t.Fatal(err)
	}
	outputs := []*memory.Memory{sparseStreamDeviceBuffer(t, 8, 0), sparseStreamDeviceBuffer(t, 8, 0)}
	for _, ctx := range contexts {
		t.Cleanup(func() {
			if err := ctx.DestroyContext(); err != nil {
				t.Error(err)
			}
		})
	}
	var wg sync.WaitGroup
	errs := make(chan error, len(contexts))
	for i, ctx := range contexts {
		wg.Add(1)
		go func(index int, ctx *SparseContext) {
			defer wg.Done()
			for range 3 {
				if err := ctx.SpMV(1, matrix, input, 0, outputs[index]); err != nil {
					errs <- err
					return
				}
			}
			got, err := readMathFloat32Memory(outputs[index], 2)
			if err != nil {
				errs <- err
				return
			}
			if !reflect.DeepEqual(got, []float32{4, 6}) {
				errs <- fmt.Errorf("shared stream worker %d got %v", index, got)
			}
		}(i, ctx)
	}
	wg.Wait()
	close(errs)
	for err := range errs {
		t.Error(err)
	}
	if err := contexts[0].DestroyContext(); err != nil {
		t.Fatal(err)
	}
	if err := contexts[1].SpMV(1, matrix, input, 0, outputs[1]); err != nil {
		t.Fatal(err)
	}
	sparseStreamCheckValues(t, outputs[1], []float32{4, 6})
}

func TestNativeSparseContextStreamConcurrent(t *testing.T) {
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
	ctx, err := CreateNativeSparseContextOnDevice(0)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := ctx.DestroyContext(); err != nil {
			t.Error(err)
		}
	})
	matrix, err := ctx.CreateCSRMatrix(2, 2, []int32{0, 2, 3}, []int32{0, 1, 1}, []float32{2, 1, 3})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := matrix.Destroy(); err != nil {
			t.Error(err)
		}
	})
	const workers = 4
	inputs := make([]*memory.Memory, workers)
	outputs := make([]*memory.Memory, workers)
	for i := range workers {
		inputs[i] = sparseStreamDeviceBuffer(t, 8, 0)
		outputs[i] = sparseStreamDeviceBuffer(t, 8, 0)
		if err := writeMathFloat32Memory(inputs[i], []float32{float32(i + 1), 2}); err != nil {
			t.Fatal(err)
		}
	}
	t.Cleanup(func() {
		if err := ctx.DestroyContext(); err != nil {
			t.Error(err)
		}
	})
	if err := ctx.SetStream(stream); err != nil {
		t.Fatal(err)
	}
	var wg sync.WaitGroup
	errs := make(chan error, workers)
	for i := range workers {
		wg.Add(1)
		go func(index int) {
			defer wg.Done()
			for range 3 {
				if err := ctx.SpMV(1, matrix, inputs[index], 0, outputs[index]); err != nil {
					errs <- err
					return
				}
			}
			got, err := readMathFloat32Memory(outputs[index], 2)
			if err != nil {
				errs <- err
				return
			}
			if got[0] != float32(2*(index+1)+2) || got[1] != 6 {
				errs <- fmt.Errorf("worker %d got %v", index, got)
			}
		}(i)
	}
	wg.Wait()
	close(errs)
	for err := range errs {
		t.Error(err)
	}
}

func TestNativeSparseContextStreamOnEachDevice(t *testing.T) {
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
	for deviceID := range devices {
		t.Run(devices[deviceID].Name, func(t *testing.T) {
			stream, err := streams.NewNativeStream(deviceID, streams.StreamNonBlocking)
			if err != nil {
				t.Fatal(err)
			}
			t.Cleanup(func() {
				if err := stream.Close(); err != nil {
					t.Error(err)
				}
			})
			ctx, err := CreateNativeSparseContextOnDevice(deviceID)
			if err != nil {
				t.Fatal(err)
			}
			t.Cleanup(func() {
				if err := ctx.DestroyContext(); err != nil {
					t.Error(err)
				}
			})
			matrix, err := ctx.CreateCSRMatrix(2, 2, []int32{0, 2, 3}, []int32{0, 1, 1}, []float32{2, 1, 3})
			if err != nil {
				t.Fatal(err)
			}
			t.Cleanup(func() {
				if err := matrix.Destroy(); err != nil {
					t.Error(err)
				}
			})
			values1 := sparseStreamPinnedBuffer(t, []float32{4, 2, 6})
			values2 := sparseStreamPinnedBuffer(t, []float32{2, 1, 3})
			vectorHost := sparseStreamPinnedBuffer(t, []float32{1, 2})
			input := sparseStreamDeviceBuffer(t, 8, deviceID)
			output := sparseStreamDeviceBuffer(t, 8, deviceID)
			denseHost := sparseStreamPinnedBuffer(t, []float32{0, 4, 0, 6})
			dense := sparseStreamDeviceBuffer(t, 16, deviceID)
			batchHost := sparseStreamPinnedBuffer(t, []float32{1, 2, 3, 4, 2, 0, 1, -1})
			batchInput := sparseStreamDeviceBuffer(t, 32, deviceID)
			batchOutput := sparseStreamDeviceBuffer(t, 32, deviceID)
			rhsHost := sparseStreamPinnedBuffer(t, []float32{4, 14})
			rhs := sparseStreamDeviceBuffer(t, 8, deviceID)
			scaled, err := ctx.CreateCSRMatrix(2, 2, []int32{0, 2, 4}, []int32{0, 1, 0, 1}, []float32{1, 2, 3, 4})
			if err != nil {
				t.Fatal(err)
			}
			t.Cleanup(func() {
				if err := scaled.Destroy(); err != nil {
					t.Error(err)
				}
			})
			t.Cleanup(func() {
				if err := ctx.DestroyContext(); err != nil {
					t.Error(err)
				}
			})
			if err := ctx.SetStream(stream); err != nil {
				t.Fatal(err)
			}
			if err := stream.CopyAsync(matrix.values, values1, 12); err != nil {
				t.Fatal(err)
			}
			if err := stream.CopyAsync(input, vectorHost, 8); err != nil {
				t.Fatal(err)
			}
			if err := ctx.SpMV(1, matrix, input, 0, output); err != nil {
				t.Fatal(err)
			}
			sparseStreamCheckValues(t, output, []float32{8, 12})
			if err := values1.Free(); err == nil {
				t.Fatal("stream retains queued copy buffers until explicit synchronization")
			}
			if err := ctx.Synchronize(); err != nil {
				t.Fatal(err)
			}
			if ctx.nativeStream != stream {
				t.Fatal("synchronization must leave the stream attached")
			}
			if err := vectorHost.Free(); err != nil {
				t.Fatalf("context synchronization must release copy buffers: %v", err)
			}
			if err := stream.CopyAsync(batchInput, batchHost, 32); err != nil {
				t.Fatal(err)
			}
			if err := ctx.SpMMDenseEx(1, matrix, batchInput, 2, 2, 2, BLASOpT, BLASOpT, 0, batchOutput); err != nil {
				t.Fatal(err)
			}
			sparseStreamCheckValues(t, batchOutput, []float32{4, 12, 14, 30})
			if err := ctx.SpMMDenseBatched(1, matrix, batchInput, 2, 2, 2, BLASOpN, BLASOpN, 2, 4, 4, 0, batchOutput); err != nil {
				t.Fatal(err)
			}
			sparseStreamCheckValues(t, batchOutput, []float32{10, 16, 18, 24, 10, -2, 6, -6})
			if err := stream.CopyAsync(rhs, rhsHost, 8); err != nil {
				t.Fatal(err)
			}
			if err := ctx.SpSVEx(matrix, rhs, output, BLASOpT); err != nil {
				t.Fatal(err)
			}
			sparseStreamCheckValues(t, output, []float32{1, 2})
			if err := stream.CopyAsync(dense, denseHost, 16); err != nil {
				t.Fatal(err)
			}
			for _, format := range []MatrixFormat{MatrixFormatCSR, MatrixFormatCOO} {
				converted, err := ctx.DenseToSparse(dense, 2, 2, format)
				if err != nil {
					t.Fatal(err)
				}
				t.Cleanup(func() {
					if err := converted.Destroy(); err != nil {
						t.Error(err)
					}
				})
				convertedRows, err := readSparseRows(converted)
				if err != nil || converted.nnz != 2 || convertedRows[0][1] != 4 || convertedRows[1][1] != 6 {
					t.Fatalf("stream dense-to-sparse output = %v, error = %v", convertedRows, err)
				}
				product, err := ctx.SpGEMM(converted, matrix)
				if err != nil {
					t.Fatal(err)
				}
				t.Cleanup(func() {
					if err := product.Destroy(); err != nil {
						t.Error(err)
					}
				})
				productRows, err := readSparseRows(product)
				if err != nil || productRows[0][1] != 24 || productRows[1][1] != 36 {
					t.Fatalf("stream mixed sparse product = %v, error = %v", productRows, err)
				}
			}
			if err := stream.CopyAsync(matrix.values, values2, 12); err != nil {
				t.Fatal(err)
			}
			denseResult, err := ctx.SparseToDense(matrix)
			if err != nil {
				t.Fatal(err)
			}
			t.Cleanup(func() {
				if err := denseResult.Free(); err != nil {
					t.Error(err)
				}
			})
			sparseStreamCheckValues(t, denseResult, []float32{2, 1, 0, 3})
			if err := stream.CopyAsync(matrix.values, values1, 12); err != nil {
				t.Fatal(err)
			}
			product, err := ctx.SpGEMM(matrix, matrix)
			if err != nil {
				t.Fatal(err)
			}
			t.Cleanup(func() {
				if err := product.Destroy(); err != nil {
					t.Error(err)
				}
			})
			productRows, err := readSparseRows(product)
			if err != nil || productRows[0][0] != 16 || productRows[0][1] != 20 || productRows[1][1] != 36 {
				t.Fatalf("stream sparse product = %v, error = %v", productRows, err)
			}
			if err := ctx.SpMM(1, matrix, matrix, 1, scaled); err != nil {
				t.Fatal(err)
			}
			scaledRows, err := readSparseRows(scaled)
			if err != nil || scaledRows[0][0] != 17 || scaledRows[0][1] != 22 || scaledRows[1][0] != 3 || scaledRows[1][1] != 40 {
				t.Fatalf("stream scaled product = %v, error = %v", scaledRows, err)
			}

			var hostCtx *SparseContext
			if err := internal.RunOnDevice(deviceID, func() error {
				var createErr error
				hostCtx, createErr = CreateSparseContext()
				return createErr
			}); err != nil {
				t.Fatal(err)
			}
			t.Cleanup(func() {
				if err := hostCtx.DestroyContext(); err != nil {
					t.Error(err)
				}
			})
			if err := hostCtx.SetStream(stream); err != nil {
				t.Fatal(err)
			}
			if err := stream.CopyAsync(matrix.values, values2, 12); err != nil {
				t.Fatal(err)
			}
			lower, upper, err := hostCtx.SpLU(matrix)
			if err != nil {
				t.Fatal(err)
			}
			t.Cleanup(func() {
				if err := errors.Join(lower.Destroy(), upper.Destroy()); err != nil {
					t.Error(err)
				}
			})
			upperRows, err := readSparseRows(upper)
			if err != nil || upperRows[0][0] != 2 || upperRows[0][1] != 1 || upperRows[1][1] != 3 {
				t.Fatalf("stream host LU = %v, error = %v", upperRows, err)
			}
			if err := hostCtx.DestroyContext(); err != nil {
				t.Fatal(err)
			}

			next, err := streams.NewNativeStream(deviceID, streams.StreamNonBlocking)
			if err != nil {
				t.Fatal(err)
			}
			t.Cleanup(func() {
				if err := next.Close(); err != nil {
					t.Error(err)
				}
			})
			if err := stream.CopyAsync(matrix.values, values1, 12); err != nil {
				t.Fatal(err)
			}
			if err := ctx.SetStream(next); err != nil {
				t.Fatal(err)
			}
			if err := values1.Free(); err != nil {
				t.Fatalf("stream change must release completed copy buffers: %v", err)
			}
			if err := ctx.SpMV(1, matrix, input, 0, output); err != nil {
				t.Fatal(err)
			}
			sparseStreamCheckValues(t, output, []float32{8, 12})
			if err := next.Close(); err != nil {
				t.Fatal(err)
			}
			if err := ctx.Synchronize(); err != nil {
				t.Fatalf("synchronizing an already closed stream: %v", err)
			}
			if err := ctx.SpMV(1, matrix, input, 0, output); !errors.Is(err, streams.ErrNativeStreamClosed) {
				t.Fatalf("operation on closed stream error = %v", err)
			}
			if err := ctx.SetStream(nil); err != nil {
				t.Fatal(err)
			}
			if err := ctx.SpMV(1, matrix, input, 0, output); err != nil {
				t.Fatal(err)
			}
			sparseStreamCheckValues(t, output, []float32{8, 12})
			if len(devices) > 1 {
				other, err := streams.NewNativeStream((deviceID+1)%len(devices), streams.StreamNonBlocking)
				if err != nil {
					t.Fatal(err)
				}
				t.Cleanup(func() {
					if err := other.Close(); err != nil {
						t.Error(err)
					}
				})
				if err := ctx.SetStream(other); err == nil {
					t.Fatal("expected a different-device stream to be rejected")
				}
			}
		})
	}
}
