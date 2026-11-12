package libraries

import (
	"errors"
	"fmt"
	"math"
	"sync"
	"unsafe"

	"github.com/stitch1968/gocuda/internal"
	"github.com/stitch1968/gocuda/memory"
	"github.com/stitch1968/gocuda/streams"
)

// NativeBLAS owns a cuBLAS handle bound to a native CUDA stream.
// Close synchronizes the stream before destroying the handle.
type NativeBLAS struct {
	mu       sync.Mutex
	handle   unsafe.Pointer
	stream   *streams.NativeStream
	deviceID int
}

// BLASOperation specifies whether a real matrix is transposed.
type BLASOperation int

const (
	BLASOpN BLASOperation = iota
	BLASOpT
)

// NewNativeBLAS creates a cuBLAS handle on the stream's device.
func NewNativeBLAS(stream *streams.NativeStream) (*NativeBLAS, error) {
	if stream == nil {
		return nil, fmt.Errorf("native stream is required")
	}
	var handle unsafe.Pointer
	err := stream.WithNativeHandle(func(nativeStream unsafe.Pointer) error {
		var err error
		handle, err = createNativeBLAS(nativeStream)
		return err
	})
	if err != nil {
		if handle != nil {
			cleanupErr := internal.RunOnDevice(stream.DeviceID(), func() error { return destroyNativeBLAS(handle) })
			return nil, errors.Join(err, cleanupErr)
		}
		return nil, err
	}
	return &NativeBLAS{handle: handle, stream: stream, deviceID: stream.DeviceID()}, nil
}

// SGEMM enqueues row-major C = alpha*A*B + beta*C for m-by-k and k-by-n inputs.
// All matrices must be live device/managed float32 buffers on the stream device.
func (blas *NativeBLAS) SGEMM(a, b, c *memory.Memory, m, n, k int, alpha, beta float32) error {
	return blas.SGEMMEx(a, b, c, m, n, k, k, n, n, BLASOpN, BLASOpN, alpha, beta)
}

// SGEMMEx enqueues row-major float32 GEMM with transpose and row-stride options.
func (blas *NativeBLAS) SGEMMEx(a, b, c *memory.Memory, m, n, k, lda, ldb, ldc int, opA, opB BLASOperation, alpha, beta float32) error {
	if err := blas.validateGEMM(a, b, c, m, n, k, lda, ldb, ldc, opA, opB, 4); err != nil {
		return err
	}
	return blas.withHandle([]*memory.Memory{a, b, c}, func(handle unsafe.Pointer) error {
		return nativeBLASSGEMMEx(handle, a, b, c, m, n, k, lda, ldb, ldc, opA, opB, alpha, beta)
	})
}

// DGEMM enqueues row-major float64 C = alpha*A*B + beta*C.
func (blas *NativeBLAS) DGEMM(a, b, c *memory.Memory, m, n, k int, alpha, beta float64) error {
	return blas.DGEMMEx(a, b, c, m, n, k, k, n, n, BLASOpN, BLASOpN, alpha, beta)
}

// DGEMMEx enqueues row-major float64 GEMM with transpose and row-stride options.
func (blas *NativeBLAS) DGEMMEx(a, b, c *memory.Memory, m, n, k, lda, ldb, ldc int, opA, opB BLASOperation, alpha, beta float64) error {
	if err := blas.validateGEMM(a, b, c, m, n, k, lda, ldb, ldc, opA, opB, 8); err != nil {
		return err
	}
	return blas.withHandle([]*memory.Memory{a, b, c}, func(handle unsafe.Pointer) error {
		return nativeBLASDGEMMEx(handle, a, b, c, m, n, k, lda, ldb, ldc, opA, opB, alpha, beta)
	})
}

func (blas *NativeBLAS) validateGEMM(a, b, c *memory.Memory, m, n, k, lda, ldb, ldc int, opA, opB BLASOperation, elementSize int) error {
	if blas == nil {
		return fmt.Errorf("nil cuBLAS handle")
	}
	if m <= 0 || n <= 0 || k <= 0 || int64(m) > math.MaxInt32 || int64(n) > math.MaxInt32 || int64(k) > math.MaxInt32 {
		return fmt.Errorf("matrix dimensions must be positive 32-bit values")
	}
	if (opA != BLASOpN && opA != BLASOpT) || (opB != BLASOpN && opB != BLASOpT) {
		return fmt.Errorf("invalid matrix transpose operation")
	}
	aRows, aCols := m, k
	bRows, bCols := k, n
	if opA == BLASOpT {
		aRows, aCols = k, m
	}
	if opB == BLASOpT {
		bRows, bCols = n, k
	}
	for _, entry := range []struct {
		name       string
		mem        *memory.Memory
		rows, cols int
		leadingDim int
	}{
		{"A", a, aRows, aCols, lda},
		{"B", b, bRows, bCols, ldb},
		{"C", c, m, n, ldc},
	} {
		if entry.leadingDim < entry.cols || int64(entry.leadingDim) > math.MaxInt32 {
			return fmt.Errorf("matrix %s has invalid row stride %d", entry.name, entry.leadingDim)
		}
		count := uint64(entry.rows-1)*uint64(entry.leadingDim) + uint64(entry.cols)
		if entry.mem == nil || entry.mem.Ptr() == nil || uint64(entry.mem.Size())/uint64(elementSize) < count {
			return fmt.Errorf("matrix %s is nil, freed, or too small", entry.name)
		}
		switch entry.mem.GetType() {
		case memory.TypeDevice, memory.TypeDeviceOnly, memory.TypeUnified:
		default:
			return fmt.Errorf("matrix %s must be device or managed memory", entry.name)
		}
		if entry.mem.GetDeviceID() != blas.deviceID {
			return fmt.Errorf("matrix %s is on device %d, expected %d", entry.name, entry.mem.GetDeviceID(), blas.deviceID)
		}
	}
	if c.Ptr() == a.Ptr() || c.Ptr() == b.Ptr() {
		return fmt.Errorf("output matrix cannot alias an input")
	}
	return nil
}

func (blas *NativeBLAS) withHandle(buffers []*memory.Memory, fn func(unsafe.Pointer) error) error {
	blas.mu.Lock()
	defer blas.mu.Unlock()
	if blas.handle == nil {
		return fmt.Errorf("cuBLAS handle is closed")
	}
	return blas.stream.EnqueueWithBuffers(buffers, func(unsafe.Pointer) error {
		return fn(blas.handle)
	})
}

// Close synchronizes the associated stream and destroys the cuBLAS handle.
func (blas *NativeBLAS) Close() error {
	if blas == nil {
		return nil
	}
	blas.mu.Lock()
	defer blas.mu.Unlock()
	if blas.handle == nil {
		return nil
	}
	if err := blas.stream.Synchronize(); err != nil && !errors.Is(err, streams.ErrNativeStreamClosed) {
		return err
	}
	return internal.RunOnDevice(blas.deviceID, func() error {
		if err := destroyNativeBLAS(blas.handle); err != nil {
			return err
		}
		blas.handle = nil
		return nil
	})
}
