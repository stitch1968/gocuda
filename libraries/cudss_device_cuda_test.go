//go:build cuda

package libraries

import (
	"errors"
	"fmt"
	"math"
	"testing"

	cuda "github.com/stitch1968/gocuda"
	"github.com/stitch1968/gocuda/memory"
)

func TestCuDSSNativeDeviceAffinityOnEachDevice(t *testing.T) {
	if err := cuda.Initialize(); err != nil {
		t.Fatal(err)
	}
	if !cuda.ShouldUseCuda() || !cudssNativeAvailable() {
		t.Skip("native cuDSS unavailable")
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
				testCuDSSNativeDevice(t, deviceID, len(devices))
				return nil
			}); err != nil {
				t.Fatal(err)
			}
		})
	}
}

func testCuDSSNativeDevice(t *testing.T, deviceID, deviceCount int) {
	t.Helper()
	allocate := func(size int64, device int) *memory.Memory {
		t.Helper()
		buffer, err := memory.AllocDeviceOnlyOnDevice(size, device)
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
	rows, columns, values := allocate(12, deviceID), allocate(8, deviceID), allocate(8, deviceID)
	rhs, output := allocate(8, deviceID), allocate(8, deviceID)
	for _, entry := range []struct {
		buffer *memory.Memory
		data   []int32
	}{{rows, []int32{0, 1, 2}}, {columns, []int32{0, 1}}} {
		if err := writeInt32Memory(entry.buffer, entry.data, len(entry.data)); err != nil {
			t.Fatal(err)
		}
	}
	for _, entry := range []struct {
		buffer *memory.Memory
		data   []float32
	}{{values, []float32{2, 4}}, {rhs, []float32{2, 4}}, {output, []float32{0, 0}}} {
		if err := writeMathFloat32Memory(entry.buffer, entry.data); err != nil {
			t.Fatal(err)
		}
	}
	matrix, err := CreateDSSMatrix(2, 2, rows, columns, values, DSSMatrixFormatCSR, false)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := matrix.Destroy(); err != nil {
			t.Error(err)
		}
	})
	handle, err := CreateNativeDSSHandleOnDevice(DSSConfig{MatrixFormat: DSSMatrixFormatCSR}, deviceID)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := handle.Destroy(); err != nil {
			t.Error(err)
		}
	})
	if handle.Backend() != ImplementationModeNativeCUDA || matrix.deviceID != deviceID {
		t.Fatalf("backend = %q, matrix device = %d, want native device %d", handle.Backend(), matrix.deviceID, deviceID)
	}
	if err := handle.Analyze(&DSSMatrix{}); !errors.Is(err, ErrDSSNativeOperationUnsupported) {
		t.Fatalf("strict fallback error = %v", err)
	}
	if err := handle.validateNativeMatrix(&DSSMatrix{native: true, deviceID: deviceID, symmetry: true}); !errors.Is(err, ErrDSSNativeOperationUnsupported) {
		t.Fatalf("strict symmetry mismatch error = %v", err)
	}
	if err := handle.Analyze(matrix); err != nil {
		t.Fatal(err)
	}
	if err := handle.Factor(matrix); err != nil {
		t.Fatal(err)
	}
	if _, err := handle.Solve(rhs, output, 1); err != nil {
		t.Fatal(err)
	}
	result, err := readMathFloat32Memory(output, 2)
	if err != nil || math.Abs(float64(result[0]-1)) > 1e-5 || math.Abs(float64(result[1]-1)) > 1e-5 {
		t.Fatalf("solution = %v, error = %v, want [1 1]", result, err)
	}
	t.Run("cholesky", func(t *testing.T) {
		spd, err := CreateDSSMatrix(2, 2, rows, columns, values, DSSMatrixFormatCSR, true)
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() {
			if err := spd.Destroy(); err != nil {
				t.Error(err)
			}
		})
		cholesky, err := CreateNativeDSSHandleOnDevice(DSSConfig{
			MatrixFormat: DSSMatrixFormatCSR, Factorization: DSSFactorizationCholesky, Symmetry: true,
		}, deviceID)
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() {
			if err := cholesky.Destroy(); err != nil {
				t.Error(err)
			}
		})
		if err := cholesky.Analyze(spd); err != nil {
			t.Fatal(err)
		}
		if err := cholesky.Factor(spd); err != nil {
			t.Fatal(err)
		}
		if _, err := cholesky.Solve(rhs, output, 1); err != nil {
			t.Fatal(err)
		}
		got, err := readMathFloat32Memory(output, 2)
		if err != nil || math.Abs(float64(got[0]-1)) > 1e-5 || math.Abs(float64(got[1]-1)) > 1e-5 {
			t.Fatalf("cholesky solution = %v, error = %v", got, err)
		}
	})
	host, err := memory.AllocWithTypeAndStream(nil, 8, memory.TypeHost)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := host.Free(); err != nil {
			t.Error(err)
		}
	})
	if _, err := handle.Solve(rhs, host, 1); err == nil {
		t.Fatal("expected host-only output to fail")
	}
	if _, err := CreateDSSMatrix(2, 2, rows, columns, host, DSSMatrixFormatCSR, false); err == nil {
		t.Fatal("expected host-only matrix values to fail")
	}
	short := allocate(4, deviceID)
	if _, err := CreateDSSMatrix(2, 2, short, columns, values, DSSMatrixFormatCSR, false); err == nil {
		t.Fatal("expected short row offset buffer to fail")
	}
	if deviceCount > 1 {
		other := (deviceID + 1) % deviceCount
		foreign := allocate(8, other)
		if _, err := handle.Solve(rhs, foreign, 1); err == nil {
			t.Fatal("expected cross-device output to fail")
		}
		if _, err := CreateDSSMatrix(2, 2, rows, columns, foreign, DSSMatrixFormatCSR, false); err == nil {
			t.Fatal("expected cross-device matrix buffer to fail")
		}
		foreignRows, foreignColumns := allocate(12, other), allocate(8, other)
		if err := writeInt32Memory(foreignRows, []int32{0, 1, 2}, 3); err != nil {
			t.Fatal(err)
		}
		if err := writeInt32Memory(foreignColumns, []int32{0, 1}, 2); err != nil {
			t.Fatal(err)
		}
		if err := writeMathFloat32Memory(foreign, []float32{2, 4}); err != nil {
			t.Fatal(err)
		}
		foreignMatrix, err := CreateDSSMatrix(2, 2, foreignRows, foreignColumns, foreign, DSSMatrixFormatCSR, false)
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() {
			if err := foreignMatrix.Destroy(); err != nil {
				t.Error(err)
			}
		})
		if err := handle.Analyze(foreignMatrix); err == nil {
			t.Fatal("expected matrix on another device to fail")
		}
		if _, err := handle.Solve(rhs, output, 1); err != nil {
			t.Fatalf("rejected analysis must preserve original factors: %v", err)
		}
	}
}
