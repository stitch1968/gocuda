//go:build !cuda

package libraries

import (
	"unsafe"

	"github.com/stitch1968/gocuda/memory"
	"github.com/stitch1968/gocuda/streams"
)

func createNativeBLAS(unsafe.Pointer) (unsafe.Pointer, error) {
	return nil, streams.ErrNativeCUDAUnavailable
}

func nativeBLASSGEMMEx(unsafe.Pointer, *memory.Memory, *memory.Memory, *memory.Memory, int, int, int, int, int, int, BLASOperation, BLASOperation, float32, float32) error {
	return streams.ErrNativeCUDAUnavailable
}

func nativeBLASDGEMMEx(unsafe.Pointer, *memory.Memory, *memory.Memory, *memory.Memory, int, int, int, int, int, int, BLASOperation, BLASOperation, float64, float64) error {
	return streams.ErrNativeCUDAUnavailable
}

func destroyNativeBLAS(unsafe.Pointer) error {
	return streams.ErrNativeCUDAUnavailable
}
