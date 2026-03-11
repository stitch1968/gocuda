//go:build cuda

package libraries

import (
	"encoding/binary"
	"math"
	"time"

	"github.com/stitch1968/gocuda/internal"
	"github.com/stitch1968/gocuda/memory"
)

func thrustNativeAvailable() bool {
	return true
}

func createNativeThrustContext() (*ThrustContext, error) {
	return &ThrustContext{handle: uintptr(time.Now().UnixNano()), native: true}, nil
}

func executeNativeThrustCopy(src, dst *memory.Memory, n int) error {
	if src == nil || dst == nil || n <= 0 {
		return errThrustUnsupported
	}
	copyBytes := int64(n * 4)
	if src.Size() < copyBytes || dst.Size() < copyBytes {
		return errThrustUnsupported
	}
	return memory.CopyDeviceToDevice(dst, src)
}

func executeNativeThrustFill(data *memory.Memory, n int, value float32) error {
	if data == nil || n <= 0 {
		return errThrustUnsupported
	}
	fillBytes := int64(n * 4)
	if data.Size() < fillBytes {
		return errThrustUnsupported
	}
	if value == 0 {
		return internal.CudaMemsetOnDevice(data.Ptr(), 0, fillBytes, data.GetDeviceID())
	}
	buffer := make([]byte, n*4)
	bits := math.Float32bits(value)
	for index := 0; index < n; index++ {
		binary.LittleEndian.PutUint32(buffer[index*4:], bits)
	}
	return memory.CopyHostToDevice(data, buffer)
}

func executeNativeThrustGenerate(data *memory.Memory, n int, generator string) error {
	if data == nil || n <= 0 {
		return errThrustUnsupported
	}
	switch generator {
	case "", "zeros":
		return executeNativeThrustFill(data, n, 0)
	case "ones":
		return executeNativeThrustFill(data, n, 1)
	case "sequence", "index":
		values := make([]byte, n*4)
		for index := 0; index < n; index++ {
			binary.LittleEndian.PutUint32(values[index*4:], math.Float32bits(float32(index)))
		}
		return memory.CopyHostToDevice(data, values)
	default:
		return errThrustUnsupported
	}
}

func destroyNativeThrustContext(ctx *ThrustContext) error {
	if ctx == nil {
		return nil
	}
	ctx.handle = 0
	ctx.native = false
	return nil
}
