//go:build !cuda

package libraries

import "github.com/stitch1968/gocuda/memory"

func thrustNativeAvailable() bool {
	return false
}

func createNativeThrustContext() (*ThrustContext, error) {
	return nil, errThrustUnsupported
}

func executeNativeThrustCopy(src, dst *memory.Memory, n int) error {
	return errThrustUnsupported
}

func executeNativeThrustFill(data *memory.Memory, n int, value float32) error {
	return errThrustUnsupported
}

func executeNativeThrustGenerate(data *memory.Memory, n int, generator string) error {
	return errThrustUnsupported
}

func destroyNativeThrustContext(ctx *ThrustContext) error {
	return nil
}
