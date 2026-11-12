//go:build !cuda

package streams

import (
	"time"
	"unsafe"
)

func createNativeStream(StreamFlags) (unsafe.Pointer, error) { return nil, ErrNativeCUDAUnavailable }
func synchronizeNativeStream(unsafe.Pointer) error           { return ErrNativeCUDAUnavailable }
func destroyNativeStream(unsafe.Pointer) error               { return ErrNativeCUDAUnavailable }
func copyNativeAsync(unsafe.Pointer, unsafe.Pointer, unsafe.Pointer, int64) error {
	return ErrNativeCUDAUnavailable
}
func createNativeEvent(NativeEventFlags) (unsafe.Pointer, error) {
	return nil, ErrNativeCUDAUnavailable
}
func recordNativeEvent(unsafe.Pointer, unsafe.Pointer) error { return ErrNativeCUDAUnavailable }
func waitNativeEvent(unsafe.Pointer, unsafe.Pointer) error   { return ErrNativeCUDAUnavailable }
func queryNativeEvent(unsafe.Pointer) (bool, error)          { return false, ErrNativeCUDAUnavailable }
func synchronizeNativeEvent(unsafe.Pointer) error            { return ErrNativeCUDAUnavailable }
func elapsedNativeEvents(unsafe.Pointer, unsafe.Pointer) (time.Duration, error) {
	return 0, ErrNativeCUDAUnavailable
}
func destroyNativeEvent(unsafe.Pointer) error { return ErrNativeCUDAUnavailable }
