//go:build !cuda

package kernels

import (
	"unsafe"

	"github.com/stitch1968/gocuda/streams"
)

func loadNativePTX(string) (unsafe.Pointer, error) { return nil, streams.ErrNativeCUDAUnavailable }
func nativeModuleFunction(unsafe.Pointer, string) (unsafe.Pointer, []int, error) {
	return nil, nil, streams.ErrNativeCUDAUnavailable
}
func launchNativeFunction(unsafe.Pointer, unsafe.Pointer, Dim3, Dim3, int, [][8]byte) error {
	return streams.ErrNativeCUDAUnavailable
}
func unloadNativeModule(unsafe.Pointer) error { return streams.ErrNativeCUDAUnavailable }
func createNativeGraph() (unsafe.Pointer, error) {
	return nil, streams.ErrNativeCUDAUnavailable
}
func addNativeGraphKernel(unsafe.Pointer, unsafe.Pointer, []unsafe.Pointer, Dim3, Dim3, int, [][8]byte) (unsafe.Pointer, error) {
	return nil, streams.ErrNativeCUDAUnavailable
}
func addNativeGraphMemcpy(unsafe.Pointer, []unsafe.Pointer, unsafe.Pointer, unsafe.Pointer, int64, bool, bool) (unsafe.Pointer, error) {
	return nil, streams.ErrNativeCUDAUnavailable
}
func addNativeGraphMemset(unsafe.Pointer, []unsafe.Pointer, unsafe.Pointer, uint32, int, int) (unsafe.Pointer, error) {
	return nil, streams.ErrNativeCUDAUnavailable
}
func instantiateNativeGraph(unsafe.Pointer) (unsafe.Pointer, error) {
	return nil, streams.ErrNativeCUDAUnavailable
}
func launchNativeGraph(unsafe.Pointer, unsafe.Pointer) error { return streams.ErrNativeCUDAUnavailable }
func destroyNativeGraphExec(unsafe.Pointer) error            { return streams.ErrNativeCUDAUnavailable }
func destroyNativeGraph(unsafe.Pointer) error                { return streams.ErrNativeCUDAUnavailable }
func nativeGraphHasDependency(unsafe.Pointer, unsafe.Pointer) (bool, error) {
	return false, streams.ErrNativeCUDAUnavailable
}
