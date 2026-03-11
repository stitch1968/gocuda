//go:build !cuda

package libraries

import "github.com/stitch1968/gocuda/memory"

func cutlassNativeAvailable() bool {
	return false
}

func createNativeCutlassGemm(desc CutlassGemmDesc) (*CutlassGemmHandle, error) {
	return nil, errCUTLASSUnsupported
}

func executeNativeCutlassGemm(handle *CutlassGemmHandle, A, B, C *memory.Memory) error {
	return errCUTLASSUnsupported
}

func destroyNativeCutlassGemm(handle *CutlassGemmHandle) error {
	return nil
}
