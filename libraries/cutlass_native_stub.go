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

func executeNativeCutlassRank2k(A, B, C *memory.Memory, N, K int, alpha, beta float32) error {
	return errCUTLASSUnsupported
}

func executeNativeCutlassTrmm(A, B *memory.Memory, M, N int, side, uplo, trans, diag string, alpha float32) error {
	return errCUTLASSUnsupported
}

func executeNativeCutlassSpmm(sparseA, denseB, denseC *memory.Memory, M, N, K int) error {
	return errCUTLASSUnsupported
}

func executeNativeCutlassConv(desc CutlassConvDesc, input, filter, output *memory.Memory, outputH, outputW int) error {
	return errCUTLASSUnsupported
}

func destroyNativeCutlassGemm(handle *CutlassGemmHandle) error {
	return nil
}
