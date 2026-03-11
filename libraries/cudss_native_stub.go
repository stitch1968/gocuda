//go:build !cuda

package libraries

import (
	"fmt"

	"github.com/stitch1968/gocuda/memory"
)

func cudssNativeAvailable() bool {
	return false
}

func createNativeDSSHandle(config DSSConfig) (*DSSHandle, error) {
	return nil, fmt.Errorf("native cuDSS backend requires a CUDA-tagged build")
}

func createNativeDSSMatrix(n, nnz int, rowPtr, colInd, values *memory.Memory, format DSSMatrixFormat, symmetry bool) (*DSSMatrix, error) {
	return nil, fmt.Errorf("native cuDSS backend requires a CUDA-tagged build")
}

func analyzeNativeDSS(handle *DSSHandle, matrix *DSSMatrix) error {
	return fmt.Errorf("native cuDSS backend requires a CUDA-tagged build")
}

func factorNativeDSS(handle *DSSHandle, matrix *DSSMatrix) error {
	return fmt.Errorf("native cuDSS backend requires a CUDA-tagged build")
}

func solveNativeDSS(handle *DSSHandle, b, x *memory.Memory, nrhs int) (*DSSSolutionInfo, error) {
	return nil, fmt.Errorf("native cuDSS backend requires a CUDA-tagged build")
}

func refactorNativeDSS(handle *DSSHandle, matrix *DSSMatrix) error {
	return fmt.Errorf("native cuDSS backend requires a CUDA-tagged build")
}

func destroyNativeDSSHandle(handle *DSSHandle) error {
	return nil
}

func destroyNativeDSSMatrix(matrix *DSSMatrix) error {
	return nil
}
