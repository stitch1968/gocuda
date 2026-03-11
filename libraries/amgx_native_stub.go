//go:build !cuda

package libraries

import (
	"fmt"

	"github.com/stitch1968/gocuda/memory"
)

func amgxNativeAvailable() bool {
	return false
}

func createNativeAmgXHandle(config AmgXConfig) (*AmgXHandle, error) {
	return nil, fmt.Errorf("native AmgX backend requires a CUDA-tagged build")
}

func createNativeAmgXMatrix(n, nnz int, rowPtr, colInd, values *memory.Memory, mode AmgXMode) (*AmgXMatrix, error) {
	return nil, fmt.Errorf("native AmgX backend requires a CUDA-tagged build")
}

func createNativeAmgXVector(size int, data *memory.Memory, mode AmgXMode) (*AmgXVector, error) {
	return nil, fmt.Errorf("native AmgX backend requires a CUDA-tagged build")
}

func setupNativeAmgX(handle *AmgXHandle, matrix *AmgXMatrix) error {
	return fmt.Errorf("native AmgX backend requires a CUDA-tagged build")
}

func solveNativeAmgX(handle *AmgXHandle, b, x *AmgXVector) (*AmgXSolveInfo, error) {
	return nil, fmt.Errorf("native AmgX backend requires a CUDA-tagged build")
}

func updateNativeAmgXMatrix(handle *AmgXHandle, matrix *AmgXMatrix, keepStructure bool) error {
	return fmt.Errorf("native AmgX backend requires a CUDA-tagged build")
}

func destroyNativeAmgXHandle(handle *AmgXHandle) error {
	return nil
}

func destroyNativeAmgXMatrix(matrix *AmgXMatrix) error {
	return nil
}

func destroyNativeAmgXVector(vector *AmgXVector) error {
	return nil
}
