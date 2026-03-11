//go:build !cuda
// +build !cuda

package libraries

import (
	"fmt"

	"github.com/stitch1968/gocuda/memory"
)

func createNativeSparseContext() (*SparseContext, error) {
	return nil, fmt.Errorf("cuSPARSE native backend requires a CUDA-tagged build")
}
func nativeSpMVCSR(ctx *SparseContext, alpha float32, A *SparseMatrix, x *memory.Memory, beta float32, y *memory.Memory) error {
	return fmt.Errorf("cuSPARSE native backend requires a CUDA-tagged build")
}
func nativeSpGEMM(ctx *SparseContext, A, B *SparseMatrix) (*SparseMatrix, error) {
	return nil, fmt.Errorf("cuSPARSE native backend requires a CUDA-tagged build")
}
func nativeSparseLU(ctx *SparseContext, A *SparseMatrix) (*SparseMatrix, *SparseMatrix, error) {
	return nil, nil, fmt.Errorf("cuSPARSE native backend requires a CUDA-tagged build")
}
func nativeSparseSolve(ctx *SparseContext, A *SparseMatrix, b, x *memory.Memory) error {
	return fmt.Errorf("cuSPARSE native backend requires a CUDA-tagged build")
}
func destroyNativeSparseContext(ctx *SparseContext) error {
	return fmt.Errorf("cuSPARSE native backend requires a CUDA-tagged build")
}
