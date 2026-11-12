//go:build !cuda

package libraries

import (
	"fmt"

	"github.com/stitch1968/gocuda/memory"
)

func createNativeSparseContext() (*SparseContext, error) {
	return nil, fmt.Errorf("cuSPARSE native backend requires a CUDA-tagged build")
}
func nativeSpMV(ctx *SparseContext, alpha float32, A *SparseMatrix, x *memory.Memory, beta float32, y *memory.Memory) error {
	return fmt.Errorf("cuSPARSE native backend requires a CUDA-tagged build")
}
func nativeSpSVCSR(ctx *SparseContext, A *SparseMatrix, b, x *memory.Memory, lower bool) error {
	return fmt.Errorf("cuSPARSE native backend requires a CUDA-tagged build")
}
func nativeSpGEMMCSR(ctx *SparseContext, A, B *SparseMatrix) (*SparseMatrix, error) {
	return nil, fmt.Errorf("cuSPARSE native backend requires a CUDA-tagged build")
}
func nativeSparseToDense(ctx *SparseContext, matrix *SparseMatrix) (*memory.Memory, error) {
	return nil, fmt.Errorf("cuSPARSE native backend requires a CUDA-tagged build")
}
func nativeDenseToSparseCSR(ctx *SparseContext, dense *memory.Memory, rows, cols int) (*SparseMatrix, error) {
	return nil, fmt.Errorf("cuSPARSE native backend requires a CUDA-tagged build")
}
func destroyNativeSparseContext(ctx *SparseContext) error {
	return fmt.Errorf("cuSPARSE native backend requires a CUDA-tagged build")
}
