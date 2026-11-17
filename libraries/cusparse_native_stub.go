//go:build !cuda

package libraries

import (
	"fmt"
	"unsafe"

	"github.com/stitch1968/gocuda/memory"
)

func createNativeSparseContext() (*SparseContext, error) {
	return nil, fmt.Errorf("cuSPARSE native backend requires a CUDA-tagged build")
}
func setNativeSparseStream(ctx *SparseContext, stream unsafe.Pointer) error {
	return fmt.Errorf("cuSPARSE native backend requires a CUDA-tagged build")
}
func nativeSpMV(ctx *SparseContext, alpha float32, A *SparseMatrix, x *memory.Memory, beta float32, y *memory.Memory, opA BLASOperation) error {
	return fmt.Errorf("cuSPARSE native backend requires a CUDA-tagged build")
}
func nativeSpMMDense(ctx *SparseContext, alpha float32, A *SparseMatrix, B *memory.Memory, beta float32, C *memory.Memory, batch sparseDenseBatch) error {
	return fmt.Errorf("cuSPARSE native backend requires a CUDA-tagged build")
}
func nativeSpSV(ctx *SparseContext, A *SparseMatrix, b, x *memory.Memory, lower bool, opA BLASOperation) error {
	return fmt.Errorf("cuSPARSE native backend requires a CUDA-tagged build")
}
func nativeSpGEMMCSR(ctx *SparseContext, A, B *SparseMatrix) (*SparseMatrix, error) {
	return nil, fmt.Errorf("cuSPARSE native backend requires a CUDA-tagged build")
}
func nativeSparseAdd(ctx *SparseContext, alpha float32, A *SparseMatrix, beta float32, B *SparseMatrix) (*SparseMatrix, error) {
	return nil, fmt.Errorf("cuSPARSE native backend requires a CUDA-tagged build")
}
func nativeSparseToDense(ctx *SparseContext, matrix *SparseMatrix) (*memory.Memory, error) {
	return nil, fmt.Errorf("cuSPARSE native backend requires a CUDA-tagged build")
}
func nativeDenseToSparse(ctx *SparseContext, dense *memory.Memory, rows, cols int, format MatrixFormat) (*SparseMatrix, error) {
	return nil, fmt.Errorf("cuSPARSE native backend requires a CUDA-tagged build")
}
func destroyNativeSparseContext(ctx *SparseContext) error {
	return fmt.Errorf("cuSPARSE native backend requires a CUDA-tagged build")
}
