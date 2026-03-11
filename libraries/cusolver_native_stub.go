//go:build !cuda

package libraries

import (
	"fmt"

	"github.com/stitch1968/gocuda/memory"
)

func createNativeSolverContext() (*SolverContext, error) {
	return nil, fmt.Errorf("cuSOLVER native backend requires a CUDA-tagged build")
}
func nativeQRFactorization(ctx *SolverContext, A *memory.Memory, m, n int) (*QRInfo, error) {
	return nil, fmt.Errorf("cuSOLVER native backend requires a CUDA-tagged build")
}
func nativeSVDDecomposition(ctx *SolverContext, A *memory.Memory, m, n int, computeUV bool) (*SVDInfo, error) {
	return nil, fmt.Errorf("cuSOLVER native backend requires a CUDA-tagged build")
}
func nativeLUFactorization(ctx *SolverContext, A *memory.Memory, m, n int) (*LUInfo, error) {
	return nil, fmt.Errorf("cuSOLVER native backend requires a CUDA-tagged build")
}
func nativeSolveLinearSystem(ctx *SolverContext, A, b *memory.Memory, n int) (*memory.Memory, error) {
	return nil, fmt.Errorf("cuSOLVER native backend requires a CUDA-tagged build")
}
func nativeEigenvalues(ctx *SolverContext, A *memory.Memory, n int, computeVectors bool) (*memory.Memory, *memory.Memory, error) {
	return nil, nil, fmt.Errorf("cuSOLVER native backend requires a CUDA-tagged build")
}
func nativeCholeskyFactorization(ctx *SolverContext, A *memory.Memory, n int) error {
	return fmt.Errorf("cuSOLVER native backend requires a CUDA-tagged build")
}
func nativePseudoInverse(ctx *SolverContext, A *memory.Memory, m, n int) (*memory.Memory, error) {
	return nil, fmt.Errorf("cuSOLVER native backend requires a CUDA-tagged build")
}
func destroyNativeSolverContext(ctx *SolverContext) error {
	return fmt.Errorf("cuSOLVER native backend requires a CUDA-tagged build")
}
