package cuda

// Re-exports from kernels package for backward compatibility

import (
	"github.com/stitch1968/gocuda/kernels"
)

// Kernel types and functions re-exported for backward compatibility
type Kernel = kernels.Kernel
type Dim3 = kernels.Dim3

// Kernel implementations re-exported
type VectorAdd = kernels.VectorAdd
type MatrixMultiply = kernels.MatrixMultiply
type Convolution2D = kernels.Convolution2D

// Convenience functions re-exported
var (
	ExecuteVectorAdd           = kernels.ExecuteVectorAdd
	ExecuteVectorAddAsync      = kernels.ExecuteVectorAddAsync
	ExecuteMatrixMultiply      = kernels.ExecuteMatrixMultiply
	ExecuteMatrixMultiplyAsync = kernels.ExecuteMatrixMultiplyAsync
)
