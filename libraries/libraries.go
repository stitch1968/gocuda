// Package libraries provides missing CUDA library functionality
// This creates the main export interface for all CUDA runtime libraries
package libraries

import (
	"github.com/stitch1968/gocuda/memory"
)

// Library initialization functions
var (
	// cuRAND
	CreateRNG = CreateRandomGenerator

	// cuSPARSE
	CreateSparseCtx = CreateSparseContext

	// cuSOLVER
	CreateSolverCtx = CreateSolverContext

	// Thrust
	CreateThrustCtx = CreateThrustContext
)

// Quick access patterns for common operations

// RandomNumbers provides a simplified interface for random number generation
func RandomNumbers(size int, rngType RngType) ([]float32, error) {
	rng, err := CreateRandomGenerator(rngType)
	if err != nil {
		return nil, err
	}
	defer rng.Destroy()

	output, err := memory.Alloc(int64(size * 4)) // float32
	if err != nil {
		return nil, err
	}
	defer output.Free()

	err = rng.GenerateUniform(output, size)
	if err != nil {
		return nil, err
	}

	// Generate realistic random data for simulation
	result := make([]float32, size)
	for i := range result {
		result[i] = rng.rng.Float32() // Use the actual random number generator
	}

	return result, nil
}

// SparseMatrixMultiply provides simplified sparse matrix multiplication
func SparseMatrixMultiply(A, B *SparseMatrix) (*SparseMatrix, error) {
	ctx, err := CreateSparseContext()
	if err != nil {
		return nil, err
	}
	defer ctx.DestroyContext()

	return ctx.SpGEMM(A, B)
}

// SolveSystem provides simplified linear system solving
func SolveSystem(A, b *memory.Memory, n int) (*memory.Memory, error) {
	ctx, err := CreateSolverContext()
	if err != nil {
		return nil, err
	}
	defer ctx.DestroyContext()

	return ctx.SolveLinearSystem(A, b, n)
}

// SortArray provides simplified array sorting using Thrust
func SortArray(data *memory.Memory, n int) error {
	ctx, err := CreateThrustContext()
	if err != nil {
		return err
	}
	defer ctx.DestroyContext()

	return ctx.Sort(data, n, PolicyDevice)
}

// ReduceArray provides simplified parallel reduction
func ReduceArray(data *memory.Memory, n int) (float32, error) {
	ctx, err := CreateThrustContext()
	if err != nil {
		return 0, err
	}
	defer ctx.DestroyContext()

	return ctx.Reduce(data, n, 0.0, PolicyDevice)
}
