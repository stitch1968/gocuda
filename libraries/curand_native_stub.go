//go:build !cuda
// +build !cuda

package libraries

import (
	"fmt"

	"github.com/stitch1968/gocuda/memory"
)

func createNativeRandomGenerator(rngType RngType) (*RandomGenerator, error) {
	return nil, fmt.Errorf("cuRAND native backend requires a CUDA-tagged build")
}

func setNativeRandomSeed(rg *RandomGenerator, seed uint64) error {
	return fmt.Errorf("cuRAND native backend requires a CUDA-tagged build")
}

func generateNativeUniform(rg *RandomGenerator, output *memory.Memory, n int) error {
	return fmt.Errorf("cuRAND native backend requires a CUDA-tagged build")
}

func generateNativeNormal(rg *RandomGenerator, output *memory.Memory, n int, mean, stddev float32) error {
	return fmt.Errorf("cuRAND native backend requires a CUDA-tagged build")
}

func generateNativeLogNormal(rg *RandomGenerator, output *memory.Memory, n int, mean, stddev float32) error {
	return fmt.Errorf("cuRAND native backend requires a CUDA-tagged build")
}

func generateNativePoisson(rg *RandomGenerator, output *memory.Memory, n int, lambda float32) error {
	return fmt.Errorf("cuRAND native backend requires a CUDA-tagged build")
}

func destroyNativeRandomGenerator(rg *RandomGenerator) error {
	return fmt.Errorf("cuRAND native backend requires a CUDA-tagged build")
}
