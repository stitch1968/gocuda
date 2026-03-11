//go:build !cuda

package libraries

import (
	"fmt"

	"github.com/stitch1968/gocuda/memory"
)

func createNativeFFTContext() (*FFTContext, error) {
	return nil, fmt.Errorf("cuFFT native backend requires a CUDA-tagged build")
}

func createNativeFFTPlan1D(ctx *FFTContext, nx int, fftType FFTType, batch int) (*FFTPlan, error) {
	return nil, fmt.Errorf("cuFFT native backend requires a CUDA-tagged build")
}

func createNativeFFTPlan2D(ctx *FFTContext, nx, ny int, fftType FFTType) (*FFTPlan, error) {
	return nil, fmt.Errorf("cuFFT native backend requires a CUDA-tagged build")
}

func createNativeFFTPlan3D(ctx *FFTContext, nx, ny, nz int, fftType FFTType) (*FFTPlan, error) {
	return nil, fmt.Errorf("cuFFT native backend requires a CUDA-tagged build")
}

func execNativeFFT(plan *FFTPlan, input, output *memory.Memory, direction FFTDirection) error {
	return fmt.Errorf("cuFFT native backend requires a CUDA-tagged build")
}

func setNativeFFTPlanStream(plan *FFTPlan, stream any) error {
	return fmt.Errorf("cuFFT native backend requires a CUDA-tagged build")
}

func destroyNativeFFTPlan(plan *FFTPlan) error {
	return fmt.Errorf("cuFFT native backend requires a CUDA-tagged build")
}

func destroyNativeFFTContext(ctx *FFTContext) error {
	return fmt.Errorf("cuFFT native backend requires a CUDA-tagged build")
}
