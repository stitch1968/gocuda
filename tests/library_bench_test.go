package tests

import (
	"testing"

	cuda "github.com/stitch1968/gocuda"
	"github.com/stitch1968/gocuda/internal"
	"github.com/stitch1968/gocuda/libraries"
	"github.com/stitch1968/gocuda/memory"
)

func BenchmarkFFT1D(b *testing.B) {
	const size = 256
	input, err := memory.Alloc(size * 8)
	if err != nil {
		b.Fatal(err)
	}
	defer input.Free()

	output, err := memory.Alloc(size * 8)
	if err != nil {
		b.Fatal(err)
	}
	defer output.Free()

	values, err := memory.View[libraries.Complex64](input, size)
	if err != nil {
		b.Fatal(err)
	}
	for i := range values {
		values[i] = libraries.Complex64{Real: float32(i + 1)}
	}

	ctx, err := libraries.CreateFFTContext()
	if err != nil {
		b.Fatal(err)
	}
	defer ctx.DestroyContext()
	plan, err := ctx.CreatePlan1D(size, libraries.FFTTypeC2C, 1)
	if err != nil {
		b.Fatal(err)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := ctx.ExecC2C(plan, input, output, libraries.FFTForward); err != nil {
			b.Fatal(err)
		}
		if cuda.ShouldUseCuda() {
			if err := internal.CudaDeviceSynchronizeOnDevice(input.GetDeviceID()); err != nil {
				b.Fatal(err)
			}
		}
	}
	b.StopTimer()
}

func BenchmarkRandomUniform(b *testing.B) {
	rng, err := libraries.CreateRandomGenerator(libraries.RngTypeXorwow)
	if err != nil {
		b.Fatal(err)
	}
	defer rng.Destroy()

	const size = 4096
	output, err := memory.Alloc(size * 4)
	if err != nil {
		b.Fatal(err)
	}
	defer output.Free()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := rng.GenerateUniform(output, size); err != nil {
			b.Fatal(err)
		}
		if cuda.ShouldUseCuda() {
			if err := internal.CudaDeviceSynchronizeOnDevice(output.GetDeviceID()); err != nil {
				b.Fatal(err)
			}
		}
	}
	b.StopTimer()
}
