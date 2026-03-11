package tests

import (
	"testing"

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

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := libraries.FFT1D(input, output, size, true); err != nil {
			b.Fatal(err)
		}
	}
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
	}
}
