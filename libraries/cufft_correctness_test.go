package libraries

import (
	"math"
	"math/cmplx"
	"testing"

	cuda "github.com/stitch1968/gocuda"
	"github.com/stitch1968/gocuda/internal"
	"github.com/stitch1968/gocuda/memory"
)

func referenceDFT(input []complex128, dims []int, batch int, direction FFTDirection) []complex128 {
	perBatch := len(input) / batch
	result := make([]complex128, len(input))
	for b := range batch {
		for k := range perBatch {
			for j := range perBatch {
				kCoord, jCoord := k, j
				phase := 0.0
				for axis := len(dims) - 1; axis >= 0; axis-- {
					n := dims[axis]
					phase += float64((kCoord%n)*(jCoord%n)) / float64(n)
					kCoord /= n
					jCoord /= n
				}
				result[b*perBatch+k] += input[b*perBatch+j] * cmplx.Exp(complex(0, float64(direction)*2*math.Pi*phase))
			}
		}
	}
	return result
}

func fftTestMemory(t *testing.T, bytes int64) *memory.Memory {
	t.Helper()
	mem, err := memory.Alloc(bytes)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := mem.Free(); err != nil {
			t.Error(err)
		}
	})
	return mem
}

func fftTestSync(t *testing.T) {
	t.Helper()
	if cuda.ShouldUseCuda() {
		if err := internal.CudaDeviceSynchronizeOnDevice(0); err != nil {
			t.Fatal(err)
		}
	}
}

func fftTestPlan(t *testing.T, ctx *FFTContext, dims []int, batch int, kind FFTType, simulated bool) *FFTPlan {
	t.Helper()
	var plan *FFTPlan
	var err error
	if simulated {
		plan = &FFTPlan{nx: dims[0], batch: batch, fftType: kind}
		if len(dims) > 1 {
			plan.ny = dims[1]
		}
		if len(dims) > 2 {
			plan.nz = dims[2]
		}
		return plan
	}
	switch len(dims) {
	case 1:
		plan, err = ctx.CreatePlan1D(dims[0], kind, batch)
	case 2:
		plan, err = ctx.CreatePlan2D(dims[0], dims[1], kind)
	case 3:
		plan, err = ctx.CreatePlan3D(dims[0], dims[1], dims[2], kind)
	}
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := plan.DestroyPlan(); err != nil {
			t.Error(err)
		}
	})
	return plan
}

func assertFFTClose(t *testing.T, got complex64, want complex128) {
	t.Helper()
	if cmplx.Abs(complex128(got)-want) > 0.002 {
		t.Fatalf("got %v, want %v", got, want)
	}
}

func TestFFTComplexMatchesReference(t *testing.T) {
	for _, tc := range []struct {
		name  string
		dims  []int
		batch int
	}{
		{"non-power-of-two-batched", []int{3}, 2},
		{"rectangular", []int{2, 3}, 1},
		{"three-dimensional", []int{2, 3, 2}, 1},
		{"power-of-two", []int{4}, 1},
	} {
		for _, simulated := range []bool{false, true} {
			for _, direction := range []FFTDirection{FFTForward, FFTInverse} {
				t.Run(tc.name+"/simulated="+boolString(simulated)+"/direction="+directionString(direction), func(t *testing.T) {
					ctx, err := CreateFFTContext()
					if err != nil {
						t.Fatal(err)
					}
					t.Cleanup(func() {
						if err := ctx.DestroyContext(); err != nil {
							t.Error(err)
						}
					})
					plan := fftTestPlan(t, ctx, tc.dims, tc.batch, FFTTypeC2C, simulated)
					size := plan.elementCount()
					input, output := fftTestMemory(t, int64(size*8)), fftTestMemory(t, int64(size*8))
					values, err := memory.View[Complex64](input, size)
					if err != nil {
						t.Fatal(err)
					}
					refInput := make([]complex128, size)
					for i := range values {
						values[i] = Complex64{Real: float32(i%5 - 2), Imag: float32(i%3 - 1)}
						refInput[i] = complex(float64(values[i].Real), float64(values[i].Imag))
					}
					if err := ctx.ExecC2C(plan, input, output, direction); err != nil {
						t.Fatal(err)
					}
					fftTestSync(t)
					got, err := memory.View[Complex64](output, size)
					if err != nil {
						t.Fatal(err)
					}
					want := referenceDFT(refInput, tc.dims, tc.batch, direction)
					for i := range got {
						assertFFTClose(t, complex(got[i].Real, got[i].Imag), want[i])
					}
				})
			}
		}
	}
}

func TestFFTRealPackedRoundTrip(t *testing.T) {
	for _, tc := range []struct {
		name  string
		dims  []int
		batch int
	}{
		{"batched-odd", []int{5}, 2},
		{"rectangular", []int{2, 3}, 1},
		{"three-dimensional", []int{2, 3, 4}, 1},
	} {
		for _, simulated := range []bool{false, true} {
			t.Run(tc.name+"/simulated="+boolString(simulated), func(t *testing.T) {
				ctx, err := CreateFFTContext()
				if err != nil {
					t.Fatal(err)
				}
				t.Cleanup(func() {
					if err := ctx.DestroyContext(); err != nil {
						t.Error(err)
					}
				})
				forward := fftTestPlan(t, ctx, tc.dims, tc.batch, FFTTypeR2C, simulated)
				inverse := fftTestPlan(t, ctx, tc.dims, tc.batch, FFTTypeC2R, simulated)
				size, packed := forward.elementCount(), forward.packedCount()
				input := fftTestMemory(t, int64(size*4))
				spectrum := fftTestMemory(t, int64(packed*8))
				output := fftTestMemory(t, int64(size*4))
				values, err := memory.View[float32](input, size)
				if err != nil {
					t.Fatal(err)
				}
				refInput := make([]complex128, size)
				for i := range values {
					values[i] = float32(i%7 - 3)
					refInput[i] = complex(float64(values[i]), 0)
				}
				if err := ctx.ExecR2C(forward, input, spectrum); err != nil {
					t.Fatal(err)
				}
				fftTestSync(t)
				got, err := memory.View[Complex64](spectrum, packed)
				if err != nil {
					t.Fatal(err)
				}
				want := referenceDFT(refInput, tc.dims, tc.batch, FFTForward)
				last := tc.dims[len(tc.dims)-1]
				for row := range size / last {
					for x := range last/2 + 1 {
						value := got[row*(last/2+1)+x]
						assertFFTClose(t, complex(value.Real, value.Imag), want[row*last+x])
					}
				}
				if err := ctx.ExecC2R(inverse, spectrum, output); err != nil {
					t.Fatal(err)
				}
				fftTestSync(t)
				result, err := memory.View[float32](output, size)
				if err != nil {
					t.Fatal(err)
				}
				for i, value := range result {
					assertFFTClose(t, complex(value, 0), refInput[i]*complex(float64(size/tc.batch), 0))
				}
			})
		}
	}
}

func TestFFTRejectsInvalidBuffersAndDirections(t *testing.T) {
	ctx, err := CreateFFTContext()
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := ctx.DestroyContext(); err != nil {
			t.Error(err)
		}
	})
	plan := fftTestPlan(t, ctx, []int{4}, 1, FFTTypeC2C, false)
	small, adequate := fftTestMemory(t, 8), fftTestMemory(t, 32)
	if err := ctx.ExecC2C(plan, small, adequate, FFTForward); err == nil {
		t.Fatal("expected undersized input to fail")
	}
	if err := ctx.ExecC2C(plan, adequate, adequate, FFTDirection(0)); err == nil {
		t.Fatal("expected invalid direction to fail")
	}
	if uint64(^uint(0)) > 1<<32-1 {
		oversized := int64(1) << 31
		if _, err := fftElementCounts(int(oversized), 0, 0, 1); err == nil {
			t.Fatal("expected dimensions beyond cuFFT's int range to fail")
		}
	}
	if !plan.native {
		if err := plan.SetStream(cuda.GetDefaultStream()); err == nil {
			t.Fatal("expected unsupported simulated stream attachment to fail")
		}
	}
	if plan.native {
		host, err := memory.AllocWithTypeAndStream(nil, 32, memory.TypeHost)
		if err != nil {
			t.Fatal(err)
		}
		defer host.Free()
		if err := ctx.ExecC2C(plan, host, adequate, FFTForward); err == nil {
			t.Fatal("expected native FFT to reject host memory")
		}
	}
}

func TestFFTDeviceOnlyMemory(t *testing.T) {
	ctx, err := CreateFFTContext()
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := ctx.DestroyContext(); err != nil {
			t.Error(err)
		}
	})
	plan := fftTestPlan(t, ctx, []int{4}, 1, FFTTypeC2C, false)
	input, err := memory.AllocDeviceOnly(32)
	if err != nil {
		t.Fatal(err)
	}
	defer input.Free()
	output, err := memory.AllocDeviceOnly(32)
	if err != nil {
		t.Fatal(err)
	}
	defer output.Free()
	values := []Complex64{{Real: 1}, {Real: 2}, {Real: 3}, {Real: 4}}
	if err := writeMathTypedMemory(input, values); err != nil {
		t.Fatal(err)
	}
	if err := ctx.ExecC2C(plan, input, output, FFTForward); err != nil {
		t.Fatal(err)
	}
	fftTestSync(t)
	result, err := readMathTypedMemory[Complex64](output, 4)
	if err != nil {
		t.Fatal(err)
	}
	want := []complex128{10, -2 + 2i, -2, -2 - 2i}
	for i := range result {
		assertFFTClose(t, complex(result[i].Real, result[i].Imag), want[i])
	}
}

func boolString(value bool) string {
	if value {
		return "true"
	}
	return "false"
}

func directionString(value FFTDirection) string {
	if value == FFTForward {
		return "forward"
	}
	return "inverse"
}
