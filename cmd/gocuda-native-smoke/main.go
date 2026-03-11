package main

import (
	"fmt"
	"os"
	"unsafe"

	cuda "github.com/stitch1968/gocuda"
	"github.com/stitch1968/gocuda/libraries"
	"github.com/stitch1968/gocuda/memory"
)

func main() {
	if err := cuda.Initialize(); err != nil {
		fmt.Fprintln(os.Stderr, err.Error())
		os.Exit(1)
	}
	if !cuda.ShouldUseCuda() {
		fmt.Fprintln(os.Stderr, "native smoke requires CUDA runtime mode")
		os.Exit(1)
	}

	if err := smokeCuFFT(); err != nil {
		fmt.Fprintln(os.Stderr, err.Error())
		os.Exit(1)
	}
	if err := smokeCuRAND(); err != nil {
		fmt.Fprintln(os.Stderr, err.Error())
		os.Exit(1)
	}
	if err := smokeCuSPARSE(); err != nil {
		fmt.Fprintln(os.Stderr, err.Error())
		os.Exit(1)
	}

	fmt.Println("GoCUDA native smoke passed")
}

func smokeCuFFT() error {
	ctx, err := libraries.CreateFFTContext()
	if err != nil {
		return fmt.Errorf("cuFFT smoke create context: %w", err)
	}
	defer ctx.DestroyContext()

	input, err := memory.Alloc(8 * 8)
	if err != nil {
		return err
	}
	defer input.Free()

	output, err := memory.Alloc(8 * 8)
	if err != nil {
		return err
	}
	defer output.Free()

	hostInput := make([]libraries.Complex64, 8)
	for i := range hostInput {
		hostInput[i] = libraries.Complex64{Real: float32(i + 1)}
	}
	if err := copyComplex64ToDevice(input, hostInput); err != nil {
		return err
	}

	plan, err := ctx.CreatePlan1D(8, libraries.FFTTypeC2C, 1)
	if err != nil {
		return err
	}
	defer plan.DestroyPlan()

	if err := ctx.ExecC2C(plan, input, output, libraries.FFTForward); err != nil {
		return err
	}

	result, err := copyComplex64FromDevice(output, 8)
	if err != nil {
		return err
	}
	if result[0].Real == 0 && result[0].Imag == 0 {
		return fmt.Errorf("cuFFT smoke produced zero DC component for non-zero input")
	}
	return nil
}

func smokeCuRAND() error {
	rng, err := libraries.CreateRandomGenerator(libraries.RngTypeXorwow)
	if err != nil {
		return fmt.Errorf("cuRAND smoke create generator: %w", err)
	}
	defer rng.Destroy()
	rng.SetSeed(42)

	output, err := memory.Alloc(16 * 4)
	if err != nil {
		return err
	}
	defer output.Free()

	if err := rng.GenerateUniform(output, 16); err != nil {
		return err
	}

	values, err := copyFloat32FromDevice(output, 16)
	if err != nil {
		return err
	}
	allZero := true
	for _, value := range values {
		if value != 0 {
			allZero = false
			break
		}
	}
	if allZero {
		return fmt.Errorf("cuRAND smoke generated all-zero output")
	}

	poisson, err := memory.Alloc(16 * int64(unsafe.Sizeof(uint32(0))))
	if err != nil {
		return err
	}
	defer poisson.Free()
	if err := rng.GeneratePoisson(poisson, 16, 4.0); err != nil {
		return err
	}
	return nil
}

func smokeCuSPARSE() error {
	ctx, err := libraries.CreateSparseContext()
	if err != nil {
		return fmt.Errorf("cuSPARSE smoke create context: %w", err)
	}
	defer ctx.DestroyContext()

	denseA, err := memory.Alloc(9 * 4)
	if err != nil {
		return err
	}
	defer denseA.Free()
	if err := copyFloat32ToDevice(denseA, []float32{
		1, 0, 2,
		0, 3, 0,
		4, 0, 5,
	}); err != nil {
		return err
	}

	A, err := ctx.DenseToSparse(denseA, 3, 3, libraries.MatrixFormatCSR)
	if err != nil {
		return err
	}
	defer A.Destroy()

	x, err := memory.Alloc(3 * 4)
	if err != nil {
		return err
	}
	defer x.Free()
	if err := copyFloat32ToDevice(x, []float32{1, 2, 3}); err != nil {
		return err
	}

	y, err := memory.Alloc(3 * 4)
	if err != nil {
		return err
	}
	defer y.Free()
	if err := ctx.SpMV(1, A, x, 0, y); err != nil {
		return err
	}
	yValues, err := copyFloat32FromDevice(y, 3)
	if err != nil {
		return err
	}
	if yValues[0] == 0 && yValues[1] == 0 && yValues[2] == 0 {
		return fmt.Errorf("cuSPARSE smoke produced all-zero SpMV output for non-zero input")
	}

	roundTrip, err := ctx.SparseToDense(A)
	if err != nil {
		return err
	}
	defer roundTrip.Free()
	roundTripValues, err := copyFloat32FromDevice(roundTrip, 9)
	if err != nil {
		return err
	}
	if roundTripValues[0] != 1 || roundTripValues[2] != 2 || roundTripValues[4] != 3 || roundTripValues[6] != 4 || roundTripValues[8] != 5 {
		return fmt.Errorf("cuSPARSE smoke dense round-trip mismatch")
	}
	return nil
}

func copyFloat32ToDevice(dst *memory.Memory, values []float32) error {
	if len(values) == 0 {
		return nil
	}
	hostBytes := unsafe.Slice((*byte)(unsafe.Pointer(&values[0])), len(values)*4)
	return memory.CopyHostToDevice(dst, hostBytes)
}

func copyFloat32FromDevice(src *memory.Memory, length int) ([]float32, error) {
	host := make([]float32, length)
	if length == 0 {
		return host, nil
	}
	hostBytes := unsafe.Slice((*byte)(unsafe.Pointer(&host[0])), length*4)
	if err := memory.CopyDeviceToHost(hostBytes, src); err != nil {
		return nil, err
	}
	return host, nil
}

func copyComplex64ToDevice(dst *memory.Memory, values []libraries.Complex64) error {
	if len(values) == 0 {
		return nil
	}
	hostBytes := unsafe.Slice((*byte)(unsafe.Pointer(&values[0])), len(values)*8)
	return memory.CopyHostToDevice(dst, hostBytes)
}

func copyComplex64FromDevice(src *memory.Memory, length int) ([]libraries.Complex64, error) {
	host := make([]libraries.Complex64, length)
	if length == 0 {
		return host, nil
	}
	hostBytes := unsafe.Slice((*byte)(unsafe.Pointer(&host[0])), length*8)
	if err := memory.CopyDeviceToHost(hostBytes, src); err != nil {
		return nil, err
	}
	return host, nil
}
