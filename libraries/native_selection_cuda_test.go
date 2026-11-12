//go:build cuda

package libraries

import (
	"testing"

	cuda "github.com/stitch1968/gocuda"
	"github.com/stitch1968/gocuda/memory"
	"github.com/stitch1968/gocuda/streams"
)

func TestNativeLibrarySelectionOnEachDevice(t *testing.T) {
	if err := cuda.Initialize(); err != nil {
		t.Fatal(err)
	}
	if !cuda.ShouldUseCuda() {
		t.Skip("CUDA hardware unavailable")
	}
	devices, err := cuda.GetDevices()
	if err != nil {
		t.Fatal(err)
	}
	caller, err := cuda.NewContext(0)
	if err != nil {
		t.Fatal(err)
	}
	for deviceID := range devices {
		t.Run(devices[deviceID].Name, func(t *testing.T) {
			err := caller.Run(func() error {
				ctx, err := CreateNativeFFTContextOnDevice(deviceID)
				if err != nil {
					return err
				}
				defer ctx.DestroyContext()
				stream, err := streams.NewNativeStream(deviceID, streams.StreamNonBlocking)
				if err != nil {
					return err
				}
				defer stream.Close()
				for _, create := range []func() (*FFTPlan, error){
					func() (*FFTPlan, error) { return ctx.CreatePlan1D(4, FFTTypeC2C, 1) },
					func() (*FFTPlan, error) { return ctx.CreatePlan2D(2, 2, FFTTypeC2C) },
					func() (*FFTPlan, error) { return ctx.CreatePlan3D(2, 2, 2, FFTTypeC2C) },
				} {
					plan, err := create()
					if err != nil {
						return err
					}
					if plan.deviceID != deviceID {
						t.Errorf("FFT plan device = %d, want %d", plan.deviceID, deviceID)
					}
					if err := plan.SetStream(stream); err != nil {
						return err
					}
					if err := plan.DestroyPlan(); err != nil {
						return err
					}
				}
				plan, err := ctx.CreatePlan1D(4, FFTTypeC2C, 1)
				if err != nil {
					return err
				}
				if err := plan.SetStream(stream); err != nil {
					return err
				}
				input, err := memory.AllocDeviceOnlyOnDevice(32, deviceID)
				if err != nil {
					return err
				}
				defer input.Free()
				spectrum, err := memory.AllocDeviceOnlyOnDevice(32, deviceID)
				if err != nil {
					return err
				}
				defer spectrum.Free()
				if err := writeMathTypedMemory(input, []Complex64{{Real: 1}, {Real: 2}, {Real: 3}, {Real: 4}}); err != nil {
					return err
				}
				if err := ctx.ExecC2C(plan, input, spectrum, FFTForward); err != nil {
					return err
				}
				if err := stream.Synchronize(); err != nil {
					return err
				}
				transformed, err := readMathTypedMemory[Complex64](spectrum, 4)
				if err != nil {
					return err
				}
				want := []Complex64{{Real: 10}, {Real: -2, Imag: 2}, {Real: -2}, {Real: -2, Imag: -2}}
				for i := range want {
					if transformed[i] != want[i] {
						t.Errorf("FFT output[%d] = %+v, want %+v", i, transformed[i], want[i])
					}
				}
				if err := plan.DestroyPlan(); err != nil {
					return err
				}
				rng, err := CreateNativeRandomGeneratorOnDevice(deviceID, RngTypeXorwow)
				if err != nil {
					return err
				}
				defer rng.Destroy()
				if rng.deviceID != deviceID {
					t.Errorf("random generator device = %d, want %d", rng.deviceID, deviceID)
				}
				output, err := memory.AllocDeviceOnlyOnDevice(64, deviceID)
				if err != nil {
					return err
				}
				defer output.Free()
				if err := rng.GenerateUniform(output, 16); err != nil {
					return err
				}
				values, err := readMathFloat32Memory(output, 16)
				if err != nil {
					return err
				}
				for _, value := range values {
					if value <= 0 || value > 1 {
						t.Errorf("uniform value outside (0,1]: %v", value)
					}
				}
				solver, err := CreateNativeSolverContextOnDevice(deviceID)
				if err != nil {
					return err
				}
				defer solver.DestroyContext()
				if solver.deviceID != deviceID || solver.Backend() != ImplementationModeNativeCUDA {
					t.Errorf("solver device/backend = %d/%s, want %d/native", solver.deviceID, solver.Backend(), deviceID)
				}
				matrix, err := memory.AllocDeviceOnlyOnDevice(16, deviceID)
				if err != nil {
					return err
				}
				defer matrix.Free()
				if err := writeMathFloat32Memory(matrix, []float32{4, 2, 2, 5}); err != nil {
					return err
				}
				if err := solver.CholeskyFactorization(matrix, 2); err != nil {
					return err
				}
				factor, err := readMathFloat32Memory(matrix, 4)
				if err != nil {
					return err
				}
				if factor[0] != 2 || factor[2] != 1 || factor[3] != 2 {
					t.Errorf("solver factor = %v, want upper triangle [2, 1, 2]", factor)
				}
				if current, err := cuda.GetCurrentDevice(); err != nil || current != 0 {
					t.Errorf("caller device = %d, error = %v; want 0", current, err)
				}
				return nil
			})
			if err != nil {
				t.Fatal(err)
			}
		})
	}
}
