//go:build cuda

package tests

import (
	"context"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"testing"
	"time"

	cuda "github.com/stitch1968/gocuda"
	"github.com/stitch1968/gocuda/kernels"
	"github.com/stitch1968/gocuda/memory"
	"github.com/stitch1968/gocuda/streams"
)

func TestCompileCUDANativeAndLaunch(t *testing.T) {
	skipOrFail := func(reason string) {
		t.Helper()
		if os.Getenv("GOCUDA_REQUIRE_NVCC_TEST") == "1" {
			t.Fatal(reason)
		}
		t.Skip(reason)
	}
	if err := cuda.Initialize(); err != nil {
		t.Fatal(err)
	}
	if !cuda.ShouldUseCuda() {
		skipOrFail("CUDA hardware unavailable")
	}
	if _, err := exec.LookPath("nvcc"); err != nil {
		skipOrFail("nvcc is unavailable")
	}
	if runtime.GOOS == "windows" {
		if _, err := exec.LookPath("cl.exe"); err != nil {
			skipOrFail("nvcc integration requires an MSVC developer environment")
		}
	}
	devices, err := cuda.GetDevices()
	if err != nil {
		t.Fatal(err)
	}
	if devices[0].Properties.Major < 7 || devices[0].Properties.Major == 7 && devices[0].Properties.Minor < 5 {
		skipOrFail("integration test requires compute capability 7.5 or newer")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 90*time.Second)
	defer cancel()
	includeDir := filepath.Join(t.TempDir(), "headers with spaces")
	if err := os.MkdirAll(includeDir, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(includeDir, "params.h"), []byte("#define BIAS 2\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	result, err := kernels.CompileCUDA(ctx, `
#include "params.h"
extern "C" __global__ void transform(float *values, unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) values[i] = values[i] * FACTOR + BIAS;
}
`, kernels.CUDACompileOptions{
		Architecture: "compute_75",
		IncludeDirs:  []string{includeDir},
		Defines:      []string{"FACTOR=3"},
	})
	if err != nil {
		t.Fatal(err)
	}
	stream, err := streams.NewNativeStream(0, streams.StreamNonBlocking)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := stream.Close(); err != nil {
			t.Error(err)
		}
	})
	module, err := kernels.LoadPTX(0, result.PTX)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := module.Close(); err != nil {
			t.Error(err)
		}
	})
	function, err := module.Function("transform")
	if err != nil {
		t.Fatal(err)
	}
	host, err := memory.AllocWithTypeAndStream(nil, 16, memory.TypePinned)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := host.Free(); err != nil {
			t.Error(err)
		}
	})
	device, err := memory.AllocDeviceOnlyOnDevice(16, 0)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := device.Free(); err != nil {
			t.Error(err)
		}
	})
	t.Cleanup(func() {
		if err := stream.Synchronize(); err != nil {
			t.Error(err)
		}
	})
	values, err := memory.View[float32](host, 4)
	if err != nil {
		t.Fatal(err)
	}
	copy(values, []float32{1, 2, 3, 4})
	if err := stream.CopyAsync(device, host, 16); err != nil {
		t.Fatal(err)
	}
	if err := function.Launch(stream, kernels.Dim3{X: 1, Y: 1, Z: 1}, kernels.Dim3{X: 32, Y: 1, Z: 1}, 0, kernels.BufferArg(device, 0), kernels.Uint32Arg(4)); err != nil {
		t.Fatal(err)
	}
	if err := stream.CopyAsync(host, device, 16); err != nil {
		t.Fatal(err)
	}
	if err := stream.Synchronize(); err != nil {
		t.Fatal(err)
	}
	for i, value := range values {
		if want := float32((i+1)*3 + 2); value != want {
			t.Fatalf("transform[%d] = %v, want %v", i, value, want)
		}
	}
}
