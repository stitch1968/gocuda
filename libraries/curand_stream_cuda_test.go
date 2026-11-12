//go:build cuda

package libraries

import (
	"reflect"
	"testing"

	cuda "github.com/stitch1968/gocuda"
	"github.com/stitch1968/gocuda/memory"
	"github.com/stitch1968/gocuda/streams"
)

func TestNativeRandomGeneratorStreamLifetime(t *testing.T) {
	if err := cuda.Initialize(); err != nil {
		t.Fatal(err)
	}
	if !cuda.ShouldUseCuda() {
		t.Skip("CUDA hardware unavailable")
	}
	for _, kind := range []RngType{RngTypeXorwow, RngTypeMtgp32} {
		t.Run(kindString(kind), func(t *testing.T) {
			const n = 1024
			stream, err := streams.NewNativeStream(0, streams.StreamNonBlocking)
			if err != nil {
				t.Fatal(err)
			}
			t.Cleanup(func() {
				if err := stream.Close(); err != nil {
					t.Error(err)
				}
			})
			rg, err := CreateNativeRandomGeneratorOnDevice(0, kind)
			if err != nil {
				t.Fatal(err)
			}
			t.Cleanup(func() {
				if err := rg.Destroy(); err != nil {
					t.Error(err)
				}
			})
			output, err := memory.AllocDeviceOnlyOnDevice(n*4, 0)
			if err != nil {
				t.Fatal(err)
			}
			t.Cleanup(func() {
				if err := output.Free(); err != nil {
					t.Error(err)
				}
			})
			host, err := memory.AllocWithTypeAndStream(nil, n*4, memory.TypePinned)
			if err != nil {
				t.Fatal(err)
			}
			t.Cleanup(func() {
				if err := host.Free(); err != nil {
					t.Error(err)
				}
			})
			t.Cleanup(func() {
				if err := stream.Synchronize(); err != nil {
					t.Error(err)
				}
			})
			hostValues, err := memory.View[float32](host, n)
			if err != nil {
				t.Fatal(err)
			}
			if err := rg.SetStream(stream); err != nil {
				t.Fatal(err)
			}
			if err := rg.SetSeed(42); err != nil {
				t.Fatal(err)
			}
			if err := rg.GenerateUniform(output, n); err != nil {
				t.Fatal(err)
			}
			if err := output.Free(); err == nil {
				t.Fatal("expected queued cuRAND output to remain retained")
			}
			if err := stream.CopyAsync(host, output, n*4); err != nil {
				t.Fatal(err)
			}
			if err := rg.SetSeed(42); err != nil {
				t.Fatal(err)
			}
			first := append([]float32(nil), hostValues...)
			if err := rg.GenerateUniform(output, n); err != nil {
				t.Fatal(err)
			}
			if err := stream.CopyAsync(host, output, n*4); err != nil {
				t.Fatal(err)
			}
			if err := stream.Synchronize(); err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(first, hostValues) {
				t.Fatal("seed reset must reproduce the stream's generated sequence")
			}
			for _, value := range hostValues {
				if value <= 0 || value > 1 {
					t.Fatalf("uniform value outside (0,1]: %v", value)
				}
			}
			if err := rg.GenerateUniform(output, n); err != nil {
				t.Fatal(err)
			}
			if err := stream.CopyAsync(host, output, n*4); err != nil {
				t.Fatal(err)
			}
			if err := rg.Destroy(); err != nil {
				t.Fatal(err)
			}
			if err := output.Free(); err != nil {
				t.Fatalf("destroy must release completed output: %v", err)
			}
			if err := host.Free(); err != nil {
				t.Fatalf("destroy must synchronize stream copies: %v", err)
			}
		})
	}
}

func kindString(kind RngType) string {
	if kind == RngTypeMtgp32 {
		return "mtgp32"
	}
	return "xorwow"
}

func TestNativeRandomGeneratorDefaultStreamCompletes(t *testing.T) {
	if err := cuda.Initialize(); err != nil {
		t.Fatal(err)
	}
	if !cuda.ShouldUseCuda() {
		t.Skip("CUDA hardware unavailable")
	}
	rg, err := CreateNativeRandomGeneratorOnDevice(0, RngTypeXorwow)
	if err != nil {
		t.Fatal(err)
	}
	defer rg.Destroy()
	stream, err := streams.NewNativeStream(0, streams.StreamNonBlocking)
	if err != nil {
		t.Fatal(err)
	}
	defer stream.Close()
	output, err := memory.AllocDeviceOnlyOnDevice(64, 0)
	if err != nil {
		t.Fatal(err)
	}
	defer output.Free()
	if err := rg.SetStream(stream); err != nil {
		t.Fatal(err)
	}
	if err := rg.GenerateUniform(output, 16); err != nil {
		t.Fatal(err)
	}
	if err := rg.SetStream(nil); err != nil {
		t.Fatal(err)
	}
	if err := output.Free(); err != nil {
		t.Fatalf("switching streams must complete previous work: %v", err)
	}
	output, err = memory.AllocDeviceOnlyOnDevice(64, 0)
	if err != nil {
		t.Fatal(err)
	}
	defer output.Free()
	if err := rg.GenerateUniform(output, 16); err != nil {
		t.Fatal(err)
	}
	if err := output.Free(); err != nil {
		t.Fatalf("default-stream generation must complete before returning: %v", err)
	}
	if err := rg.SetStream(stream); err != nil {
		t.Fatal(err)
	}
	if err := stream.Close(); err != nil {
		t.Fatal(err)
	}
	if err := rg.SetStream(nil); err != nil {
		t.Fatalf("detaching a closed stream: %v", err)
	}
}
