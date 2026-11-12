//go:build cuda

package tests

import (
	"os"
	"runtime"
	"testing"

	cuda "github.com/stitch1968/gocuda"
	"github.com/stitch1968/gocuda/libraries"
	"github.com/stitch1968/gocuda/memory"
	"github.com/stitch1968/gocuda/streams"
)

func TestHardwareDeviceSynchronizeWaitsForNativeStreams(t *testing.T) {
	if err := cuda.Initialize(); err != nil {
		t.Fatal(err)
	}
	if !cuda.ShouldUseCuda() {
		t.Skip("CUDA hardware unavailable")
	}
	stream, err := streams.NewNativeStream(0, streams.StreamNonBlocking)
	if err != nil {
		t.Fatal(err)
	}
	defer stream.Close()
	host, err := memory.AllocWithTypeAndStream(nil, 4, memory.TypePinned)
	if err != nil {
		t.Fatal(err)
	}
	defer host.Free()
	device, err := memory.AllocDeviceOnlyOnDevice(4, 0)
	if err != nil {
		t.Fatal(err)
	}
	defer device.Free()
	event, err := streams.NewNativeEvent(0, streams.NativeEventDefault)
	if err != nil {
		t.Fatal(err)
	}
	defer event.Close()
	defer stream.Synchronize()
	values, err := memory.View[byte](host, 4)
	if err != nil {
		t.Fatal(err)
	}
	copy(values, []byte{1, 2, 3, 4})
	if err := stream.CopyAsync(device, host, 4); err != nil {
		t.Fatal(err)
	}
	if err := event.Record(stream); err != nil {
		t.Fatal(err)
	}
	ctx, err := cuda.NewContext(0)
	if err != nil {
		t.Fatal(err)
	}
	if err := ctx.Run(cuda.DeviceSynchronize); err != nil {
		t.Fatal(err)
	}
	completed, err := event.Query()
	if err != nil || !completed {
		t.Fatalf("native event complete = %v, error = %v", completed, err)
	}
}

func TestHardwareDeviceAffinity(t *testing.T) {
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
	for deviceID := range devices {
		t.Run(devices[deviceID].Name, func(t *testing.T) {
			ctx, err := cuda.NewContext(deviceID)
			if err != nil {
				t.Fatal(err)
			}
			stream, err := ctx.NewStream()
			if err != nil {
				t.Fatal(err)
			}
			defer stream.Close()
			err = ctx.Run(func() error {
				for _, allocate := range []func() (*memory.Memory, error){
					func() (*memory.Memory, error) { return memory.Alloc(16) },
					func() (*memory.Memory, error) { return memory.AllocDeviceOnly(16) },
					func() (*memory.Memory, error) { return memory.AllocWithStream(stream, 16) },
					func() (*memory.Memory, error) { return cuda.Alloc(16).Unified().Allocate() },
				} {
					mem, err := allocate()
					if err != nil {
						return err
					}
					if got := mem.GetDeviceID(); got != deviceID {
						t.Errorf("allocation device = %d, want %d", got, deviceID)
					}
					if err := mem.Free(); err != nil {
						return err
					}
				}
				fft, err := libraries.CreateFFTContext()
				if err != nil {
					return err
				}
				defer fft.DestroyContext()
				if current, err := cuda.GetCurrentDevice(); err != nil || current != deviceID {
					t.Errorf("library initialization changed device: got %d, error %v", current, err)
				}
				return nil
			})
			if err != nil {
				t.Fatal(err)
			}
		})
	}
}

func TestHardwareSetDeviceOnLockedThread(t *testing.T) {
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
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	original, err := cuda.GetCurrentDevice()
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := cuda.SetDevice(original); err != nil {
			t.Errorf("restore device: %v", err)
		}
	}()
	for id := range devices {
		if err := cuda.SetDevice(id); err != nil {
			t.Fatal(err)
		}
		current, err := cuda.GetCurrentDevice()
		if err != nil || current != id {
			t.Fatalf("selected device %d, got %d, error %v", id, current, err)
		}
	}

}

func TestHardwareStreamAllocationUsesStreamDevice(t *testing.T) {
	if err := cuda.Initialize(); err != nil {
		t.Fatal(err)
	}
	devices, err := cuda.GetDevices()
	if err != nil {
		t.Fatal(err)
	}
	if !cuda.ShouldUseCuda() || len(devices) < 2 {
		if os.Getenv("GOCUDA_REQUIRE_MULTI_GPU") == "1" {
			t.Fatal("multi-GPU validation requires at least two CUDA devices")
		}
		t.Skip("requires at least two CUDA devices")
	}
	source, err := cuda.NewContext(0)
	if err != nil {
		t.Fatal(err)
	}
	target, err := cuda.NewContext(1)
	if err != nil {
		t.Fatal(err)
	}
	stream, err := target.NewStream()
	if err != nil {
		t.Fatal(err)
	}
	defer stream.Close()
	if err := source.Run(func() error {
		for _, kind := range []memory.Type{memory.TypeDevice, memory.TypeDeviceOnly, memory.TypeUnified} {
			allocation, err := memory.AllocWithTypeAndStream(stream, 16, kind)
			if err != nil {
				return err
			}
			if got := allocation.GetDeviceID(); got != 1 {
				t.Errorf("stream allocation type %d device = %d, want 1", kind, got)
			}
			if err := allocation.Free(); err != nil {
				return err
			}
		}
		return nil
	}); err != nil {
		t.Fatal(err)
	}
}
