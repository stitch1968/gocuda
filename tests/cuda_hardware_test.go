//go:build cuda

package tests

import (
	"context"
	"testing"

	cuda "github.com/stitch1968/gocuda"
)

func TestHardwareContextRunUsesSelectedDevice(t *testing.T) {
	if err := cuda.Initialize(); err != nil {
		t.Fatalf("Failed to initialize CUDA: %v", err)
	}

	devices, err := cuda.GetDevices()
	if err != nil {
		t.Fatalf("Failed to get devices: %v", err)
	}
	if len(devices) == 0 {
		t.Fatal("expected at least one CUDA device in hardware mode")
	}

	targetDevice := 0
	if len(devices) > 1 {
		targetDevice = 1
	}

	ctx, err := cuda.NewContext(targetDevice)
	if err != nil {
		t.Fatalf("Failed to create context: %v", err)
	}

	observedDevice := -1
	if err := ctx.Run(func() error {
		currentDevice, err := cuda.GetCurrentDevice()
		if err != nil {
			return err
		}
		observedDevice = currentDevice
		return nil
	}); err != nil {
		t.Fatalf("Context-bound execution failed: %v", err)
	}

	if observedDevice != targetDevice {
		t.Fatalf("expected context to run on device %d, got %d", targetDevice, observedDevice)
	}
}

func TestHardwareStreamExecutionUsesStreamDevice(t *testing.T) {
	if err := cuda.Initialize(); err != nil {
		t.Fatalf("Failed to initialize CUDA: %v", err)
	}

	devices, err := cuda.GetDevices()
	if err != nil {
		t.Fatalf("Failed to get devices: %v", err)
	}
	if len(devices) == 0 {
		t.Fatal("expected at least one CUDA device in hardware mode")
	}

	targetDevice := 0
	if len(devices) > 1 {
		targetDevice = 1
	}

	ctx, err := cuda.NewContext(targetDevice)
	if err != nil {
		t.Fatalf("Failed to create context: %v", err)
	}

	stream, err := ctx.NewStream()
	if err != nil {
		t.Fatalf("Failed to create stream: %v", err)
	}
	defer stream.Close()

	if stream.DeviceID() != targetDevice {
		t.Fatalf("expected stream on device %d, got %d", targetDevice, stream.DeviceID())
	}

	observedDevice := -1
	err = cuda.GoWithStream(stream, func(ctx context.Context, args ...any) error {
		currentDevice, err := cuda.GetCurrentDevice()
		if err != nil {
			return err
		}
		observedDevice = currentDevice
		return nil
	})
	if err != nil {
		t.Fatalf("Failed to execute stream-bound work: %v", err)
	}

	if err := stream.Synchronize(); err != nil {
		t.Fatalf("Failed to synchronize stream: %v", err)
	}

	if observedDevice != targetDevice {
		t.Fatalf("expected stream execution on device %d, got %d", targetDevice, observedDevice)
	}
}