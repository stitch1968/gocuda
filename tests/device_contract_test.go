package tests

import (
	"errors"
	"runtime"
	"testing"

	cuda "github.com/stitch1968/gocuda"
)

func TestDeviceSelectionAndSimpleContext(t *testing.T) {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	if err := cuda.Initialize(); err != nil {
		t.Fatal(err)
	}
	original, err := cuda.GetCurrentDevice()
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := cuda.SetDevice(original); err != nil {
			t.Errorf("restore device: %v", err)
		}
	}()
	if err := cuda.SetDevice(0); err != nil {
		t.Fatal(err)
	}
	if current, err := cuda.GetCurrentDevice(); err != nil || current != 0 {
		t.Fatalf("current device = %d, error = %v; want 0", current, err)
	}
	if err := cuda.SetDevice(-1); err == nil {
		t.Fatal("negative device must fail")
	}
	if err := cuda.SetCudaDevice(-1); err == nil {
		t.Fatal("negative CUDA device must fail")
	}
	simple, err := cuda.NewSimpleContext()
	if err != nil {
		t.Fatal(err)
	}
	want := errors.New("context callback")
	if err := simple.Run(func() error {
		device, err := cuda.GetCurrentDevice()
		if err != nil {
			return err
		}
		if device != 0 {
			t.Errorf("simple context device = %d, want 0", device)
		}
		return want
	}); !errors.Is(err, want) {
		t.Fatalf("callback error = %v, want %v", err, want)
	}
	if err := simple.Run(nil); err == nil {
		t.Fatal("nil context callback must fail")
	}
}
