//go:build cuda

package tests

import (
	"os"
	"os/exec"
	"sync"
	"testing"

	cuda "github.com/stitch1968/gocuda"
)

func TestHardwareFallbackBeforeRuntimeInitialization(t *testing.T) {
	const marker = "GOCUDA_FALLBACK_CHILD"
	if os.Getenv(marker) != "1" {
		cmd := exec.Command(os.Args[0], "-test.run=^TestHardwareFallbackBeforeRuntimeInitialization$")
		cmd.Env = append(os.Environ(), marker+"=1")
		output, err := cmd.CombinedOutput()
		if err != nil {
			t.Fatalf("fallback initialization failed: %v\n%s", err, output)
		}
		return
	}
	cuda.ForceFallbackToSimulation(true)
	_ = cuda.InitializeCudaRuntime()
	if cuda.ShouldUseCuda() {
		t.Fatal("explicit fallback before initialization was lost")
	}
	var wg sync.WaitGroup
	for range 8 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for range 100 {
				cuda.ForceFallbackToSimulation(true)
				_ = cuda.ShouldUseCuda()
				cuda.ForceFallbackToSimulation(false)
			}
		}()
	}
	wg.Wait()
	cuda.ForceFallbackToSimulation(false)
	if cuda.ShouldUseCuda() != cuda.IsCudaAvailable() {
		t.Fatal("disabling fallback must restore detected runtime mode")
	}
}
