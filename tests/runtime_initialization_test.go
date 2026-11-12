package tests

import (
	"os"
	"os/exec"
	"sync"
	"testing"

	cuda "github.com/stitch1968/gocuda"
)

func TestRuntimeConcurrentDiscovery(t *testing.T) {
	const marker = "GOCUDA_RUNTIME_DISCOVERY_CHILD"
	if os.Getenv(marker) != "1" {
		cmd := exec.Command(os.Args[0], "-test.run=^TestRuntimeConcurrentDiscovery$")
		cmd.Env = append(os.Environ(), marker+"=1")
		output, err := cmd.CombinedOutput()
		if err != nil {
			t.Fatalf("concurrent runtime discovery failed: %v\n%s", err, output)
		}
		return
	}
	var wg sync.WaitGroup
	for range 16 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for range 100 {
				if cuda.GetCudaRuntime() == nil {
					t.Error("runtime must not be nil")
				}
				_ = cuda.ShouldUseCuda()
				_ = cuda.IsCudaAvailable()
			}
		}()
	}
	wg.Wait()
}
