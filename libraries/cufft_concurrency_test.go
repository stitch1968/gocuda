package libraries

import (
	"fmt"
	"sync"
	"testing"

	"github.com/stitch1968/gocuda/memory"
)

func TestFFTPlanConcurrentExecution(t *testing.T) {
	ctx, err := CreateFFTContext()
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := ctx.DestroyContext(); err != nil {
			t.Error(err)
		}
	})
	plan, err := ctx.CreatePlan1D(4, FFTTypeC2C, 1)
	if err != nil {
		t.Fatal(err)
	}
	const workers = 4
	inputs := make([]*memory.Memory, workers)
	outputs := make([]*memory.Memory, workers)
	for i := range workers {
		inputs[i] = fftTestMemory(t, 32)
		outputs[i] = fftTestMemory(t, 32)
		if err := writeMathTypedMemory(inputs[i], []Complex64{{Real: float32(i + 1)}, {}, {}, {}}); err != nil {
			t.Fatal(err)
		}
	}
	var wg sync.WaitGroup
	errs := make(chan error, workers)
	for i := range workers {
		wg.Add(1)
		go func(index int) {
			defer wg.Done()
			for range 4 {
				if err := ctx.ExecC2C(plan, inputs[index], outputs[index], FFTForward); err != nil {
					errs <- err
					return
				}
			}
			values, err := readMathTypedMemory[Complex64](outputs[index], 4)
			if err != nil {
				errs <- err
				return
			}
			for _, value := range values {
				if value.Real != float32(index+1) || value.Imag != 0 {
					errs <- fmt.Errorf("worker %d got %+v", index, value)
					return
				}
			}
		}(i)
	}
	wg.Wait()
	close(errs)
	for err := range errs {
		t.Error(err)
	}
}
