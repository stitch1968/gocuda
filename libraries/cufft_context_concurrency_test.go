package libraries

import (
	"sync"
	"testing"
)

func TestFFTContextConcurrentPlanCreation(t *testing.T) {
	ctx, err := CreateFFTContext()
	if err != nil {
		t.Fatal(err)
	}
	const workers = 4
	var wg sync.WaitGroup
	errs := make(chan error, workers)
	for range workers {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for range 4 {
				plan, err := ctx.CreatePlan1D(4, FFTTypeC2C, 1)
				if err != nil {
					errs <- err
					return
				}
				if err := plan.DestroyPlan(); err != nil {
					errs <- err
					return
				}
			}
		}()
	}
	wg.Wait()
	close(errs)
	for err := range errs {
		t.Error(err)
	}
	if err := ctx.DestroyContext(); err != nil {
		t.Fatal(err)
	}
	if _, err := ctx.CreatePlan1D(4, FFTTypeC2C, 1); err == nil {
		t.Fatal("expected destroyed context to reject new plans")
	}
}
