package libraries

import (
	"errors"
	"fmt"
	"sync"
	"testing"

	cuda "github.com/stitch1968/gocuda"
	"github.com/stitch1968/gocuda/memory"
	"github.com/stitch1968/gocuda/streams"
)

func TestCuDNNHandlePolicy(t *testing.T) {
	handle, err := CreateDNNHandle()
	if err != nil {
		t.Fatal(err)
	}
	want := ImplementationModeValidatedGo
	if cuda.ShouldUseCuda() && cudnnNativeAvailable() {
		want = ImplementationModeNativeCUDA
	}
	if got := handle.Backend(); got != want {
		t.Fatalf("cuDNN backend = %q, want %q", got, want)
	}
	if err := handle.Synchronize(); err != nil {
		t.Fatal(err)
	}
	if err := handle.SetStream(nil); err != nil {
		t.Fatal(err)
	}
	if !handle.native {
		if err := handle.SetStream(&streams.NativeStream{}); err == nil {
			t.Fatal("simulation must reject native streams")
		}
	}
	if err := handle.DestroyHandle(); err != nil {
		t.Fatal(err)
	}
	if err := handle.Synchronize(); err == nil {
		t.Fatal("destroyed handle must reject synchronization")
	}
	if err := handle.SetStream(nil); err == nil {
		t.Fatal("destroyed handle must reject stream changes")
	}
	if !cuda.ShouldUseCuda() || !cudnnNativeAvailable() {
		if _, err := CreateNativeDNNHandleOnDevice(0); !errors.Is(err, streams.ErrNativeCUDAUnavailable) {
			t.Fatalf("native cuDNN creation error = %v", err)
		}
	} else if _, err := CreateNativeDNNHandleOnDevice(-1); err == nil {
		t.Fatal("expected invalid cuDNN device to fail")
	}
}

func TestCuDNNConvenienceCleanup(t *testing.T) {
	handle := &DNNHandle{}
	first := errors.New("first descriptor cleanup")
	second := errors.New("second descriptor cleanup")
	var order []int
	err := destroyDNNConvenienceResources(handle, []func() error{
		func() error {
			if !handle.destroyed {
				t.Error("handle must be destroyed before descriptors")
			}
			order = append(order, 1)
			return first
		},
		func() error {
			if !handle.destroyed {
				t.Error("handle must be destroyed before descriptors")
			}
			order = append(order, 2)
			return second
		},
	})
	if !errors.Is(err, first) || !errors.Is(err, second) || fmt.Sprint(order) != "[2 1]" {
		t.Fatalf("cleanup error = %v, descriptor order = %v", err, order)
	}
}

func TestCuDNNDescriptorAndBufferRetention(t *testing.T) {
	handle := &DNNHandle{}
	tensor := &TensorDescriptor{}
	filter := &FilterDescriptor{}
	conv := &ConvolutionDescriptor{}
	pool := &PoolingDescriptor{}
	activation := &ActivationDescriptor{}
	if err := tensor.SetTensor4dDescriptor(DNNTensorNCHW, DNNDataFloat, 1, 1, 1, 1); err != nil {
		t.Fatal(err)
	}
	if err := filter.SetFilter4dDescriptor(DNNDataFloat, DNNTensorNCHW, 1, 1, 1, 1); err != nil {
		t.Fatal(err)
	}
	if err := conv.SetConvolution2dDescriptor(0, 0, 1, 1, 1, 1, DNNCrossCorrelation, DNNDataFloat); err != nil {
		t.Fatal(err)
	}
	if err := pool.SetPooling2dDescriptor(DNNPoolingMax, 1, 1, 0, 0, 1, 1); err != nil {
		t.Fatal(err)
	}
	if err := activation.SetActivationDescriptor(DNNActivationRelu, DNNPropagateNaN, 0); err != nil {
		t.Fatal(err)
	}
	buffer, err := memory.Alloc(4)
	if err != nil {
		t.Fatal(err)
	}
	defer buffer.Free()
	started, finish, result := make(chan struct{}), make(chan struct{}), make(chan error, 1)
	go func() {
		result <- handle.execute([]*dnnDescriptorState{&tensor.dnnDescriptorState, &tensor.dnnDescriptorState,
			&filter.dnnDescriptorState, &conv.dnnDescriptorState, &pool.dnnDescriptorState, &activation.dnnDescriptorState},
			[]*memory.Memory{buffer}, func() (*memory.Memory, error) {
				close(started)
				<-finish
				return nil, nil
			})
	}()
	<-started
	for _, tc := range []struct {
		name string
		run  func() error
	}{
		{"tensor update", func() error { return tensor.SetTensor4dDescriptor(DNNTensorNCHW, DNNDataFloat, 1, 1, 1, 1) }},
		{"tensor destroy", tensor.DestroyTensorDescriptor},
		{"filter update", func() error { return filter.SetFilter4dDescriptor(DNNDataFloat, DNNTensorNCHW, 1, 1, 1, 1) }},
		{"filter destroy", filter.DestroyFilterDescriptor},
		{"convolution update", func() error {
			return conv.SetConvolution2dDescriptor(0, 0, 1, 1, 1, 1, DNNCrossCorrelation, DNNDataFloat)
		}},
		{"convolution destroy", conv.DestroyConvolutionDescriptor},
		{"pool update", func() error { return pool.SetPooling2dDescriptor(DNNPoolingMax, 1, 1, 0, 0, 1, 1) }},
		{"pool destroy", pool.DestroyPoolingDescriptor},
		{"activation update", func() error { return activation.SetActivationDescriptor(DNNActivationRelu, DNNPropagateNaN, 0) }},
		{"activation destroy", activation.DestroyActivationDescriptor},
		{"buffer free", buffer.Free},
	} {
		if err := tc.run(); err == nil {
			close(finish)
			<-result
			t.Fatalf("%s must fail while in use", tc.name)
		}
	}
	destroyStarted, destroyed := make(chan struct{}), make(chan error, 1)
	go func() {
		close(destroyStarted)
		destroyed <- handle.DestroyHandle()
	}()
	<-destroyStarted
	select {
	case err := <-destroyed:
		close(finish)
		<-result
		t.Fatalf("handle was destroyed while in use: %v", err)
	default:
	}
	close(finish)
	if err := <-result; err != nil {
		t.Fatal(err)
	}
	if err := <-destroyed; err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct {
		name string
		run  func() error
	}{
		{"tensor", tensor.DestroyTensorDescriptor},
		{"filter", filter.DestroyFilterDescriptor},
		{"convolution", conv.DestroyConvolutionDescriptor},
		{"pool", pool.DestroyPoolingDescriptor},
		{"activation", activation.DestroyActivationDescriptor},
		{"buffer", buffer.Free},
	} {
		if err := tc.run(); err != nil {
			t.Fatalf("%s must be released after completion: %v", tc.name, err)
		}
	}
	if err := handle.Synchronize(); err == nil {
		t.Fatal("destroyed handle must reject synchronization")
	}
}

func TestCuDNNConcurrentActivationAndDescriptorUpdates(t *testing.T) {
	handle := &DNNHandle{}
	desc := &TensorDescriptor{}
	activation := &ActivationDescriptor{}
	if err := desc.SetTensor4dDescriptor(DNNTensorNCHW, DNNDataFloat, 1, 1, 1, 2); err != nil {
		t.Fatal(err)
	}
	if err := activation.SetActivationDescriptor(DNNActivationRelu, DNNPropagateNaN, 0); err != nil {
		t.Fatal(err)
	}
	input := solverTestBuffer(t, []float32{-1, 2})
	output := solverTestBuffer(t, []float32{0, 0})
	const iterations = 32
	var wg sync.WaitGroup
	errs := make(chan error, 2)
	wg.Add(2)
	go func() {
		defer wg.Done()
		for range iterations {
			if err := handle.ActivationForward(activation, 1, desc, input, 0, desc, output); err != nil {
				errs <- err
				return
			}
		}
	}()
	go func() {
		defer wg.Done()
		for range iterations {
			err := activation.SetActivationDescriptor(DNNActivationRelu, DNNPropagateNaN, 0)
			if err != nil && !errors.Is(err, ErrDNNDescriptorInUse) {
				errs <- fmt.Errorf("unexpected descriptor update error: %w", err)
				return
			}
		}
	}()
	wg.Wait()
	close(errs)
	for err := range errs {
		t.Error(err)
	}
	if err := handle.DestroyHandle(); err != nil {
		t.Fatal(err)
	}
	if err := desc.DestroyTensorDescriptor(); err != nil {
		t.Fatal(err)
	}
	if err := activation.DestroyActivationDescriptor(); err != nil {
		t.Fatal(err)
	}
}
