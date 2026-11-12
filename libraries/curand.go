// Package libraries provides CUDA runtime library bindings for GoCUDA
// This implements cuRAND functionality for random number generation
package libraries

import (
	"errors"
	"fmt"
	"math"
	"math/rand"
	"sync"
	"time"
	"unsafe"

	cuda "github.com/stitch1968/gocuda"
	"github.com/stitch1968/gocuda/internal"
	"github.com/stitch1968/gocuda/memory"
	"github.com/stitch1968/gocuda/streams"
)

// cuRAND - Random Number Generation Library

// RandomGenerator types
type RngType int

const (
	RngTypePseudoDefault RngType = iota
	RngTypeXorwow
	RngTypeMrg32k3a
	RngTypeMtgp32
	RngTypeSobol32
	RngTypeScrambledSobol32
	RngTypeSobol64
	RngTypeScrambledSobol64
)

// RandomGenerator manages random number generation
type RandomGenerator struct {
	rngType           RngType
	seed              uint64
	handle            unsafe.Pointer
	state             *memory.Memory
	rng               *rand.Rand
	native            bool
	deviceID          int
	destroyed         bool
	mu                sync.Mutex
	nativeStream      *streams.NativeStream
	defaultRetentions []*memory.Retention
}

// CreateRandomGenerator creates a new random number generator
func CreateRandomGenerator(rngType RngType) (*RandomGenerator, error) {
	if err := ensureCudaReady(); err != nil {
		return nil, err
	}
	if cuda.ShouldUseCuda() {
		return createNativeRandomGenerator(rngType)
	}
	if !isPseudoRandomType(rngType) {
		return nil, fmt.Errorf("simulation does not support random generator type %d", rngType)
	}

	seed := uint64(time.Now().UnixNano())
	rg := &RandomGenerator{
		rngType: rngType,
		seed:    seed,
		rng:     rand.New(rand.NewSource(int64(seed))),
	}

	// Allocate state memory (simulated)
	var err error
	rg.state, err = memory.Alloc(1024) // State memory for generator
	if err != nil {
		return nil, fmt.Errorf("failed to allocate generator state: %v", err)
	}

	return rg, nil
}

// CreateNativeRandomGeneratorOnDevice creates a native-only cuRAND generator
// on deviceID. It never falls back to simulation.
func CreateNativeRandomGeneratorOnDevice(deviceID int, rngType RngType) (*RandomGenerator, error) {
	if err := ensureCudaReady(); err != nil {
		return nil, err
	}
	if !cuda.ShouldUseCuda() {
		return nil, streams.ErrNativeCUDAUnavailable
	}
	if deviceID < 0 || deviceID >= cuda.GetCudaDeviceCount() {
		return nil, fmt.Errorf("device ID %d out of range", deviceID)
	}
	var rg *RandomGenerator
	err := internal.RunOnDevice(deviceID, func() error {
		var createErr error
		rg, createErr = createNativeRandomGenerator(rngType)
		return createErr
	})
	if err != nil {
		if rg != nil {
			return nil, errors.Join(err, rg.Destroy())
		}
		return nil, err
	}
	return rg, nil
}

// Backend reports whether this generator uses native CUDA or Go simulation.
func (rg *RandomGenerator) Backend() LibraryImplementationMode {
	if rg == nil {
		return ""
	}
	if rg.native {
		return ImplementationModeNativeCUDA
	}
	return ImplementationModeValidatedGo
}

// SetSeed sets a pseudorandom generator's seed and resets its sequence offset.
// Outstanding generation work on the generator's stream is synchronized first.
func (rg *RandomGenerator) SetSeed(seed uint64) error {
	if rg == nil {
		return fmt.Errorf("nil random generator")
	}
	rg.mu.Lock()
	defer rg.mu.Unlock()
	if rg.destroyed {
		return fmt.Errorf("random generator has been destroyed")
	}
	if !isPseudoRandomType(rg.rngType) {
		return fmt.Errorf("seed is not supported for quasi-random generators")
	}
	if rg.native {
		if err := rg.synchronizeLocked(); err != nil {
			return err
		}
		return internal.RunOnDevice(rg.deviceID, func() error { return setNativeRandomSeed(rg, seed) })
	}
	rg.seed = seed
	rg.rng = rand.New(rand.NewSource(int64(seed)))
	return nil
}

// SetStream attaches a native CUDA stream. Passing nil selects the default
// stream. Existing work is synchronized before changing streams.
func (rg *RandomGenerator) SetStream(stream *streams.NativeStream) error {
	if rg == nil {
		return fmt.Errorf("nil random generator")
	}
	rg.mu.Lock()
	defer rg.mu.Unlock()
	if rg.destroyed {
		return fmt.Errorf("random generator has been destroyed")
	}
	if !rg.native {
		if stream != nil {
			return fmt.Errorf("native stream attachment requires a CUDA random generator")
		}
		return nil
	}
	if stream != nil && stream.DeviceID() != rg.deviceID {
		return fmt.Errorf("stream device %d does not match random generator device %d", stream.DeviceID(), rg.deviceID)
	}
	if err := rg.synchronizeLocked(); err != nil {
		return err
	}
	setStream := func(handle unsafe.Pointer) error {
		if err := setNativeRandomStream(rg, handle); err != nil {
			return err
		}
		rg.nativeStream = stream
		return nil
	}
	if stream != nil {
		return stream.WithNativeHandle(setStream)
	}
	return internal.RunOnDevice(rg.deviceID, func() error { return setStream(nil) })
}

func (rg *RandomGenerator) synchronizeLocked() error {
	if rg.nativeStream != nil {
		err := rg.nativeStream.Synchronize()
		if errors.Is(err, streams.ErrNativeStreamClosed) {
			return nil
		}
		return err
	}
	if len(rg.defaultRetentions) == 0 {
		return nil
	}
	if err := internal.CudaDefaultStreamSynchronizeOnDevice(rg.deviceID); err != nil {
		return err
	}
	for _, retention := range rg.defaultRetentions {
		retention.Release()
	}
	rg.defaultRetentions = nil
	return nil
}

func isPseudoRandomType(rngType RngType) bool {
	return rngType >= RngTypePseudoDefault && rngType <= RngTypeMtgp32
}

func (rg *RandomGenerator) generate(output *memory.Memory, n int, native, simulated func() error) error {
	if rg == nil {
		return fmt.Errorf("nil random generator")
	}
	rg.mu.Lock()
	defer rg.mu.Unlock()
	if rg.destroyed {
		return fmt.Errorf("random generator has been destroyed")
	}
	if output == nil || output.Ptr() == nil {
		return fmt.Errorf("output must be a live memory allocation")
	}
	if n <= 0 || int64(n) > output.Size()/4 {
		return fmt.Errorf("invalid number of samples %d for output size %d", n, output.Size())
	}
	if rg.native {
		if output.Data() != nil || output.GetDeviceID() != rg.deviceID {
			return fmt.Errorf("native random output must be CUDA memory on device %d", rg.deviceID)
		}
		if rg.nativeStream != nil {
			return rg.nativeStream.EnqueueWithBuffers([]*memory.Memory{output}, func(unsafe.Pointer) error {
				return native()
			})
		}
		retention, err := output.Retain()
		if err != nil {
			return err
		}
		rg.defaultRetentions = append(rg.defaultRetentions, retention)
		err = internal.RunOnDevice(rg.deviceID, native)
		return errors.Join(err, rg.synchronizeLocked())
	}
	retention, err := output.Retain()
	if err != nil {
		return err
	}
	defer retention.Release()
	return simulated()
}

// GenerateUniform generates uniform random numbers in (0, 1].
func (rg *RandomGenerator) GenerateUniform(output *memory.Memory, n int) error {
	return rg.generate(output, n, func() error {
		return generateNativeUniform(rg, output, n)
	}, func() error {
		values := make([]float32, n)
		for i := range values {
			values[i] = 1 - rg.rng.Float32()
		}
		return writeMathFloat32Memory(output, values)
	})
}

// GenerateNormal generates normally distributed float32 values.
func (rg *RandomGenerator) GenerateNormal(output *memory.Memory, n int, mean, stddev float32) error {
	if err := validateNormalParameters(n, mean, stddev); err != nil {
		return err
	}
	return rg.generate(output, n, func() error {
		return generateNativeNormal(rg, output, n, mean, stddev)
	}, func() error {
		values := make([]float32, n)
		for i := range values {
			values[i] = mean + stddev*float32(rg.rng.NormFloat64())
		}
		return writeMathFloat32Memory(output, values)
	})
}

// GenerateLogNormal generates log-normal random numbers
func (rg *RandomGenerator) GenerateLogNormal(output *memory.Memory, n int, mean, stddev float32) error {
	if err := validateNormalParameters(n, mean, stddev); err != nil {
		return err
	}
	return rg.generate(output, n, func() error {
		return generateNativeLogNormal(rg, output, n, mean, stddev)
	}, func() error {
		values := make([]float32, n)
		for i := range values {
			values[i] = float32(math.Exp(float64(mean) + float64(stddev)*rg.rng.NormFloat64()))
		}
		return writeMathFloat32Memory(output, values)
	})
}

func validateNormalParameters(n int, mean, stddev float32) error {
	if n <= 0 || n%2 != 0 {
		return fmt.Errorf("normal generation requires a positive even sample count")
	}
	if math.IsNaN(float64(mean)) || math.IsInf(float64(mean), 0) || math.IsNaN(float64(stddev)) || math.IsInf(float64(stddev), 0) || stddev < 0 {
		return fmt.Errorf("mean and standard deviation must be finite, with non-negative standard deviation")
	}
	return nil
}

// GeneratePoisson generates Poisson-distributed uint32 values.
func (rg *RandomGenerator) GeneratePoisson(output *memory.Memory, n int, lambda float32) error {
	if lambda <= 0 || math.IsNaN(float64(lambda)) || math.IsInf(float64(lambda), 0) {
		return fmt.Errorf("lambda must be finite and positive")
	}
	return rg.generate(output, n, func() error {
		return generateNativePoisson(rg, output, n, lambda)
	}, func() error {
		if lambda > 1024 {
			return fmt.Errorf("simulated Poisson generation supports lambda up to 1024")
		}
		values := make([]uint32, n)
		for i := range values {
			for remaining := float64(lambda); remaining > 0; {
				part := math.Min(remaining, 32)
				threshold := math.Exp(-part)
				product := 1.0
				var count uint32
				for {
					product *= rg.rng.Float64()
					count++
					if product <= threshold {
						break
					}
				}
				values[i] += count - 1
				remaining -= part
			}
		}
		return writeMathTypedMemory(output, values)
	})
}

// Destroy synchronizes outstanding generation work and cleans up the generator.
func (rg *RandomGenerator) Destroy() error {
	if rg == nil {
		return nil
	}
	rg.mu.Lock()
	defer rg.mu.Unlock()
	if rg.destroyed {
		return nil
	}
	var err error
	if rg.native {
		if err := rg.synchronizeLocked(); err != nil {
			return err
		}
		err = internal.RunOnDevice(rg.deviceID, func() error {
			if err := destroyNativeRandomGenerator(rg); err != nil {
				return err
			}
			rg.destroyed = true
			return nil
		})
	} else if rg.state != nil {
		err = rg.state.Free()
	}
	if err == nil {
		rg.destroyed = true
	}
	return err
}

// Utility function to simulate kernel execution
func simulateKernelExecution(kernelName string, operations int, complexity int) error {
	// Simulate execution time based on operation complexity
	simulationTime := time.Duration(operations*complexity) * time.Nanosecond
	time.Sleep(simulationTime)

	fmt.Printf("Simulated %s: %d operations\n", kernelName, operations)
	return nil
}
