package libraries

import (
	"math"
	"reflect"
	"testing"

	"github.com/stitch1968/gocuda/memory"
)

func TestRandomGeneratorProducesValues(t *testing.T) {
	const n = 4096
	rng, err := CreateRandomGenerator(RngTypeXorwow)
	if err != nil {
		t.Fatal(err)
	}
	defer rng.Destroy()
	output, err := memory.AllocDeviceOnly(int64(n * 4))
	if err != nil {
		t.Fatal(err)
	}
	defer output.Free()

	uniform := func() []float32 {
		t.Helper()
		if err := rng.SetSeed(1234); err != nil {
			t.Fatal(err)
		}
		if err := rng.GenerateUniform(output, n); err != nil {
			t.Fatal(err)
		}
		values, err := readMathFloat32Memory(output, n)
		if err != nil {
			t.Fatal(err)
		}
		return values
	}
	first, second := uniform(), uniform()
	if !reflect.DeepEqual(first, second) {
		t.Fatal("resetting the seed must reproduce the uniform sequence")
	}
	var sum float64
	for _, value := range first {
		if value <= 0 || value > 1 {
			t.Fatalf("uniform value outside (0,1]: %v", value)
		}
		sum += float64(value)
	}
	if mean := sum / n; mean < 0.45 || mean > 0.55 {
		t.Fatalf("uniform sample mean %v outside expected range", mean)
	}

	if err := rng.GenerateNormal(output, n, -1.5, 2); err != nil {
		t.Fatal(err)
	}
	normal, err := readMathFloat32Memory(output, n)
	if err != nil {
		t.Fatal(err)
	}
	sum = 0
	for _, value := range normal {
		sum += float64(value)
	}
	if mean := sum / n; mean < -1.7 || mean > -1.3 {
		t.Fatalf("normal sample mean %v outside expected range", mean)
	}

	if err := rng.GenerateLogNormal(output, n, 0, 1); err != nil {
		t.Fatal(err)
	}
	lognormal, err := readMathFloat32Memory(output, n)
	if err != nil {
		t.Fatal(err)
	}
	for _, value := range lognormal {
		if value <= 0 || math.IsNaN(float64(value)) {
			t.Fatalf("invalid log-normal sample %v", value)
		}
	}

	if err := rng.GeneratePoisson(output, n, 4); err != nil {
		t.Fatal(err)
	}
	poisson, err := readMathTypedMemory[uint32](output, n)
	if err != nil {
		t.Fatal(err)
	}
	sum = 0
	for _, value := range poisson {
		sum += float64(value)
	}
	if mean := sum / n; mean < 3.7 || mean > 4.3 {
		t.Fatalf("Poisson sample mean %v outside expected range", mean)
	}
}

func TestRandomGeneratorValidatesBuffersAndParameters(t *testing.T) {
	rng, err := CreateRandomGenerator(RngTypeXorwow)
	if err != nil {
		t.Fatal(err)
	}
	output, err := memory.Alloc(16)
	if err != nil {
		t.Fatal(err)
	}
	defer output.Free()
	if err := rng.GenerateUniform(output, 5); err == nil {
		t.Fatal("expected undersized output to fail")
	}
	if err := rng.GenerateNormal(output, 3, 0, 1); err == nil {
		t.Fatal("expected odd normal sample count to fail")
	}
	if err := rng.GenerateLogNormal(output, 4, 0, -1); err == nil {
		t.Fatal("expected negative standard deviation to fail")
	}
	if err := rng.GeneratePoisson(output, 4, float32(math.NaN())); err == nil {
		t.Fatal("expected NaN lambda to fail")
	}
	if !rng.native {
		if err := rng.GeneratePoisson(output, 4, math.SmallestNonzeroFloat32); err != nil {
			t.Fatal(err)
		}
		values, err := readMathTypedMemory[uint32](output, 4)
		if err != nil {
			t.Fatal(err)
		}
		for _, value := range values {
			if value != 0 {
				t.Fatalf("unexpected tiny-lambda Poisson value %d", value)
			}
		}
		if err := rng.GeneratePoisson(output, 4, 1025); err == nil {
			t.Fatal("expected unsupported simulated lambda to fail")
		}
		if _, err := CreateRandomGenerator(RngTypeSobol32); err == nil {
			t.Fatal("expected unsupported simulated quasi-random generator to fail")
		}
	} else {
		host, err := memory.AllocWithTypeAndStream(nil, 16, memory.TypeHost)
		if err != nil {
			t.Fatal(err)
		}
		defer host.Free()
		if err := rng.GenerateUniform(host, 4); err == nil {
			t.Fatal("expected native generation into Go memory to fail")
		}
	}
	if err := rng.Destroy(); err != nil {
		t.Fatal(err)
	}
	if err := rng.SetSeed(1); err == nil {
		t.Fatal("expected seed after destroy to fail")
	}
	if err := rng.GenerateUniform(output, 4); err == nil {
		t.Fatal("expected generation after destroy to fail")
	}
}

func TestRandomNumbersReturnsGeneratedValues(t *testing.T) {
	values, err := RandomNumbers(32, RngTypeXorwow)
	if err != nil {
		t.Fatal(err)
	}

	if len(values) != 32 {
		t.Fatalf("got %d values, want 32", len(values))
	}
	for _, value := range values {
		if value <= 0 || value > 1 {
			t.Fatalf("uniform value outside (0,1]: %v", value)
		}
	}
	if _, err := RandomNumbers(0, RngTypeXorwow); err == nil {
		t.Fatal("expected invalid size to fail")
	}
}

func TestRandomGeneratorSeedsPseudoTypes(t *testing.T) {
	const n = 1024
	output, err := memory.AllocDeviceOnly(n * 4)
	if err != nil {
		t.Fatal(err)
	}
	defer output.Free()
	for _, kind := range []RngType{RngTypePseudoDefault, RngTypeXorwow, RngTypeMrg32k3a, RngTypeMtgp32} {
		rng, err := CreateRandomGenerator(kind)
		if err != nil {
			t.Fatalf("generator type %d: %v", kind, err)
		}
		generate := func() []float32 {
			t.Helper()
			if err := rng.SetSeed(42); err != nil {
				t.Fatalf("generator type %d seed: %v", kind, err)
			}
			if err := rng.GenerateUniform(output, n); err != nil {
				t.Fatalf("generator type %d generate: %v", kind, err)
			}
			values, err := readMathFloat32Memory(output, n)
			if err != nil {
				t.Fatal(err)
			}
			return values
		}
		if first, second := generate(), generate(); !reflect.DeepEqual(first, second) {
			t.Errorf("generator type %d did not reset its sequence", kind)
		}
		if err := rng.Destroy(); err != nil {
			t.Errorf("generator type %d destroy: %v", kind, err)
		}
	}
}
