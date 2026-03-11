//go:build cuda
// +build cuda

package libraries

/*
#cgo linux CFLAGS: -I/usr/local/cuda/include -I/opt/cuda/include
#cgo windows CFLAGS: -I"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.1/include" -I"D:/NVIDIA/include"
#cgo linux LDFLAGS: -L/usr/local/cuda/lib64 -L/opt/cuda/lib64 -lcurand
#cgo windows LDFLAGS: -L${SRCDIR}/../lib_mingw -lcurand

#include <curand.h>

static curandStatus_t createGeneratorWrapper(curandGenerator_t* generator, int rngType) {
	return curandCreateGenerator(generator, (curandRngType_t)rngType);
}
*/
import "C"

import (
	"fmt"
	"unsafe"

	"github.com/stitch1968/gocuda/memory"
)

func createNativeRandomGenerator(rngType RngType) (*RandomGenerator, error) {
	nativeType, err := nativeCurandType(rngType)
	if err != nil {
		return nil, err
	}

	var handle C.curandGenerator_t
	if status := C.createGeneratorWrapper(&handle, C.int(nativeType)); status != C.CURAND_STATUS_SUCCESS {
		return nil, curandError("curandCreateGenerator", status)
	}

	rg := &RandomGenerator{
		rngType: rngType,
		seed:    0,
		handle:  unsafe.Pointer(handle),
		native:  true,
	}

	if isPseudoRandomType(rngType) {
		rg.seed = 1
		if status := C.curandSetPseudoRandomGeneratorSeed(handle, C.ulonglong(rg.seed)); status != C.CURAND_STATUS_SUCCESS {
			_ = C.curandDestroyGenerator(handle)
			return nil, curandError("curandSetPseudoRandomGeneratorSeed", status)
		}
	}

	return rg, nil
}

func setNativeRandomSeed(rg *RandomGenerator, seed uint64) error {
	if !isPseudoRandomType(rg.rngType) {
		rg.seed = seed
		return nil
	}
	status := C.curandSetPseudoRandomGeneratorSeed(C.curandGenerator_t(rg.handle), C.ulonglong(seed))
	if status != C.CURAND_STATUS_SUCCESS {
		return curandError("curandSetPseudoRandomGeneratorSeed", status)
	}
	rg.seed = seed
	return nil
}

func generateNativeUniform(rg *RandomGenerator, output *memory.Memory, n int) error {
	status := C.curandGenerateUniform(C.curandGenerator_t(rg.handle), (*C.float)(output.Ptr()), C.size_t(n))
	if status != C.CURAND_STATUS_SUCCESS {
		return curandError("curandGenerateUniform", status)
	}
	return nil
}

func generateNativeNormal(rg *RandomGenerator, output *memory.Memory, n int, mean, stddev float32) error {
	status := C.curandGenerateNormal(C.curandGenerator_t(rg.handle), (*C.float)(output.Ptr()), C.size_t(n), C.float(mean), C.float(stddev))
	if status != C.CURAND_STATUS_SUCCESS {
		return curandError("curandGenerateNormal", status)
	}
	return nil
}

func generateNativeLogNormal(rg *RandomGenerator, output *memory.Memory, n int, mean, stddev float32) error {
	status := C.curandGenerateLogNormal(C.curandGenerator_t(rg.handle), (*C.float)(output.Ptr()), C.size_t(n), C.float(mean), C.float(stddev))
	if status != C.CURAND_STATUS_SUCCESS {
		return curandError("curandGenerateLogNormal", status)
	}
	return nil
}

func generateNativePoisson(rg *RandomGenerator, output *memory.Memory, n int, lambda float32) error {
	status := C.curandGeneratePoisson(C.curandGenerator_t(rg.handle), (*C.uint)(output.Ptr()), C.size_t(n), C.double(lambda))
	if status != C.CURAND_STATUS_SUCCESS {
		return curandError("curandGeneratePoisson", status)
	}
	return nil
}

func destroyNativeRandomGenerator(rg *RandomGenerator) error {
	status := C.curandDestroyGenerator(C.curandGenerator_t(rg.handle))
	if status != C.CURAND_STATUS_SUCCESS {
		return curandError("curandDestroyGenerator", status)
	}
	return nil
}

func nativeCurandType(rngType RngType) (C.int, error) {
	switch rngType {
	case RngTypePseudoDefault:
		return C.CURAND_RNG_PSEUDO_DEFAULT, nil
	case RngTypeXorwow:
		return C.CURAND_RNG_PSEUDO_XORWOW, nil
	case RngTypeMrg32k3a:
		return C.CURAND_RNG_PSEUDO_MRG32K3A, nil
	case RngTypeMtgp32:
		return C.CURAND_RNG_PSEUDO_MTGP32, nil
	case RngTypeSobol32:
		return C.CURAND_RNG_QUASI_SOBOL32, nil
	case RngTypeScrambledSobol32:
		return C.CURAND_RNG_QUASI_SCRAMBLED_SOBOL32, nil
	case RngTypeSobol64:
		return C.CURAND_RNG_QUASI_SOBOL64, nil
	case RngTypeScrambledSobol64:
		return C.CURAND_RNG_QUASI_SCRAMBLED_SOBOL64, nil
	default:
		return 0, fmt.Errorf("unsupported cuRAND generator type %v", rngType)
	}
}

func isPseudoRandomType(rngType RngType) bool {
	switch rngType {
	case RngTypePseudoDefault, RngTypeXorwow, RngTypeMrg32k3a, RngTypeMtgp32:
		return true
	default:
		return false
	}
}

func curandError(operation string, status C.curandStatus_t) error {
	return fmt.Errorf("%s failed: %s (%d)", operation, curandStatusString(status), int(status))
}

func curandStatusString(status C.curandStatus_t) string {
	switch status {
	case C.CURAND_STATUS_SUCCESS:
		return "success"
	case C.CURAND_STATUS_VERSION_MISMATCH:
		return "version mismatch"
	case C.CURAND_STATUS_NOT_INITIALIZED:
		return "not initialized"
	case C.CURAND_STATUS_ALLOCATION_FAILED:
		return "allocation failed"
	case C.CURAND_STATUS_TYPE_ERROR:
		return "type error"
	case C.CURAND_STATUS_OUT_OF_RANGE:
		return "out of range"
	case C.CURAND_STATUS_LENGTH_NOT_MULTIPLE:
		return "length not multiple"
	case C.CURAND_STATUS_DOUBLE_PRECISION_REQUIRED:
		return "double precision required"
	case C.CURAND_STATUS_LAUNCH_FAILURE:
		return "launch failure"
	case C.CURAND_STATUS_PREEXISTING_FAILURE:
		return "preexisting failure"
	case C.CURAND_STATUS_INITIALIZATION_FAILED:
		return "initialization failed"
	case C.CURAND_STATUS_ARCH_MISMATCH:
		return "architecture mismatch"
	case C.CURAND_STATUS_INTERNAL_ERROR:
		return "internal error"
	default:
		return "unknown error"
	}
}
