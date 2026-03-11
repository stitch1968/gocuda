//go:build cuda

package libraries

/*
#cgo linux CFLAGS: -I/usr/local/cuda/include -I/opt/cuda/include
#cgo windows CFLAGS: -I"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.1/include" -I"D:/NVIDIA/include"
#cgo linux LDFLAGS: -L/usr/local/cuda/lib64 -L/opt/cuda/lib64 -lcufft
#cgo windows LDFLAGS: -L${SRCDIR}/../lib_mingw -lcufft

#include <cuda_runtime.h>
#include <cufft.h>

static cufftResult createPlan1d(cufftHandle* plan, int nx, int fftType, int batch) {
	return cufftPlan1d(plan, nx, (cufftType)fftType, batch);
}

static cufftResult createPlan2d(cufftHandle* plan, int nx, int ny, int fftType) {
	return cufftPlan2d(plan, nx, ny, (cufftType)fftType);
}

static cufftResult createPlan3d(cufftHandle* plan, int nx, int ny, int nz, int fftType) {
	return cufftPlan3d(plan, nx, ny, nz, (cufftType)fftType);
}

static cufftResult execC2C(cufftHandle plan, void* input, void* output, int direction) {
	return cufftExecC2C(plan, (cufftComplex*)input, (cufftComplex*)output, direction);
}

static cufftResult execR2C(cufftHandle plan, void* input, void* output) {
	return cufftExecR2C(plan, (cufftReal*)input, (cufftComplex*)output);
}

static cufftResult execC2R(cufftHandle plan, void* input, void* output) {
	return cufftExecC2R(plan, (cufftComplex*)input, (cufftReal*)output);
}

static cufftResult execZ2Z(cufftHandle plan, void* input, void* output, int direction) {
	return cufftExecZ2Z(plan, (cufftDoubleComplex*)input, (cufftDoubleComplex*)output, direction);
}

static cufftResult execD2Z(cufftHandle plan, void* input, void* output) {
	return cufftExecD2Z(plan, (cufftDoubleReal*)input, (cufftDoubleComplex*)output);
}

static cufftResult execZ2D(cufftHandle plan, void* input, void* output) {
	return cufftExecZ2D(plan, (cufftDoubleComplex*)input, (cufftDoubleReal*)output);
}

static cufftResult setPlanStream(cufftHandle plan, cudaStream_t stream) {
	return cufftSetStream(plan, stream);
}
*/
import "C"

import (
	"fmt"
	"unsafe"

	cuda "github.com/stitch1968/gocuda"
	"github.com/stitch1968/gocuda/internal"
	"github.com/stitch1968/gocuda/memory"
)

func createNativeFFTContext() (*FFTContext, error) {
	return &FFTContext{
		handle: 1,
		plans:  make([]*FFTPlan, 0),
		native: true,
	}, nil
}

func createNativeFFTPlan1D(ctx *FFTContext, nx int, fftType FFTType, batch int) (*FFTPlan, error) {
	cufftType, err := nativeCuFFTType(fftType)
	if err != nil {
		return nil, err
	}
	var handle C.cufftHandle
	if result := C.createPlan1d(&handle, C.int(nx), C.int(cufftType), C.int(batch)); result != C.CUFFT_SUCCESS {
		return nil, cufftError("cufftPlan1d", result)
	}
	return &FFTPlan{handle: uintptr(handle), fftType: fftType, nx: nx, batch: batch, native: true}, nil
}

func createNativeFFTPlan2D(ctx *FFTContext, nx, ny int, fftType FFTType) (*FFTPlan, error) {
	cufftType, err := nativeCuFFTType(fftType)
	if err != nil {
		return nil, err
	}
	var handle C.cufftHandle
	if result := C.createPlan2d(&handle, C.int(nx), C.int(ny), C.int(cufftType)); result != C.CUFFT_SUCCESS {
		return nil, cufftError("cufftPlan2d", result)
	}
	return &FFTPlan{handle: uintptr(handle), fftType: fftType, nx: nx, ny: ny, batch: 1, native: true}, nil
}

func createNativeFFTPlan3D(ctx *FFTContext, nx, ny, nz int, fftType FFTType) (*FFTPlan, error) {
	cufftType, err := nativeCuFFTType(fftType)
	if err != nil {
		return nil, err
	}
	var handle C.cufftHandle
	if result := C.createPlan3d(&handle, C.int(nx), C.int(ny), C.int(nz), C.int(cufftType)); result != C.CUFFT_SUCCESS {
		return nil, cufftError("cufftPlan3d", result)
	}
	return &FFTPlan{handle: uintptr(handle), fftType: fftType, nx: nx, ny: ny, nz: nz, batch: 1, native: true}, nil
}

func execNativeFFT(plan *FFTPlan, input, output *memory.Memory, direction FFTDirection) error {
	if input == nil || output == nil {
		return fmt.Errorf("input and output memory cannot be nil")
	}

	handle := C.cufftHandle(plan.handle)
	var result C.cufftResult

	switch plan.fftType {
	case FFTTypeC2C:
		result = C.execC2C(handle, input.Ptr(), output.Ptr(), C.int(direction))
	case FFTTypeR2C:
		result = C.execR2C(handle, input.Ptr(), output.Ptr())
	case FFTTypeC2R:
		result = C.execC2R(handle, input.Ptr(), output.Ptr())
	case FFTTypeZ2Z:
		result = C.execZ2Z(handle, input.Ptr(), output.Ptr(), C.int(direction))
	case FFTTypeD2Z:
		result = C.execD2Z(handle, input.Ptr(), output.Ptr())
	case FFTTypeZ2D:
		result = C.execZ2D(handle, input.Ptr(), output.Ptr())
	default:
		return fmt.Errorf("unsupported FFT type %v", plan.fftType)
	}

	if result != C.CUFFT_SUCCESS {
		return cufftError("cufftExec", result)
	}
	return nil
}

func setNativeFFTPlanStream(plan *FFTPlan, stream interface{}) error {
	var nativeStream unsafe.Pointer

	switch typed := stream.(type) {
	case nil:
		nativeStream = nil
	case *cuda.Stream:
		if typed != internal.GetDefaultStream() {
			handle := internal.NativeStreamHandle(typed)
			if handle == nil {
				return fmt.Errorf("custom CUDA stream attachment for cuFFT requires a CUDA-backed stream handle")
			}
			nativeStream = handle
		}
	default:
		return fmt.Errorf("unsupported stream type %T", stream)
	}

	if result := C.setPlanStream(C.cufftHandle(plan.handle), (C.cudaStream_t)(nativeStream)); result != C.CUFFT_SUCCESS {
		return cufftError("cufftSetStream", result)
	}
	return nil
}

func destroyNativeFFTPlan(plan *FFTPlan) error {
	if result := C.cufftDestroy(C.cufftHandle(plan.handle)); result != C.CUFFT_SUCCESS {
		return cufftError("cufftDestroy", result)
	}
	return nil
}

func destroyNativeFFTContext(ctx *FFTContext) error {
	return nil
}

func nativeCuFFTType(fftType FFTType) (C.int, error) {
	switch fftType {
	case FFTTypeC2C:
		return C.CUFFT_C2C, nil
	case FFTTypeR2C:
		return C.CUFFT_R2C, nil
	case FFTTypeC2R:
		return C.CUFFT_C2R, nil
	case FFTTypeD2Z:
		return C.CUFFT_D2Z, nil
	case FFTTypeZ2D:
		return C.CUFFT_Z2D, nil
	case FFTTypeZ2Z:
		return C.CUFFT_Z2Z, nil
	default:
		return 0, fmt.Errorf("unsupported FFT type %v", fftType)
	}
}

func cufftError(operation string, result C.cufftResult) error {
	return fmt.Errorf("%s failed: %s (%d)", operation, cufftResultString(result), int(result))
}

func cufftResultString(result C.cufftResult) string {
	switch result {
	case C.CUFFT_SUCCESS:
		return "success"
	case C.CUFFT_INVALID_PLAN:
		return "invalid plan"
	case C.CUFFT_ALLOC_FAILED:
		return "allocation failed"
	case C.CUFFT_INVALID_TYPE:
		return "invalid type"
	case C.CUFFT_INVALID_VALUE:
		return "invalid value"
	case C.CUFFT_INTERNAL_ERROR:
		return "internal error"
	case C.CUFFT_EXEC_FAILED:
		return "execution failed"
	case C.CUFFT_SETUP_FAILED:
		return "setup failed"
	case C.CUFFT_INVALID_SIZE:
		return "invalid size"
	case C.CUFFT_UNALIGNED_DATA:
		return "unaligned data"
	case C.CUFFT_INVALID_DEVICE:
		return "invalid device"
	case C.CUFFT_NO_WORKSPACE:
		return "no workspace"
	case C.CUFFT_NOT_IMPLEMENTED:
		return "not implemented"
	case C.CUFFT_NOT_SUPPORTED:
		return "not supported"
	default:
		return "unknown error"
	}
}
