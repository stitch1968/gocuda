//go:build cuda

package libraries

/*
#cgo linux CFLAGS: -I/usr/local/cuda/include -I/opt/cuda/include
#cgo windows CFLAGS: -I"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.1/include" -I"D:/NVIDIA/include"
#cgo linux LDFLAGS: -L/usr/local/cuda/lib64 -L/opt/cuda/lib64 -lnvjpeg
#cgo windows LDFLAGS: -L${SRCDIR}/../lib_mingw -lnvjpeg

#include <cuda_runtime.h>
#include <nvjpeg.h>
#include <string.h>

static nvjpegStatus_t createNVJPEGHandle(nvjpegHandle_t* handle) {
	return nvjpegCreateSimple(handle);
}

static nvjpegStatus_t createJpegStateWrapper(nvjpegHandle_t handle, nvjpegJpegState_t* state) {
	return nvjpegJpegStateCreate(handle, state);
}

static nvjpegStatus_t destroyJpegStateWrapper(nvjpegJpegState_t state) {
	return nvjpegJpegStateDestroy(state);
}

static nvjpegStatus_t createEncoderStateWrapper(nvjpegHandle_t handle, nvjpegEncoderState_t* state) {
	return nvjpegEncoderStateCreate(handle, state, NULL);
}

static nvjpegStatus_t createEncoderStateWithBackendWrapper(nvjpegHandle_t handle, nvjpegEncoderState_t* state, nvjpegEncBackend_t backend) {
	return nvjpegEncoderStateCreateWithBackend(handle, state, backend, NULL);
}

static nvjpegStatus_t destroyEncoderStateWrapper(nvjpegEncoderState_t state) {
	return nvjpegEncoderStateDestroy(state);
}

static nvjpegStatus_t createEncoderParamsWrapper(nvjpegHandle_t handle, nvjpegEncoderParams_t* params) {
	return nvjpegEncoderParamsCreate(handle, params, NULL);
}

static nvjpegStatus_t destroyEncoderParamsWrapper(nvjpegEncoderParams_t params) {
	return nvjpegEncoderParamsDestroy(params);
}

static nvjpegStatus_t destroyNVJPEGHandle(nvjpegHandle_t handle) {
	return nvjpegDestroy(handle);
}

static nvjpegStatus_t getJpegInfoWrapper(nvjpegHandle_t handle, const unsigned char* data, size_t length, int* components, nvjpegChromaSubsampling_t* subsampling, int* widths, int* heights) {
	return nvjpegGetImageInfo(handle, data, length, components, subsampling, widths, heights);
}

static void prepareInterleavedImage(nvjpegImage_t* image, unsigned char* data, unsigned int pitch) {
	memset(image, 0, sizeof(nvjpegImage_t));
	image->channel[0] = data;
	image->pitch[0] = pitch;
}

static nvjpegStatus_t decodeJpegWrapper(nvjpegHandle_t handle, nvjpegJpegState_t state, const unsigned char* data, size_t length, int format, nvjpegImage_t* image) {
	return nvjpegDecode(handle, state, data, length, (nvjpegOutputFormat_t)format, image, NULL);
}

static nvjpegStatus_t setEncoderQualityWrapper(nvjpegEncoderParams_t params, int quality) {
	return nvjpegEncoderParamsSetQuality(params, quality, NULL);
}

static nvjpegStatus_t setEncoderSamplingFactorsWrapper(nvjpegEncoderParams_t params, nvjpegChromaSubsampling_t subsampling) {
	return nvjpegEncoderParamsSetSamplingFactors(params, subsampling, NULL);
}

static nvjpegStatus_t encodeImageWrapper(nvjpegHandle_t handle, nvjpegEncoderState_t state, nvjpegEncoderParams_t params, nvjpegImage_t* image, int format, int width, int height) {
	return nvjpegEncodeImage(handle, state, params, image, (nvjpegInputFormat_t)format, width, height, NULL);
}

static nvjpegStatus_t retrieveBitstreamLengthWrapper(nvjpegHandle_t handle, nvjpegEncoderState_t state, size_t* length) {
	return nvjpegEncodeRetrieveBitstream(handle, state, NULL, length, NULL);
}

static nvjpegStatus_t retrieveBitstreamWrapper(nvjpegHandle_t handle, nvjpegEncoderState_t state, unsigned char* data, size_t* length) {
	return nvjpegEncodeRetrieveBitstream(handle, state, data, length, NULL);
}
*/
import "C"

import (
	"fmt"
	"unsafe"

	"github.com/stitch1968/gocuda/memory"
)

func nvjpegHandleToUintptr(handle C.nvjpegHandle_t) uintptr {
	return uintptr(unsafe.Pointer(handle))
}

func nvjpegHandleFromUintptr(handle uintptr) C.nvjpegHandle_t {
	return (C.nvjpegHandle_t)(unsafe.Pointer(handle))
}

func nvjpegJpegStateToUintptr(state C.nvjpegJpegState_t) uintptr {
	return uintptr(unsafe.Pointer(state))
}

func nvjpegJpegStateFromUintptr(state uintptr) C.nvjpegJpegState_t {
	return (C.nvjpegJpegState_t)(unsafe.Pointer(state))
}

func nvjpegEncoderStateToUintptr(state C.nvjpegEncoderState_t) uintptr {
	return uintptr(unsafe.Pointer(state))
}

func nvjpegEncoderStateFromUintptr(state uintptr) C.nvjpegEncoderState_t {
	return (C.nvjpegEncoderState_t)(unsafe.Pointer(state))
}

func nvjpegEncoderParamsToUintptr(params C.nvjpegEncoderParams_t) uintptr {
	return uintptr(unsafe.Pointer(params))
}

func nvjpegEncoderParamsFromUintptr(params uintptr) C.nvjpegEncoderParams_t {
	return (C.nvjpegEncoderParams_t)(unsafe.Pointer(params))
}

func nvjpegNativeAvailable() bool {
	return true
}

func createNativeJpegDecoder(backend JpegBackend) (*JpegDecoderState, error) {
	var handle C.nvjpegHandle_t
	if status := C.createNVJPEGHandle(&handle); status != C.NVJPEG_STATUS_SUCCESS {
		return nil, nvjpegError("nvjpegCreateSimple", status)
	}

	var state C.nvjpegJpegState_t
	if status := C.createJpegStateWrapper(handle, &state); status != C.NVJPEG_STATUS_SUCCESS {
		_ = C.destroyNVJPEGHandle(handle)
		return nil, nvjpegError("nvjpegJpegStateCreate", status)
	}

	return &JpegDecoderState{
		backend:      backend,
		nativeHandle: nvjpegHandleToUintptr(handle),
		nativeState:  nvjpegJpegStateToUintptr(state),
		native:       true,
	}, nil
}

func createNativeJpegEncoder(backend JpegBackend, quality int) (*JpegEncoderState, error) {
	var handle C.nvjpegHandle_t
	if status := C.createNVJPEGHandle(&handle); status != C.NVJPEG_STATUS_SUCCESS {
		return nil, nvjpegError("nvjpegCreateSimple", status)
	}

	var state C.nvjpegEncoderState_t
	if status := C.createEncoderStateWithBackendWrapper(handle, &state, nativeNVJPEGEncoderBackend(backend)); status != C.NVJPEG_STATUS_SUCCESS {
		_ = C.destroyNVJPEGHandle(handle)
		return nil, nvjpegError("nvjpegEncoderStateCreateWithBackend", status)
	}

	var params C.nvjpegEncoderParams_t
	if status := C.createEncoderParamsWrapper(handle, &params); status != C.NVJPEG_STATUS_SUCCESS {
		_ = C.destroyEncoderStateWrapper(state)
		_ = C.destroyNVJPEGHandle(handle)
		return nil, nvjpegError("nvjpegEncoderParamsCreate", status)
	}

	if status := C.setEncoderQualityWrapper(params, C.int(quality)); status != C.NVJPEG_STATUS_SUCCESS {
		_ = C.destroyEncoderParamsWrapper(params)
		_ = C.destroyEncoderStateWrapper(state)
		_ = C.destroyNVJPEGHandle(handle)
		return nil, nvjpegError("nvjpegEncoderParamsSetQuality", status)
	}
	if status := C.setEncoderSamplingFactorsWrapper(params, C.NVJPEG_CSS_444); status != C.NVJPEG_STATUS_SUCCESS {
		_ = C.destroyEncoderParamsWrapper(params)
		_ = C.destroyEncoderStateWrapper(state)
		_ = C.destroyNVJPEGHandle(handle)
		return nil, nvjpegError("nvjpegEncoderParamsSetSamplingFactors", status)
	}

	return &JpegEncoderState{
		backend:      backend,
		quality:      quality,
		nativeHandle: nvjpegHandleToUintptr(handle),
		nativeState:  nvjpegEncoderStateToUintptr(state),
		nativeParams: nvjpegEncoderParamsToUintptr(params),
		native:       true,
	}, nil
}

func decodeNativeJpeg(decoder *JpegDecoderState, jpegData []byte, params JpegDecodeParams) (*memory.Memory, int, int, error) {
	if len(jpegData) == 0 {
		return nil, 0, 0, fmt.Errorf("empty JPEG data")
	}
	if params.CropWidth > 0 || params.CropHeight > 0 || params.ScaleWidth > 0 || params.ScaleHeight > 0 {
		return nil, 0, 0, errNVJPEGUnsupported
	}

	nativeFormat, channels, err := nativeNVJPEGOutputFormat(params.OutputFormat)
	if err != nil {
		return nil, 0, 0, err
	}

	width, height, _, err := getNativeJpegImageInfoWithHandle(nvjpegHandleFromUintptr(decoder.nativeHandle), jpegData)
	if err != nil {
		return nil, 0, 0, err
	}

	output, err := memory.Alloc(int64(width * height * channels))
	if err != nil {
		return nil, 0, 0, fmt.Errorf("failed to allocate output memory: %v", err)
	}

	var image C.nvjpegImage_t
	C.prepareInterleavedImage(&image, (*C.uchar)(output.Ptr()), C.uint(width*channels))
	status := C.decodeJpegWrapper(
		nvjpegHandleFromUintptr(decoder.nativeHandle),
		nvjpegJpegStateFromUintptr(decoder.nativeState),
		(*C.uchar)(unsafe.Pointer(&jpegData[0])),
		C.size_t(len(jpegData)),
		C.int(nativeFormat),
		&image,
	)
	if status != C.NVJPEG_STATUS_SUCCESS {
		_ = output.Free()
		return nil, 0, 0, nvjpegError("nvjpegDecode", status)
	}

	return output, width, height, nil
}

func encodeNativeJpeg(encoder *JpegEncoderState, imageData *memory.Memory, width, height int, params JpegEncodeParams) ([]byte, error) {
	if imageData == nil {
		return nil, fmt.Errorf("image data cannot be nil")
	}
	if width <= 0 || height <= 0 {
		return nil, fmt.Errorf("invalid image dimensions: %dx%d", width, height)
	}

	nativeFormat, channels, err := nativeNVJPEGInputFormat(params.InputFormat)
	if err != nil {
		return nil, err
	}

	quality := params.Quality
	if quality == 0 {
		quality = encoder.quality
	}
	nativeParams := nvjpegEncoderParamsFromUintptr(encoder.nativeParams)
	if status := C.setEncoderQualityWrapper(nativeParams, C.int(quality)); status != C.NVJPEG_STATUS_SUCCESS {
		return nil, nvjpegError("nvjpegEncoderParamsSetQuality", status)
	}
	subsampling, err := nativeNVJPEGEncodeSampling(params.InputFormat)
	if err != nil {
		return nil, err
	}
	if status := C.setEncoderSamplingFactorsWrapper(nativeParams, subsampling); status != C.NVJPEG_STATUS_SUCCESS {
		return nil, nvjpegError("nvjpegEncoderParamsSetSamplingFactors", status)
	}

	var image C.nvjpegImage_t
	C.prepareInterleavedImage(&image, (*C.uchar)(imageData.Ptr()), C.uint(width*channels))
	if status := C.encodeImageWrapper(
		nvjpegHandleFromUintptr(encoder.nativeHandle),
		nvjpegEncoderStateFromUintptr(encoder.nativeState),
		nvjpegEncoderParamsFromUintptr(encoder.nativeParams),
		&image,
		C.int(nativeFormat),
		C.int(width),
		C.int(height),
	); status != C.NVJPEG_STATUS_SUCCESS {
		return nil, nvjpegError("nvjpegEncodeImage", status)
	}

	var length C.size_t
	if status := C.retrieveBitstreamLengthWrapper(nvjpegHandleFromUintptr(encoder.nativeHandle), nvjpegEncoderStateFromUintptr(encoder.nativeState), &length); status != C.NVJPEG_STATUS_SUCCESS {
		return nil, nvjpegError("nvjpegEncodeRetrieveBitstream(length)", status)
	}
	if length == 0 {
		return nil, fmt.Errorf("nvjpegEncodeRetrieveBitstream returned empty output")
	}

	encoded := make([]byte, int(length))
	if status := C.retrieveBitstreamWrapper(
		nvjpegHandleFromUintptr(encoder.nativeHandle),
		nvjpegEncoderStateFromUintptr(encoder.nativeState),
		(*C.uchar)(unsafe.Pointer(&encoded[0])),
		&length,
	); status != C.NVJPEG_STATUS_SUCCESS {
		return nil, nvjpegError("nvjpegEncodeRetrieveBitstream", status)
	}

	return encoded[:int(length)], nil
}

func getNativeJpegImageInfo(jpegData []byte) (int, int, int, error) {
	var handle C.nvjpegHandle_t
	if status := C.createNVJPEGHandle(&handle); status != C.NVJPEG_STATUS_SUCCESS {
		return 0, 0, 0, nvjpegError("nvjpegCreateSimple", status)
	}
	defer C.destroyNVJPEGHandle(handle)

	return getNativeJpegImageInfoWithHandle(handle, jpegData)
}

func getNativeJpegImageInfoWithHandle(handle C.nvjpegHandle_t, jpegData []byte) (int, int, int, error) {
	if len(jpegData) == 0 {
		return 0, 0, 0, fmt.Errorf("empty JPEG data")
	}

	var components C.int
	var subsampling C.nvjpegChromaSubsampling_t
	var widths [C.NVJPEG_MAX_COMPONENT]C.int
	var heights [C.NVJPEG_MAX_COMPONENT]C.int
	if status := C.getJpegInfoWrapper(
		handle,
		(*C.uchar)(unsafe.Pointer(&jpegData[0])),
		C.size_t(len(jpegData)),
		&components,
		&subsampling,
		(*C.int)(unsafe.Pointer(&widths[0])),
		(*C.int)(unsafe.Pointer(&heights[0])),
	); status != C.NVJPEG_STATUS_SUCCESS {
		return 0, 0, 0, nvjpegError("nvjpegGetImageInfo", status)
	}

	return int(widths[0]), int(heights[0]), int(components), nil
}

func destroyNativeJpegDecoder(decoder *JpegDecoderState) error {
	if decoder.nativeState != 0 {
		if status := C.destroyJpegStateWrapper(nvjpegJpegStateFromUintptr(decoder.nativeState)); status != C.NVJPEG_STATUS_SUCCESS {
			return nvjpegError("nvjpegJpegStateDestroy", status)
		}
		decoder.nativeState = 0
	}
	if decoder.nativeHandle != 0 {
		if status := C.destroyNVJPEGHandle(nvjpegHandleFromUintptr(decoder.nativeHandle)); status != C.NVJPEG_STATUS_SUCCESS {
			return nvjpegError("nvjpegDestroy", status)
		}
		decoder.nativeHandle = 0
	}
	decoder.native = false
	return nil
}

func destroyNativeJpegEncoder(encoder *JpegEncoderState) error {
	if encoder.nativeParams != 0 {
		if status := C.destroyEncoderParamsWrapper(nvjpegEncoderParamsFromUintptr(encoder.nativeParams)); status != C.NVJPEG_STATUS_SUCCESS {
			return nvjpegError("nvjpegEncoderParamsDestroy", status)
		}
		encoder.nativeParams = 0
	}
	if encoder.nativeState != 0 {
		if status := C.destroyEncoderStateWrapper(nvjpegEncoderStateFromUintptr(encoder.nativeState)); status != C.NVJPEG_STATUS_SUCCESS {
			return nvjpegError("nvjpegEncoderStateDestroy", status)
		}
		encoder.nativeState = 0
	}
	if encoder.nativeHandle != 0 {
		if status := C.destroyNVJPEGHandle(nvjpegHandleFromUintptr(encoder.nativeHandle)); status != C.NVJPEG_STATUS_SUCCESS {
			return nvjpegError("nvjpegDestroy", status)
		}
		encoder.nativeHandle = 0
	}
	encoder.native = false
	return nil
}

func nativeNVJPEGOutputFormat(format JpegFormat) (C.int, int, error) {
	switch format {
	case JpegFormatRGB:
		return C.NVJPEG_OUTPUT_RGBI, 3, nil
	case JpegFormatBGR:
		return C.NVJPEG_OUTPUT_BGRI, 3, nil
	case JpegFormatGrayscale:
		return C.NVJPEG_OUTPUT_Y, 1, nil
	default:
		return 0, 0, errNVJPEGUnsupported
	}
}

func nativeNVJPEGInputFormat(format JpegFormat) (C.int, int, error) {
	switch format {
	case JpegFormatRGB:
		return C.NVJPEG_INPUT_RGBI, 3, nil
	case JpegFormatBGR:
		return C.NVJPEG_INPUT_BGRI, 3, nil
	case JpegFormatGrayscale:
		return 0, 0, errNVJPEGUnsupported
	default:
		return 0, 0, errNVJPEGUnsupported
	}
}

func nativeNVJPEGEncoderBackend(backend JpegBackend) C.nvjpegEncBackend_t {
	if backend == JpegBackendHardware {
		return C.NVJPEG_ENC_BACKEND_HARDWARE
	}
	return C.NVJPEG_ENC_BACKEND_GPU
}

func nativeNVJPEGEncodeSampling(format JpegFormat) (C.nvjpegChromaSubsampling_t, error) {
	switch format {
	case JpegFormatRGB, JpegFormatBGR, JpegFormatRGBI, JpegFormatBGRI:
		return C.NVJPEG_CSS_444, nil
	default:
		return 0, errNVJPEGUnsupported
	}
}

func nvjpegError(operation string, status C.nvjpegStatus_t) error {
	return fmt.Errorf("%s failed: %s (%d)", operation, nvjpegStatusString(status), int(status))
}

func nvjpegStatusString(status C.nvjpegStatus_t) string {
	switch status {
	case C.NVJPEG_STATUS_SUCCESS:
		return "success"
	case C.NVJPEG_STATUS_NOT_INITIALIZED:
		return "not initialized"
	case C.NVJPEG_STATUS_INVALID_PARAMETER:
		return "invalid parameter"
	case C.NVJPEG_STATUS_BAD_JPEG:
		return "bad jpeg"
	case C.NVJPEG_STATUS_JPEG_NOT_SUPPORTED:
		return "jpeg not supported"
	case C.NVJPEG_STATUS_ALLOCATOR_FAILURE:
		return "allocator failure"
	case C.NVJPEG_STATUS_EXECUTION_FAILED:
		return "execution failed"
	case C.NVJPEG_STATUS_ARCH_MISMATCH:
		return "architecture mismatch"
	case C.NVJPEG_STATUS_INTERNAL_ERROR:
		return "internal error"
	case C.NVJPEG_STATUS_IMPLEMENTATION_NOT_SUPPORTED:
		return "implementation not supported"
	default:
		return "unknown error"
	}
}
