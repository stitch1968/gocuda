//go:build cuda

package libraries

/*
#cgo linux CFLAGS: -I/usr/local/cuda/include -I/opt/cuda/include
#cgo windows CFLAGS: -I"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.1/include" -I"D:/NVIDIA/include"
#cgo linux LDFLAGS: -L/usr/local/cuda/lib64 -L/opt/cuda/lib64 -lnvjpeg2k
#cgo windows LDFLAGS: -L${SRCDIR}/../lib_mingw -lnvjpeg2k

#include <cuda_runtime.h>
#include <nvjpeg2k.h>
#include <stdlib.h>
#include <string.h>

static nvjpeg2kStatus_t createNVJPEG2KHandleWrapper(nvjpeg2kHandle_t* handle) {
	return nvjpeg2kCreateSimple(handle);
}

static nvjpeg2kStatus_t destroyNVJPEG2KHandleWrapper(nvjpeg2kHandle_t handle) {
	return nvjpeg2kDestroy(handle);
}

static nvjpeg2kStatus_t createDecodeStateWrapper(nvjpeg2kHandle_t handle, nvjpeg2kDecodeState_t* state) {
	return nvjpeg2kDecodeStateCreate(handle, state);
}

static nvjpeg2kStatus_t destroyDecodeStateWrapper(nvjpeg2kDecodeState_t state) {
	return nvjpeg2kDecodeStateDestroy(state);
}

static nvjpeg2kStatus_t createStreamWrapper(nvjpeg2kStream_t* stream) {
	return nvjpeg2kStreamCreate(stream);
}

static nvjpeg2kStatus_t destroyStreamWrapper(nvjpeg2kStream_t stream) {
	return nvjpeg2kStreamDestroy(stream);
}

static nvjpeg2kStatus_t createDecodeParamsWrapper(nvjpeg2kDecodeParams_t* params) {
	return nvjpeg2kDecodeParamsCreate(params);
}

static nvjpeg2kStatus_t destroyDecodeParamsWrapper(nvjpeg2kDecodeParams_t params) {
	return nvjpeg2kDecodeParamsDestroy(params);
}

static nvjpeg2kStatus_t parseStreamWrapper(nvjpeg2kHandle_t handle, const unsigned char* data, size_t length, nvjpeg2kStream_t stream) {
	return nvjpeg2kStreamParse(handle, data, length, 0, 0, stream);
}

static nvjpeg2kStatus_t setDecodeAreaWrapper(nvjpeg2kDecodeParams_t params, unsigned int startX, unsigned int endX, unsigned int startY, unsigned int endY) {
	return nvjpeg2kDecodeParamsSetDecodeArea(params, startX, endX, startY, endY);
}

static nvjpeg2kStatus_t setDecodeRGBOutputWrapper(nvjpeg2kDecodeParams_t params, int enableRGB) {
	return nvjpeg2kDecodeParamsSetRGBOutput(params, enableRGB);
}

static nvjpeg2kStatus_t setDecodeOutputFormatWrapper(nvjpeg2kDecodeParams_t params, nvjpeg2kImageFormat_t format) {
	return nvjpeg2kDecodeParamsSetOutputFormat(params, format);
}

static nvjpeg2kStatus_t decodeImageWrapper(nvjpeg2kHandle_t handle, nvjpeg2kDecodeState_t state, nvjpeg2kStream_t stream, nvjpeg2kDecodeParams_t params, nvjpeg2kImage_t* output, cudaStream_t cudaStream) {
	return nvjpeg2kDecodeImage(handle, state, stream, params, output, cudaStream);
}

static nvjpeg2kStatus_t getImageInfoWrapper(nvjpeg2kStream_t stream, nvjpeg2kImageInfo_t* info) {
	return nvjpeg2kStreamGetImageInfo(stream, info);
}

static nvjpeg2kStatus_t getComponentInfoWrapper(nvjpeg2kStream_t stream, nvjpeg2kImageComponentInfo_t* info, unsigned int componentID) {
	return nvjpeg2kStreamGetImageComponentInfo(stream, info, componentID);
}

static nvjpeg2kStatus_t getResolutionsInTileWrapper(nvjpeg2kStream_t stream, unsigned int tileID, unsigned int* numRes) {
	return nvjpeg2kStreamGetResolutionsInTile(stream, tileID, numRes);
}

static nvjpeg2kStatus_t createEncoderHandleWrapper(nvjpeg2kEncoder_t* handle) {
	return nvjpeg2kEncoderCreateSimple(handle);
}

static nvjpeg2kStatus_t destroyEncoderHandleWrapper(nvjpeg2kEncoder_t handle) {
	return nvjpeg2kEncoderDestroy(handle);
}

static nvjpeg2kStatus_t createEncodeStateWrapper(nvjpeg2kEncoder_t handle, nvjpeg2kEncodeState_t* state) {
	return nvjpeg2kEncodeStateCreate(handle, state);
}

static nvjpeg2kStatus_t destroyEncodeStateWrapper(nvjpeg2kEncodeState_t state) {
	return nvjpeg2kEncodeStateDestroy(state);
}

static nvjpeg2kStatus_t createEncodeParamsWrapper(nvjpeg2kEncodeParams_t* params) {
	return nvjpeg2kEncodeParamsCreate(params);
}

static nvjpeg2kStatus_t destroyEncodeParamsWrapper(nvjpeg2kEncodeParams_t params) {
	return nvjpeg2kEncodeParamsDestroy(params);
}

static nvjpeg2kStatus_t setEncodeConfigWrapper(nvjpeg2kEncodeParams_t params, nvjpeg2kEncodeConfig_t* config) {
	return nvjpeg2kEncodeParamsSetEncodeConfig(params, config);
}

static nvjpeg2kStatus_t setEncodeQualityWrapper(nvjpeg2kEncodeParams_t params, nvjpeg2kQualityType qualityType, double value) {
	return nvjpeg2kEncodeParamsSpecifyQuality(params, qualityType, value);
}

static nvjpeg2kStatus_t setEncodeInputFormatWrapper(nvjpeg2kEncodeParams_t params, nvjpeg2kImageFormat_t format) {
	return nvjpeg2kEncodeParamsSetInputFormat(params, format);
}

static nvjpeg2kStatus_t encodeImageWrapper(nvjpeg2kEncoder_t handle, nvjpeg2kEncodeState_t state, const nvjpeg2kEncodeParams_t params, const nvjpeg2kImage_t* input, cudaStream_t stream) {
	return nvjpeg2kEncode(handle, state, params, input, stream);
}

static nvjpeg2kStatus_t retrieveBitstreamWrapper(nvjpeg2kEncoder_t handle, nvjpeg2kEncodeState_t state, unsigned char* data, size_t* length, cudaStream_t stream) {
	return nvjpeg2kEncodeRetrieveBitstream(handle, state, data, length, stream);
}

static void initImageWrapper(nvjpeg2kImage_t* image, void** pixelData, size_t* pitches, nvjpeg2kImageType_t pixelType, unsigned int components) {
	memset(image, 0, sizeof(nvjpeg2kImage_t));
	image->pixel_data = pixelData;
	image->pitch_in_bytes = pitches;
	image->pixel_type = pixelType;
	image->num_components = components;
}

static cudaError_t syncDefaultStreamWrapper() {
	return cudaStreamSynchronize((cudaStream_t)0);
}
*/
import "C"

import (
	"fmt"
	"unsafe"

	"github.com/stitch1968/gocuda/memory"
)

func nvjpeg2000NativeAvailable() bool {
	return true
}

func createNativeJpeg2000Decoder(codec Jpeg2000Codec) (*Jpeg2000DecoderState, error) {
	var handle C.nvjpeg2kHandle_t
	if status := C.createNVJPEG2KHandleWrapper(&handle); status != C.NVJPEG2K_STATUS_SUCCESS {
		return nil, nvjpeg2000Error("nvjpeg2kCreateSimple", status)
	}

	var state C.nvjpeg2kDecodeState_t
	if status := C.createDecodeStateWrapper(handle, &state); status != C.NVJPEG2K_STATUS_SUCCESS {
		_ = C.destroyNVJPEG2KHandleWrapper(handle)
		return nil, nvjpeg2000Error("nvjpeg2kDecodeStateCreate", status)
	}

	var stream C.nvjpeg2kStream_t
	if status := C.createStreamWrapper(&stream); status != C.NVJPEG2K_STATUS_SUCCESS {
		_ = C.destroyDecodeStateWrapper(state)
		_ = C.destroyNVJPEG2KHandleWrapper(handle)
		return nil, nvjpeg2000Error("nvjpeg2kStreamCreate", status)
	}

	var params C.nvjpeg2kDecodeParams_t
	if status := C.createDecodeParamsWrapper(&params); status != C.NVJPEG2K_STATUS_SUCCESS {
		_ = C.destroyStreamWrapper(stream)
		_ = C.destroyDecodeStateWrapper(state)
		_ = C.destroyNVJPEG2KHandleWrapper(handle)
		return nil, nvjpeg2000Error("nvjpeg2kDecodeParamsCreate", status)
	}

	return &Jpeg2000DecoderState{
		codec:        codec,
		nativeHandle: uintptr(handle),
		nativeState:  uintptr(state),
		nativeStream: uintptr(stream),
		nativeParams: uintptr(params),
		native:       true,
	}, nil
}

func createNativeJpeg2000Encoder(codec Jpeg2000Codec) (*Jpeg2000EncoderState, error) {
	if codec != Jpeg2000CodecJ2K && codec != Jpeg2000CodecJP2 {
		return nil, errNVJPEG2000Unsupported
	}

	var handle C.nvjpeg2kEncoder_t
	if status := C.createEncoderHandleWrapper(&handle); status != C.NVJPEG2K_STATUS_SUCCESS {
		return nil, nvjpeg2000Error("nvjpeg2kEncoderCreateSimple", status)
	}

	var state C.nvjpeg2kEncodeState_t
	if status := C.createEncodeStateWrapper(handle, &state); status != C.NVJPEG2K_STATUS_SUCCESS {
		_ = C.destroyEncoderHandleWrapper(handle)
		return nil, nvjpeg2000Error("nvjpeg2kEncodeStateCreate", status)
	}

	var params C.nvjpeg2kEncodeParams_t
	if status := C.createEncodeParamsWrapper(&params); status != C.NVJPEG2K_STATUS_SUCCESS {
		_ = C.destroyEncodeStateWrapper(state)
		_ = C.destroyEncoderHandleWrapper(handle)
		return nil, nvjpeg2000Error("nvjpeg2kEncodeParamsCreate", status)
	}

	return &Jpeg2000EncoderState{
		codec:            codec,
		compressionRatio: 10.0,
		numLayers:        1,
		numLevels:        5,
		nativeHandle:     uintptr(handle),
		nativeState:      uintptr(state),
		nativeParams:     uintptr(params),
		native:           true,
	}, nil
}

func decodeNativeJpeg2000(decoder *Jpeg2000DecoderState, j2kData []byte, params Jpeg2000DecodeParams) (*memory.Memory, int, int, error) {
	if len(j2kData) == 0 {
		return nil, 0, 0, fmt.Errorf("empty JPEG2000 data")
	}
	if params.DecodeLayer > 0 || params.DecodeLevel > 0 || params.ReduceFactor > 0 {
		return nil, 0, 0, errNVJPEG2000Unsupported
	}
	if (params.CropWidth > 0 || params.CropHeight > 0) && (params.CropWidth <= 0 || params.CropHeight <= 0) {
		return nil, 0, 0, errNVJPEG2000Unsupported
	}

	stream := C.nvjpeg2kStream_t(decoder.nativeStream)
	if status := C.parseStreamWrapper(
		C.nvjpeg2kHandle_t(decoder.nativeHandle),
		(*C.uchar)(unsafe.Pointer(&j2kData[0])),
		C.size_t(len(j2kData)),
		stream,
	); status != C.NVJPEG2K_STATUS_SUCCESS {
		return nil, 0, 0, nvjpeg2000Error("nvjpeg2kStreamParse", status)
	}

	info, componentInfo, numLevels, err := getNativeJpeg2000StreamInfo(stream, j2kData)
	if err != nil {
		return nil, 0, 0, err
	}
	if componentInfo.precision != 8 || componentInfo.sgn != 0 {
		return nil, 0, 0, errNVJPEG2000Unsupported
	}

	channels, enableRGB, err := nativeJpeg2000DecodeSupport(params.OutputFormat, int(info.num_components))
	if err != nil {
		return nil, 0, 0, err
	}

	width := int(info.image_width)
	height := int(info.image_height)
	pitch := width * channels
	output, err := memory.Alloc(int64(pitch * height))
	if err != nil {
		return nil, 0, 0, fmt.Errorf("failed to allocate output memory: %v", err)
	}

	pixelData := C.malloc(C.size_t(info.num_components) * C.size_t(unsafe.Sizeof(uintptr(0))))
	pitches := C.malloc(C.size_t(info.num_components) * C.size_t(unsafe.Sizeof(C.size_t(0))))
	if pixelData == nil || pitches == nil {
		if pixelData != nil {
			C.free(pixelData)
		}
		if pitches != nil {
			C.free(pitches)
		}
		_ = output.Free()
		return nil, 0, 0, fmt.Errorf("failed to allocate nvjpeg2k image descriptors")
	}
	defer C.free(pixelData)
	defer C.free(pitches)

	pixelArray := (*[1 << 20]unsafe.Pointer)(pixelData)[:int(info.num_components):int(info.num_components)]
	pitchArray := (*[1 << 20]C.size_t)(pitches)[:int(info.num_components):int(info.num_components)]
	pixelArray[0] = output.Ptr()
	pitchArray[0] = C.size_t(pitch)
	for index := 1; index < int(info.num_components); index++ {
		pixelArray[index] = nil
		pitchArray[index] = 0
	}

	decodeParams := C.nvjpeg2kDecodeParams_t(decoder.nativeParams)
	if status := C.setDecodeOutputFormatWrapper(decodeParams, C.NVJPEG2K_FORMAT_INTERLEAVED); status != C.NVJPEG2K_STATUS_SUCCESS {
		_ = output.Free()
		return nil, 0, 0, nvjpeg2000Error("nvjpeg2kDecodeParamsSetOutputFormat", status)
	}
	if status := C.setDecodeRGBOutputWrapper(decodeParams, C.int(enableRGB)); status != C.NVJPEG2K_STATUS_SUCCESS {
		_ = output.Free()
		return nil, 0, 0, nvjpeg2000Error("nvjpeg2kDecodeParamsSetRGBOutput", status)
	}
	if params.CropWidth > 0 && params.CropHeight > 0 {
		if status := C.setDecodeAreaWrapper(
			decodeParams,
			C.uint(params.CropX),
			C.uint(params.CropX+params.CropWidth),
			C.uint(params.CropY),
			C.uint(params.CropY+params.CropHeight),
		); status != C.NVJPEG2K_STATUS_SUCCESS {
			_ = output.Free()
			return nil, 0, 0, nvjpeg2000Error("nvjpeg2kDecodeParamsSetDecodeArea", status)
		}
	}

	var image C.nvjpeg2kImage_t
	C.initImageWrapper(&image, (**C.void)(pixelData), (*C.size_t)(pitches), C.NVJPEG2K_UINT8, C.uint(info.num_components))
	if status := C.decodeImageWrapper(
		C.nvjpeg2kHandle_t(decoder.nativeHandle),
		C.nvjpeg2kDecodeState_t(decoder.nativeState),
		stream,
		decodeParams,
		&image,
		(C.cudaStream_t)(nil),
	); status != C.NVJPEG2K_STATUS_SUCCESS {
		_ = output.Free()
		return nil, 0, 0, nvjpeg2000Error("nvjpeg2kDecodeImage", status)
	}
	if status := C.syncDefaultStreamWrapper(); status != C.cudaSuccess {
		_ = output.Free()
		return nil, 0, 0, fmt.Errorf("cudaStreamSynchronize failed with status %d", int(status))
	}

	_ = numLevels
	return output, width, height, nil
}

func encodeNativeJpeg2000(encoder *Jpeg2000EncoderState, imageData *memory.Memory, width, height int, params Jpeg2000EnodeParams) ([]byte, error) {
	if imageData == nil {
		return nil, fmt.Errorf("image data cannot be nil")
	}
	if width <= 0 || height <= 0 {
		return nil, fmt.Errorf("invalid image dimensions: %dx%d", width, height)
	}
	if params.NumLayers > 1 || len(params.PrecinctWidth) > 0 || len(params.PrecinctHeight) > 0 {
		return nil, errNVJPEG2000Unsupported
	}

	streamType, colorSpace, mctMode, components, err := nativeJpeg2000EncodeSupport(params.InputFormat, params.Codec)
	if err != nil {
		return nil, err
	}

	pixelData := C.malloc(C.size_t(components) * C.size_t(unsafe.Sizeof(uintptr(0))))
	pitches := C.malloc(C.size_t(components) * C.size_t(unsafe.Sizeof(C.size_t(0))))
	if pixelData == nil || pitches == nil {
		if pixelData != nil {
			C.free(pixelData)
		}
		if pitches != nil {
			C.free(pitches)
		}
		return nil, fmt.Errorf("failed to allocate nvjpeg2k encode descriptors")
	}
	defer C.free(pixelData)
	defer C.free(pitches)

	pixelArray := (*[1 << 20]unsafe.Pointer)(pixelData)[:components:components]
	pitchArray := (*[1 << 20]C.size_t)(pitches)[:components:components]
	pixelArray[0] = imageData.Ptr()
	pitchArray[0] = C.size_t(width * components)
	for index := 1; index < components; index++ {
		pixelArray[index] = nil
		pitchArray[index] = 0
	}

	componentInfo := make([]C.nvjpeg2kImageComponentInfo_t, components)
	for index := range componentInfo {
		componentInfo[index].component_width = C.uint(width)
		componentInfo[index].component_height = C.uint(height)
		componentInfo[index].precision = 8
		componentInfo[index].sgn = 0
	}

	resolutionCount := nativeJpeg2000ResolutionCount(width, height, params.NumLevels)
	codeblockSize := nativeJpeg2000CodeblockSize(params.CodeblockWidth)

	var config C.nvjpeg2kEncodeConfig_t
	config.stream_type = streamType
	config.color_space = colorSpace
	config.rsiz = 0
	config.image_width = C.uint(width)
	config.image_height = C.uint(height)
	config.enable_tiling = 0
	config.tile_width = 0
	config.tile_height = 0
	config.num_components = C.uint(components)
	config.image_comp_info = (*C.nvjpeg2kImageComponentInfo_t)(unsafe.Pointer(&componentInfo[0]))
	config.enable_SOP_marker = 0
	config.enable_EPH_marker = 0
	config.prog_order = nativeJpeg2000ProgressionOrder(params.ProgressionOrder)
	config.num_layers = 1
	config.mct_mode = C.uint(mctMode)
	config.num_resolutions = C.uint(resolutionCount)
	config.code_block_w = C.uint(codeblockSize)
	config.code_block_h = C.uint(codeblockSize)
	config.encode_modes = 0
	if params.Lossless {
		config.irreversible = 0
	} else {
		config.irreversible = 1
	}
	config.enable_custom_precincts = 0
	config.num_precincts_init = 0

	encodeParams := C.nvjpeg2kEncodeParams_t(encoder.nativeParams)
	if status := C.setEncodeConfigWrapper(encodeParams, &config); status != C.NVJPEG2K_STATUS_SUCCESS {
		return nil, nvjpeg2000Error("nvjpeg2kEncodeParamsSetEncodeConfig", status)
	}
	if status := C.setEncodeInputFormatWrapper(encodeParams, C.NVJPEG2K_FORMAT_INTERLEAVED); status != C.NVJPEG2K_STATUS_SUCCESS {
		return nil, nvjpeg2000Error("nvjpeg2kEncodeParamsSetInputFormat", status)
	}
	if !params.Lossless {
		quality := nativeJpeg2000Quality(params.CompressionRatio)
		if status := C.setEncodeQualityWrapper(encodeParams, C.NVJPEG2K_QUALITY_TYPE_Q_FACTOR, C.double(quality)); status != C.NVJPEG2K_STATUS_SUCCESS {
			return nil, nvjpeg2000Error("nvjpeg2kEncodeParamsSpecifyQuality", status)
		}
	}

	var image C.nvjpeg2kImage_t
	C.initImageWrapper(&image, (**C.void)(pixelData), (*C.size_t)(pitches), C.NVJPEG2K_UINT8, C.uint(components))
	if status := C.encodeImageWrapper(
		C.nvjpeg2kEncoder_t(encoder.nativeHandle),
		C.nvjpeg2kEncodeState_t(encoder.nativeState),
		encodeParams,
		&image,
		(C.cudaStream_t)(nil),
	); status != C.NVJPEG2K_STATUS_SUCCESS {
		return nil, nvjpeg2000Error("nvjpeg2kEncode", status)
	}

	var length C.size_t
	if status := C.retrieveBitstreamWrapper(C.nvjpeg2kEncoder_t(encoder.nativeHandle), C.nvjpeg2kEncodeState_t(encoder.nativeState), nil, &length, (C.cudaStream_t)(nil)); status != C.NVJPEG2K_STATUS_SUCCESS {
		return nil, nvjpeg2000Error("nvjpeg2kEncodeRetrieveBitstream(length)", status)
	}
	if length == 0 {
		return nil, fmt.Errorf("nvjpeg2kEncodeRetrieveBitstream returned empty output")
	}

	encoded := make([]byte, int(length))
	if status := C.retrieveBitstreamWrapper(
		C.nvjpeg2kEncoder_t(encoder.nativeHandle),
		C.nvjpeg2kEncodeState_t(encoder.nativeState),
		(*C.uchar)(unsafe.Pointer(&encoded[0])),
		&length,
		(C.cudaStream_t)(nil),
	); status != C.NVJPEG2K_STATUS_SUCCESS {
		return nil, nvjpeg2000Error("nvjpeg2kEncodeRetrieveBitstream", status)
	}
	if status := C.syncDefaultStreamWrapper(); status != C.cudaSuccess {
		return nil, fmt.Errorf("cudaStreamSynchronize failed with status %d", int(status))
	}

	return encoded[:int(length)], nil
}

func getNativeJpeg2000ImageInfo(j2kData []byte) (*Jpeg2000ImageInfo, error) {
	if len(j2kData) == 0 {
		return nil, fmt.Errorf("empty JPEG2000 data")
	}

	var handle C.nvjpeg2kHandle_t
	if status := C.createNVJPEG2KHandleWrapper(&handle); status != C.NVJPEG2K_STATUS_SUCCESS {
		return nil, nvjpeg2000Error("nvjpeg2kCreateSimple", status)
	}
	defer C.destroyNVJPEG2KHandleWrapper(handle)

	var stream C.nvjpeg2kStream_t
	if status := C.createStreamWrapper(&stream); status != C.NVJPEG2K_STATUS_SUCCESS {
		return nil, nvjpeg2000Error("nvjpeg2kStreamCreate", status)
	}
	defer C.destroyStreamWrapper(stream)

	if status := C.parseStreamWrapper(handle, (*C.uchar)(unsafe.Pointer(&j2kData[0])), C.size_t(len(j2kData)), stream); status != C.NVJPEG2K_STATUS_SUCCESS {
		return nil, nvjpeg2000Error("nvjpeg2kStreamParse", status)
	}

	info, componentInfo, numLevels, err := getNativeJpeg2000StreamInfo(stream, j2kData)
	if err != nil {
		return nil, err
	}

	return &Jpeg2000ImageInfo{
		Width:      int(info.image_width),
		Height:     int(info.image_height),
		Components: int(info.num_components),
		BitDepth:   int(componentInfo.precision),
		NumLayers:  1,
		NumLevels:  numLevels,
		Codec:      codecFromBytes(j2kData),
	}, nil
}

func destroyNativeJpeg2000Decoder(decoder *Jpeg2000DecoderState) error {
	if decoder.nativeParams != 0 {
		if status := C.destroyDecodeParamsWrapper(C.nvjpeg2kDecodeParams_t(decoder.nativeParams)); status != C.NVJPEG2K_STATUS_SUCCESS {
			return nvjpeg2000Error("nvjpeg2kDecodeParamsDestroy", status)
		}
		decoder.nativeParams = 0
	}
	if decoder.nativeStream != 0 {
		if status := C.destroyStreamWrapper(C.nvjpeg2kStream_t(decoder.nativeStream)); status != C.NVJPEG2K_STATUS_SUCCESS {
			return nvjpeg2000Error("nvjpeg2kStreamDestroy", status)
		}
		decoder.nativeStream = 0
	}
	if decoder.nativeState != 0 {
		if status := C.destroyDecodeStateWrapper(C.nvjpeg2kDecodeState_t(decoder.nativeState)); status != C.NVJPEG2K_STATUS_SUCCESS {
			return nvjpeg2000Error("nvjpeg2kDecodeStateDestroy", status)
		}
		decoder.nativeState = 0
	}
	if decoder.nativeHandle != 0 {
		if status := C.destroyNVJPEG2KHandleWrapper(C.nvjpeg2kHandle_t(decoder.nativeHandle)); status != C.NVJPEG2K_STATUS_SUCCESS {
			return nvjpeg2000Error("nvjpeg2kDestroy", status)
		}
		decoder.nativeHandle = 0
	}
	decoder.native = false
	return nil
}

func destroyNativeJpeg2000Encoder(encoder *Jpeg2000EncoderState) error {
	if encoder.nativeParams != 0 {
		if status := C.destroyEncodeParamsWrapper(C.nvjpeg2kEncodeParams_t(encoder.nativeParams)); status != C.NVJPEG2K_STATUS_SUCCESS {
			return nvjpeg2000Error("nvjpeg2kEncodeParamsDestroy", status)
		}
		encoder.nativeParams = 0
	}
	if encoder.nativeState != 0 {
		if status := C.destroyEncodeStateWrapper(C.nvjpeg2kEncodeState_t(encoder.nativeState)); status != C.NVJPEG2K_STATUS_SUCCESS {
			return nvjpeg2000Error("nvjpeg2kEncodeStateDestroy", status)
		}
		encoder.nativeState = 0
	}
	if encoder.nativeHandle != 0 {
		if status := C.destroyEncoderHandleWrapper(C.nvjpeg2kEncoder_t(encoder.nativeHandle)); status != C.NVJPEG2K_STATUS_SUCCESS {
			return nvjpeg2000Error("nvjpeg2kEncoderDestroy", status)
		}
		encoder.nativeHandle = 0
	}
	encoder.native = false
	return nil
}

func getNativeJpeg2000StreamInfo(stream C.nvjpeg2kStream_t, j2kData []byte) (C.nvjpeg2kImageInfo_t, C.nvjpeg2kImageComponentInfo_t, int, error) {
	var imageInfo C.nvjpeg2kImageInfo_t
	if status := C.getImageInfoWrapper(stream, &imageInfo); status != C.NVJPEG2K_STATUS_SUCCESS {
		return imageInfo, C.nvjpeg2kImageComponentInfo_t{}, 0, nvjpeg2000Error("nvjpeg2kStreamGetImageInfo", status)
	}

	var componentInfo C.nvjpeg2kImageComponentInfo_t
	if status := C.getComponentInfoWrapper(stream, &componentInfo, 0); status != C.NVJPEG2K_STATUS_SUCCESS {
		return imageInfo, componentInfo, 0, nvjpeg2000Error("nvjpeg2kStreamGetImageComponentInfo", status)
	}

	var numRes C.uint
	if status := C.getResolutionsInTileWrapper(stream, 0, &numRes); status != C.NVJPEG2K_STATUS_SUCCESS {
		return imageInfo, componentInfo, 0, nvjpeg2000Error("nvjpeg2kStreamGetResolutionsInTile", status)
	}
	if numRes == 0 {
		numRes = 1
	}

	return imageInfo, componentInfo, int(numRes) - 1, nil
}

func nativeJpeg2000DecodeSupport(format Jpeg2000Format, components int) (channels int, enableRGB int, err error) {
	switch format {
	case Jpeg2000FormatGrayscale:
		if components != 1 {
			return 0, 0, errNVJPEG2000Unsupported
		}
		return 1, 0, nil
	case Jpeg2000FormatRGB:
		if components != 3 {
			return 0, 0, errNVJPEG2000Unsupported
		}
		return 3, 1, nil
	default:
		return 0, 0, errNVJPEG2000Unsupported
	}
}

func nativeJpeg2000EncodeSupport(format Jpeg2000Format, codec Jpeg2000Codec) (C.nvjpeg2kBitstreamType_t, C.nvjpeg2kColorSpace_t, int, int, error) {
	streamType, err := nativeJpeg2000BitstreamType(codec)
	if err != nil {
		return 0, 0, 0, 0, err
	}

	switch format {
	case Jpeg2000FormatGrayscale:
		return streamType, C.NVJPEG2K_COLORSPACE_GRAY, 0, 1, nil
	case Jpeg2000FormatRGB:
		return streamType, C.NVJPEG2K_COLORSPACE_SRGB, 1, 3, nil
	default:
		return 0, 0, 0, 0, errNVJPEG2000Unsupported
	}
}

func nativeJpeg2000BitstreamType(codec Jpeg2000Codec) (C.nvjpeg2kBitstreamType_t, error) {
	switch codec {
	case Jpeg2000CodecJ2K:
		return C.NVJPEG2K_STREAM_J2K, nil
	case Jpeg2000CodecJP2:
		return C.NVJPEG2K_STREAM_JP2, nil
	default:
		return 0, errNVJPEG2000Unsupported
	}
}

func nativeJpeg2000ProgressionOrder(order Jpeg2000ProgressionOrder) C.nvjpeg2kProgOrder_t {
	switch order {
	case Jpeg2000ProgressionRLCP:
		return C.NVJPEG2K_RLCP
	case Jpeg2000ProgressionRPCL:
		return C.NVJPEG2K_RPCL
	case Jpeg2000ProgressionPCRL:
		return C.NVJPEG2K_PCRL
	case Jpeg2000ProgressionCPRL:
		return C.NVJPEG2K_CPRL
	default:
		return C.NVJPEG2K_LRCP
	}
}

func nativeJpeg2000ResolutionCount(width, height, numLevels int) int {
	maxDim := width
	if height < maxDim {
		maxDim = height
	}
	maxRes := 1
	for maxDim > 1 {
		maxRes++
		maxDim = (maxDim + 1) / 2
	}
	requested := numLevels + 1
	if requested <= 0 {
		requested = 1
	}
	if requested > maxRes {
		return maxRes
	}
	return requested
}

func nativeJpeg2000CodeblockSize(size int) int {
	if size == 64 {
		return 64
	}
	return 32
}

func nativeJpeg2000Quality(compressionRatio float32) float64 {
	if compressionRatio <= 0 {
		return 90
	}
	quality := 100.0 / float64(compressionRatio)
	if quality < 1 {
		return 1
	}
	if quality > 100 {
		return 100
	}
	return quality
}

func nvjpeg2000Error(name string, status C.nvjpeg2kStatus_t) error {
	return fmt.Errorf("%s failed: %s (%d)", name, nvjpeg2000StatusString(status), int(status))
}

func nvjpeg2000StatusString(status C.nvjpeg2kStatus_t) string {
	switch status {
	case C.NVJPEG2K_STATUS_SUCCESS:
		return "success"
	case C.NVJPEG2K_STATUS_NOT_INITIALIZED:
		return "not initialized"
	case C.NVJPEG2K_STATUS_INVALID_PARAMETER:
		return "invalid parameter"
	case C.NVJPEG2K_STATUS_BAD_JPEG:
		return "bad jpeg2000 bitstream"
	case C.NVJPEG2K_STATUS_JPEG_NOT_SUPPORTED:
		return "jpeg2000 bitstream not supported"
	case C.NVJPEG2K_STATUS_ALLOCATOR_FAILURE:
		return "allocator failure"
	case C.NVJPEG2K_STATUS_EXECUTION_FAILED:
		return "execution failed"
	case C.NVJPEG2K_STATUS_ARCH_MISMATCH:
		return "architecture mismatch"
	case C.NVJPEG2K_STATUS_INTERNAL_ERROR:
		return "internal error"
	case C.NVJPEG2K_STATUS_IMPLEMENTATION_NOT_SUPPORTED:
		return "implementation not supported"
	default:
		return "unknown error"
	}
}
