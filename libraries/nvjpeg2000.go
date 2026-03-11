// Package libraries provides CUDA runtime library bindings for GoCUDA
// This implements nvJPEG2000 functionality for JPEG2000 encoding/decoding
package libraries

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"

	cuda "github.com/stitch1968/gocuda"
	"github.com/stitch1968/gocuda/memory"
)

// nvJPEG2000 - High-Performance JPEG2000 Encoder/Decoder Library

// Jpeg2000Format formats and configurations
type Jpeg2000Format int

const (
	Jpeg2000FormatRGB Jpeg2000Format = iota
	Jpeg2000FormatBGR
	Jpeg2000FormatRGBA
	Jpeg2000FormatBGRA
	Jpeg2000FormatGrayscale
	Jpeg2000FormatYUV420
	Jpeg2000FormatYUV422
	Jpeg2000FormatYUV444
)

// Jpeg2000Codec codec types
type Jpeg2000Codec int

const (
	Jpeg2000CodecJ2K Jpeg2000Codec = iota // Raw JPEG2000 codestream
	Jpeg2000CodecJP2                      // JPEG2000 with JP2 container
	Jpeg2000CodecJPT                      // JPEG2000 with JPT container
	Jpeg2000CodecJPX                      // JPEG2000 with JPX container
)

// Jpeg2000ProgressionOrder progression orders
type Jpeg2000ProgressionOrder int

const (
	Jpeg2000ProgressionLRCP Jpeg2000ProgressionOrder = iota // Layer-Resolution-Component-Position
	Jpeg2000ProgressionRLCP                                 // Resolution-Layer-Component-Position
	Jpeg2000ProgressionRPCL                                 // Resolution-Position-Component-Layer
	Jpeg2000ProgressionPCRL                                 // Position-Component-Resolution-Layer
	Jpeg2000ProgressionCPRL                                 // Component-Position-Resolution-Layer
)

// Jpeg2000DecoderState decoder state
type Jpeg2000DecoderState struct {
	handle       *memory.Memory
	stream       *memory.Memory
	codec        Jpeg2000Codec
	nativeHandle uintptr
	nativeState  uintptr
	nativeStream uintptr
	nativeParams uintptr
	native       bool
}

// Jpeg2000EncoderState encoder state
type Jpeg2000EncoderState struct {
	handle           *memory.Memory
	stream           *memory.Memory
	codec            Jpeg2000Codec
	compressionRatio float32
	numLayers        int
	numLevels        int
	nativeHandle     uintptr
	nativeState      uintptr
	nativeParams     uintptr
	native           bool
}

var errNVJPEG2000Unsupported = errors.New("nvjpeg2000 native path unsupported for requested parameters")

// Jpeg2000DecodeParams decode parameters
type Jpeg2000DecodeParams struct {
	OutputFormat Jpeg2000Format
	Codec        Jpeg2000Codec
	DecodeLayer  int // Decode up to this layer (-1 for all)
	DecodeLevel  int // Decode up to this resolution level (-1 for all)
	CropX        int
	CropY        int
	CropWidth    int
	CropHeight   int
	ReduceFactor int // Reduce image by 2^reduce_factor
}

// Jpeg2000EnodeParams encode parameters
type Jpeg2000EnodeParams struct {
	InputFormat      Jpeg2000Format
	Codec            Jpeg2000Codec
	CompressionRatio float32
	Lossless         bool
	NumLayers        int
	NumLevels        int
	ProgressionOrder Jpeg2000ProgressionOrder
	CodeblockWidth   int
	CodeblockHeight  int
	PrecinctWidth    []int
	PrecinctHeight   []int
}

// Jpeg2000ImageInfo image information
type Jpeg2000ImageInfo struct {
	Width      int
	Height     int
	Components int
	BitDepth   int
	NumLayers  int
	NumLevels  int
	Codec      Jpeg2000Codec
}

// CreateJpeg2000Decoder creates a new JPEG2000 decoder
func CreateJpeg2000Decoder(codec Jpeg2000Codec) (*Jpeg2000DecoderState, error) {
	if err := ensureCudaReady(); err != nil {
		return nil, err
	}
	if cuda.ShouldUseCuda() && nvjpeg2000NativeAvailable() {
		decoder, err := createNativeJpeg2000Decoder(codec)
		if err == nil {
			return decoder, nil
		}
	}
	return createDeterministicJpeg2000Decoder(codec)
}

func createDeterministicJpeg2000Decoder(codec Jpeg2000Codec) (*Jpeg2000DecoderState, error) {
	if err := requireJpeg2000Tooling(); err != nil {
		return nil, err
	}

	decoder := &Jpeg2000DecoderState{
		codec: codec,
	}

	// Allocate decoder handle (simulated)
	var err error
	decoder.handle, err = memory.Alloc(4096) // JPEG2000 decoder needs more state memory
	if err != nil {
		return nil, fmt.Errorf("failed to allocate decoder handle: %v", err)
	}

	// Allocate stream memory
	decoder.stream, err = memory.Alloc(2048)
	if err != nil {
		decoder.handle.Free()
		return nil, fmt.Errorf("failed to allocate stream memory: %v", err)
	}

	return decoder, nil
}

// CreateJpeg2000Encoder creates a new JPEG2000 encoder
func CreateJpeg2000Encoder(codec Jpeg2000Codec) (*Jpeg2000EncoderState, error) {
	if err := ensureCudaReady(); err != nil {
		return nil, err
	}
	if cuda.ShouldUseCuda() && nvjpeg2000NativeAvailable() {
		encoder, err := createNativeJpeg2000Encoder(codec)
		if err == nil {
			return encoder, nil
		}
	}
	return createDeterministicJpeg2000Encoder(codec)
}

func createDeterministicJpeg2000Encoder(codec Jpeg2000Codec) (*Jpeg2000EncoderState, error) {
	if err := requireJpeg2000Tooling(); err != nil {
		return nil, err
	}

	encoder := &Jpeg2000EncoderState{
		codec:            codec,
		compressionRatio: 10.0, // Default 10:1 compression
		numLayers:        1,    // Default single layer
		numLevels:        5,    // Default 5 decomposition levels
	}

	// Allocate encoder handle (simulated)
	var err error
	encoder.handle, err = memory.Alloc(4096) // JPEG2000 encoder needs more state memory
	if err != nil {
		return nil, fmt.Errorf("failed to allocate encoder handle: %v", err)
	}

	// Allocate stream memory
	encoder.stream, err = memory.Alloc(2048)
	if err != nil {
		encoder.handle.Free()
		return nil, fmt.Errorf("failed to allocate stream memory: %v", err)
	}

	return encoder, nil
}

// DecodeJpeg2000 decodes a JPEG2000 image from byte data
func (decoder *Jpeg2000DecoderState) DecodeJpeg2000(j2kData []byte, params Jpeg2000DecodeParams) (*memory.Memory, int, int, error) {
	if decoder != nil && decoder.native {
		output, width, height, err := decodeNativeJpeg2000(decoder, j2kData, params)
		if err == nil {
			return output, width, height, nil
		}
		if !errors.Is(err, errNVJPEG2000Unsupported) {
			return nil, 0, 0, err
		}
	}
	return decodeJpeg2000Deterministic(j2kData, params)
}

func decodeJpeg2000Deterministic(j2kData []byte, params Jpeg2000DecodeParams) (*memory.Memory, int, int, error) {
	if len(j2kData) == 0 {
		return nil, 0, 0, fmt.Errorf("empty JPEG2000 data")
	}

	info, err := parseJpeg2000Header(j2kData)
	if err != nil {
		return nil, 0, 0, fmt.Errorf("failed to parse JPEG2000 header: %v", err)
	}

	width := info.Width
	height := info.Height

	// Apply reduce factor
	if params.ReduceFactor > 0 {
		for i := 0; i < params.ReduceFactor; i++ {
			width /= 2
			height /= 2
		}
	}

	// Apply cropping if specified
	if params.CropWidth > 0 && params.CropHeight > 0 {
		width = params.CropWidth
		height = params.CropHeight
	}

	pixelFormat, err := jpeg2000PixelFormat(params.OutputFormat)
	if err != nil {
		return nil, 0, 0, err
	}
	rawData, err := runJpeg2000Decode(j2kData, width, height, pixelFormat, params)
	if err != nil {
		return nil, 0, 0, err
	}
	outputMem, err := memory.Alloc(int64(len(rawData)))
	if err != nil {
		return nil, 0, 0, fmt.Errorf("failed to allocate output memory: %v", err)
	}
	if err := memory.CopyHostToDevice(outputMem, rawData); err != nil {
		outputMem.Free()
		return nil, 0, 0, err
	}

	return outputMem, width, height, nil
}

// DecodeJpeg2000Batch decodes multiple JPEG2000 images in batch
func (decoder *Jpeg2000DecoderState) DecodeJpeg2000Batch(j2kDataList [][]byte, params Jpeg2000DecodeParams) ([]*memory.Memory, []int, []int, error) {
	if len(j2kDataList) == 0 {
		return nil, nil, nil, fmt.Errorf("empty JPEG2000 data list")
	}

	outputs := make([]*memory.Memory, len(j2kDataList))
	widths := make([]int, len(j2kDataList))
	heights := make([]int, len(j2kDataList))

	for i, j2kData := range j2kDataList {
		output, width, height, err := decoder.DecodeJpeg2000(j2kData, params)
		if err != nil {
			// Clean up previously allocated memory
			for j := range i {
				if outputs[j] != nil {
					outputs[j].Free()
				}
			}
			return nil, nil, nil, fmt.Errorf("failed to decode JPEG2000 %d: %v", i, err)
		}
		outputs[i] = output
		widths[i] = width
		heights[i] = height
	}

	return outputs, widths, heights, nil
}

// EncodeJpeg2000 encodes image data to JPEG2000 format
func (encoder *Jpeg2000EncoderState) EncodeJpeg2000(imageData *memory.Memory, width, height int, params Jpeg2000EnodeParams) ([]byte, error) {
	if encoder != nil && encoder.native {
		encoded, err := encodeNativeJpeg2000(encoder, imageData, width, height, params)
		if err == nil {
			return encoded, nil
		}
		if !errors.Is(err, errNVJPEG2000Unsupported) {
			return nil, err
		}
	}
	return encodeJpeg2000Deterministic(imageData, width, height, params)
}

func encodeJpeg2000Deterministic(imageData *memory.Memory, width, height int, params Jpeg2000EnodeParams) ([]byte, error) {
	if imageData == nil {
		return nil, fmt.Errorf("image data cannot be nil")
	}
	if width <= 0 || height <= 0 {
		return nil, fmt.Errorf("invalid image dimensions: %dx%d", width, height)
	}

	pixelFormat, err := jpeg2000PixelFormat(params.InputFormat)
	if err != nil {
		return nil, err
	}
	expectedSize := jpeg2000RawImageSize(width, height, params.InputFormat)
	hostData := make([]byte, expectedSize)
	if err := memory.CopyDeviceToHost(hostData, imageData); err != nil {
		return nil, err
	}
	return runJpeg2000Encode(hostData, width, height, pixelFormat, params)
}

// SetCompressionRatio sets the JPEG2000 compression ratio
func (encoder *Jpeg2000EncoderState) SetCompressionRatio(ratio float32) error {
	if ratio <= 0 {
		return fmt.Errorf("compression ratio must be positive, got %f", ratio)
	}
	encoder.compressionRatio = ratio
	return nil
}

// SetNumLayers sets the number of quality layers
func (encoder *Jpeg2000EncoderState) SetNumLayers(layers int) error {
	if layers <= 0 {
		return fmt.Errorf("number of layers must be positive, got %d", layers)
	}
	encoder.numLayers = layers
	return nil
}

// SetNumLevels sets the number of wavelet decomposition levels
func (encoder *Jpeg2000EncoderState) SetNumLevels(levels int) error {
	if levels < 0 || levels > 10 {
		return fmt.Errorf("number of levels must be between 0 and 10, got %d", levels)
	}
	encoder.numLevels = levels
	return nil
}

// GetJpeg2000ImageInfo extracts detailed information from JPEG2000 data
func GetJpeg2000ImageInfo(j2kData []byte) (*Jpeg2000ImageInfo, error) {
	if len(j2kData) == 0 {
		return nil, fmt.Errorf("empty JPEG2000 data")
	}
	if cuda.ShouldUseCuda() && nvjpeg2000NativeAvailable() {
		info, err := getNativeJpeg2000ImageInfo(j2kData)
		if err == nil {
			return info, nil
		}
	}

	return parseJpeg2000Header(j2kData)
}

// parseJpeg2000Header is a helper function to parse JPEG2000 headers
func parseJpeg2000Header(j2kData []byte) (*Jpeg2000ImageInfo, error) {
	if len(j2kData) < 12 {
		return nil, fmt.Errorf("invalid JPEG2000 data: too short")
	}
	if err := requireJpeg2000Tooling(); err != nil {
		return nil, err
	}
	return runJpeg2000Probe(j2kData)
}

type jpeg2000ProbeResponse struct {
	Streams []struct {
		Width            int    `json:"width"`
		Height           int    `json:"height"`
		PixFmt           string `json:"pix_fmt"`
		BitsPerRawSample string `json:"bits_per_raw_sample"`
	} `json:"streams"`
}

func requireJpeg2000Tooling() error {
	if _, err := exec.LookPath("ffmpeg"); err != nil {
		return fmt.Errorf("ffmpeg is required for nvJPEG2000 operations: %w", err)
	}
	if _, err := exec.LookPath("ffprobe"); err != nil {
		return fmt.Errorf("ffprobe is required for nvJPEG2000 operations: %w", err)
	}
	return nil
}

func runJpeg2000Probe(j2kData []byte) (*Jpeg2000ImageInfo, error) {
	inputPath, cleanup, err := writeJpeg2000TempFile(j2kData, codecExtensionFromBytes(j2kData))
	if err != nil {
		return nil, err
	}
	defer cleanup()
	cmd := exec.Command("ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height,pix_fmt,bits_per_raw_sample", "-of", "json", inputPath)
	output, err := cmd.Output()
	if err != nil {
		return nil, fmt.Errorf("ffprobe failed: %w", err)
	}
	var response jpeg2000ProbeResponse
	if err := json.Unmarshal(output, &response); err != nil {
		return nil, err
	}
	if len(response.Streams) == 0 {
		return nil, fmt.Errorf("ffprobe returned no video streams")
	}
	stream := response.Streams[0]
	bitDepth := 8
	if stream.BitsPerRawSample == "16" {
		bitDepth = 16
	}
	return &Jpeg2000ImageInfo{
		Width:      stream.Width,
		Height:     stream.Height,
		Components: componentsForPixelFormat(stream.PixFmt),
		BitDepth:   bitDepth,
		NumLayers:  1,
		NumLevels:  0,
		Codec:      codecFromBytes(j2kData),
	}, nil
}

func runJpeg2000Decode(j2kData []byte, width, height int, pixelFormat string, params Jpeg2000DecodeParams) ([]byte, error) {
	inputPath, cleanup, err := writeJpeg2000TempFile(j2kData, codecExtensionFromBytes(j2kData))
	if err != nil {
		return nil, err
	}
	defer cleanup()
	args := []string{"-v", "error", "-i", inputPath}
	filter := jpeg2000Filter(params, width, height)
	if filter != "" {
		args = append(args, "-vf", filter)
	}
	args = append(args, "-frames:v", "1", "-pix_fmt", pixelFormat, "-f", "rawvideo", "pipe:1")
	cmd := exec.Command("ffmpeg", args...)
	var stdout bytes.Buffer
	var stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	if err := cmd.Run(); err != nil {
		return nil, fmt.Errorf("ffmpeg decode failed: %s: %w", stderr.String(), err)
	}
	return stdout.Bytes(), nil
}

func runJpeg2000Encode(rawData []byte, width, height int, pixelFormat string, params Jpeg2000EnodeParams) ([]byte, error) {
	inputFile, err := os.CreateTemp("", "gocuda-j2k-input-*.raw")
	if err != nil {
		return nil, err
	}
	inputPath := inputFile.Name()
	defer os.Remove(inputPath)
	defer inputFile.Close()
	if _, err := inputFile.Write(rawData); err != nil {
		return nil, err
	}
	outputFile, err := os.CreateTemp("", "gocuda-j2k-output-*"+codecExtension(params.Codec))
	if err != nil {
		return nil, err
	}
	outputPath := outputFile.Name()
	outputFile.Close()
	defer os.Remove(outputPath)
	args := []string{"-v", "error", "-y", "-f", "rawvideo", "-pixel_format", pixelFormat, "-video_size", fmt.Sprintf("%dx%d", width, height), "-i", inputPath, "-frames:v", "1", "-c:v", "jpeg2000"}
	if params.Lossless {
		args = append(args, "-pred", "1")
	}
	if params.CompressionRatio > 0 {
		args = append(args, "-q:v", fmt.Sprintf("%.2f", params.CompressionRatio))
	}
	args = append(args, outputPath)
	cmd := exec.Command("ffmpeg", args...)
	var stderr bytes.Buffer
	cmd.Stderr = &stderr
	if err := cmd.Run(); err != nil {
		return nil, fmt.Errorf("ffmpeg encode failed: %s: %w", stderr.String(), err)
	}
	return os.ReadFile(outputPath)
}

func writeJpeg2000TempFile(data []byte, extension string) (string, func(), error) {
	file, err := os.CreateTemp("", "gocuda-j2k-*"+extension)
	if err != nil {
		return "", nil, err
	}
	if _, err := file.Write(data); err != nil {
		file.Close()
		os.Remove(file.Name())
		return "", nil, err
	}
	if err := file.Close(); err != nil {
		os.Remove(file.Name())
		return "", nil, err
	}
	return file.Name(), func() { _ = os.Remove(file.Name()) }, nil
}

func jpeg2000Filter(params Jpeg2000DecodeParams, width, height int) string {
	filters := make([]string, 0, 2)
	if params.ReduceFactor > 0 {
		scaleDivisor := 1 << params.ReduceFactor
		filters = append(filters, fmt.Sprintf("scale=%d:%d", maxInt(width/scaleDivisor, 1), maxInt(height/scaleDivisor, 1)))
	}
	if params.CropWidth > 0 && params.CropHeight > 0 {
		filters = append(filters, fmt.Sprintf("crop=%d:%d:%d:%d", params.CropWidth, params.CropHeight, params.CropX, params.CropY))
	}
	if len(filters) == 0 {
		return ""
	}
	return filepath.ToSlash(filters[0] + func() string {
		if len(filters) == 1 {
			return ""
		}
		return "," + filters[1]
	}())
}

func jpeg2000PixelFormat(format Jpeg2000Format) (string, error) {
	switch format {
	case Jpeg2000FormatRGB:
		return "rgb24", nil
	case Jpeg2000FormatBGR:
		return "bgr24", nil
	case Jpeg2000FormatRGBA:
		return "rgba", nil
	case Jpeg2000FormatBGRA:
		return "bgra", nil
	case Jpeg2000FormatGrayscale:
		return "gray", nil
	case Jpeg2000FormatYUV420:
		return "yuv420p", nil
	case Jpeg2000FormatYUV422:
		return "yuv422p", nil
	case Jpeg2000FormatYUV444:
		return "yuv444p", nil
	default:
		return "", fmt.Errorf("unsupported JPEG2000 format: %d", format)
	}
}

func jpeg2000RawImageSize(width, height int, format Jpeg2000Format) int {
	switch format {
	case Jpeg2000FormatGrayscale:
		return width * height
	case Jpeg2000FormatRGB, Jpeg2000FormatBGR:
		return width * height * 3
	case Jpeg2000FormatRGBA, Jpeg2000FormatBGRA:
		return width * height * 4
	case Jpeg2000FormatYUV420:
		return width * height * 3 / 2
	case Jpeg2000FormatYUV422:
		return width * height * 2
	case Jpeg2000FormatYUV444:
		return width * height * 3
	default:
		return width * height * 3
	}
}

func componentsForPixelFormat(pixelFormat string) int {
	switch pixelFormat {
	case "gray", "gray16le":
		return 1
	case "rgba", "bgra", "argb", "abgr":
		return 4
	default:
		return 3
	}
}

func codecFromBytes(j2kData []byte) Jpeg2000Codec {
	if len(j2kData) >= 12 && string(j2kData[4:8]) == "jP  " {
		return Jpeg2000CodecJP2
	}
	return Jpeg2000CodecJ2K
}

func codecExtension(codec Jpeg2000Codec) string {
	switch codec {
	case Jpeg2000CodecJP2:
		return ".jp2"
	case Jpeg2000CodecJPT:
		return ".jpt"
	case Jpeg2000CodecJPX:
		return ".jpx"
	default:
		return ".j2k"
	}
}

func codecExtensionFromBytes(j2kData []byte) string {
	return codecExtension(codecFromBytes(j2kData))
}

// Destroy cleans up decoder resources
func (decoder *Jpeg2000DecoderState) Destroy() error {
	if decoder != nil && decoder.native {
		return destroyNativeJpeg2000Decoder(decoder)
	}
	if decoder.handle != nil {
		_ = decoder.handle.Free()
		decoder.handle = nil
	}
	if decoder.stream != nil {
		_ = decoder.stream.Free()
		decoder.stream = nil
	}
	return nil
}

// Destroy cleans up encoder resources
func (encoder *Jpeg2000EncoderState) Destroy() error {
	if encoder != nil && encoder.native {
		return destroyNativeJpeg2000Encoder(encoder)
	}
	if encoder.handle != nil {
		_ = encoder.handle.Free()
		encoder.handle = nil
	}
	if encoder.stream != nil {
		_ = encoder.stream.Free()
		encoder.stream = nil
	}
	return nil
}

// Convenience functions for quick operations

// DecodeJpeg2000Quick provides a simple interface for JPEG2000 decoding
func DecodeJpeg2000Quick(j2kData []byte, outputFormat Jpeg2000Format) (*memory.Memory, int, int, error) {
	decoder, err := CreateJpeg2000Decoder(Jpeg2000CodecJ2K)
	if err != nil {
		return nil, 0, 0, err
	}
	defer decoder.Destroy()

	params := Jpeg2000DecodeParams{
		OutputFormat: outputFormat,
		Codec:        Jpeg2000CodecJ2K,
		DecodeLayer:  -1, // All layers
		DecodeLevel:  -1, // All levels
	}

	return decoder.DecodeJpeg2000(j2kData, params)
}

// EncodeJpeg2000Quick provides a simple interface for JPEG2000 encoding
func EncodeJpeg2000Quick(imageData *memory.Memory, width, height int, inputFormat Jpeg2000Format, compressionRatio float32) ([]byte, error) {
	encoder, err := CreateJpeg2000Encoder(Jpeg2000CodecJ2K)
	if err != nil {
		return nil, err
	}
	defer encoder.Destroy()

	params := Jpeg2000EnodeParams{
		InputFormat:      inputFormat,
		Codec:            Jpeg2000CodecJ2K,
		CompressionRatio: compressionRatio,
		Lossless:         false,
		NumLayers:        1,
		NumLevels:        5,
		ProgressionOrder: Jpeg2000ProgressionLRCP,
		CodeblockWidth:   64,
		CodeblockHeight:  64,
	}

	return encoder.EncodeJpeg2000(imageData, width, height, params)
}

// EncodeJpeg2000Lossless provides lossless JPEG2000 encoding
func EncodeJpeg2000Lossless(imageData *memory.Memory, width, height int, inputFormat Jpeg2000Format) ([]byte, error) {
	encoder, err := CreateJpeg2000Encoder(Jpeg2000CodecJ2K)
	if err != nil {
		return nil, err
	}
	defer encoder.Destroy()

	params := Jpeg2000EnodeParams{
		InputFormat:      inputFormat,
		Codec:            Jpeg2000CodecJ2K,
		CompressionRatio: 1.0, // Lossless
		Lossless:         true,
		NumLayers:        1,
		NumLevels:        5,
		ProgressionOrder: Jpeg2000ProgressionLRCP,
		CodeblockWidth:   64,
		CodeblockHeight:  64,
	}

	return encoder.EncodeJpeg2000(imageData, width, height, params)
}
