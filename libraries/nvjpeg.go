// Package libraries provides CUDA runtime library bindings for GoCUDA
// This implements nvJPEG functionality for JPEG encoding/decoding
package libraries

import (
	"bytes"
	"fmt"
	"image"
	"image/color"
	"image/jpeg"

	"github.com/stitch1968/gocuda/memory"
)

// nvJPEG - High-Performance JPEG Encoder/Decoder Library

// JpegFormat formats and configurations
type JpegFormat int

const (
	JpegFormatRGB JpegFormat = iota
	JpegFormatBGR
	JpegFormatRGBI
	JpegFormatBGRI
	JpegFormatGrayscale
	JpegFormatYUV420
	JpegFormatYUV422
	JpegFormatYUV444
)

// JpegBackend decoder backend types
type JpegBackend int

const (
	JpegBackendDefault JpegBackend = iota
	JpegBackendHybrid
	JpegBackendGPUHybrid
	JpegBackendHardware
)

// JpegDecoderState decoder state
type JpegDecoderState struct {
	handle  *memory.Memory
	backend JpegBackend
	stream  *memory.Memory
}

// JpegEncoderState encoder state
type JpegEncoderState struct {
	handle  *memory.Memory
	backend JpegBackend
	stream  *memory.Memory
	quality int
}

// JpegDecodeParams decode parameters
type JpegDecodeParams struct {
	OutputFormat JpegFormat
	Backend      JpegBackend
	CropX        int
	CropY        int
	CropWidth    int
	CropHeight   int
	ScaleWidth   int
	ScaleHeight  int
}

// JpegEncodeParams encode parameters
type JpegEncodeParams struct {
	InputFormat      JpegFormat
	Quality          int
	OptimizedHuffman bool
	RestartInterval  int
}

// CreateJpegDecoder creates a new JPEG decoder
func CreateJpegDecoder(backend JpegBackend) (*JpegDecoderState, error) {
	if err := ensureCudaReady(); err != nil {
		return nil, err
	}

	decoder := &JpegDecoderState{
		backend: backend,
	}

	// Allocate decoder handle (simulated)
	var err error
	decoder.handle, err = memory.Alloc(2048) // Decoder state memory
	if err != nil {
		return nil, fmt.Errorf("failed to allocate decoder handle: %v", err)
	}

	// Allocate stream memory
	decoder.stream, err = memory.Alloc(1024)
	if err != nil {
		decoder.handle.Free()
		return nil, fmt.Errorf("failed to allocate stream memory: %v", err)
	}

	return decoder, nil
}

// CreateJpegEncoder creates a new JPEG encoder
func CreateJpegEncoder(backend JpegBackend) (*JpegEncoderState, error) {
	if err := ensureCudaReady(); err != nil {
		return nil, err
	}

	encoder := &JpegEncoderState{
		backend: backend,
		quality: 90, // Default quality
	}

	// Allocate encoder handle (simulated)
	var err error
	encoder.handle, err = memory.Alloc(2048) // Encoder state memory
	if err != nil {
		return nil, fmt.Errorf("failed to allocate encoder handle: %v", err)
	}

	// Allocate stream memory
	encoder.stream, err = memory.Alloc(1024)
	if err != nil {
		encoder.handle.Free()
		return nil, fmt.Errorf("failed to allocate stream memory: %v", err)
	}

	return encoder, nil
}

// DecodeJpeg decodes a JPEG image from byte data
func (decoder *JpegDecoderState) DecodeJpeg(jpegData []byte, params JpegDecodeParams) (*memory.Memory, int, int, error) {
	if len(jpegData) == 0 {
		return nil, 0, 0, fmt.Errorf("empty JPEG data")
	}

	// Simulate JPEG decoding using Go's standard library for CPU fallback
	reader := bytes.NewReader(jpegData)
	img, err := jpeg.Decode(reader)
	if err != nil {
		return nil, 0, 0, fmt.Errorf("failed to decode JPEG: %v", err)
	}

	bounds := img.Bounds()
	cropRect := resolveJpegCrop(bounds, params)
	width := cropRect.Dx()
	height := cropRect.Dy()
	if params.ScaleWidth > 0 {
		width = params.ScaleWidth
	}
	if params.ScaleHeight > 0 {
		height = params.ScaleHeight
	}

	// Calculate output size based on format
	var channels int
	switch params.OutputFormat {
	case JpegFormatGrayscale:
		channels = 1
	case JpegFormatRGB, JpegFormatBGR:
		channels = 3
	case JpegFormatRGBI, JpegFormatBGRI:
		channels = 4
	default:
		channels = 3
	}

	outputSize := width * height * channels
	outputMem, err := memory.Alloc(int64(outputSize))
	if err != nil {
		return nil, 0, 0, fmt.Errorf("failed to allocate output memory: %v", err)
	}

	outputBytes := convertImageToJpegFormat(img, cropRect, width, height, params.OutputFormat)
	if err := memory.CopyHostToDevice(outputMem, outputBytes); err != nil {
		outputMem.Free()
		return nil, 0, 0, err
	}

	return outputMem, width, height, nil
}

// DecodeJpegBatch decodes multiple JPEG images in batch
func (decoder *JpegDecoderState) DecodeJpegBatch(jpegDataList [][]byte, params JpegDecodeParams) ([]*memory.Memory, []int, []int, error) {
	if len(jpegDataList) == 0 {
		return nil, nil, nil, fmt.Errorf("empty JPEG data list")
	}

	outputs := make([]*memory.Memory, len(jpegDataList))
	widths := make([]int, len(jpegDataList))
	heights := make([]int, len(jpegDataList))

	for i, jpegData := range jpegDataList {
		output, width, height, err := decoder.DecodeJpeg(jpegData, params)
		if err != nil {
			// Clean up previously allocated memory
			for j := 0; j < i; j++ {
				if outputs[j] != nil {
					outputs[j].Free()
				}
			}
			return nil, nil, nil, fmt.Errorf("failed to decode JPEG %d: %v", i, err)
		}
		outputs[i] = output
		widths[i] = width
		heights[i] = height
	}

	return outputs, widths, heights, nil
}

// EncodeJpeg encodes image data to JPEG format
func (encoder *JpegEncoderState) EncodeJpeg(imageData *memory.Memory, width, height int, params JpegEncodeParams) ([]byte, error) {
	if imageData == nil {
		return nil, fmt.Errorf("image data cannot be nil")
	}
	if width <= 0 || height <= 0 {
		return nil, fmt.Errorf("invalid image dimensions: %dx%d", width, height)
	}

	// Calculate expected input size
	var channels int
	switch params.InputFormat {
	case JpegFormatGrayscale:
		channels = 1
	case JpegFormatRGB, JpegFormatBGR:
		channels = 3
	case JpegFormatRGBI, JpegFormatBGRI:
		channels = 4
	default:
		channels = 3
	}

	expectedSize := width * height * channels

	hostBytes := make([]byte, expectedSize)
	if err := memory.CopyDeviceToHost(hostBytes, imageData); err != nil {
		return nil, err
	}

	var buf bytes.Buffer
	img, err := buildImageFromJpegBytes(hostBytes, width, height, params.InputFormat)
	if err != nil {
		return nil, err
	}

	// Set encoder quality
	quality := params.Quality
	if quality == 0 {
		quality = encoder.quality
	}
	options := &jpeg.Options{Quality: quality}
	err = jpeg.Encode(&buf, img, options)
	if err != nil {
		return nil, fmt.Errorf("failed to encode JPEG: %v", err)
	}

	return buf.Bytes(), nil
}

// SetQuality sets the JPEG encoding quality (0-100)
func (encoder *JpegEncoderState) SetQuality(quality int) error {
	if quality < 0 || quality > 100 {
		return fmt.Errorf("quality must be between 0 and 100, got %d", quality)
	}
	encoder.quality = quality
	return nil
}

// GetJpegImageInfo extracts basic information from JPEG data without full decoding
func GetJpegImageInfo(jpegData []byte) (width, height, channels int, err error) {
	if len(jpegData) == 0 {
		return 0, 0, 0, fmt.Errorf("empty JPEG data")
	}

	reader := bytes.NewReader(jpegData)
	config, err := jpeg.DecodeConfig(reader)
	if err != nil {
		return 0, 0, 0, fmt.Errorf("failed to decode JPEG config: %v", err)
	}

	// Most JPEG images are RGB (3 channels) or Grayscale (1 channel)
	channels = 3
	if config.ColorModel != nil {
		// This is a simplified check - real nvJPEG would provide more detailed info
		channels = 3
	}

	return config.Width, config.Height, channels, nil
}

// Destroy cleans up decoder resources
func (decoder *JpegDecoderState) Destroy() error {
	if decoder.handle != nil {
		decoder.handle.Free()
		decoder.handle = nil
	}
	if decoder.stream != nil {
		decoder.stream.Free()
		decoder.stream = nil
	}
	return nil
}

// Destroy cleans up encoder resources
func (encoder *JpegEncoderState) Destroy() error {
	if encoder.handle != nil {
		encoder.handle.Free()
		encoder.handle = nil
	}
	if encoder.stream != nil {
		encoder.stream.Free()
		encoder.stream = nil
	}
	return nil
}

// Convenience functions for quick operations

// DecodeJpegQuick provides a simple interface for JPEG decoding
func DecodeJpegQuick(jpegData []byte, outputFormat JpegFormat) (*memory.Memory, int, int, error) {
	decoder, err := CreateJpegDecoder(JpegBackendDefault)
	if err != nil {
		return nil, 0, 0, err
	}
	defer decoder.Destroy()

	params := JpegDecodeParams{
		OutputFormat: outputFormat,
		Backend:      JpegBackendDefault,
	}

	return decoder.DecodeJpeg(jpegData, params)
}

// EncodeJpegQuick provides a simple interface for JPEG encoding
func EncodeJpegQuick(imageData *memory.Memory, width, height int, inputFormat JpegFormat, quality int) ([]byte, error) {
	encoder, err := CreateJpegEncoder(JpegBackendDefault)
	if err != nil {
		return nil, err
	}
	defer encoder.Destroy()

	params := JpegEncodeParams{
		InputFormat: inputFormat,
		Quality:     quality,
	}

	return encoder.EncodeJpeg(imageData, width, height, params)
}

func resolveJpegCrop(bounds image.Rectangle, params JpegDecodeParams) image.Rectangle {
	if params.CropWidth <= 0 || params.CropHeight <= 0 {
		return bounds
	}
	minX := clampInt(bounds.Min.X+params.CropX, bounds.Min.X, bounds.Max.X)
	minY := clampInt(bounds.Min.Y+params.CropY, bounds.Min.Y, bounds.Max.Y)
	maxX := clampInt(minX+params.CropWidth, minX, bounds.Max.X)
	maxY := clampInt(minY+params.CropHeight, minY, bounds.Max.Y)
	return image.Rect(minX, minY, maxX, maxY)
}

func convertImageToJpegFormat(img image.Image, cropRect image.Rectangle, width, height int, format JpegFormat) []byte {
	channels := jpegChannelCount(format)
	output := make([]byte, width*height*channels)
	for y := 0; y < height; y++ {
		sourceY := cropRect.Min.Y + (y*cropRect.Dy())/height
		for x := 0; x < width; x++ {
			sourceX := cropRect.Min.X + (x*cropRect.Dx())/width
			r, g, b, _ := img.At(sourceX, sourceY).RGBA()
			r8 := byte(r >> 8)
			g8 := byte(g >> 8)
			b8 := byte(b >> 8)
			offset := (y*width + x) * channels
			switch format {
			case JpegFormatGrayscale:
				output[offset] = byte((299*uint16(r8) + 587*uint16(g8) + 114*uint16(b8)) / 1000)
			case JpegFormatBGR:
				output[offset], output[offset+1], output[offset+2] = b8, g8, r8
			case JpegFormatRGBI:
				output[offset], output[offset+1], output[offset+2], output[offset+3] = r8, g8, b8, 255
			case JpegFormatBGRI:
				output[offset], output[offset+1], output[offset+2], output[offset+3] = b8, g8, r8, 255
			default:
				output[offset], output[offset+1], output[offset+2] = r8, g8, b8
			}
		}
	}
	return output
}

func buildImageFromJpegBytes(data []byte, width, height int, format JpegFormat) (image.Image, error) {
	channels := jpegChannelCount(format)
	if len(data) < width*height*channels {
		return nil, fmt.Errorf("image buffer too small: need %d bytes, have %d", width*height*channels, len(data))
	}
	if format == JpegFormatGrayscale {
		img := image.NewGray(image.Rect(0, 0, width, height))
		copy(img.Pix, data[:width*height])
		return img, nil
	}
	img := image.NewRGBA(image.Rect(0, 0, width, height))
	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			source := (y*width + x) * channels
			var r, g, b byte
			switch format {
			case JpegFormatBGR:
				b, g, r = data[source], data[source+1], data[source+2]
			case JpegFormatRGBI:
				r, g, b = data[source], data[source+1], data[source+2]
			case JpegFormatBGRI:
				b, g, r = data[source], data[source+1], data[source+2]
			default:
				r, g, b = data[source], data[source+1], data[source+2]
			}
			img.SetRGBA(x, y, color.RGBA{R: r, G: g, B: b, A: 255})
		}
	}
	return img, nil
}

func jpegChannelCount(format JpegFormat) int {
	switch format {
	case JpegFormatGrayscale:
		return 1
	case JpegFormatRGBI, JpegFormatBGRI:
		return 4
	default:
		return 3
	}
}

func clampInt(value, minValue, maxValue int) int {
	if value < minValue {
		return minValue
	}
	if value > maxValue {
		return maxValue
	}
	return value
}
