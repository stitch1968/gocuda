//go:build !cuda

package libraries

import (
	"fmt"

	"github.com/stitch1968/gocuda/memory"
)

func nvjpegNativeAvailable() bool {
	return false
}

func createNativeJpegDecoder(backend JpegBackend) (*JpegDecoderState, error) {
	return nil, fmt.Errorf("native nvJPEG backend requires a CUDA-tagged build")
}

func createNativeJpegEncoder(backend JpegBackend, quality int) (*JpegEncoderState, error) {
	return nil, fmt.Errorf("native nvJPEG backend requires a CUDA-tagged build")
}

func decodeNativeJpeg(decoder *JpegDecoderState, jpegData []byte, params JpegDecodeParams) (*memory.Memory, int, int, error) {
	return nil, 0, 0, fmt.Errorf("native nvJPEG backend requires a CUDA-tagged build")
}

func encodeNativeJpeg(encoder *JpegEncoderState, imageData *memory.Memory, width, height int, params JpegEncodeParams) ([]byte, error) {
	return nil, fmt.Errorf("native nvJPEG backend requires a CUDA-tagged build")
}

func getNativeJpegImageInfo(jpegData []byte) (int, int, int, error) {
	return 0, 0, 0, fmt.Errorf("native nvJPEG backend requires a CUDA-tagged build")
}

func destroyNativeJpegDecoder(decoder *JpegDecoderState) error {
	return nil
}

func destroyNativeJpegEncoder(encoder *JpegEncoderState) error {
	return nil
}
