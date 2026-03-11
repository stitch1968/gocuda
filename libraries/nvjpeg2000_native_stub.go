//go:build !cuda

package libraries

import "github.com/stitch1968/gocuda/memory"

func nvjpeg2000NativeAvailable() bool {
	return false
}

func createNativeJpeg2000Decoder(codec Jpeg2000Codec) (*Jpeg2000DecoderState, error) {
	return nil, errNVJPEG2000Unsupported
}

func createNativeJpeg2000Encoder(codec Jpeg2000Codec) (*Jpeg2000EncoderState, error) {
	return nil, errNVJPEG2000Unsupported
}

func decodeNativeJpeg2000(decoder *Jpeg2000DecoderState, j2kData []byte, params Jpeg2000DecodeParams) (*memory.Memory, int, int, error) {
	return nil, 0, 0, errNVJPEG2000Unsupported
}

func encodeNativeJpeg2000(encoder *Jpeg2000EncoderState, imageData *memory.Memory, width, height int, params Jpeg2000EnodeParams) ([]byte, error) {
	return nil, errNVJPEG2000Unsupported
}

func getNativeJpeg2000ImageInfo(j2kData []byte) (*Jpeg2000ImageInfo, error) {
	return nil, errNVJPEG2000Unsupported
}

func destroyNativeJpeg2000Decoder(decoder *Jpeg2000DecoderState) error {
	return nil
}

func destroyNativeJpeg2000Encoder(encoder *Jpeg2000EncoderState) error {
	return nil
}
