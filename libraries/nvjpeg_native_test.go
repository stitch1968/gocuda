//go:build cuda

package libraries

import (
	"bytes"
	"image"
	"image/color"
	"image/jpeg"
	"testing"

	cuda "github.com/stitch1968/gocuda"
	"github.com/stitch1968/gocuda/memory"
)

func TestNVJPEGNativeBackendSelection(t *testing.T) {
	if err := cuda.Initialize(); err != nil {
		t.Fatalf("Initialize failed: %v", err)
	}
	if !cuda.ShouldUseCuda() {
		t.Skip("CUDA runtime not available")
	}
	if !nvjpegNativeAvailable() {
		t.Skip("nvJPEG native backend not available")
	}

	decoder, err := CreateJpegDecoder(JpegBackendDefault)
	if err != nil {
		t.Fatalf("CreateJpegDecoder failed: %v", err)
	}
	defer decoder.Destroy()
	if !decoder.native {
		t.Fatal("expected native nvJPEG decoder in CUDA build")
	}

	encoder, err := CreateJpegEncoder(JpegBackendDefault)
	if err != nil {
		t.Fatalf("CreateJpegEncoder failed: %v", err)
	}
	defer encoder.Destroy()
	if !encoder.native {
		t.Fatal("expected native nvJPEG encoder in CUDA build")
	}
}

func TestNVJPEGNativeDecodeAndEncode(t *testing.T) {
	if err := cuda.Initialize(); err != nil {
		t.Fatalf("Initialize failed: %v", err)
	}
	if !cuda.ShouldUseCuda() {
		t.Skip("CUDA runtime not available")
	}
	if !nvjpegNativeAvailable() {
		t.Skip("nvJPEG native backend not available")
	}

	img := image.NewRGBA(image.Rect(0, 0, 2, 2))
	img.SetRGBA(0, 0, color.RGBA{R: 255, A: 255})
	img.SetRGBA(1, 0, color.RGBA{G: 255, A: 255})
	img.SetRGBA(0, 1, color.RGBA{B: 255, A: 255})
	img.SetRGBA(1, 1, color.RGBA{R: 255, G: 255, B: 255, A: 255})

	var encoded bytes.Buffer
	if err := jpeg.Encode(&encoded, img, &jpeg.Options{Quality: 100}); err != nil {
		t.Fatalf("jpeg.Encode failed: %v", err)
	}

	decoder, err := CreateJpegDecoder(JpegBackendDefault)
	if err != nil {
		t.Fatalf("CreateJpegDecoder failed: %v", err)
	}
	defer decoder.Destroy()

	decoded, width, height, err := decoder.DecodeJpeg(encoded.Bytes(), JpegDecodeParams{OutputFormat: JpegFormatRGB})
	if err != nil {
		t.Fatalf("DecodeJpeg failed: %v", err)
	}
	defer decoded.Free()
	if width != 2 || height != 2 {
		t.Fatalf("unexpected decoded dimensions: %dx%d", width, height)
	}

	host := make([]byte, width*height*3)
	if err := memory.CopyDeviceToHost(host, decoded); err != nil {
		t.Fatalf("CopyDeviceToHost failed: %v", err)
	}
	allZero := true
	for _, value := range host {
		if value != 0 {
			allZero = false
			break
		}
	}
	if allZero {
		t.Fatal("expected non-zero decoded image data")
	}

	encoder, err := CreateJpegEncoder(JpegBackendDefault)
	if err != nil {
		t.Fatalf("CreateJpegEncoder failed: %v", err)
	}
	defer encoder.Destroy()

	reencoded, err := encoder.EncodeJpeg(decoded, width, height, JpegEncodeParams{InputFormat: JpegFormatRGB, Quality: 90})
	if err != nil {
		t.Fatalf("EncodeJpeg failed: %v", err)
	}
	if len(reencoded) == 0 {
		t.Fatal("expected encoded JPEG output")
	}
	if _, err := jpeg.Decode(bytes.NewReader(reencoded)); err != nil {
		t.Fatalf("jpeg.Decode failed for encoded output: %v", err)
	}
}
