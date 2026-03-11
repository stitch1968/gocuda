package libraries

import (
	"bytes"
	"image"
	"image/color"
	"image/jpeg"
	"testing"

	"github.com/stitch1968/gocuda/memory"
)

func TestNVJPEGDecodeAndInfo(t *testing.T) {
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
		t.Fatalf("unexpected dimensions: got %dx%d want 2x2", width, height)
	}
	bytesOut := make([]byte, width*height*3)
	if err := memory.CopyDeviceToHost(bytesOut, decoded); err != nil {
		t.Fatalf("CopyDeviceToHost failed: %v", err)
	}
	nonZero := false
	for _, value := range bytesOut {
		if value != 0 {
			nonZero = true
			break
		}
	}
	if !nonZero {
		t.Fatal("expected decoded pixel data to contain non-zero values")
	}

	infoWidth, infoHeight, channels, err := GetJpegImageInfo(encoded.Bytes())
	if err != nil {
		t.Fatalf("GetJpegImageInfo failed: %v", err)
	}
	if infoWidth != 2 || infoHeight != 2 || channels != 3 {
		t.Fatalf("unexpected image info: got %dx%d channels=%d", infoWidth, infoHeight, channels)
	}
}

func TestNVJPEGEncode(t *testing.T) {
	encoder, err := CreateJpegEncoder(JpegBackendDefault)
	if err != nil {
		t.Fatalf("CreateJpegEncoder failed: %v", err)
	}
	defer encoder.Destroy()

	imgData, err := memory.Alloc(2 * 2 * 3)
	if err != nil {
		t.Fatalf("alloc image data failed: %v", err)
	}
	defer imgData.Free()
	pixels := []byte{
		255, 0, 0,
		0, 255, 0,
		0, 0, 255,
		255, 255, 255,
	}
	if err := memory.CopyHostToDevice(imgData, pixels); err != nil {
		t.Fatalf("CopyHostToDevice failed: %v", err)
	}

	encoded, err := encoder.EncodeJpeg(imgData, 2, 2, JpegEncodeParams{InputFormat: JpegFormatRGB, Quality: 95})
	if err != nil {
		t.Fatalf("EncodeJpeg failed: %v", err)
	}
	if len(encoded) == 0 {
		t.Fatal("expected JPEG output bytes")
	}
	if _, err := jpeg.Decode(bytes.NewReader(encoded)); err != nil {
		t.Fatalf("encoded JPEG is invalid: %v", err)
	}
}
