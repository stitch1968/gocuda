package libraries

import (
	"bytes"
	"image"
	"image/color"
	"image/jpeg"
	"strings"
	"sync"
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

func TestNVJPEGDecoderLifecycle(t *testing.T) {
	if _, err := CreateJpegDecoder(JpegBackend(-1)); err == nil {
		t.Fatal("expected invalid decoder backend to fail")
	}
	var nilDecoder *JpegDecoderState
	if _, _, _, err := nilDecoder.DecodeJpeg(nil, JpegDecodeParams{}); err == nil {
		t.Fatal("expected nil decoder to fail")
	}
	if err := nilDecoder.Synchronize(); err == nil {
		t.Fatal("expected nil decoder synchronization to fail")
	}
	decoder, err := CreateJpegDecoder(JpegBackendDefault)
	if err != nil {
		t.Fatal(err)
	}
	if err := decoder.Synchronize(); err != nil {
		t.Fatal(err)
	}
	if err := decoder.Destroy(); err != nil {
		t.Fatal(err)
	}
	if err := decoder.Destroy(); err != nil {
		t.Fatal(err)
	}
	if _, _, _, err := decoder.DecodeJpeg(nil, JpegDecodeParams{}); err == nil {
		t.Fatal("expected destroyed decoder to reject decoding")
	}
	if err := decoder.Synchronize(); err == nil {
		t.Fatal("expected destroyed decoder synchronization to fail")
	}
}

func TestNVJPEGConcurrentDecodeAndDestroy(t *testing.T) {
	var encoded bytes.Buffer
	if err := jpeg.Encode(&encoded, image.NewRGBA(image.Rect(0, 0, 2, 2)), nil); err != nil {
		t.Fatal(err)
	}
	decoder, err := CreateJpegDecoder(JpegBackendDefault)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := decoder.Destroy(); err != nil {
			t.Error(err)
		}
	})
	output, _, _, err := decoder.DecodeJpeg(encoded.Bytes(), JpegDecodeParams{OutputFormat: JpegFormatRGB})
	if err != nil {
		t.Fatal(err)
	}
	if err := output.Free(); err != nil {
		t.Fatal(err)
	}
	const workers = 4
	start := make(chan struct{})
	errs := make(chan error, workers+1)
	var wg sync.WaitGroup
	for range workers {
		wg.Add(1)
		go func() {
			defer wg.Done()
			<-start
			output, _, _, err := decoder.DecodeJpeg(encoded.Bytes(), JpegDecodeParams{OutputFormat: JpegFormatRGB})
			if err != nil {
				if !strings.Contains(err.Error(), "JPEG decoder has been destroyed") {
					errs <- err
				}
				return
			}
			if err := output.Free(); err != nil {
				errs <- err
			}
		}()
	}
	wg.Add(1)
	go func() {
		defer wg.Done()
		<-start
		if err := decoder.Destroy(); err != nil {
			errs <- err
		}
	}()
	close(start)
	wg.Wait()
	close(errs)
	for err := range errs {
		t.Error(err)
	}
}
