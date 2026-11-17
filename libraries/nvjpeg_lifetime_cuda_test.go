//go:build cuda

package libraries

import (
	"bytes"
	"fmt"
	"image"
	"image/color"
	"image/jpeg"
	"sync"
	"testing"

	cuda "github.com/stitch1968/gocuda"
	"github.com/stitch1968/gocuda/memory"
)

func TestNVJPEGNativeDecodeOnEachDevice(t *testing.T) {
	if err := cuda.Initialize(); err != nil {
		t.Fatal(err)
	}
	if !cuda.ShouldUseCuda() || !nvjpegNativeAvailable() {
		t.Skip("native nvJPEG unavailable")
	}
	devices, err := cuda.GetDevices()
	if err != nil {
		t.Fatal(err)
	}
	imageData := image.NewRGBA(image.Rect(0, 0, 2, 2))
	imageData.SetRGBA(0, 0, color.RGBA{R: 255, A: 255})
	var encoded bytes.Buffer
	if err := jpeg.Encode(&encoded, imageData, nil); err != nil {
		t.Fatal(err)
	}
	reference, err := jpeg.Decode(bytes.NewReader(encoded.Bytes()))
	if err != nil {
		t.Fatal(err)
	}
	expected := make([]byte, 12)
	for y := range 2 {
		for x := range 2 {
			r, g, b, _ := reference.At(x, y).RGBA()
			index := (y*2 + x) * 3
			expected[index], expected[index+1], expected[index+2] = byte(r>>8), byte(g>>8), byte(b>>8)
		}
	}
	for deviceID := range devices {
		t.Run(fmt.Sprintf("%d_%s", deviceID, devices[deviceID].Name), func(t *testing.T) {
			deviceContext, err := cuda.NewContext(deviceID)
			if err != nil {
				t.Fatal(err)
			}
			var decoder *JpegDecoderState
			if err := deviceContext.Run(func() error {
				var createErr error
				decoder, createErr = CreateJpegDecoder(JpegBackendDefault)
				return createErr
			}); err != nil {
				t.Fatal(err)
			}
			t.Cleanup(func() {
				if err := decoder.Destroy(); err != nil {
					t.Error(err)
				}
			})
			if decoder.deviceID != deviceID {
				t.Fatalf("decoder device = %d, want %d", decoder.deviceID, deviceID)
			}
			const workers = 4
			var wg sync.WaitGroup
			errs := make(chan error, workers*3)
			for range workers {
				wg.Add(1)
				go func() {
					defer wg.Done()
					output, width, height, err := decoder.DecodeJpeg(encoded.Bytes(), JpegDecodeParams{OutputFormat: JpegFormatRGB})
					if err != nil {
						errs <- err
						return
					}
					if output.GetDeviceID() != deviceID || width != 2 || height != 2 {
						errs <- fmt.Errorf("decoded output device=%d size=%dx%d", output.GetDeviceID(), width, height)
					}
					pixels := make([]byte, 12)
					if err := memory.CopyDeviceToHost(pixels, output); err != nil {
						errs <- err
					} else {
						for i, pixel := range pixels {
							difference := int(pixel) - int(expected[i])
							if difference < -5 || difference > 5 {
								errs <- fmt.Errorf("pixel component %d = %d, want approximately %d", i, pixel, expected[i])
								break
							}
						}
					}
					if err := output.Free(); err != nil {
						errs <- err
					}
				}()
			}
			wg.Wait()
			close(errs)
			for err := range errs {
				t.Error(err)
			}
		})
	}
}

func TestNVJPEGNativeGrayscaleDecode(t *testing.T) {
	if err := cuda.Initialize(); err != nil {
		t.Fatal(err)
	}
	if !cuda.ShouldUseCuda() || !nvjpegNativeAvailable() {
		t.Skip("native nvJPEG unavailable")
	}
	img := image.NewGray(image.Rect(0, 0, 16, 16))
	for y := range 16 {
		for x := range 16 {
			img.SetGray(x, y, color.Gray{Y: byte(x*8 + y*3)})
		}
	}
	var encoded bytes.Buffer
	if err := jpeg.Encode(&encoded, img, nil); err != nil {
		t.Fatal(err)
	}
	reference, err := jpeg.Decode(bytes.NewReader(encoded.Bytes()))
	if err != nil {
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
	output, width, height, err := decoder.DecodeJpeg(encoded.Bytes(), JpegDecodeParams{OutputFormat: JpegFormatGrayscale})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := output.Free(); err != nil {
			t.Error(err)
		}
	})
	if width != 16 || height != 16 {
		t.Fatalf("decoded dimensions = %dx%d, want 16x16", width, height)
	}
	pixels := make([]byte, width*height)
	if err := memory.CopyDeviceToHost(pixels, output); err != nil {
		t.Fatal(err)
	}
	for y := range height {
		for x := range width {
			r, _, _, _ := reference.At(x, y).RGBA()
			difference := int(pixels[y*width+x]) - int(r>>8)
			if difference < -3 || difference > 3 {
				t.Fatalf("pixel (%d,%d) = %d, want approximately %d", x, y, pixels[y*width+x], r>>8)
			}
		}
	}
}

func TestNVJPEGNativePendingCleanupRetry(t *testing.T) {
	if err := cuda.Initialize(); err != nil {
		t.Fatal(err)
	}

	if !cuda.ShouldUseCuda() || !nvjpegNativeAvailable() {
		t.Skip("native nvJPEG unavailable")
	}
	decoder, err := CreateJpegDecoder(JpegBackendDefault)
	if err != nil {
		t.Fatal(err)
	}
	buffer, err := memory.AllocDeviceOnlyOnDevice(4, decoder.deviceID)
	if err != nil {
		t.Fatal(err)
	}
	retention, err := buffer.Retain()
	if err != nil {
		t.Fatal(err)
	}
	decoder.pending = append(decoder.pending, buffer)
	t.Cleanup(func() {
		retention.Release()
		if err := decoder.Destroy(); err != nil {
			t.Error(err)
		}
		if err := buffer.Free(); err != nil {
			t.Error(err)
		}
	})
	if err := decoder.Synchronize(); err == nil {
		t.Fatal("expected retained pending buffer cleanup to fail")
	}
	if len(decoder.pending) != 1 || buffer.Ptr() == nil {
		t.Fatal("failed cleanup must remain pending")
	}
	if err := decoder.Destroy(); err == nil {
		t.Fatal("expected destruction to preserve pending cleanup")
	}
	retention.Release()
	if err := decoder.Synchronize(); err != nil {
		t.Fatal(err)
	}
	if len(decoder.pending) != 0 || buffer.Ptr() != nil {
		t.Fatal("successful retry must release pending buffer")
	}
}

func TestNVJPEGNativeHybridDecode(t *testing.T) {
	if err := cuda.Initialize(); err != nil {
		t.Fatal(err)
	}
	if !cuda.ShouldUseCuda() || !nvjpegNativeAvailable() {
		t.Skip("native nvJPEG unavailable")
	}
	img := image.NewRGBA(image.Rect(0, 0, 16, 16))
	for y := range 16 {
		for x := range 16 {
			img.SetRGBA(x, y, color.RGBA{R: 220, G: 20, B: 10, A: 255})
		}
	}
	var encoded bytes.Buffer
	if err := jpeg.Encode(&encoded, img, nil); err != nil {
		t.Fatal(err)
	}
	for _, backend := range []JpegBackend{JpegBackendHybrid, JpegBackendGPUHybrid} {
		t.Run(fmt.Sprintf("backend_%d", backend), func(t *testing.T) {
			decoder, err := CreateJpegDecoder(backend)
			if err != nil {
				t.Fatal(err)
			}
			t.Cleanup(func() {
				if err := decoder.Destroy(); err != nil {
					t.Error(err)
				}
			})
			output, width, height, err := decoder.DecodeJpeg(encoded.Bytes(), JpegDecodeParams{OutputFormat: JpegFormatRGB})
			if err != nil {
				t.Fatal(err)
			}
			t.Cleanup(func() {
				if err := output.Free(); err != nil {
					t.Error(err)
				}
			})
			if width != 16 || height != 16 {
				t.Fatalf("decoded dimensions = %dx%d, want 16x16", width, height)
			}
			pixels := make([]byte, width*height*3)
			if err := memory.CopyDeviceToHost(pixels, output); err != nil {
				t.Fatal(err)
			}
			for i := 0; i < len(pixels); i += 3 {
				if pixels[i] < 200 || pixels[i+1] > 40 || pixels[i+2] > 30 {
					t.Fatalf("unexpected RGB pixel: %v", pixels[i:i+3])
				}
			}
		})
	}
}
