//go:build cuda

package libraries

import (
	"testing"

	cuda "github.com/stitch1968/gocuda"
	"github.com/stitch1968/gocuda/memory"
)

func TestNVJPEG2000NativeBackendSelection(t *testing.T) {
	if err := cuda.Initialize(); err != nil {
		t.Fatalf("Initialize failed: %v", err)
	}
	if !cuda.ShouldUseCuda() {
		t.Skip("CUDA runtime not available")
	}
	if !nvjpeg2000NativeAvailable() {
		t.Skip("nvJPEG2000 native backend not available")
	}

	decoder, err := CreateJpeg2000Decoder(Jpeg2000CodecJ2K)
	if err != nil {
		t.Fatalf("CreateJpeg2000Decoder failed: %v", err)
	}
	defer decoder.Destroy()
	if !decoder.native {
		t.Fatal("expected native nvJPEG2000 decoder in CUDA build")
	}

	encoder, err := CreateJpeg2000Encoder(Jpeg2000CodecJ2K)
	if err != nil {
		t.Fatalf("CreateJpeg2000Encoder failed: %v", err)
	}
	defer encoder.Destroy()
	if !encoder.native {
		t.Fatal("expected native nvJPEG2000 encoder in CUDA build")
	}
}

func TestNVJPEG2000NativeEncodeProbeDecode(t *testing.T) {
	if err := cuda.Initialize(); err != nil {
		t.Fatalf("Initialize failed: %v", err)
	}
	if !cuda.ShouldUseCuda() {
		t.Skip("CUDA runtime not available")
	}
	if !nvjpeg2000NativeAvailable() {
		t.Skip("nvJPEG2000 native backend not available")
	}

	input, err := memory.Alloc(2 * 2 * 3)
	if err != nil {
		t.Fatalf("alloc input failed: %v", err)
	}
	defer input.Free()
	if err := memory.CopyHostToDevice(input, []byte{
		255, 0, 0,
		0, 255, 0,
		0, 0, 255,
		255, 255, 0,
	}); err != nil {
		t.Fatalf("CopyHostToDevice failed: %v", err)
	}

	encoder, err := CreateJpeg2000Encoder(Jpeg2000CodecJ2K)
	if err != nil {
		t.Fatalf("CreateJpeg2000Encoder failed: %v", err)
	}
	defer encoder.Destroy()

	encoded, err := encoder.EncodeJpeg2000(input, 2, 2, Jpeg2000EnodeParams{
		InputFormat:      Jpeg2000FormatRGB,
		Codec:            Jpeg2000CodecJ2K,
		CompressionRatio: 1,
		Lossless:         true,
		NumLayers:        1,
		NumLevels:        1,
		ProgressionOrder: Jpeg2000ProgressionLRCP,
		CodeblockWidth:   32,
		CodeblockHeight:  32,
	})
	if err != nil {
		t.Fatalf("EncodeJpeg2000 failed: %v", err)
	}
	if len(encoded) == 0 {
		t.Fatal("expected encoded JPEG2000 output")
	}

	info, err := GetJpeg2000ImageInfo(encoded)
	if err != nil {
		t.Fatalf("GetJpeg2000ImageInfo failed: %v", err)
	}
	if info.Width != 2 || info.Height != 2 {
		t.Fatalf("unexpected probed dimensions: %+v", info)
	}

	decoder, err := CreateJpeg2000Decoder(Jpeg2000CodecJ2K)
	if err != nil {
		t.Fatalf("CreateJpeg2000Decoder failed: %v", err)
	}
	defer decoder.Destroy()

	decoded, width, height, err := decoder.DecodeJpeg2000(encoded, Jpeg2000DecodeParams{OutputFormat: Jpeg2000FormatRGB, Codec: Jpeg2000CodecJ2K})
	if err != nil {
		t.Fatalf("DecodeJpeg2000 failed: %v", err)
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
}
