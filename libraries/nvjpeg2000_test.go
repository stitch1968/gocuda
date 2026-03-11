package libraries

import (
	"os/exec"
	"testing"

	"github.com/stitch1968/gocuda/memory"
)

func TestNvJPEG2000EncodeProbeDecode(t *testing.T) {
	if _, err := exec.LookPath("ffmpeg"); err != nil {
		t.Skip("ffmpeg not available")
	}
	if _, err := exec.LookPath("ffprobe"); err != nil {
		t.Skip("ffprobe not available")
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

	encoded, err := EncodeJpeg2000Lossless(input, 2, 2, Jpeg2000FormatRGB)
	if err != nil {
		t.Fatalf("EncodeJpeg2000Lossless failed: %v", err)
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

	decoded, width, height, err := DecodeJpeg2000Quick(encoded, Jpeg2000FormatRGB)
	if err != nil {
		t.Fatalf("DecodeJpeg2000Quick failed: %v", err)
	}
	defer decoded.Free()
	if width != 2 || height != 2 {
		t.Fatalf("unexpected decoded dimensions: %dx%d", width, height)
	}
	host := make([]byte, 2*2*3)
	if err := memory.CopyDeviceToHost(host, decoded); err != nil {
		t.Fatalf("CopyDeviceToHost failed: %v", err)
	}
	if len(host) != 12 {
		t.Fatalf("unexpected decoded byte count: %d", len(host))
	}
}
