//go:build windows

package tests

import (
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

func TestWindowsHardwareWrapperConfiguresAndRestoresCgo(t *testing.T) {
	temp := t.TempDir()
	cudaRoot := filepath.Join(temp, "CUDA Toolkit")
	include := filepath.Join(cudaRoot, "include")
	extra := filepath.Join(temp, "extra headers")
	for _, path := range []string{include, extra} {
		if err := os.MkdirAll(path, 0o755); err != nil {
			t.Fatal(err)
		}
	}
	if err := os.WriteFile(filepath.Join(include, "cuda.h"), []byte("#define CUDA_VERSION 13020\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	root, err := filepath.Abs("..")
	if err != nil {
		t.Fatal(err)
	}
	script := `& .\run_hardware_mode.ps1 -SkipVerify -Command @('go','env','CGO_CFLAGS'); ` +
		`if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }; ` +
		`if ($env:CGO_CFLAGS -ne '-DWRAPPER_TEST=1 -O1' -or $env:CGO_ENABLED -ne '0') { throw 'cgo environment not restored' }; ` +
		`Write-Output 'WRAPPER_RESTORED'`
	cmd := exec.Command("powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", script)
	cmd.Dir = root
	cmd.Env = append(os.Environ(),
		"CUDA_PATH="+cudaRoot,
		"GOCUDA_EXTRA_INCLUDE_DIRS="+extra,
		"CGO_CFLAGS=-DWRAPPER_TEST=1 -O1",
		"CGO_ENABLED=0",
	)
	output, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("wrapper failed: %v\n%s", err, output)
	}
	text := string(output)
	cudaFlag := strings.Index(text, "-I"+include)
	extraFlag := strings.Index(text, "-I"+extra)
	originalFlag := strings.Index(text, "-DWRAPPER_TEST=1")
	if cudaFlag < 0 || extraFlag < 0 || originalFlag < 0 || cudaFlag > originalFlag || extraFlag > originalFlag {
		t.Fatalf("expected CUDA and extra includes before original flags:\n%s", text)
	}
	if !strings.Contains(text, "WRAPPER_RESTORED") {
		t.Fatalf("wrapper did not restore caller environment:\n%s", text)
	}
}
