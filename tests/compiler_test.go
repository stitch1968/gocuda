package tests

import (
	"context"
	"errors"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"

	"github.com/stitch1968/gocuda/kernels"
)

func TestCompileCUDAWithLocalCompiler(t *testing.T) {
	temp := t.TempDir()
	executable := filepath.Join(temp, "fake-nvcc")
	if runtime.GOOS == "windows" {
		executable += ".exe"
	}
	build := exec.Command("go", "build", "-o", executable, "."+string(os.PathSeparator)+filepath.Join("testdata", "fake_nvcc"))
	if output, err := build.CombinedOutput(); err != nil {
		t.Fatalf("build fake compiler: %v\n%s", err, output)
	}
	marker := filepath.Join(temp, "arguments.txt")
	t.Setenv("GOCUDA_FAKE_NVCC_MARKER", marker)
	options := kernels.CUDACompileOptions{
		Architecture: "compute_75",
		CompilerPath: executable,
		IncludeDirs:  []string{filepath.Join(temp, "include with spaces")},
		Defines:      []string{"VALUE=42"},
	}
	checkCleanup := func() {
		t.Helper()
		data, err := os.ReadFile(marker)
		if err != nil {
			t.Fatal(err)
		}
		lines := strings.Split(string(data), "\n")
		if _, err := os.Stat(filepath.Dir(lines[0])); !errors.Is(err, os.ErrNotExist) {
			t.Fatalf("compiler temporary directory remains: %v", err)
		}
		text := string(data)
		for _, expected := range []string{"--ptx", "--gpu-architecture=compute_75", "--include-path", options.IncludeDirs[0], "--define-macro", "VALUE=42"} {
			if !strings.Contains(text, expected) {
				t.Fatalf("compiler arguments missing %q:\n%s", expected, text)
			}
		}
	}
	result, err := kernels.CompileCUDA(context.Background(), "__global__ void test() {}", options)
	if err != nil {
		t.Fatal(err)
	}
	if result.PTX != "// fake PTX\n" || !strings.Contains(result.Diagnostics, "synthetic compiler diagnostic") {
		t.Fatalf("unexpected compiler result: %+v", result)
	}
	checkCleanup()
	if _, err := kernels.CompileCUDA(context.Background(), "FAIL", options); err == nil || !strings.Contains(err.Error(), "synthetic compilation error") {
		t.Fatalf("expected compiler diagnostic, got %v", err)
	}
	checkCleanup()
	ctx, cancel := context.WithTimeout(context.Background(), 500*time.Millisecond)
	defer cancel()
	if _, err := kernels.CompileCUDA(ctx, "SLOW", options); !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("expected compilation deadline, got %v", err)
	}
	checkCleanup()
}

func TestCompileCUDARejectsInvalidInput(t *testing.T) {
	options := kernels.CUDACompileOptions{Architecture: "compute_75"}
	if _, err := kernels.CompileCUDA(nil, "kernel", options); err == nil {
		t.Fatal("expected nil context to fail")
	}
	if _, err := kernels.CompileCUDA(context.Background(), "", options); err == nil {
		t.Fatal("expected empty source to fail")
	}
	options.Architecture = "sm_75"
	if _, err := kernels.CompileCUDA(context.Background(), "kernel", options); err == nil {
		t.Fatal("expected invalid architecture to fail")
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := kernels.CompileCUDA(ctx, "kernel", options); !errors.Is(err, context.Canceled) {
		t.Fatalf("expected canceled context, got %v", err)
	}
}
