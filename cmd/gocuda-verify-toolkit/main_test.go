package main

import (
	"os"
	"path/filepath"
	"runtime"
	"testing"
)

func TestCheckCUDAHomeMissing(t *testing.T) {
	missing := filepath.Join(t.TempDir(), "missing-cuda")
	result := checkCUDAHome(missing)
	if result.ok {
		t.Fatalf("expected missing CUDA home to fail, got %+v", result)
	}
}

func TestCheckNVCCFallsBackToToolkitBin(t *testing.T) {
	toolkit := t.TempDir()
	binDir := filepath.Join(toolkit, "bin")
	if err := os.MkdirAll(binDir, 0o755); err != nil {
		t.Fatal(err)
	}
	nvccName := "nvcc"
	if runtime.GOOS == "windows" {
		nvccName = "nvcc.exe"
	}
	nvccPath := filepath.Join(binDir, nvccName)
	if err := os.WriteFile(nvccPath, []byte(""), 0o755); err != nil {
		t.Fatal(err)
	}
	oldPath := os.Getenv("PATH")
	t.Setenv("PATH", t.TempDir())
	defer os.Setenv("PATH", oldPath)

	result := checkNVCC(toolkit)
	if !result.ok {
		t.Fatalf("expected toolkit bin fallback to succeed, got %+v", result)
	}
	if result.detail != nvccPath {
		t.Fatalf("expected nvcc path %s, got %s", nvccPath, result.detail)
	}
}

func TestCheckCUDAHeadersMissingFiles(t *testing.T) {
	toolkit := t.TempDir()
	if err := os.MkdirAll(filepath.Join(toolkit, "include"), 0o755); err != nil {
		t.Fatal(err)
	}
	results := checkCUDAHeaders(toolkit)
	for _, result := range results {
		if result.ok {
			t.Fatalf("expected missing header check to fail, got %+v", result)
		}
	}
}

func TestCheckCUDALibrariesMissingFiles(t *testing.T) {
	toolkit := t.TempDir()
	libDir := filepath.Join(toolkit, "lib64")
	if runtime.GOOS == "windows" {
		libDir = filepath.Join(toolkit, "lib", "x64")
	}
	if err := os.MkdirAll(libDir, 0o755); err != nil {
		t.Fatal(err)
	}
	results := checkCUDALibraries(toolkit)
	for _, result := range results {
		if result.ok {
			t.Fatalf("expected missing library check to fail, got %+v", result)
		}
	}
}
