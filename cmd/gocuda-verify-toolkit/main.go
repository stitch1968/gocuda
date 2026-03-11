package main

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
)

type check struct {
	name   string
	ok     bool
	detail string
}

func main() {
	checks, err := runChecks()
	for _, item := range checks {
		status := "FAIL"
		if item.ok {
			status = "OK"
		}
		fmt.Printf("[%s] %s: %s\n", status, item.name, item.detail)
	}
	if err != nil {
		fmt.Fprintln(os.Stderr, err.Error())
		os.Exit(1)
	}
	fmt.Println("GoCUDA toolkit verification passed")
}

func runChecks() ([]check, error) {
	checks := []check{
		checkCommand("go", "Go toolchain available"),
	}

	cudaHome := detectCUDAHome()
	checks = append(checks, checkCUDAHome(cudaHome))
	checks = append(checks, checkNVCC(cudaHome))
	checks = append(checks, checkCUDAHeaders(cudaHome)...)
	checks = append(checks, checkCUDALibraries(cudaHome)...)
	checks = append(checks, checkRepoImportLibraries())

	failed := make([]string, 0)
	for _, item := range checks {
		if !item.ok {
			failed = append(failed, item.name)
		}
	}
	if len(failed) > 0 {
		return checks, fmt.Errorf("toolkit verification failed: %s", strings.Join(failed, ", "))
	}
	return checks, nil
}

func checkCommand(command string, label string) check {
	path, err := exec.LookPath(command)
	if err != nil {
		return check{name: label, ok: false, detail: err.Error()}
	}
	return check{name: label, ok: true, detail: path}
}

func detectCUDAHome() string {
	if runtime.GOOS == "windows" {
		if value := os.Getenv("CUDA_PATH"); value != "" {
			return value
		}
		return `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1`
	}
	if value := os.Getenv("CUDA_HOME"); value != "" {
		return value
	}
	if value := os.Getenv("CUDA_PATH"); value != "" {
		return value
	}
	return "/usr/local/cuda"
}

func checkCUDAHome(cudaHome string) check {
	if info, err := os.Stat(cudaHome); err == nil && info.IsDir() {
		return check{name: "CUDA toolkit root", ok: true, detail: cudaHome}
	}
	return check{name: "CUDA toolkit root", ok: false, detail: fmt.Sprintf("not found: %s", cudaHome)}
}

func checkNVCC(cudaHome string) check {
	nvccName := "nvcc"
	if runtime.GOOS == "windows" {
		nvccName = "nvcc.exe"
	}
	if path, err := exec.LookPath(nvccName); err == nil {
		return check{name: "nvcc compiler", ok: true, detail: path}
	}
	path := filepath.Join(cudaHome, "bin", nvccName)
	if _, err := os.Stat(path); err == nil {
		return check{name: "nvcc compiler", ok: true, detail: path}
	}
	return check{name: "nvcc compiler", ok: false, detail: fmt.Sprintf("not found in PATH or %s", path)}
}

func checkCUDAHeaders(cudaHome string) []check {
	headers := []string{"cuda.h", "cufft.h", "curand.h", "cusolverDn.h"}
	results := make([]check, 0, len(headers))
	for _, header := range headers {
		path := filepath.Join(cudaHome, "include", header)
		if _, err := os.Stat(path); err == nil {
			results = append(results, check{name: fmt.Sprintf("Header %s", header), ok: true, detail: path})
		} else {
			results = append(results, check{name: fmt.Sprintf("Header %s", header), ok: false, detail: fmt.Sprintf("missing %s", path)})
		}
	}
	return results
}

func checkCUDALibraries(cudaHome string) []check {
	var libDir string
	if runtime.GOOS == "windows" {
		libDir = filepath.Join(cudaHome, "lib", "x64")
	} else {
		libDir = filepath.Join(cudaHome, "lib64")
	}

	libraries := map[string][]string{
		"CUDA runtime library": {libraryFileName("cudart")},
		"cuFFT library":        {libraryFileName("cufft")},
		"cuRAND library":       {libraryFileName("curand")},
		"cuSOLVER library":     {libraryFileName("cusolver")},
	}

	results := make([]check, 0, len(libraries))
	for label, candidates := range libraries {
		found := ""
		for _, candidate := range candidates {
			path := filepath.Join(libDir, candidate)
			if _, err := os.Stat(path); err == nil {
				found = path
				break
			}
		}
		if found != "" {
			results = append(results, check{name: label, ok: true, detail: found})
		} else {
			results = append(results, check{name: label, ok: false, detail: fmt.Sprintf("missing in %s", libDir)})
		}
	}
	return results
}

func libraryFileName(base string) string {
	if runtime.GOOS == "windows" {
		return base + ".lib"
	}
	return "lib" + base + ".so"
}

func checkRepoImportLibraries() check {
	if runtime.GOOS != "windows" {
		return check{name: "Repository import libraries", ok: true, detail: "not required on this platform"}
	}
	required := []string{
		filepath.Join("lib_mingw", "libcuda.a"),
		filepath.Join("lib_mingw", "libcudart.a"),
		filepath.Join("lib_mingw", "libcufft.a"),
		filepath.Join("lib_mingw", "libcurand.a"),
		filepath.Join("lib_mingw", "libcusolver.a"),
		filepath.Join("lib_mingw", "libcusparse.a"),
	}
	missing := make([]string, 0)
	for _, path := range required {
		if _, err := os.Stat(path); err != nil {
			missing = append(missing, path)
		}
	}
	if len(missing) > 0 {
		return check{name: "Repository import libraries", ok: false, detail: fmt.Sprintf("missing %s", strings.Join(missing, ", "))}
	}
	return check{name: "Repository import libraries", ok: true, detail: strings.Join(required, ", ")}
}
