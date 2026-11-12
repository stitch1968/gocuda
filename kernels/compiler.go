package kernels

import (
	"context"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strings"
	"unicode/utf8"
)

var computeArchitecturePattern = regexp.MustCompile(`^compute_[0-9]{2,3}[a-z]?$`)

// CUDACompileOptions configures compilation with the local nvcc executable.
type CUDACompileOptions struct {
	Architecture string
	CompilerPath string
	IncludeDirs  []string
	Defines      []string
}

// CUDACompileResult contains PTX and any compiler diagnostics.
type CUDACompileResult struct {
	PTX         string
	Diagnostics string
}

// CompileCUDA compiles CUDA source to PTX with a local nvcc installation.
// On Windows, nvcc also requires a configured MSVC host compiler. Compilation
// is local and does not require the CUDA build tag or a GPU.
func CompileCUDA(ctx context.Context, source string, options CUDACompileOptions) (result *CUDACompileResult, err error) {
	if ctx == nil {
		return nil, fmt.Errorf("compilation context must not be nil")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if source == "" || strings.IndexByte(source, 0) >= 0 {
		return nil, fmt.Errorf("CUDA source must be nonempty and contain no NUL bytes")
	}
	if !computeArchitecturePattern.MatchString(options.Architecture) {
		return nil, fmt.Errorf("invalid PTX architecture %q; expected e.g. compute_75", options.Architecture)
	}
	for _, include := range options.IncludeDirs {
		if include == "" || strings.IndexByte(include, 0) >= 0 {
			return nil, fmt.Errorf("include directories must be nonempty and contain no NUL bytes")
		}
	}
	for _, define := range options.Defines {
		if define == "" || strings.IndexByte(define, 0) >= 0 {
			return nil, fmt.Errorf("compiler definitions must be nonempty and contain no NUL bytes")
		}
	}
	compiler := options.CompilerPath
	if strings.IndexByte(compiler, 0) >= 0 {
		return nil, fmt.Errorf("compiler path must not contain NUL bytes")
	}
	if compiler == "" {
		compiler, err = exec.LookPath("nvcc")
		if err != nil {
			return nil, fmt.Errorf("find nvcc: %w", err)
		}
	}
	dir, err := os.MkdirTemp("", "gocuda-nvcc-*")
	if err != nil {
		return nil, fmt.Errorf("create compiler temporary directory: %w", err)
	}
	defer func() {
		if cleanupErr := os.RemoveAll(dir); cleanupErr != nil {
			result = nil
			err = errors.Join(err, fmt.Errorf("remove compiler temporary directory: %w", cleanupErr))
		}
	}()
	input := filepath.Join(dir, "kernel.cu")
	output := filepath.Join(dir, "kernel.ptx")
	if err := os.WriteFile(input, []byte(source), 0o600); err != nil {
		return nil, fmt.Errorf("write CUDA source: %w", err)
	}
	args := []string{"--ptx", "--gpu-architecture=" + options.Architecture, "--output-file", output}
	for _, include := range options.IncludeDirs {
		args = append(args, "--include-path", include)
	}
	for _, define := range options.Defines {
		args = append(args, "--define-macro", define)
	}
	args = append(args, input)
	cmd := exec.CommandContext(ctx, compiler, args...)
	diagnostics, runErr := cmd.CombinedOutput()
	if runErr != nil {
		if ctx.Err() != nil {
			runErr = ctx.Err()
		}
		return nil, fmt.Errorf("nvcc compilation failed: %w\n%s", runErr, strings.TrimSpace(string(diagnostics)))
	}
	ptx, err := os.ReadFile(output)
	if err != nil {
		return nil, fmt.Errorf("read compiled PTX: %w", err)
	}
	if len(ptx) == 0 || !utf8.Valid(ptx) || strings.IndexByte(string(ptx), 0) >= 0 {
		return nil, fmt.Errorf("nvcc produced empty or invalid PTX")
	}
	return &CUDACompileResult{PTX: string(ptx), Diagnostics: string(diagnostics)}, nil
}
