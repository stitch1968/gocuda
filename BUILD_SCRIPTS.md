# GoCUDA Build Scripts Documentation

This document explains all the available build methods for the GoCUDA project.

## Build Scripts Overview

### 1. **build.bat** (Windows Batch Script)
**Primary build script for Windows users**

```cmd
# Basic usage
build.bat [cuda|nocuda] [demo]

# Examples
build.bat nocuda           # Build with simulation mode
build.bat cuda             # Build with CUDA support
build.bat nocuda demo      # Build and run comprehensive demo
```

**Features:**
- ✅ CUDA detection and validation
- ✅ Automatic testing after build
- ✅ Demo execution option
- ✅ Comprehensive error handling
- ✅ Build information display

### 2. **build.sh** (Unix Shell Script)
**Primary build script for Linux/macOS users**

```bash
# Basic usage
./build.sh [cuda|nocuda] [demo]

# Examples
./build.sh nocuda          # Build with simulation mode
./build.sh cuda            # Build with CUDA support
./build.sh nocuda demo     # Build and run comprehensive demo
```

**Features:**
- ✅ CUDA detection and validation
- ✅ Automatic testing after build
- ✅ Demo execution option
- ✅ Comprehensive error handling
- ✅ Build information display

### 3. **Makefile** (GNU Make)
**Advanced build script for developers familiar with make**

```make
# Available targets
make build                 # Build the library
make test                  # Run tests
make examples              # Run basic demo
make demos                 # Run all demos
make demo-missing-features # Run comprehensive library demo
make demo-advanced         # Run advanced features demo
make clean                 # Clean build artifacts
make format                # Format code
make test-coverage         # Run tests with coverage
make build-all             # Build for all platforms
make help                  # Show all available targets
```

**Features:**
- ✅ Multiple build targets
- ✅ Platform-specific builds
- ✅ Code formatting and linting
- ✅ Coverage reporting
- ✅ Development tools integration

## Verification Scripts

### verify_build.bat / verify_build.sh
**Comprehensive verification of all build methods**

```cmd
# Windows
.\verify_build.bat

# Linux/macOS
chmod +x verify_build.sh && ./verify_build.sh
```

**What it checks:**
- ✅ Basic Go build functionality
- ✅ Go tests execution
- ✅ Build script availability and functionality
- ✅ Demo availability
- ✅ Project structure integrity

## Quick Start Recommendations

### For New Users
**Use the primary build scripts:**
- **Windows:** `build.bat nocuda demo`
- **Linux/macOS:** `./build.sh nocuda demo`

### For Developers
**Use make for advanced features:**
- `make build && make demos`
- `make test-coverage`
- `make format`

### For CI/CD
**Use direct Go commands:**
```bash
go build ./...
go test ./...
go run demos/missing_features/main.go
```

## Build Modes Explained

### CPU Simulation Mode (`nocuda`)
- **No GPU required** - Works on any system
- **High-quality simulation** - Realistic behavior and performance modeling
- **Functional behavior coverage** - Tested operations run without requiring a GPU
- **Perfect for development** - Test CUDA code without GPU hardware
- **Default mode** - Automatically selected when CUDA not available

### CUDA Hardware Mode (`cuda`)
- **Requires NVIDIA GPU** with CUDA drivers installed
- **Windows also requires** Visual Studio Build Tools or Visual Studio with the C++ workload
- **Windows cgo linking uses** the repository's `lib_mingw` import libraries
- **Windows import-lib generation helper:** `setup_windows_cuda_import_libs.bat <cudart_dll> <cudnn_dll> [nvjpeg_dll] [nvjpeg2k_dll] [cublas_dll] [cudss_dll] [amgx_dll] [cutensor_dll] [cuda_driver_dll]`
- **Real GPU acceleration** - Direct hardware execution
- **Managed memory is used for high-level allocations** so CPU-backed helper paths can safely inspect buffers in CUDA builds
- **Some `libraries/` wrappers remain compatibility layers** backed by CPU-side helper logic on managed buffers rather than direct vendor library calls
- **Production deployment** - Real-world performance characteristics

### Windows CUDA Import Libraries

GoCUDA includes `generate_mingw_import_lib.py` to build MinGW import libraries from installed CUDA DLLs. The convenience wrapper below generates the current Windows link inputs used by the repository:

```cmd
setup_windows_cuda_import_libs.bat "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin\cudart64_13.dll" "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin\cudnn64_9.dll" "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin\nvjpeg64_13.dll" "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin\nvjpeg2k64_0.dll" "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin\cublas64_13.dll" "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin\cudss64_0.dll" "C:\amgx\bin\amgxsh.dll" "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin\cutensor64_2.dll" "C:\Windows\System32\nvcuda.dll"
```

This prepares the Windows-side `lib_mingw` inputs for CUDA runtime, cuDNN, nvJPEG, nvJPEG2000, cuBLAS-backed CUTLASS GEMM, cuDSS, AmgX, and cuTENSOR work. After `lib_mingw\libcudnn.a`, `lib_mingw\libnvjpeg.a`, `lib_mingw\libnvjpeg2k.a`, `lib_mingw\libcublas.a`, `lib_mingw\libcudss.a`, `lib_mingw\libamgxsh.a`, and `lib_mingw\libcutensor.a` are generated, CUDA-tagged Windows builds can link those native backends.

## Troubleshooting

### Common Issues

**1. "go: command not found"**
- Install Go programming language from https://golang.org/dl/

**2. "nvcc: command not found" (CUDA mode)**
- Install NVIDIA CUDA Toolkit
- Ensure `nvcc` is in your system PATH

**3. "make: command not found" (Windows)**
- Install make utility (e.g., via Chocolatey: `choco install make`)
- Or use the batch script instead: `build.bat`

### Build Script Preferences

**Windows Users:**
1. `build.bat` (recommended)
2. `make` (if installed)

**Linux/macOS Users:**
1. `./build.sh` (recommended)
2. `make` (for advanced users)

**Cross-platform:**
1. Direct Go commands (always works)
2. Verification scripts (for testing)

## Examples

### Complete Build and Test Workflow

**Windows:**
```cmd
# 1. Verify everything works
.\verify_build.bat

# 2. Build and test
.\build.bat nocuda

# 3. Run comprehensive demo
cd demos\missing_features && go run main.go

# 4. Run advanced features demo
cd ..\advanced_features && go run main.go
```

**Linux/macOS:**
```bash
# 1. Verify everything works
chmod +x verify_build.sh && ./verify_build.sh

# 2. Build and test
./build.sh nocuda

# 3. Run comprehensive demo
cd demos/missing_features && go run main.go

# 4. Run advanced features demo
cd ../advanced_features && go run main.go
```

### Advanced Development Workflow

```make
# Format code
make format

# Build with coverage
make test-coverage

# Run all demos
make demos

# Clean and rebuild
make clean && make build

# Build for all platforms
make build-all
```

## Integration with IDEs

### VS Code
Add to `.vscode/tasks.json`:
```json
{
    "version": "2.0.0",
    "tasks": [
        {
            "label": "Build GoCUDA",
            "type": "shell",
            "command": "build.bat",
            "args": ["nocuda"],
            "group": "build",
            "windows": {
                "command": "build.bat"
            },
            "linux": {
                "command": "./build.sh"
            },
            "osx": {
                "command": "./build.sh"
            }
        }
    ]
}
```

### GoLand/IntelliJ
Configure external tools to run build scripts directly.

## Summary

The GoCUDA project provides multiple build methods to accommodate different development environments and preferences:

- **Simple:** Direct Go commands
- **Convenient:** Platform-specific build scripts
- **Advanced:** Makefile with multiple targets
- **Verification:** Automated testing scripts

All methods support both CPU simulation and real CUDA hardware modes, ensuring maximum compatibility and flexibility for all users.

For most users, the recommended approach is:
1. Run the verification script to ensure everything works
2. Use the primary build script for your platform
3. Start with CPU simulation mode (`nocuda`)
4. Upgrade to CUDA hardware mode when needed

The build system is designed to be robust, user-friendly, and production-ready while maintaining compatibility across all major platforms.
