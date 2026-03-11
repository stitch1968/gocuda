@echo off
REM Build script for gocuda library on Windows
REM Usage: build.bat [cuda|nocuda] [demo]

setlocal

set MODE=%1
if "%MODE%"=="" set MODE=nocuda

set RUN_DEMO=%2

echo Building gocuda library...

if "%MODE%"=="cuda" (
    echo Building with CUDA support...
    echo Requirements:
    echo   - NVIDIA CUDA Toolkit installed
    echo   - CGO enabled
    echo.
    
    REM Check if CUDA is available
    nvcc --version >nul 2>&1
    if errorlevel 1 (
        echo X CUDA compiler ^(nvcc^) not found
        echo Please install NVIDIA CUDA Toolkit
        exit /b 1
    ) else (
        echo √ CUDA compiler found
        nvcc --version | findstr "release"
    )
    
    REM Build with CUDA support
    echo Building with CUDA tags...
    set CGO_ENABLED=1
    if not exist "lib_mingw\libcudnn.a" (
        echo WARNING: lib_mingw\libcudnn.a not found
        echo Run setup_windows_cuda_import_libs.bat with your local cudart/cudnn DLL paths before expecting Windows cuDNN native builds to link.
    )
    if not exist "lib_mingw\libnvjpeg.a" (
        echo WARNING: lib_mingw\libnvjpeg.a not found
        echo Run setup_windows_cuda_import_libs.bat with your local cudart/cudnn/nvjpeg DLL paths before expecting Windows nvJPEG native builds to link.
    )
    if not exist "lib_mingw\libnvjpeg2k.a" (
        echo WARNING: lib_mingw\libnvjpeg2k.a not found
        echo Run setup_windows_cuda_import_libs.bat with your local nvjpeg2k DLL path before expecting Windows nvJPEG2000 native builds to link.
    )
    if not exist "lib_mingw\libcublas.a" (
        echo WARNING: lib_mingw\libcublas.a not found
        echo Run setup_windows_cuda_import_libs.bat with your local cublas DLL path before expecting Windows CUTLASS native GEMM builds to link.
    )
    if not exist "lib_mingw\libcudss.a" (
        echo WARNING: lib_mingw\libcudss.a not found
        echo Run setup_windows_cuda_import_libs.bat with your local cudart/cudnn/cudss DLL paths before expecting Windows cuDSS native builds to link.
    )
    if not exist "lib_mingw\libamgxsh.a" (
        echo WARNING: lib_mingw\libamgxsh.a not found
        echo Run setup_windows_cuda_import_libs.bat with your local amgxsh.dll path before expecting Windows AmgX native builds to link.
    )
    if not exist "lib_mingw\libcutensor.a" (
        echo WARNING: lib_mingw\libcutensor.a not found
        echo Run setup_windows_cuda_import_libs.bat with your local cutensor DLL path before expecting Windows cuTENSOR native builds to link.
    )
    go build -tags cuda -v ./...
    if errorlevel 1 (
        echo Build failed!
        exit /b 1
    )
    
    REM Run tests with CUDA
    echo Running tests with CUDA support...
    set CGO_ENABLED=1
    go test -tags cuda -v ./...
    
) else if "%MODE%"=="nocuda" (
    echo Building without CUDA support ^(simulation mode only^)...
    echo This mode works on any system without CUDA installation.
    echo.
    
    REM Build without CUDA support
    echo Building with simulation-only mode...
    go build -v ./...
    if errorlevel 1 (
        echo Build failed!
        exit /b 1
    )
    
    REM Run tests in simulation mode
    echo Running tests in simulation mode...
    go test -v ./...
    
) else (
    echo Usage: %0 [cuda^|nocuda] [demo]
    echo.
    echo Options:
    echo   cuda    - Build with real CUDA support ^(requires CUDA toolkit^)
    echo   nocuda  - Build with CPU simulation only ^(default^)
    echo   demo    - Run comprehensive demo after successful build
    exit /b 1
)

echo.
echo Build completed successfully!
echo.

REM Run demo if requested
if "%RUN_DEMO%"=="demo" (
    echo Running comprehensive demo...
    echo.
    cd demos\missing_features && go run main.go
    echo.
)

REM Show build info
echo === Build Information ===
echo Mode: %MODE%
go version
if "%MODE%"=="cuda" (
    echo CUDA version: 
    nvcc --version | findstr "release" 2>nul || echo Unknown
    echo CGO: Enabled
) else (
    echo CUDA: Not required
    echo CGO: Disabled
)
echo =========================

endlocal
