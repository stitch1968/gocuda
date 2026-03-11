@echo off
REM Generate MinGW import libraries for CUDA runtime DLLs used by GoCUDA.
REM Usage:
REM   setup_windows_cuda_import_libs.bat <cudart_dll> <cudnn_dll> [nvjpeg_dll] [cuda_driver_dll]
REM Example:
REM   setup_windows_cuda_import_libs.bat "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin\cudart64_13.dll" "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin\cudnn64_9.dll" "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin\nvjpeg64_13.dll" "C:\Windows\System32\nvcuda.dll"

setlocal

if "%~1"=="" goto :usage
if "%~2"=="" goto :usage

set SCRIPT_DIR=%~dp0
set CUDART_DLL=%~1
set CUDNN_DLL=%~2
set NVJPEG_DLL=
set CUDA_DLL=

if not "%~3"=="" (
    echo %~nx3 | findstr /i "nvjpeg" >nul
    if errorlevel 1 (
        set CUDA_DLL=%~3
    ) else (
        set NVJPEG_DLL=%~3
        set CUDA_DLL=%~4
    )
)

if "%CUDA_DLL%"=="" set CUDA_DLL=%SystemRoot%\System32\nvcuda.dll

python --version >nul 2>&1
if errorlevel 1 (
    echo Python is required to generate import libraries.
    exit /b 1
)

echo Generating import library for %CUDART_DLL%
python "%SCRIPT_DIR%generate_mingw_import_lib.py" "%CUDART_DLL%"
if errorlevel 1 exit /b 1

echo Generating import library for %CUDNN_DLL%
python "%SCRIPT_DIR%generate_mingw_import_lib.py" "%CUDNN_DLL%"
if errorlevel 1 exit /b 1

if not "%NVJPEG_DLL%"=="" (
    echo Generating import library for %NVJPEG_DLL%
    python "%SCRIPT_DIR%generate_mingw_import_lib.py" "%NVJPEG_DLL%"
    if errorlevel 1 exit /b 1
)

if exist "%CUDA_DLL%" (
    echo Generating import library for %CUDA_DLL%
    python "%SCRIPT_DIR%generate_mingw_import_lib.py" "%CUDA_DLL%"
    if errorlevel 1 exit /b 1
) else (
    echo Skipping CUDA driver DLL because it was not found: %CUDA_DLL%
)

echo.
echo Generated import libraries are available in lib_mingw.
echo Current Windows native backends may require generated import libraries such as libcudnn.a and libnvjpeg.a before CUDA-tagged builds can link successfully.
exit /b 0

:usage
echo Usage: %~n0 ^<cudart_dll^> ^<cudnn_dll^> [nvjpeg_dll] [cuda_driver_dll]
echo Example:
echo   %~n0 "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin\cudart64_13.dll" "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin\cudnn64_9.dll" "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin\nvjpeg64_13.dll" "C:\Windows\System32\nvcuda.dll"
exit /b 1