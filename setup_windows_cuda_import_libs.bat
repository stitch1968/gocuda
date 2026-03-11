@echo off
REM Generate MinGW import libraries for CUDA runtime DLLs used by GoCUDA.
REM Usage:
REM   setup_windows_cuda_import_libs.bat <cudart_dll> <cudnn_dll> [nvjpeg_dll] [nvjpeg2k_dll] [cublas_dll] [cudss_dll] [amgx_dll] [cutensor_dll] [cuda_driver_dll]
REM Example:
REM   setup_windows_cuda_import_libs.bat "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin\x64\cudart64_13.dll" "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin\x64\cudnn64_9.dll" "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin\x64\nvjpeg64_13.dll" "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin\x64\nvjpeg2k64_0.dll" "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin\x64\cublas64_13.dll" "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin\x64\cudss64_0.dll" "C:\amgx\bin\amgxsh.dll" "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin\x64\cutensor64_2.dll" "C:\Windows\System32\nvcuda.dll"

setlocal
setlocal EnableDelayedExpansion

if "%~1"=="" goto :usage
if "%~2"=="" goto :usage

set SCRIPT_DIR=%~dp0
set CUDART_DLL=%~1
set CUDNN_DLL=%~2
set NVJPEG_DLL=
set NVJPEG2K_DLL=
set CUBLAS_DLL=
set CUDSS_DLL=
set AMGX_DLL=
set CUTENSOR_DLL=
set CUDA_DLL=

for %%F in (%3 %4 %5 %6 %7 %8 %9) do (
    if not "%%~F"=="" (
        echo %%~nxF | findstr /i "nvjpeg2k" >nul
        if not errorlevel 1 set NVJPEG2K_DLL=%%~F
        echo %%~nxF | findstr /i "nvjpeg" >nul
        if not errorlevel 1 if /i not "%%~F"=="!NVJPEG2K_DLL!" set NVJPEG_DLL=%%~F
        echo %%~nxF | findstr /i "cublas" >nul
        if not errorlevel 1 set CUBLAS_DLL=%%~F
        echo %%~nxF | findstr /i "cudss" >nul
        if not errorlevel 1 set CUDSS_DLL=%%~F
        echo %%~nxF | findstr /i "amgx" >nul
        if not errorlevel 1 set AMGX_DLL=%%~F
        echo %%~nxF | findstr /i "cutensor" >nul
        if not errorlevel 1 set CUTENSOR_DLL=%%~F
        echo %%~nxF | findstr /i "nvcuda" >nul
        if not errorlevel 1 set CUDA_DLL=%%~F
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

if not "%NVJPEG2K_DLL%"=="" (
    echo Generating import library for %NVJPEG2K_DLL%
    python "%SCRIPT_DIR%generate_mingw_import_lib.py" "%NVJPEG2K_DLL%"
    if errorlevel 1 exit /b 1
)

if not "%CUBLAS_DLL%"=="" (
    echo Generating import library for %CUBLAS_DLL%
    python "%SCRIPT_DIR%generate_mingw_import_lib.py" "%CUBLAS_DLL%"
    if errorlevel 1 exit /b 1
)

if not "%CUDSS_DLL%"=="" (
    echo Generating import library for %CUDSS_DLL%
    python "%SCRIPT_DIR%generate_mingw_import_lib.py" "%CUDSS_DLL%"
    if errorlevel 1 exit /b 1
)

if not "%AMGX_DLL%"=="" (
    echo Generating import library for %AMGX_DLL%
    python "%SCRIPT_DIR%generate_mingw_import_lib.py" "%AMGX_DLL%"
    if errorlevel 1 exit /b 1
)

if not "%CUTENSOR_DLL%"=="" (
    echo Generating import library for %CUTENSOR_DLL%
    python "%SCRIPT_DIR%generate_mingw_import_lib.py" "%CUTENSOR_DLL%"
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
echo Current Windows native backends may require generated import libraries such as libcudnn.a, libnvjpeg.a, libnvjpeg2k.a, libcublas.a, libcudss.a, libamgxsh.a, and libcutensor.a before CUDA-tagged builds can link successfully.
exit /b 0

:usage
echo Usage: %~n0 ^<cudart_dll^> ^<cudnn_dll^> [nvjpeg_dll] [nvjpeg2k_dll] [cublas_dll] [cudss_dll] [amgx_dll] [cutensor_dll] [cuda_driver_dll]
echo Example:
echo   %~n0 "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin\x64\cudart64_13.dll" "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin\x64\cudnn64_9.dll" "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin\x64\nvjpeg64_13.dll" "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin\x64\nvjpeg2k64_0.dll" "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin\x64\cublas64_13.dll" "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin\x64\cudss64_0.dll" "C:\amgx\bin\amgxsh.dll" "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin\x64\cutensor64_2.dll" "C:\Windows\System32\nvcuda.dll"
exit /b 1