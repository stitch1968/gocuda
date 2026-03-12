# Windows Native SDK Setup

This checklist is for the primary Windows CUDA validation machine used to run `go test -tags cuda ./...`.

## Current Machine State

The local verifier currently reports these base components as installed:

- `cuda_runtime.h`
- `cudart*.dll`
- `cudart.lib`
- `cublas_v2.h`
- `cublas*.dll`
- `cublas.lib`
- `nvjpeg.h`
- `nvjpeg*.dll`
- `nvjpeg.lib`
- `nvcuda.dll`

The verifier currently reports these native backlog components as missing:

- `cudnn.h`
- `cudnn*.dll`
- `cudnn*.lib`
- `nvjpeg2k.h`
- `nvjpeg2k*.dll`
- `nvjpeg2k*.lib`
- `cutensor.h`
- `cutensor*.dll`
- `cutensor*.lib`
- `cudss.h`
- `cudss*.dll`
- `cudss*.lib`
- `amgxsh.dll`

## What To Install

Install these NVIDIA SDK packages on the Windows validation machine:

1. NVIDIA cuDNN for the CUDA toolkit version in use.
2. NVIDIA nvJPEG2000 SDK for the CUDA toolkit version in use.
3. NVIDIA cuTENSOR for the CUDA toolkit version in use.
4. NVIDIA cuDSS for the CUDA toolkit version in use.
5. NVIDIA AmgX if Windows AmgX native validation is required on this machine.

Use the matching CUDA-major toolkit build for each package. This machine is currently using `CUDA v13.1`, so avoid mixing headers or DLLs from a different CUDA major version.

## Where Files Can Live

The repository tooling already looks in these places:

- `%CUDA_PATH%\include`
- `%CUDA_PATH%\bin`
- `%CUDA_PATH%\bin\x64`
- `%CUDA_PATH%\lib\x64`
- `D:\NVIDIA\include`
- `D:\NVIDIA\bin`
- `D:\NVIDIA\lib\x64`
- `C:\amgx\bin`

If you install an SDK somewhere else, point the verifier and auto-helper at it with environment variables:

- `GOCUDA_EXTRA_INCLUDE_DIRS`
- `GOCUDA_EXTRA_BIN_DIRS`
- `GOCUDA_EXTRA_LIB_DIRS`

Each variable accepts a semicolon-separated list of directories.

## Install Verification Sequence

After installing or extracting each SDK, run these steps from the repo root.

1. Verify the environment:

```powershell
powershell -ExecutionPolicy Bypass -File .\verify_windows_cuda_native_env.ps1
```

2. Auto-generate import libraries for whatever DLLs are now present:

```powershell
powershell -ExecutionPolicy Bypass -File .\setup_windows_cuda_import_libs_auto.ps1
```

3. Re-run the verifier and confirm the missing items list has shrunk.

4. Retry the CUDA-tagged library build:

```powershell
go test -tags cuda ./libraries
```

5. If that succeeds, retry the broader CUDA-tagged test/build targets.

## Required End State For This Repo

Before Windows CUDA-tagged validation is expected to pass for the current native backlog, the verifier should report the following as present:

- `CUDA runtime header`
- `CUDA runtime DLL`
- `CUDA runtime import lib`
- `CUDA driver DLL`
- `cuBLAS header`
- `cuBLAS DLL`
- `cuBLAS import lib`
- `nvJPEG header`
- `nvJPEG DLL`
- `nvJPEG import lib`
- `cuDNN header`
- `cuDNN DLL`
- `cuDNN import lib`
- `nvJPEG2000 header`
- `nvJPEG2000 DLL`
- `nvJPEG2000 import lib`
- `cuTENSOR header`
- `cuTENSOR DLL`
- `cuTENSOR import lib`
- `cuDSS header`
- `cuDSS DLL`
- `cuDSS import lib`

AmgX is still optional unless Windows AmgX validation is specifically being exercised on this machine.

