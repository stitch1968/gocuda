$ErrorActionPreference = "Stop"

# Paths found during debugging
$CudaInclude = "C:/PROGRA~1/NVIDIA~2/CUDA/v13.1/include"
$LibMingw = "$PSScriptRoot/lib_mingw"
$CudaBin = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1\bin\x64"

# Add CUDA bin to PATH for runtime DLLs
$env:PATH = "$env:PATH;$CudaBin"

# Set CGO variables for MinGW
$env:CGO_CFLAGS = "-I$CudaInclude"
$env:CGO_LDFLAGS = "-L$LibMingw -lcudart -lcuda -Wl,--no-as-needed"
$env:CGO_ENABLED = "1"

Write-Host "Compile and Run in Hardware Mode (RTX 5090)..." -ForegroundColor Cyan
go run -tags cuda demos/advanced_features/main.go
