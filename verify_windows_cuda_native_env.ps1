param(
    [switch]$Strict
)

$ErrorActionPreference = 'Stop'

function Resolve-SearchRoots {
    param(
        [string[]]$Candidates,
        [string]$ExtraEnvVar
    )

    $roots = New-Object System.Collections.Generic.List[string]
    foreach ($candidate in $Candidates) {
        if (-not [string]::IsNullOrWhiteSpace($candidate) -and (Test-Path $candidate)) {
            $roots.Add((Resolve-Path $candidate).Path)
        }
    }

    $extraValue = [Environment]::GetEnvironmentVariable($ExtraEnvVar)
    if (-not [string]::IsNullOrWhiteSpace($extraValue)) {
        foreach ($path in ($extraValue -split ';')) {
            if (-not [string]::IsNullOrWhiteSpace($path) -and (Test-Path $path)) {
                $roots.Add((Resolve-Path $path).Path)
            }
        }
    }

    return $roots | Select-Object -Unique
}

function Find-Match {
    param(
        [string[]]$Roots,
        [string[]]$Patterns
    )

    foreach ($root in $Roots) {
        foreach ($pattern in $Patterns) {
            if ($pattern -notmatch '[\*\?]') {
                $directPath = Join-Path $root $pattern
                if (Test-Path $directPath) {
                    return (Resolve-Path $directPath).Path
                }
            }
            $match = Get-ChildItem -Path $root -Filter $pattern -File -ErrorAction SilentlyContinue | Select-Object -First 1
            if ($null -ne $match) {
                return $match.FullName
            }
        }
    }

    return $null
}

$cudaPath = if ($env:CUDA_PATH) { $env:CUDA_PATH } else { 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1' }
$includeRoots = Resolve-SearchRoots -Candidates @(
    (Join-Path $cudaPath 'include'),
    'D:\NVIDIA\include'
) -ExtraEnvVar 'GOCUDA_EXTRA_INCLUDE_DIRS'
$binRoots = Resolve-SearchRoots -Candidates @(
    (Join-Path $cudaPath 'bin'),
    (Join-Path $cudaPath 'bin\x64'),
    'D:\NVIDIA\bin'
) -ExtraEnvVar 'GOCUDA_EXTRA_BIN_DIRS'
$libRoots = Resolve-SearchRoots -Candidates @(
    (Join-Path $cudaPath 'lib\x64'),
    (Join-Path $cudaPath 'lib64'),
    'D:\NVIDIA\lib\x64'
) -ExtraEnvVar 'GOCUDA_EXTRA_LIB_DIRS'

$checks = @(
    @{ Name = 'CUDA runtime header'; Required = $true; Type = 'header'; Patterns = @('cuda_runtime.h') },
    @{ Name = 'CUDA runtime DLL'; Required = $true; Type = 'dll'; Patterns = @('cudart*.dll') },
    @{ Name = 'cuBLAS header'; Required = $true; Type = 'header'; Patterns = @('cublas_v2.h') },
    @{ Name = 'nvJPEG header'; Required = $false; Type = 'header'; Patterns = @('nvjpeg.h') },
    @{ Name = 'cuDNN header'; Required = $false; Type = 'header'; Patterns = @('cudnn.h') },
    @{ Name = 'nvJPEG2000 header'; Required = $false; Type = 'header'; Patterns = @('nvjpeg2k.h') },
    @{ Name = 'cuTENSOR header'; Required = $false; Type = 'header'; Patterns = @('cutensor.h') },
    @{ Name = 'cuDSS header'; Required = $false; Type = 'header'; Patterns = @('cudss.h') },
    @{ Name = 'CUDA driver DLL'; Required = $true; Type = 'dll'; Patterns = @('nvcuda.dll') },
    @{ Name = 'cuBLAS DLL'; Required = $false; Type = 'dll'; Patterns = @('cublas*.dll') },
    @{ Name = 'nvJPEG DLL'; Required = $false; Type = 'dll'; Patterns = @('nvjpeg*.dll') },
    @{ Name = 'cuDNN DLL'; Required = $false; Type = 'dll'; Patterns = @('cudnn*.dll') },
    @{ Name = 'nvJPEG2000 DLL'; Required = $false; Type = 'dll'; Patterns = @('nvjpeg2k*.dll') },
    @{ Name = 'cuTENSOR DLL'; Required = $false; Type = 'dll'; Patterns = @('cutensor*.dll') },
    @{ Name = 'cuDSS DLL'; Required = $false; Type = 'dll'; Patterns = @('cudss*.dll') },
    @{ Name = 'cuBLAS import lib'; Required = $true; Type = 'lib'; Patterns = @('cublas.lib') },
    @{ Name = 'CUDA runtime import lib'; Required = $true; Type = 'lib'; Patterns = @('cudart.lib') },
    @{ Name = 'nvJPEG import lib'; Required = $false; Type = 'lib'; Patterns = @('nvjpeg.lib') },
    @{ Name = 'cuDNN import lib'; Required = $false; Type = 'lib'; Patterns = @('cudnn*.lib') },
    @{ Name = 'nvJPEG2000 import lib'; Required = $false; Type = 'lib'; Patterns = @('nvjpeg2k*.lib') },
    @{ Name = 'cuTENSOR import lib'; Required = $false; Type = 'lib'; Patterns = @('cutensor*.lib') },
    @{ Name = 'cuDSS import lib'; Required = $false; Type = 'lib'; Patterns = @('cudss*.lib') }
)

$results = foreach ($check in $checks) {
    $roots = switch ($check.Type) {
        'header' { @($includeRoots) }
        'dll' { @($binRoots) + @('C:\Windows\System32', 'C:\Windows\Sysnative') }
        'lib' { @($libRoots) }
    }

    $found = Find-Match -Roots ($roots | Select-Object -Unique) -Patterns $check.Patterns
    [PSCustomObject]@{
        Name = $check.Name
        Required = $check.Required
        Type = $check.Type
        Found = [bool]$found
        Path = $found
    }
}

Write-Host 'GoCUDA Windows Native CUDA Environment Check'
Write-Host '=========================================='
Write-Host "CUDA root: $cudaPath"
Write-Host "Include roots: $($includeRoots -join '; ')"
Write-Host "Bin roots: $($binRoots -join '; ')"
Write-Host "Lib roots: $($libRoots -join '; ')"
Write-Host ''

foreach ($result in $results) {
    $prefix = if ($result.Found) { '[ok]' } elseif ($result.Required) { '[missing-required]' } else { '[missing-optional]' }
    Write-Host ("{0} {1}" -f $prefix, $result.Name)
    if ($result.Path) {
        Write-Host ("       {0}" -f $result.Path)
    }
}

$missingRequired = $results | Where-Object { -not $_.Found -and $_.Required }
$missingOptional = $results | Where-Object { -not $_.Found -and -not $_.Required }

Write-Host ''
if ($missingOptional.Count -gt 0) {
    Write-Host 'Missing optional native backlog dependencies:'
    foreach ($result in $missingOptional) {
        Write-Host ("  - {0}" -f $result.Name)
    }
}

$helperArgs = @()
foreach ($name in 'CUDA runtime DLL','cuDNN DLL','nvJPEG DLL','nvJPEG2000 DLL','cuBLAS DLL','cuDSS DLL','cuTENSOR DLL') {
    $match = $results | Where-Object { $_.Name -eq $name } | Select-Object -First 1
    if ($match -and $match.Path -and $match.Type -eq 'dll') {
        $helperArgs += ('"{0}"' -f $match.Path)
    }
}
$driverMatch = $results | Where-Object { $_.Name -eq 'CUDA driver DLL' } | Select-Object -First 1
if ($driverMatch -and $driverMatch.Path) {
    $helperArgs += ('"{0}"' -f $driverMatch.Path)
}

Write-Host ''
Write-Host 'Suggested next steps:'
Write-Host '  1. Install the missing optional vendor SDKs whose headers are absent.'
Write-Host '  2. Re-run this script after installation.'
if ($helperArgs.Count -ge 2) {
    Write-Host ("  3. Regenerate MinGW import libs with: setup_windows_cuda_import_libs.bat {0}" -f ($helperArgs -join ' '))
}
Write-Host '  4. Or auto-detect available DLLs with: powershell -ExecutionPolicy Bypass -File setup_windows_cuda_import_libs_auto.ps1'

if ($Strict -and $missingRequired.Count -gt 0) {
    exit 1
}

exit 0