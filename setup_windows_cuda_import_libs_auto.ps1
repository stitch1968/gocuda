param(
    [switch]$Strict
)

$ErrorActionPreference = 'Stop'

function Resolve-SearchRoots {
    param(
        [string[]]$Candidates,
        [string]$ExtraEnvVar,
        [switch]$IncludePathEnv
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

    if ($IncludePathEnv) {
        $pathValue = $env:PATH
        if ($null -eq $pathValue) {
            $pathValue = ''
        }
        foreach ($path in ($pathValue -split ';')) {
            if (-not [string]::IsNullOrWhiteSpace($path) -and (Test-Path $path)) {
                $roots.Add((Resolve-Path $path).Path)
            }
        }
    }

    return $roots | Select-Object -Unique
}

function Resolve-GlobRoots {
    param(
        [string[]]$Patterns
    )

    $roots = New-Object System.Collections.Generic.List[string]
    foreach ($pattern in $Patterns) {
        if ([string]::IsNullOrWhiteSpace($pattern)) {
            continue
        }

        foreach ($match in @(Get-ChildItem -Path $pattern -Directory -ErrorAction SilentlyContinue)) {
            $roots.Add($match.FullName)
        }
    }

    return $roots | Select-Object -Unique
}

function Find-Dll {
    param(
        [string[]]$Roots,
        [string[]]$Patterns,
        [string]$ExcludeRegex
    )

    foreach ($root in @($Roots)) {
        foreach ($pattern in $Patterns) {
            $matches = Get-ChildItem -Path $root -Filter $pattern -File -ErrorAction SilentlyContinue | Sort-Object Name
            foreach ($match in $matches) {
                if ($ExcludeRegex -and $match.Name -match $ExcludeRegex) {
                    continue
                }
                return $match.FullName
            }
        }
    }

    return $null
}

function Invoke-ImportLibGeneration {
    param(
        [string]$PythonExe,
        [string]$Generator,
        [string]$DllPath,
        [string]$ImportLibName
    )

    Write-Host ("Generating import library for {0}" -f $DllPath)
    $arguments = @($Generator, $DllPath)
    if (-not [string]::IsNullOrWhiteSpace($ImportLibName)) {
        $arguments += @('--import-lib-name', $ImportLibName)
    }
    & $PythonExe @arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Import-lib generation failed for $DllPath"
    }
}

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$generator = Join-Path $scriptDir 'generate_mingw_import_lib.py'
if (-not (Test-Path $generator)) {
    throw "Missing generator script: $generator"
}

$pythonCommand = Get-Command python -ErrorAction SilentlyContinue | Select-Object -First 1
$pythonExe = if ($pythonCommand) { $pythonCommand.Source } else { $null }
if (-not $pythonExe) {
    throw 'Python is required to generate import libraries.'
}

$cudaPath = if ($env:CUDA_PATH) { $env:CUDA_PATH } else { 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1' }
$defaultBinRoots = @(
    (Join-Path $cudaPath 'bin'),
    (Join-Path $cudaPath 'bin\x64'),
    'D:\NVIDIA\bin',
    'C:\amgx\bin'
) + @(Resolve-GlobRoots -Patterns @(
    'C:\Program Files\NVIDIA\CUDNN\*\bin\*\x64',
    'C:\Program Files\NVIDIA nvJPEG2K\*\bin\*',
    'C:\Program Files\NVIDIA cuDSS\*\bin\*',
    'C:\Program Files\NVIDIA cuTensor\*\bin\*',
    'C:\Program Files\NVIDIA cuTENSOR\*\bin\*'
))
$binRoots = Resolve-SearchRoots -Candidates $defaultBinRoots -ExtraEnvVar 'GOCUDA_EXTRA_BIN_DIRS' -IncludePathEnv

$targets = @(
    @{ Name = 'cudart'; Required = $true; ImportLibName = 'cudart'; Patterns = @('cudart*.dll') },
    @{ Name = 'cudnn'; Required = $false; ImportLibName = 'cudnn'; Patterns = @('cudnn64_*.dll', 'cudnn*.dll') },
    @{ Name = 'nvjpeg2k'; Required = $false; ImportLibName = 'nvjpeg2k'; Patterns = @('nvjpeg2k*.dll') },
    @{ Name = 'nvjpeg'; Required = $false; ImportLibName = 'nvjpeg'; Patterns = @('nvjpeg*.dll'); ExcludeRegex = 'nvjpeg2k' },
    @{ Name = 'cublas'; Required = $false; ImportLibName = 'cublas'; Patterns = @('cublas*.dll'); ExcludeRegex = 'cublasLt' },
    @{ Name = 'cudss'; Required = $false; ImportLibName = 'cudss'; Patterns = @('cudss64_*.dll', 'cudss*.dll') },
    @{ Name = 'amgxsh'; Required = $false; ImportLibName = 'amgxsh'; Patterns = @('amgxsh.dll') },
    @{ Name = 'cutensor'; Required = $false; ImportLibName = 'cutensor'; Patterns = @('cutensor*.dll') },
    @{ Name = 'nvcuda'; Required = $true; ImportLibName = 'nvcuda'; Patterns = @('nvcuda.dll') }
)

$missingRequired = New-Object System.Collections.Generic.List[string]
$generated = New-Object System.Collections.Generic.List[string]
$skipped = New-Object System.Collections.Generic.List[string]

Write-Host 'GoCUDA Windows Auto Import-Lib Generation'
Write-Host '========================================='
Write-Host ("Search roots: {0}" -f ($binRoots -join '; '))
Write-Host ''

foreach ($target in $targets) {
    $match = Find-Dll -Roots $binRoots -Patterns $target.Patterns -ExcludeRegex $target.ExcludeRegex
    if ($match) {
        Invoke-ImportLibGeneration -PythonExe $pythonExe -Generator $generator -DllPath $match -ImportLibName $target.ImportLibName
        $generated.Add($match)
    } else {
        if ($target.Required) {
            $missingRequired.Add($target.Name)
        } else {
            $skipped.Add($target.Name)
        }
    }
}

Write-Host ''
Write-Host 'Generation summary:'
foreach ($item in $generated) {
    Write-Host ("  [generated] {0}" -f $item)
}
foreach ($item in $skipped) {
    Write-Host ("  [skipped-optional] {0}" -f $item)
}
foreach ($item in $missingRequired) {
    Write-Host ("  [missing-required] {0}" -f $item)
}

Write-Host ''
if ($missingRequired.Count -eq 0) {
    Write-Host 'Environment status: required import libraries generated.'
    if ($skipped.Count -gt 0) {
        Write-Host ("Optional DLLs not found: {0}" -f ($skipped -join ', '))
    }
    Write-Host 'Suggested next steps:'
    Write-Host '  1. Run powershell -ExecutionPolicy Bypass -File verify_windows_cuda_native_env.ps1'
    Write-Host '  2. Retry go test -tags cuda ./libraries after the required headers and import libs are in place'
} else {
    Write-Host 'Next steps:'
    Write-Host '  1. Run powershell -ExecutionPolicy Bypass -File verify_windows_cuda_native_env.ps1'
    Write-Host '  2. Install any missing vendor SDKs still reported as optional gaps'
    Write-Host '  3. Retry go test -tags cuda ./libraries after the required headers and import libs are in place'
}

if ($Strict -and $missingRequired.Count -gt 0) {
    exit 1
}

exit 0