param(
	[string[]]$Command = @('go', 'test', '-tags', 'cuda', './libraries'),
	[switch]$SkipVerify
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

function Resolve-GlobRoots {
	param(
		[string[]]$Patterns,
		[switch]$LatestOnly
	)

	$roots = New-Object System.Collections.Generic.List[string]
	foreach ($pattern in $Patterns) {
		if ([string]::IsNullOrWhiteSpace($pattern)) {
			continue
		}

		$matches = @(Get-ChildItem -Path $pattern -Directory -ErrorAction SilentlyContinue | Sort-Object FullName -Descending)
		if ($LatestOnly) {
			$matches = @($matches | Select-Object -First 1)
		}

		foreach ($match in $matches) {
			$roots.Add($match.FullName)
		}
	}

	return $roots | Select-Object -Unique
}

$cudaPath = if ($env:CUDA_PATH) { $env:CUDA_PATH } else { 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1' }
$cudaHeader = Join-Path (Join-Path $cudaPath 'include') 'cuda.h'
$cudaMajorSelector = '*'
if (Test-Path $cudaHeader) {
	$versionLine = Select-String -Path $cudaHeader -Pattern '^\s*#define\s+CUDA_VERSION\s+(\d+)' | Select-Object -First 1
	if ($null -ne $versionLine) {
		$cudaVersion = [int]$versionLine.Matches[0].Groups[1].Value
		$cudaMajorSelector = '{0}*' -f [int][math]::Floor($cudaVersion / 1000)
	}
}
$defaultIncludeRoots = @(
	(Join-Path $cudaPath 'include'),
	'C:\amgx\include',
	'D:\NVIDIA\include'
) + @(Resolve-GlobRoots -Patterns @(
	"C:\Program Files\NVIDIA\CUDNN\*\include\$cudaMajorSelector",
	'C:\Program Files\NVIDIA nvJPEG2K\*\include',
	'C:\Program Files\NVIDIA cuDSS\*\include',
	'C:\Program Files\NVIDIA cuTensor\*\include',
	'C:\Program Files\NVIDIA cuTENSOR\*\include'
) -LatestOnly)
$includeRoots = @(Resolve-SearchRoots -Candidates $defaultIncludeRoots -ExtraEnvVar 'GOCUDA_EXTRA_INCLUDE_DIRS')
$defaultBinRoots = @(
	(Join-Path $cudaPath 'bin'),
	(Join-Path $cudaPath 'bin\x64'),
	'C:\amgx\bin',
	'D:\NVIDIA\bin'
) + @(Resolve-GlobRoots -Patterns @(
	"C:\Program Files\NVIDIA\CUDNN\*\bin\$cudaMajorSelector\x64",
	"C:\Program Files\NVIDIA nvJPEG2K\*\bin\$cudaMajorSelector",
	"C:\Program Files\NVIDIA cuDSS\*\bin\$cudaMajorSelector",
	"C:\Program Files\NVIDIA cuTensor\*\bin\$cudaMajorSelector",
	"C:\Program Files\NVIDIA cuTENSOR\*\bin\$cudaMajorSelector"
) -LatestOnly)
$binRoots = Resolve-SearchRoots -Candidates $defaultBinRoots -ExtraEnvVar 'GOCUDA_EXTRA_BIN_DIRS'

if (-not $SkipVerify) {
	$verifyScript = Join-Path $PSScriptRoot 'verify_windows_cuda_native_env.ps1'
	if (Test-Path $verifyScript) {
		& $verifyScript -Strict
	}
}

$originalPath = $env:PATH
$originalCgoEnabled = $env:CGO_ENABLED
$originalCgoCflags = $env:CGO_CFLAGS
$baseCgoCflags = if ([string]::IsNullOrWhiteSpace($originalCgoCflags)) { (& go env CGO_CFLAGS) } else { $originalCgoCflags }
if ($binRoots.Count -gt 0) {
	$env:PATH = (($binRoots -join ';') + ';' + $originalPath)
}
if ($includeRoots.Count -gt 0) {
	$includeFlags = @($includeRoots | ForEach-Object { '"-I' + $_ + '"' })
	$env:CGO_CFLAGS = (($includeFlags -join ' ') + ' ' + $baseCgoCflags).Trim()
}
$env:CGO_ENABLED = '1'

Write-Host 'Windows CUDA runtime wrapper' -ForegroundColor Cyan
Write-Host ("CUDA root: {0}" -f $cudaPath)
Write-Host ("Prepended include roots: {0}" -f ($includeRoots -join '; '))
Write-Host ("Prepended runtime DLL roots: {0}" -f ($binRoots -join '; '))
Write-Host ("Command: {0}" -f ($Command -join ' '))

Push-Location $PSScriptRoot
try {
	if ($Command.Count -eq 0) {
		throw 'No command specified.'
	}

	$commandName = $Command[0]
	$commandArgs = @()
	if ($Command.Count -gt 1) {
		$commandArgs = $Command[1..($Command.Count - 1)]
	}

	& $commandName @commandArgs
	$exitCode = $LASTEXITCODE
	if ($null -eq $exitCode) {
		$exitCode = 0
	}
}
finally {
	Pop-Location
	$env:PATH = $originalPath
	$env:CGO_ENABLED = $originalCgoEnabled
	$env:CGO_CFLAGS = $originalCgoCflags
}

exit $exitCode
