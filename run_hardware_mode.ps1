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
$defaultBinRoots = @(
	(Join-Path $cudaPath 'bin'),
	(Join-Path $cudaPath 'bin\x64'),
	'C:\amgx\bin',
	'D:\NVIDIA\bin'
) + @(Resolve-GlobRoots -Patterns @(
	'C:\Program Files\NVIDIA\CUDNN\*\bin\*\x64',
	'C:\Program Files\NVIDIA nvJPEG2K\*\bin\*',
	'C:\Program Files\NVIDIA cuDSS\*\bin\*',
	'C:\Program Files\NVIDIA cuTensor\*\bin\*',
	'C:\Program Files\NVIDIA cuTENSOR\*\bin\*'
) -LatestOnly)
$binRoots = Resolve-SearchRoots -Candidates $defaultBinRoots -ExtraEnvVar 'GOCUDA_EXTRA_BIN_DIRS'

if (-not $SkipVerify) {
	$verifyScript = Join-Path $PSScriptRoot 'verify_windows_cuda_native_env.ps1'
	if (Test-Path $verifyScript) {
		& $verifyScript -Strict
	}
}

$originalPath = $env:PATH
if ($binRoots.Count -gt 0) {
	$env:PATH = (($binRoots -join ';') + ';' + $originalPath)
}
$env:CGO_ENABLED = '1'

Write-Host 'Windows CUDA runtime wrapper' -ForegroundColor Cyan
Write-Host ("CUDA root: {0}" -f $cudaPath)
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
}

exit $exitCode
