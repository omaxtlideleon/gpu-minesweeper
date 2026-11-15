param(
    [string]$Source = 'minesweep_enumerate.cu',
    [string]$Output = 'minesweep_enumerate.exe',
    [string]$Arch = 'sm_89',
    [string[]]$ExtraFlags = @()
)
$ErrorActionPreference = 'Stop'
Push-Location $PSScriptRoot
try {
    if (-not (Get-Command nvcc -ErrorAction SilentlyContinue)) {
        throw 'nvcc was not found. Make sure the CUDA Toolkit is on PATH.'
    }
    & nvcc -std=c++17 -O3 "-arch=$Arch" -lineinfo -Xptxas=-v @ExtraFlags $Source -o $Output
    if ($LASTEXITCODE -ne 0) { throw "nvcc failed: $LASTEXITCODE" }
} finally { Pop-Location }
