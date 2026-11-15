# Run the full board set a few times and compare speeds.
# Build minesweep_enumerate.exe first. Each run also checks the counts.
param([int]$Repeats = 5)
$ErrorActionPreference = 'Stop'
Push-Location $PSScriptRoot
try {
    $expected = '2606835066264,3286482376320,1718041685580,482808902400,77911023960,7123152960,343783440,7555680,53460'
    $rates = @()
    for ($i = 1; $i -le $Repeats; $i++) {
        $output = & '.\minesweep_enumerate.exe' 2>&1
        if ($LASTEXITCODE -ne 0) { throw "Simulator failed on run $i with exit code $LASTEXITCODE" }
        $joined = $output -join "`n"
        $counts = @([regex]::Matches($joined, '(?m)^\s+[0-8]: (\d+) appearances') | ForEach-Object { $_.Groups[1].Value })
        if (($counts -join ',') -ne $expected) { throw "histogram mismatch on run $i" }
        $speed = [double]([regex]::Match($joined, 'Speed: ~(\d+) boards/s').Groups[1].Value)
        if ($speed -le 0) { throw "Could not read throughput on run $i" }
        $rates += $speed / 1e9
        Write-Host ('Run {0}: {1:N3} billion boards/s (histogram verified)' -f $i, $rates[-1])
    }
    $sorted = @($rates | Sort-Object)
    $middle = [int][math]::Floor($sorted.Count / 2)
    if ($sorted.Count % 2) { $median = $sorted[$middle] }
    else { $median = ($sorted[$middle - 1] + $sorted[$middle]) / 2 }
    Write-Host ('Median: {0:N3} billion boards/s; range {1:N3}-{2:N3}' -f $median,$sorted[0],$sorted[-1])
} finally { Pop-Location }
