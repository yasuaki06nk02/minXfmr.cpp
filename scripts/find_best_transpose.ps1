# Short script to try all combinations of transpose flags (wq,wk,wv,wo)
# Usage: .\find_best_transpose.ps1 -ModelPath .\model.gguf -CliPath .\build\src\Release\minxfmr_cli.exe -Prompt "hello"
param(
    [string]$ModelPath,
    [string]$CliPath = ".\build\src\Release\minxfmr_cli.exe",
    [string]$Prompt = "hello",
    [int]$StartMask = 0,
    [int]$MaxMasks = 4,
    [int]$TimeoutSec = 60,
    [int]$MaxGenTokens = 8,
    [int]$CpuThreads = 0,
    [switch]$AllowHighLoad,
    [switch]$DryRun,
    [string]$LogDir = ".\transpose_search_logs"
)

if (-not (Test-Path $LogDir)) { New-Item -ItemType Directory -Path $LogDir | Out-Null }

# Delegate to Python helper that implements robust scoring and JSON-mode. This wrapper enforces safe defaults
# and forwards arguments to the Python script. If Python is not present, fall back to the simple PS loop.

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$py = Join-Path $scriptDir ".\.venv\Scripts\python.exe"
if (-not (Test-Path $py)) { $py = "python" }

$pyScript = Join-Path $scriptDir "find_best_transpose.py"
if (Test-Path $pyScript) {
    $argsList = @()
    $argsList += "`"$ModelPath`""
    $argsList += "`"$CliPath`""
    $argsList += "`"$Prompt`""
    $argsList += "--start-mask"; $argsList += "$StartMask"
    $argsList += "--max-masks"; $argsList += "$MaxMasks"
    $argsList += "--timeout-sec"; $argsList += "$TimeoutSec"
    $argsList += "--max-gen-tokens"; $argsList += "$MaxGenTokens"
    if ($CpuThreads -gt 0) { $argsList += "--cpu-threads"; $argsList += "$CpuThreads" }
    if ($AllowHighLoad) { $argsList += "--allow-high-load" }
    if ($DryRun) { $argsList += "--dry-run" }
    $argsList += "--emit-json"
    $argsLine = $argsList -join ' '
    Write-Host "Invoking Python helper: $py $argsLine"
    & $py $pyScript @argsList
    exit $LASTEXITCODE
}

# Fallback: simple safe loop if Python helper not available
$best = @{score= -1; combo = $null; log = $null}
$flags = @("wq","wk","wv","wo")
$safeMax = [math]::Min(16 - $StartMask, [math]::Max(1, $MaxMasks))
if ($safeMax -gt 4 -and -not $AllowHighLoad) {
    Write-Host "[safety] max-masks > 4 requires -AllowHighLoad. Clamping to 4."
    $safeMax = 4
}

for ($mask = $StartMask; $mask -lt ($StartMask + $safeMax); $mask++) {
    $args = @("$ModelPath","--temp", "0", "--top_k", "1", "--max-gen-tokens", "$MaxGenTokens", "--prompt", "$Prompt")
    for ($i=0;$i -lt 4;$i++) {
        $name = $flags[$i]
        if (($mask -band (1 -shl $i)) -ne 0) { $args += "--transpose-$name" } else { $args += "--no-transpose-$name" }
    }
    $logfile = Join-Path $LogDir ("result_{0}.log" -f $mask)

    Write-Host "Running mask=$mask -> $($args -join ' ')"
    if ($DryRun) { Write-Host "dry-run: would write $logfile"; continue }

    $psi = New-Object System.Diagnostics.ProcessStartInfo
    $psi.FileName = $CliPath
    $psi.Arguments = $args -join ' '
    $psi.RedirectStandardOutput = $true
    $psi.RedirectStandardError = $true
    $psi.UseShellExecute = $false
    $proc = New-Object System.Diagnostics.Process
    $proc.StartInfo = $psi
    $proc.Start() | Out-Null
    $finished = $proc.WaitForExit($TimeoutSec * 1000)
    if (-not $finished) {
        Write-Host "[timeout] mask=$mask exceeded ${TimeoutSec}s, killing process"
        try { $proc.Kill() } catch {}
    }
    $out = $proc.StandardOutput.ReadToEnd()
    $err = $proc.StandardError.ReadToEnd()
    $full = "[STDOUT]`n" + $out + "`n[STDERR]`n" + $err
    Set-Content -Path $logfile -Value $full -Encoding UTF8

    $assist = ($out | Select-String "assistant" -SimpleMatch) -join "`n"
    $letters = 0
    foreach ($l in ($assist -split "`n")) { $letters += (($l -replace '[^A-Za-z]','').Length) }
    if ($letters -gt $best.score) {
        $best.score = $letters
        $best.combo = $args -join ' '
        $best.log = $logfile
    }
    Write-Host "mask=$mask letters=$letters sample:`n$assist`n"
}

Write-Host "Best score=$($best.score)`nBest combo: $($best.combo)`nLog: $($best.log)"
