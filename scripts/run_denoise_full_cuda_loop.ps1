param([string[]]$Methods = @('all7'))

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$work = Join-Path $root "outputs\denoise_full_cuda"
if (!(Test-Path -LiteralPath $work)) { throw "Benchmark work directory is missing" }
if ((Test-Path -LiteralPath (Join-Path $work "STOP")) -or (Test-Path -LiteralPath (Join-Path $work "BLOCKED.json"))) {
    throw "Benchmark is paused. Resolve STOP/BLOCKED state before resuming; no job was launched."
}
$python = (Get-Command python -ErrorAction Stop).Source
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$out = Join-Path $work "background_$stamp.log"
$err = Join-Path $work "background_$stamp.err.log"
$runner = Join-Path $root "scripts\run_denoise_full_cuda.py"
$arguments = @('"' + $runner + '"', '--execute', '--methods') + $Methods
$process = Start-Process -FilePath $python -ArgumentList $arguments -WorkingDirectory $root -RedirectStandardOutput $out -RedirectStandardError $err -PassThru -Wait
if ($process.ExitCode -ne 0) {
    throw "Benchmark exited with code $($process.ExitCode). Progress retained. Inspect $out; no automatic retry."
}
