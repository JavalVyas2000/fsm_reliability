# Revision item 2b (docs/revision_prereg.md): R0 + validate rejections, on the closedloop2 episodes of item 2.
# Waits for the main revision queue (GPU) to finish, then runs Qwen2.5-7B (primary) and SmolLM2-1.7B.
# Logs to the same outputs\revision_queue.log; writes "ITEM 2B DONE" at the end.
$ErrorActionPreference = "Continue"
$env:PYTHONIOENCODING = "utf-8"
$py = "$env:LOCALAPPDATA\Programs\Python\Python313\python.exe"
Set-Location "C:\Users\jv624\Desktop\fsm_reliability"
$log = "outputs\revision_queue.log"

function Note($msg) { "$(Get-Date -Format s) $msg" | Out-File -Append -Encoding utf8 $log }
function Run($name, [string[]]$pyArgs) {
    Note "START $name"
    & $py @pyArgs 2>&1 | Out-File -Append -Encoding utf8 "outputs\queue_rev_$name.log"
    $code = $LASTEXITCODE
    Note "END   $name (exit $code)"
    return $code
}
function LatestDir($root, $pattern) {
    (Get-ChildItem $root -Directory | Where-Object Name -match $pattern | Sort-Object Name | Select-Object -Last 1).Name
}

while (-not (Select-String -Path $log -Pattern "REVISION QUEUE DONE" -Quiet)) { Start-Sleep -Seconds 300 }

$models = @(
    @{key = "qwen25-7b"; hf = "Qwen/Qwen2.5-7B-Instruct"; quant = @("--quantization", "4bit");
      frozen = "outputs/certification/20261003_164452_cstr_v4_qwen25-7b_skip_ucb"},
    @{key = "smollm2-17b"; hf = "HuggingFaceTB/SmolLM2-1.7B-Instruct"; quant = @();
      frozen = "outputs/certification/20261009_153848_cstr_v4_smollm2-17b_skip_ucb"}
)
if (-not (Select-String -Path $log -Pattern "ITEM 2B START" -Quiet)) { Note "ITEM 2B START" }
foreach ($m in $models) {
    $args42 = @("-m", "scripts.42_cstr_closed_loop", "--dataset_dir", "data/cstr/closedloop2_v4", "--frozen_dir", $m.frozen,
                "--model", $m.hf, "--policies", "always_validate", "observables_probe_r0v", "combined_probe_r0v",
                "internals_probe_r0v", "--tag", "$($m.key)_fresh_r0v", "--local_files_only") + $m.quant
    # restart after a stop: skip finished models; resume a partial run into its own directory
    if (Select-String -Path $log -Pattern "END   closedloop2v_$($m.key)(_resume)? \(exit 0\)" -Quiet) { continue }
    $rd = LatestDir "outputs\cstr_closed_loop" "_$($m.key)_fresh_r0v$"
    if ($rd) { $code = Run "closedloop2v_$($m.key)_resume" ($args42 + @("--run_dir", "outputs/cstr_closed_loop/$rd")) }
    else { $code = Run "closedloop2v_$($m.key)" $args42 }
    if ($code -ne 0) {
        $rd = LatestDir "outputs\cstr_closed_loop" "_$($m.key)_fresh_r0v$"
        if ($rd) { Run "closedloop2v_$($m.key)_resume" ($args42 + @("--run_dir", "outputs/cstr_closed_loop/$rd")) | Out-Null }
    }
}
Note "ITEM 2B DONE"
