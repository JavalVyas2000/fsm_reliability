# First-proposal-only rule (docs/cstr_closed_loop_prereg.md, Amendment 4): SmolLM2-1.7B then Qwen2.5-7B (4-bit). Detached.
$ErrorActionPreference = "Continue"
$env:PYTHONIOENCODING = "utf-8"
$py = "$env:LOCALAPPDATA\Programs\Python\Python313\python.exe"
Set-Location "C:\Users\jv624\Desktop\fsm_reliability"
$log = "outputs\data_queue.log"
function Note($msg) { "$(Get-Date -Format s) [r0 rule] $msg" | Out-File -Append -Encoding utf8 $log }
$runs = @(
    @{tag = "smollm2-17b"; model = "HuggingFaceTB/SmolLM2-1.7B-Instruct"; frozen = "outputs/certification/20261009_153848_cstr_v4_smollm2-17b_skip_ucb"; extra = @()},
    @{tag = "qwen25-7b"; model = "Qwen/Qwen2.5-7B-Instruct"; frozen = "outputs/certification/20261003_164452_cstr_v4_qwen25-7b_skip_ucb"; extra = @("--quantization", "4bit")}
)
foreach ($r in $runs) {
    Note "START $($r.tag)"
    & $py @(@("-m", "scripts.42_cstr_closed_loop", "--dataset_dir", "data/cstr/closedloop_v4", "--frozen_dir", $r.frozen,
              "--model", $r.model, "--policies", "observables_probe_r0", "combined_probe_r0", "internals_probe_r0",
              "--tag", "$($r.tag)_r0rule", "--local_files_only") + $r.extra) 2>&1 |
        Out-File -Encoding utf8 "outputs\queue_closed_loop_$($r.tag)_r0rule.log"
    Note "END   $($r.tag) (exit $LASTEXITCODE)"
}
Note "R0 RULE DONE"
