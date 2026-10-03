# Amendment 2 of docs/cstr_closed_loop_prereg.md: supplementary internals-only closed-loop runs for the two models
# whose main runs started before the policy existed. Detached; starts after "MODEL SET DONE".
$ErrorActionPreference = "Continue"
$env:PYTHONIOENCODING = "utf-8"
$py = "$env:LOCALAPPDATA\Programs\Python\Python313\python.exe"
Set-Location "C:\Users\jv624\Desktop\fsm_reliability"
$log = "outputs\data_queue.log"
function Note($msg) { "$(Get-Date -Format s) [internals closed loop] $msg" | Out-File -Append -Encoding utf8 $log }
while (-not (Select-String -Path $log -Pattern "MODEL SET DONE" -Quiet)) { Start-Sleep -Seconds 600 }
$runs = @(
    @{tag = "qwen25-3b_internals"; model = "Qwen/Qwen2.5-3B-Instruct"; frozen = "outputs/certification/20260929_221853_cstr_v4_qwen25-3b_skip_ucb_EXPLORATORY"},
    @{tag = "qwen25-15b_internals"; model = "Qwen/Qwen2.5-1.5B-Instruct"; frozen = "outputs/certification/20261002_094601_cstr_v4_qwen25-15b_skip_ucb"}
)
foreach ($r in $runs) {
    Note "START $($r.tag)"
    & $py @("-m", "scripts.42_cstr_closed_loop", "--dataset_dir", "data/cstr/closedloop_v4", "--frozen_dir", $r.frozen,
            "--model", $r.model, "--policies", "internals_probe", "--tag", $r.tag, "--local_files_only") 2>&1 |
        Out-File -Encoding utf8 "outputs\queue_closed_loop_$($r.tag).log"
    Note "END   $($r.tag) (exit $LASTEXITCODE)"
}
Note "INTERNALS CLOSED LOOP DONE"
