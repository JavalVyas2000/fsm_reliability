# Closed loop for SmolLM2-1.7B (docs/cstr_closed_loop_prereg.md, Amendment 3). Starts after "SMOLLM DONE"
# (its probes are frozen by then), writes "CAMPAIGN COMPLETE" at the end. Detached.
$ErrorActionPreference = "Continue"
$env:PYTHONIOENCODING = "utf-8"
$py = "$env:LOCALAPPDATA\Programs\Python\Python313\python.exe"
Set-Location "C:\Users\jv624\Desktop\fsm_reliability"
$log = "outputs\data_queue.log"
function Note($msg) { "$(Get-Date -Format s) [smollm closed loop] $msg" | Out-File -Append -Encoding utf8 $log }
while (-not (Select-String -Path $log -Pattern "SMOLLM DONE" -Quiet)) { Start-Sleep -Seconds 300 }
$f = (Get-ChildItem outputs\certification -Directory | Where-Object { $_.Name -match "cstr_v4_smollm2-17b_skip_ucb$" } | Sort-Object Name | Select-Object -Last 1).Name
if (-not $f -or -not (Test-Path "outputs\certification\$f\rules.json")) { Note "STOP: no complete SmolLM2 freeze found"; Note "CAMPAIGN COMPLETE"; exit 1 }
Note "START closed-loop smollm2-17b (frozen $f)"
& $py -m scripts.42_cstr_closed_loop --dataset_dir data/cstr/closedloop_v4 --frozen_dir "outputs/certification/$f" `
    --model HuggingFaceTB/SmolLM2-1.7B-Instruct --tag smollm2-17b --local_files_only 2>&1 |
    Out-File -Encoding utf8 outputs\queue_closed_loop_smollm2-17b.log
Note "END   closed-loop smollm2-17b (exit $LASTEXITCODE)"
Note "CAMPAIGN COMPLETE"
