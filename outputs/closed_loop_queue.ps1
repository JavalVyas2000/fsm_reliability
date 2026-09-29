# After the data queue: closed-loop CSTR (docs/cstr_closed_loop_prereg.md).
# 1. wait for "QUEUE DONE" in outputs\data_queue.log and for the closed-loop episodes build
# 2. GPU smoke test (2 episodes, 2 proposals); 3. full run only if the smoke test wrote 2 x 5 episode rows
$ErrorActionPreference = "Continue"
$env:PYTHONIOENCODING = "utf-8"
$py = "$env:LOCALAPPDATA\Programs\Python\Python313\python.exe"
Set-Location "C:\Users\jv624\Desktop\fsm_reliability"
$log = "outputs\data_queue.log"
function Note($msg) { "$(Get-Date -Format s) [closed-loop] $msg" | Out-File -Append -Encoding utf8 $log }
$frozen = "outputs/certification/20260929_221853_cstr_v4_qwen25-3b_skip_ucb_EXPLORATORY"
$common = @("-m", "scripts.42_cstr_closed_loop", "--frozen_dir", $frozen, "--local_files_only")

while (-not (Select-String -Path $log -Pattern "QUEUE DONE" -Quiet)) { Start-Sleep -Seconds 600 }
while (-not (Test-Path "data\cstr\closedloop_v4\dataset_manifest.json")) { Start-Sleep -Seconds 300 }

Note "START smoke"
& $py @($common + @("--dataset_dir", "data/cstr/pilot_v4", "--limit", "2", "--max_proposals", "2", "--tag", "SMOKE_gpu")) 2>&1 |
    Out-File -Encoding utf8 outputs\queue_closed_loop_smoke.log
$smoke = (Get-ChildItem outputs\cstr_closed_loop -Directory | Where-Object Name -match "SMOKE_gpu" | Sort-Object Name | Select-Object -Last 1)
$rows = if ($smoke -and (Test-Path "$($smoke.FullName)\episodes.jsonl")) { (Get-Content "$($smoke.FullName)\episodes.jsonl" | Measure-Object -Line).Lines } else { 0 }
Note "END   smoke ($rows episode rows)"
if ($rows -lt 10) { Note "STOP: smoke test incomplete, see outputs\queue_closed_loop_smoke.log"; exit 1 }

Note "START closed-loop qwen25-3b (400 episodes)"
& $py @($common + @("--dataset_dir", "data/cstr/closedloop_v4", "--tag", "qwen25-3b")) 2>&1 |
    Out-File -Encoding utf8 outputs\queue_closed_loop_qwen25-3b.log
Note "END   closed-loop qwen25-3b (exit $LASTEXITCODE)"
Note "CLOSED LOOP DONE"
