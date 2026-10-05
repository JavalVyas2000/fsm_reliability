# Resume the crashed Llama closed loop (CUDA OOM at episode 79, 2026-10-04 22:52) after Qwen-7B, then the model-set
# queue (Amendment 4) unchanged. Detached.
$ErrorActionPreference = "Continue"
$env:PYTHONIOENCODING = "utf-8"
$py = "$env:LOCALAPPDATA\Programs\Python\Python313\python.exe"
Set-Location "C:\Users\jv624\Desktop\fsm_reliability"
$log = "outputs\data_queue.log"
function Note($msg) { "$(Get-Date -Format s) [llama resume] $msg" | Out-File -Append -Encoding utf8 $log }
while (-not (Select-String -Path $log -Pattern "CLOSED LOOP MODELS DONE" -Quiet)) { Start-Sleep -Seconds 300 }
Note "START resume closed-loop llama-32-3b (20261004_151943_llama-32-3b) after CUDA OOM at episode 79"
& $py -m scripts.42_cstr_closed_loop --dataset_dir data/cstr/closedloop_v4 --frozen_dir outputs/certification/20261001_112113_cstr_v4_llama-32-3b_skip_ucb `
    --model meta-llama/Llama-3.2-3B-Instruct --run_dir outputs/cstr_closed_loop/20261004_151943_llama-32-3b --local_files_only 2>&1 |
    Out-File -Append -Encoding utf8 outputs\queue_closed_loop_llama-32-3b.log
Note "END   resume closed-loop llama-32-3b (exit $LASTEXITCODE)"
& powershell -NoProfile -ExecutionPolicy Bypass -File outputs\model_set_queue.ps1
