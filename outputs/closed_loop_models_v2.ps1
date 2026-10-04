# Closed loop for Llama-3.2-3B (restart) and Qwen2.5-7B, after the GPU-memory fix in scripts/42 (2026-10-04).
# The first Llama run spilled ~3 GB into shared memory (generation ~135 s/proposal) and was aborted after 66
# episodes (outputs/cstr_closed_loop/*_llama-32-3b_ABORTED_gpu_memory_spill). Qwen2.5-1.5B finished earlier.
# Writes "CLOSED LOOP MODELS DONE" (the later queues wait for it).
$ErrorActionPreference = "Continue"
$env:PYTHONIOENCODING = "utf-8"
$py = "$env:LOCALAPPDATA\Programs\Python\Python313\python.exe"
Set-Location "C:\Users\jv624\Desktop\fsm_reliability"
$log = "outputs\data_queue.log"
function Note($msg) { "$(Get-Date -Format s) [closed-loop models] $msg" | Out-File -Append -Encoding utf8 $log }
Note "RESTART after GPU-memory fix (Llama run aborted at 66/400 episodes)"
$runs = @(
    @{tag = "llama-32-3b"; model = "meta-llama/Llama-3.2-3B-Instruct"; frozen = "outputs/certification/20261001_112113_cstr_v4_llama-32-3b_skip_ucb"; extra = @()},
    @{tag = "qwen25-7b"; model = "Qwen/Qwen2.5-7B-Instruct"; frozen = "outputs/certification/20261003_164452_cstr_v4_qwen25-7b_skip_ucb"; extra = @("--quantization", "4bit")}
)
foreach ($r in $runs) {
    Note "START closed-loop $($r.tag)"
    & $py @(@("-m", "scripts.42_cstr_closed_loop", "--dataset_dir", "data/cstr/closedloop_v4", "--frozen_dir", $r.frozen,
              "--model", $r.model, "--tag", $r.tag, "--local_files_only") + $r.extra) 2>&1 |
        Out-File -Encoding utf8 "outputs\queue_closed_loop_$($r.tag).log"
    Note "END   closed-loop $($r.tag) (exit $LASTEXITCODE)"
}
Note "CLOSED LOOP MODELS DONE"
