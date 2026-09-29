# Data-collection queue (2026-09-29). Runs after the Llama-3.2-3B CSTR collection finishes:
#   1. Llama CSTR grounding           2. Qwen2.5-1.5B CSTR collection + grounding
#   3. fresh FSM cert set (seed 20260930): inference + grounding for the 4 FSM models
#   4. Qwen2.5-7B (4-bit) CSTR collection + grounding
# Each step logs to outputs\queue_<step>.log; progress lines go to outputs\data_queue.log.
$ErrorActionPreference = "Continue"
$env:PYTHONIOENCODING = "utf-8"
$py = "$env:LOCALAPPDATA\Programs\Python\Python313\python.exe"
Set-Location "C:\Users\jv624\Desktop\fsm_reliability"
$log = "outputs\data_queue.log"

function Note($msg) { "$(Get-Date -Format s) $msg" | Out-File -Append -Encoding utf8 $log }
function Run($name, [string[]]$pyArgs) {
    Note "START $name"
    & $py @pyArgs 2>&1 | Out-File -Encoding utf8 "outputs\queue_$name.log"
    Note "END   $name (exit $LASTEXITCODE)"
}
function LatestDir($root, $pattern) {
    (Get-ChildItem $root -Directory | Where-Object Name -match $pattern | Sort-Object Name | Select-Object -Last 1).Name
}
function CountLines($file) { if (Test-Path $file) { (Get-Content $file | Measure-Object -Line).Lines } else { 0 } }
$cstr = @("--dataset_dir", "data/cstr/episodes_v4", "--partitions", "train", "dev_cal", "dev_thr", "test_iid", "cert",
          "--max_reprompts", "0", "--prompt", "v3.1", "--max_new_tokens", "512", "--tag", "v4_v31_r0", "--local_files_only")

# ---- 1. wait for the Llama collection, then its grounding (train/dev/test_iid; cert only after the freeze)
$llama = "outputs/cstr_collect/20260929_214136_llama-32-3b-instruct_v4_v31_r0"
Note "waiting for $llama"
while (Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -match "llama-32-3b-instruct_v4_v31_r0" }) { Start-Sleep -Seconds 300 }
$n = CountLines "$llama\records.jsonl"
if ($n -lt 5000) { Note "STOP: Llama collection has $n/5000 records (interrupted?). Resume it with --run_dir, then restart this queue."; exit 1 }
Run "llama_cstr_grounding" @("-m", "scripts.31_cstr_field_grounding", "--collect_dir", $llama, "--prompt", "v3.1")

# ---- 2. Qwen2.5-1.5B on CSTR
Run "qwen15_cstr_collect" (@("-m", "scripts.25_cstr_collect", "--model", "Qwen/Qwen2.5-1.5B-Instruct") + $cstr)
$d = LatestDir "outputs\cstr_collect" "qwen25-15b-instruct_v4_v31_r0"
if ($d -and (CountLines "outputs\cstr_collect\$d\records.jsonl") -ge 5000) {
    Run "qwen15_cstr_grounding" @("-m", "scripts.31_cstr_field_grounding", "--collect_dir", "outputs/cstr_collect/$d", "--prompt", "v3.1")
} else { Note "SKIP qwen15 grounding: collection incomplete ($d)" }

# ---- 3. fresh FSM certification set, same arguments as the original cert runs
$fsm = @(
    @{tag = "qwen25-3b-instruct"; args = @()},
    @{tag = "llama-32-3b-instruct"; args = @("--model", "meta-llama/Llama-3.2-3B-Instruct", "--max_new_tokens", "96")},
    @{tag = "qwen25-15b-instruct"; args = @("--model", "Qwen/Qwen2.5-1.5B-Instruct", "--max_new_tokens", "96")},
    @{tag = "smollm2-17b-instruct"; args = @("--model", "HuggingFaceTB/SmolLM2-1.7B-Instruct", "--max_new_tokens", "96")}
)
foreach ($m in $fsm) {
    Run "fsm_cert2_$($m.tag)" (@("-m", "scripts.11_run_fsm_inference_v2", "--dataset_dir", "data/v2/fsm_cert2_seed20260930",
                               "--partitions", "cert", "--tag", "cert2_3000") + $m.args + @("--local_files_only"))
    $d = LatestDir "outputs\fsm_inference" "$($m.tag)_cert2_3000"
    if ($d) { Run "fsm_cert2_grounding_$($m.tag)" @("-m", "scripts.27_fsm_grounding", "--inference_dir", "outputs/fsm_inference/$d") }
    else { Note "SKIP FSM grounding $($m.tag): no inference run" }
}

# ---- 4. Qwen2.5-7B (4-bit) on CSTR
Run "qwen7_cstr_collect" (@("-m", "scripts.25_cstr_collect", "--model", "Qwen/Qwen2.5-7B-Instruct", "--quantization", "4bit") + $cstr)
$d = LatestDir "outputs\cstr_collect" "qwen25-7b-instruct_v4_v31_r0"
if ($d -and (CountLines "outputs\cstr_collect\$d\records.jsonl") -ge 5000) {
    Run "qwen7_cstr_grounding" @("-m", "scripts.31_cstr_field_grounding", "--collect_dir", "outputs/cstr_collect/$d", "--prompt", "v3.1")
} else { Note "SKIP qwen7 grounding: collection incomplete ($d)" }
Note "QUEUE DONE"
