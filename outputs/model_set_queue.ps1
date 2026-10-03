# Amendment 4 of docs/cstr_v4_prereg.md: Qwen2.5-7B on FSM, SmolLM2-1.7B on CSTR (gated). Detached; starts after
# the closed-loop runs for the other models ("CLOSED LOOP MODELS DONE").
$ErrorActionPreference = "Continue"
$env:PYTHONIOENCODING = "utf-8"
$py = "$env:LOCALAPPDATA\Programs\Python\Python313\python.exe"
Set-Location "C:\Users\jv624\Desktop\fsm_reliability"
$log = "outputs\data_queue.log"
function Note($msg) { "$(Get-Date -Format s) [model set] $msg" | Out-File -Append -Encoding utf8 $log }
function Run($name, [string[]]$pyArgs) {
    Note "START $name"
    & $py @pyArgs 2>&1 | Out-File -Encoding utf8 "outputs\queue_$name.log"
    Note "END   $name (exit $LASTEXITCODE)"
}
function LatestDir($root, $pattern, $exclude = "^$") {
    (Get-ChildItem $root -Directory | Where-Object { $_.Name -match $pattern -and $_.Name -notmatch $exclude } | Sort-Object Name | Select-Object -Last 1).Name
}
while (-not (Select-String -Path $log -Pattern "CLOSED LOOP MODELS DONE" -Quiet)) { Start-Sleep -Seconds 600 }

# ---- 1. Qwen2.5-7B (4-bit) on FSM
$q7 = @("--model", "Qwen/Qwen2.5-7B-Instruct", "--max_new_tokens", "96", "--quantization", "4bit", "--local_files_only")
Run "fsm_qwen7_pilot" (@("-m", "scripts.11_run_fsm_inference_v2", "--dataset_dir", "data/v2/fsm_pilot_seed20260923", "--tag", "pilot3000") + $q7)
$pi = LatestDir "outputs\fsm_inference" "qwen25-7b-instruct_pilot3000"
if ($pi) { Run "fsm_qwen7_pilot_grounding" @("-m", "scripts.27_fsm_grounding", "--inference_dir", "outputs/fsm_inference/$pi") }
Run "fsm_qwen7_cert2" (@("-m", "scripts.11_run_fsm_inference_v2", "--dataset_dir", "data/v2/fsm_cert2_seed20260930", "--partitions", "cert", "--tag", "cert2_3000") + $q7)
$ci = LatestDir "outputs\fsm_inference" "qwen25-7b-instruct_cert2_3000"
if ($ci) { Run "fsm_qwen7_cert2_grounding" @("-m", "scripts.27_fsm_grounding", "--inference_dir", "outputs/fsm_inference/$ci") }
$pg = LatestDir "outputs\fsm_grounding" "qwen25-7b-instruct_pilot3000"
$cg = LatestDir "outputs\fsm_grounding" "qwen25-7b-instruct_cert2_3000"
if ($pg -and $cg) {
    Run "fsm_qwen7_freeze" @("-m", "scripts.39_cstr_freeze_skip_rules", "--domain", "fsm", "--threshold_rule", "ucb",
                             "--aug_dir", "outputs/fsm_grounding/$pg/aug", "--tag", "fsm_qwen25-7b_skip_ucb_cert2")
    $f = LatestDir "outputs\certification" "fsm_qwen25-7b_skip_ucb_cert2"
    Run "fsm_qwen7_cert2_eval" @("-m", "scripts.40_cstr_evaluate_skip_rules", "--frozen_dir", "outputs/certification/$f",
                                 "--aug_dir", "outputs/fsm_grounding/$cg/aug", "--partition", "cert")
} else { Note "SKIP FSM Qwen-7B freeze/eval: grounding missing" }

# ---- 2. SmolLM2-1.7B on CSTR, gated by a 100-episode pilot
$smol = @("--model", "HuggingFaceTB/SmolLM2-1.7B-Instruct", "--max_reprompts", "0", "--prompt", "v3.1", "--max_new_tokens", "512", "--local_files_only")
Run "cstr_smollm_pilot" (@("-m", "scripts.25_cstr_collect", "--dataset_dir", "data/cstr/pilot_v4", "--partitions", "pilot", "--tag", "smollm_pilot100") + $smol)
$sp = LatestDir "outputs\cstr_collect" "smollm2-17b-instruct_smollm_pilot100"
$gate = & $py -c "import json,sys; r=[json.loads(l) for l in open(sys.argv[1],encoding='utf-8')]; v=[x for x in r if x.get('schema_valid')==1 and x.get('verifier_pass') is not None]; p=sum(bool(x['verifier_pass']) for x in v)/max(1,len(v)); f=1-len([x for x in r if x.get('schema_valid')==1])/max(1,len(r)); print(f'{p:.3f} {f:.3f}')" "outputs/cstr_collect/$sp/records.jsonl"
Note "SmolLM2 CSTR pilot: pass rate among valid, format-failure rate = $gate"
$parts = "$gate".Trim().Split(" ")
if ($parts.Count -eq 2 -and [double]$parts[0] -ge 0.05 -and [double]$parts[1] -le 0.5) {
    Run "cstr_smollm_collect" (@("-m", "scripts.25_cstr_collect", "--dataset_dir", "data/cstr/episodes_v4", "--partitions",
                                 "train", "dev_cal", "dev_thr", "test_iid", "cert", "--tag", "v4_v31_r0") + $smol)
    $c = LatestDir "outputs\cstr_collect" "smollm2-17b-instruct_v4_v31_r0"
    Run "cstr_smollm_grounding" @("-m", "scripts.31_cstr_field_grounding", "--collect_dir", "outputs/cstr_collect/$c", "--prompt", "v3.1")
    $g = LatestDir "outputs\cstr_grounding" "smollm2-17b-instruct_v4_v31_r0" "withcert"
    Run "cstr_smollm_freeze" @("-m", "scripts.39_cstr_freeze_skip_rules", "--aug_dir", "outputs/cstr_grounding/$g/aug",
                               "--tag", "cstr_v4_smollm2-17b_skip_ucb", "--threshold_rule", "ucb")
    $f = LatestDir "outputs\certification" "cstr_v4_smollm2-17b_skip_ucb"
    Run "cstr_smollm_test_eval" @("-m", "scripts.40_cstr_evaluate_skip_rules", "--frozen_dir", "outputs/certification/$f",
                                  "--aug_dir", "outputs/cstr_grounding/$g/aug", "--partition", "test_iid")
    Run "cstr_smollm_cert_grounding" @("-m", "scripts.31_cstr_field_grounding", "--collect_dir", "outputs/cstr_collect/$c",
                                       "--prompt", "v3.1", "--partitions", "cert", "--include_cert")
    $gc = LatestDir "outputs\cstr_grounding" "smollm2-17b-instruct_v4_v31_r0_withcert"
    Run "cstr_smollm_cert_eval" @("-m", "scripts.40_cstr_evaluate_skip_rules", "--frozen_dir", "outputs/certification/$f",
                                  "--aug_dir", "outputs/cstr_grounding/$gc/aug", "--partition", "cert")
} else { Note "SmolLM2 CSTR gate not met: full collection skipped (Amendment 4)" }
Note "MODEL SET DONE"
