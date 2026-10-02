# Follow-up after the detached pipeline (2026-10-02). Waits until the closed loop is done (or stopped), then:
#   1. Qwen2.5-1.5B CSTR: cert grounding + one-time cert evaluation (rules frozen 20261002_094601)
#   2. Qwen2.5-7B CSTR: freeze (UCB, Amendment 3) + test_iid once + cert grounding + cert once
$ErrorActionPreference = "Continue"
$env:PYTHONIOENCODING = "utf-8"
$py = "$env:LOCALAPPDATA\Programs\Python\Python313\python.exe"
Set-Location "C:\Users\jv624\Desktop\fsm_reliability"
$log = "outputs\data_queue.log"
function Note($msg) { "$(Get-Date -Format s) [followup] $msg" | Out-File -Append -Encoding utf8 $log }
function Run($name, [string[]]$pyArgs) {
    Note "START $name"
    & $py @pyArgs 2>&1 | Out-File -Encoding utf8 "outputs\queue_$name.log"
    Note "END   $name (exit $LASTEXITCODE)"
}
function LatestDir($root, $pattern, $exclude = "^$") {
    (Get-ChildItem $root -Directory | Where-Object { $_.Name -match $pattern -and $_.Name -notmatch $exclude } | Sort-Object Name | Select-Object -Last 1).Name
}
while (-not (Select-String -Path $log -Pattern "CLOSED LOOP DONE|STOP: smoke" -Quiet)) { Start-Sleep -Seconds 600 }
Note "starting follow-up"

# ---- 1. Qwen2.5-1.5B cert
Run "qwen15_cstr_cert_grounding" @("-m", "scripts.31_cstr_field_grounding", "--collect_dir",
    "outputs/cstr_collect/20261001_115207_qwen25-15b-instruct_v4_v31_r0", "--prompt", "v3.1", "--partitions", "cert", "--include_cert")
$g = LatestDir "outputs\cstr_grounding" "qwen25-15b-instruct_v4_v31_r0_withcert"
if ($g) { Run "qwen15_cstr_cert_eval" @("-m", "scripts.40_cstr_evaluate_skip_rules", "--frozen_dir",
          "outputs/certification/20261002_094601_cstr_v4_qwen25-15b_skip_ucb", "--aug_dir", "outputs/cstr_grounding/$g/aug", "--partition", "cert") }

# ---- 2. Qwen2.5-7B
$c = LatestDir "outputs\cstr_collect" "qwen25-7b-instruct_v4_v31_r0"
$g7 = LatestDir "outputs\cstr_grounding" "qwen25-7b-instruct_v4_v31_r0" "withcert"
if ($c -and $g7) {
    Run "qwen7_cstr_freeze" @("-m", "scripts.39_cstr_freeze_skip_rules", "--aug_dir", "outputs/cstr_grounding/$g7/aug",
                              "--tag", "cstr_v4_qwen25-7b_skip_ucb", "--threshold_rule", "ucb")
    $f7 = LatestDir "outputs\certification" "cstr_v4_qwen25-7b_skip_ucb"
    Run "qwen7_cstr_test_eval" @("-m", "scripts.40_cstr_evaluate_skip_rules", "--frozen_dir", "outputs/certification/$f7",
                                 "--aug_dir", "outputs/cstr_grounding/$g7/aug", "--partition", "test_iid")
    Run "qwen7_cstr_cert_grounding" @("-m", "scripts.31_cstr_field_grounding", "--collect_dir", "outputs/cstr_collect/$c",
                                      "--prompt", "v3.1", "--partitions", "cert", "--include_cert")
    $g7c = LatestDir "outputs\cstr_grounding" "qwen25-7b-instruct_v4_v31_r0_withcert"
    if ($g7c) { Run "qwen7_cstr_cert_eval" @("-m", "scripts.40_cstr_evaluate_skip_rules", "--frozen_dir", "outputs/certification/$f7",
                "--aug_dir", "outputs/cstr_grounding/$g7c/aug", "--partition", "cert") }
} else { Note "SKIP Qwen2.5-7B follow-up: collection or grounding missing" }
Note "FOLLOWUP DONE"
