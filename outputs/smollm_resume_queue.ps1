# SmolLM2-1.7B CSTR (prereg Amendment 4): the collection died at 1073/5000 (BrokenProcessPool: a simulator worker
# terminated abruptly, 2026-10-07 17:46). Resume it (up to 5 times) after the internals-only closed loops, and run
# grounding / freeze / test / cert only once all 5000 records exist. Detached.
$ErrorActionPreference = "Continue"
$env:PYTHONIOENCODING = "utf-8"
$py = "$env:LOCALAPPDATA\Programs\Python\Python313\python.exe"
Set-Location "C:\Users\jv624\Desktop\fsm_reliability"
$log = "outputs\data_queue.log"
function Note($msg) { "$(Get-Date -Format s) [smollm resume] $msg" | Out-File -Append -Encoding utf8 $log }
function Run($name, [string[]]$pyArgs) {
    Note "START $name"
    & $py @pyArgs 2>&1 | Out-File -Append -Encoding utf8 "outputs\queue_$name.log"
    Note "END   $name (exit $LASTEXITCODE)"
}
function CountLines($f) { if (Test-Path $f) { (Get-Content $f | Measure-Object -Line).Lines } else { 0 } }
function LatestDir($root, $pattern, $exclude = "^$") {
    (Get-ChildItem $root -Directory | Where-Object { $_.Name -match $pattern -and $_.Name -notmatch $exclude } | Sort-Object Name | Select-Object -Last 1).Name
}
while (-not (Select-String -Path $log -Pattern "INTERNALS CLOSED LOOP DONE" -Quiet)) { Start-Sleep -Seconds 600 }
$c = "outputs/cstr_collect/20261007_094026_smollm2-17b-instruct_v4_v31_r0"
$smol = @("-m", "scripts.25_cstr_collect", "--model", "HuggingFaceTB/SmolLM2-1.7B-Instruct", "--dataset_dir", "data/cstr/episodes_v4",
          "--partitions", "train", "dev_cal", "dev_thr", "test_iid", "cert", "--max_reprompts", "0", "--prompt", "v3.1",
          "--max_new_tokens", "512", "--run_dir", $c, "--local_files_only")
for ($i = 1; $i -le 5 -and (CountLines "$c\records.jsonl") -lt 5000; $i++) { Run "cstr_smollm_collect_resume" $smol }
$n = CountLines "$c\records.jsonl"
if ($n -lt 5000) { Note "STOP: SmolLM2 collection has $n/5000 records after 5 resumes"; Note "CAMPAIGN DONE"; exit 1 }
Run "cstr_smollm_grounding_full" @("-m", "scripts.31_cstr_field_grounding", "--collect_dir", $c, "--prompt", "v3.1")
$g = LatestDir "outputs\cstr_grounding" "smollm2-17b-instruct_v4_v31_r0" "withcert|PARTIAL"
Run "cstr_smollm_freeze_full" @("-m", "scripts.39_cstr_freeze_skip_rules", "--aug_dir", "outputs/cstr_grounding/$g/aug",
                                "--tag", "cstr_v4_smollm2-17b_skip_ucb", "--threshold_rule", "ucb")
$f = LatestDir "outputs\certification" "cstr_v4_smollm2-17b_skip_ucb"
Run "cstr_smollm_test_eval_full" @("-m", "scripts.40_cstr_evaluate_skip_rules", "--frozen_dir", "outputs/certification/$f",
                                   "--aug_dir", "outputs/cstr_grounding/$g/aug", "--partition", "test_iid")
Run "cstr_smollm_cert_grounding_full" @("-m", "scripts.31_cstr_field_grounding", "--collect_dir", $c, "--prompt", "v3.1",
                                        "--partitions", "cert", "--include_cert")
$gc = LatestDir "outputs\cstr_grounding" "smollm2-17b-instruct_v4_v31_r0_withcert" "PARTIAL"
Run "cstr_smollm_cert_eval_full" @("-m", "scripts.40_cstr_evaluate_skip_rules", "--frozen_dir", "outputs/certification/$f",
                                   "--aug_dir", "outputs/cstr_grounding/$gc/aug", "--partition", "cert")
Note "SMOLLM DONE"
Note "CAMPAIGN DONE"
