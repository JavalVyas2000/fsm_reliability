# Revision queue (docs/revision_prereg.md, items 2 and 1). Detached via Start-Process.
#   0. build data/cstr/closedloop2_v4 (400 fresh episodes); data/cstr/cert2_v4 (2000) is built by a parallel process
#   1. item 2: closed loop on closedloop2 for SmolLM2-1.7B and Qwen2.5-7B (4-bit): AV, combined, internals, 3 x R0
#   2. item 1: per CSTR model: cert2 first-proposal collection -> grounding -> one-time evaluation of the frozen rules
# Writes "REVISION QUEUE DONE" to outputs\revision_queue.log at the end.
$ErrorActionPreference = "Continue"
$env:PYTHONIOENCODING = "utf-8"
$py = "$env:LOCALAPPDATA\Programs\Python\Python313\python.exe"
Set-Location "C:\Users\jv624\Desktop\fsm_reliability"
$log = "outputs\revision_queue.log"

function Note($msg) { "$(Get-Date -Format s) $msg" | Out-File -Append -Encoding utf8 $log }
function Run($name, [string[]]$pyArgs) {
    Note "START $name"
    & $py @pyArgs 2>&1 | Out-File -Append -Encoding utf8 "outputs\queue_rev_$name.log"
    $code = $LASTEXITCODE
    Note "END   $name (exit $code)"
    return $code
}
function LatestDir($root, $pattern) {
    (Get-ChildItem $root -Directory | Where-Object Name -match $pattern | Sort-Object Name | Select-Object -Last 1).Name
}

$models = @(  # order: fastest first
    @{key = "smollm2-17b"; hf = "HuggingFaceTB/SmolLM2-1.7B-Instruct"; slug = "smollm2-17b-instruct"; quant = @();
      frozen = "outputs/certification/20261009_153848_cstr_v4_smollm2-17b_skip_ucb"},
    @{key = "qwen25-15b"; hf = "Qwen/Qwen2.5-1.5B-Instruct"; slug = "qwen25-15b-instruct"; quant = @();
      frozen = "outputs/certification/20261002_094601_cstr_v4_qwen25-15b_skip_ucb"},
    @{key = "qwen25-3b"; hf = "Qwen/Qwen2.5-3B-Instruct"; slug = "qwen25-3b-instruct"; quant = @();
      frozen = "outputs/certification/20260929_221853_cstr_v4_qwen25-3b_skip_ucb_EXPLORATORY"},
    @{key = "llama-32-3b"; hf = "meta-llama/Llama-3.2-3B-Instruct"; slug = "llama-32-3b-instruct"; quant = @();
      frozen = "outputs/certification/20261001_112113_cstr_v4_llama-32-3b_skip_ucb"},
    @{key = "qwen25-7b"; hf = "Qwen/Qwen2.5-7B-Instruct"; slug = "qwen25-7b-instruct"; quant = @("--quantization", "4bit");
      frozen = "outputs/certification/20261003_164452_cstr_v4_qwen25-7b_skip_ucb"}
)

Note "REVISION QUEUE START"

# ---- 0. fresh closed-loop episodes
if (-not (Test-Path "data\cstr\closedloop2_v4\dataset_manifest.json")) {
    Run "build_closedloop2" @("-m", "scripts.35_cstr_build_from_specs", "--specs", "data/cstr/closedloop2_v4_specs.xlsx",
        "--ranges", "configs/cstr_severity_ranges_v4.json", "--out_dir", "data/cstr/closedloop2_v4",
        "--splits", "closedloop2", "--replace_seed", "20261030", "--workers", "8") | Out-Null
}

# ---- 1. item 2: R0 on fresh episodes (SmolLM2 first, then Qwen2.5-7B)
foreach ($m in @($models[0], $models[4])) {
    $args42 = @("-m", "scripts.42_cstr_closed_loop", "--dataset_dir", "data/cstr/closedloop2_v4", "--frozen_dir", $m.frozen,
                "--model", $m.hf, "--policies", "always_validate", "combined_probe", "internals_probe",
                "observables_probe_r0", "combined_probe_r0", "internals_probe_r0",
                "--tag", "$($m.key)_fresh_r0", "--local_files_only") + $m.quant
    $code = Run "closedloop2_$($m.key)" $args42
    if ($code -ne 0) {  # resume once into the same run directory (completed episodes are kept)
        $rd = LatestDir "outputs\cstr_closed_loop" "_$($m.key)_fresh_r0$"
        if ($rd) { Run "closedloop2_$($m.key)_resume" ($args42 + @("--run_dir", "outputs/cstr_closed_loop/$rd")) | Out-Null }
    }
}
Note "ITEM 2 DONE"

# ---- 2. item 1: cert2 (wait for the parallel build)
while (-not (Test-Path "data\cstr\cert2_v4\dataset_manifest.json")) { Start-Sleep -Seconds 300 }
foreach ($m in $models) {
    $collect = @("-m", "scripts.25_cstr_collect", "--model", $m.hf, "--dataset_dir", "data/cstr/cert2_v4", "--partitions", "cert2",
                 "--max_reprompts", "0", "--prompt", "v3.1", "--max_new_tokens", "512", "--tag", "cert2_v31_r0",
                 "--local_files_only") + $m.quant
    $code = Run "cert2_collect_$($m.key)" $collect
    $cd = LatestDir "outputs\cstr_collect" "_$($m.slug)_cert2_v31_r0$"
    if ($code -ne 0 -and $cd) {
        $code = Run "cert2_collect_$($m.key)_resume" ($collect + @("--run_dir", "outputs/cstr_collect/$cd"))
    }
    if (-not $cd) { Note "SKIP $($m.key): no collection directory"; continue }
    Run "cert2_grounding_$($m.key)" @("-m", "scripts.31_cstr_field_grounding", "--collect_dir", "outputs/cstr_collect/$cd",
        "--prompt", "v3.1", "--partitions", "cert2") | Out-Null
    $gd = LatestDir "outputs\cstr_grounding" "_$([regex]::Escape($cd))$"
    if (-not $gd) { Note "SKIP eval $($m.key): no grounding directory"; continue }
    Run "cert2_eval_$($m.key)" @("-m", "scripts.40_cstr_evaluate_skip_rules", "--frozen_dir", $m.frozen,
        "--aug_dir", "outputs/cstr_grounding/$gd/aug", "--partition", "cert2") | Out-Null
}
Note "REVISION QUEUE DONE"
