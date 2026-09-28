# Wait for the v4 snapshot build (scripts/35), assign partitions (scripts/36), then collect
# first proposals with internals for all partitions (scripts/25, prompt v3.1, round 0 only).
$ErrorActionPreference = "Stop"
$env:PYTHONIOENCODING = "utf-8"
$py = "$env:LOCALAPPDATA\Programs\Python\Python313\python.exe"
Set-Location "C:\Users\jv624\Desktop\fsm_reliability"
$manifest = "data\cstr\episodes_v4\dataset_manifest.json"
while (-not (Test-Path $manifest)) { Start-Sleep -Seconds 120 }
Start-Sleep -Seconds 30
$ErrorActionPreference = "Continue"  # python writes progress to stderr
& $py -m scripts.36_cstr_assign_partitions --build_dir data/cstr/episodes_v4 2>&1 | Out-File -Encoding utf8 outputs\cstr_v4_partitions.log
if (-not (Test-Path "data\cstr\episodes_v4\snapshots_cert.pkl")) { throw "partition assignment failed" }
& $py -m scripts.25_cstr_collect --dataset_dir data/cstr/episodes_v4 --partitions train dev_cal dev_thr test_iid cert `
    --max_reprompts 0 --prompt v3.1 --max_new_tokens 512 --tag v4_v31_r0 --local_files_only 2>&1 |
    Out-File -Encoding utf8 outputs\cstr_v4_collect.log
