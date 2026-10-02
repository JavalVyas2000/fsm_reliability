# Pause the pipeline right after the Llama grounding step ends (user restart, 2026-10-01):
# stop both queue scripts, and any collection / FSM inference the queue started afterwards.
$log = "C:\Users\jv624\Desktop\fsm_reliability\outputs\data_queue.log"
while (-not (Select-String -Path $log -Pattern "END   llama_cstr_grounding|STOP" -Quiet)) { Start-Sleep -Seconds 5 }
$targets = Get-CimInstance Win32_Process | Where-Object {
    $_.CommandLine -match "data_queue\.ps1|closed_loop_queue\.ps1|scripts\.25_cstr_collect|scripts\.11_run_fsm_inference|scripts\.42_cstr_closed_loop" -and
    $_.CommandLine -notmatch "pause_after_grounding" }
foreach ($t in $targets) { Stop-Process -Id $t.ProcessId -Force -ErrorAction SilentlyContinue; "stopped $($t.ProcessId): $($t.CommandLine.Substring(0, [Math]::Min(80, $t.CommandLine.Length)))" }
Start-Sleep -Seconds 5
# spawned verifier workers of a stopped collection
Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -match "multiprocessing.spawn" } |
    ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue; "stopped worker $($_.ProcessId)" }
"$(Get-Date -Format s) [pause] queues stopped after Llama grounding for the laptop restart" | Out-File -Append -Encoding utf8 $log
"paused"
