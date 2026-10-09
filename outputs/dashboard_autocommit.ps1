# Commit + push DASHBOARD.md every 2 hours while the campaign runs (the pre-commit hook regenerates it).
# Stops after the final queue marker appears. Detached.
Set-Location "C:\Users\jv624\Desktop\fsm_reliability"
$log = "outputs\data_queue.log"
while ($true) {
    $done = Select-String -Path $log -Pattern "CAMPAIGN COMPLETE" -Quiet
    & "$env:LOCALAPPDATA\Programs\Python\Python313\python.exe" -m scripts.48_dashboard | Out-Null
    git add DASHBOARD.md 2>$null
    git diff --cached --quiet -- DASHBOARD.md
    if ($LASTEXITCODE -ne 0) {
        git commit -q -m "Dashboard: automatic status update" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" -- DASHBOARD.md 2>$null
        git push -q origin main 2>$null
    }
    if ($done) { break }
    Start-Sleep -Seconds 7200
}
