param(
    [string]$TaskName = "IndianSwingTradeFinder"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$Runner = Join-Path $PSScriptRoot "scheduled_swing_run.py"

if (-not (Test-Path $Python)) {
    throw "Virtual-environment Python not found: $Python"
}
if (-not (Test-Path $Runner)) {
    throw "Scheduled runner not found: $Runner"
}

$UserId = "$env:USERDOMAIN\$env:USERNAME"
$Action = New-ScheduledTaskAction `
    -Execute $Python `
    -Argument "`"$Runner`"" `
    -WorkingDirectory $ProjectRoot

$WeekdayTrigger = New-ScheduledTaskTrigger `
    -Weekly `
    -WeeksInterval 1 `
    -DaysOfWeek Monday, Tuesday, Wednesday, Thursday, Friday `
    -At "4:00 PM"

# The runner itself exits harmlessly before 16:00, on weekends, or when today's
# successful scan is already recorded.
$LogonTrigger = New-ScheduledTaskTrigger -AtLogOn -User $UserId

$Settings = New-ScheduledTaskSettingsSet `
    -StartWhenAvailable `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -MultipleInstances IgnoreNew `
    -ExecutionTimeLimit (New-TimeSpan -Hours 2)

$Principal = New-ScheduledTaskPrincipal `
    -UserId $UserId `
    -LogonType Interactive `
    -RunLevel Limited

Register-ScheduledTask `
    -TaskName $TaskName `
    -Action $Action `
    -Trigger @($WeekdayTrigger, $LogonTrigger) `
    -Settings $Settings `
    -Principal $Principal `
    -Description "Run NSE swing scan at 4 PM weekdays or next login after 4 PM; update Excel performance journal." `
    -Force | Out-Null

$Task = Get-ScheduledTask -TaskName $TaskName
Write-Host "Installed task: $($Task.TaskName)"
Write-Host "State: $($Task.State)"
Write-Host "Excel journal: $(Join-Path $ProjectRoot 'reports\swing_performance.xlsx')"
