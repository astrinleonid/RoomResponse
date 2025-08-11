# Batch Room Response Data Collection Script for PowerShell
# Collects scenarios 0.1 through 0.5 with 200 measurements each

Write-Host "Room Response Batch Data Collection" -ForegroundColor Cyan
Write-Host "====================================" -ForegroundColor Cyan

# Confirm start
$response = Read-Host "Proceed with batch collection? (y/n)"
if ($response -ne "y" -and $response -ne "Y") {
    Write-Host "Batch collection cancelled." -ForegroundColor Yellow
    exit 0
}

# Scenario parameters
$scenarios = @("1.11", "1.12", "1.13", "1.14", "1.15")
$numMeasurements = 100
$measurementInterval = 1.0

# Track results
$completed = @()
$failed = @()

# Start time
$startTime = Get-Date

# Run each scenario
for ($i = 0; $i -lt $scenarios.Count; $i++) {
    $scenario = $scenarios[$i]
    $scenarioNum = $i + 1
    $totalScenarios = $scenarios.Count

    Write-Host ""
    Write-Host "🚀 Starting scenario $scenarioNum/$totalScenarios`: $scenario" -ForegroundColor Green
    Write-Host "Time: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"

    # Build command arguments
    $arguments = @(
        "collect_dataset.py",
        "--quiet",
        "--scenario-number", $scenario,
        "--description", "Automated_collection_scenario_$scenario",
        "--num-measurements", $numMeasurements,
        "--measurement-interval", $measurementInterval
    )

    # Run the collection
    try {
        $process = Start-Process -FilePath "python" -ArgumentList $arguments -Wait -PassThru -NoNewWindow

        if ($process.ExitCode -eq 0) {
            Write-Host "✅ Scenario $scenario completed successfully!" -ForegroundColor Green
            $completed += $scenario
        } else {
            Write-Host "❌ Scenario $scenario failed with exit code $($process.ExitCode)" -ForegroundColor Red
            $failed += $scenario
        }
    }
    catch {
        Write-Host "❌ Scenario $scenario failed with error: $($_.Exception.Message)" -ForegroundColor Red
        $failed += $scenario
    }

    # Brief pause between scenarios (optional)
    if ($scenarioNum -lt $totalScenarios) {
        Write-Host "Proceeding to next scenario in 3 seconds..." -ForegroundColor Cyan
        Start-Sleep -Seconds 30
    }
}

# Calculate duration
$endTime = Get-Date
$duration = $endTime - $startTime
$durationMinutes = [math]::Round($duration.TotalMinutes, 1)

# Print summary
Write-Host ""
Write-Host "===============================================" -ForegroundColor Cyan
Write-Host "BATCH COLLECTION SUMMARY" -ForegroundColor Cyan
Write-Host "===============================================" -ForegroundColor Cyan
Write-Host "Total duration: $durationMinutes minutes"
Write-Host "Completed scenarios: $($completed.Count)"

if ($completed.Count -gt 0) {
    Write-Host "  ✅ $($completed -join ', ')" -ForegroundColor Green
}

if ($failed.Count -gt 0) {
    Write-Host "Failed scenarios: $($failed.Count)"
    Write-Host "  ❌ $($failed -join ', ')" -ForegroundColor Red
}

$totalMeasurements = $completed.Count * $numMeasurements
Write-Host "Total measurements collected: ~$totalMeasurements"
Write-Host "===============================================" -ForegroundColor Cyan

# Exit with appropriate code
if ($failed.Count -gt 0) {
    Write-Host ""
    Write-Host "⚠️ Some scenarios failed. Check the output above for details." -ForegroundColor Yellow
    exit 1
} elseif ($completed.Count -gt 0) {
    Write-Host ""
    Write-Host "🎉 Batch collection completed successfully!" -ForegroundColor Green
    exit 0
} else {
    Write-Host ""
    Write-Host "❌ No scenarios completed successfully." -ForegroundColor Red
    exit 1
}