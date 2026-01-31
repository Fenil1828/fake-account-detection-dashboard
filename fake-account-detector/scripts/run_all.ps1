# Fake Account Detection - Run All Services (Windows)
# Run this script in PowerShell

Write-Host "🎯 Starting Fake Account Detection System..." -ForegroundColor Cyan
Write-Host ""

# Check if model exists
$modelPath = "models\detector.pkl"
if (-Not (Test-Path $modelPath)) {
    Write-Host "⚠️  Model not found. Training now..." -ForegroundColor Yellow
    python backend\model_training.py
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "❌ Model training failed. Please check errors above." -ForegroundColor Red
        exit 1
    }
}

# Start backend API in background
Write-Host ""
Write-Host "🔧 Starting Backend API..." -ForegroundColor Yellow
$backendJob = Start-Job -ScriptBlock {
    Set-Location $using:PWD
    python backend\app.py
}

Write-Host "   Waiting for API to initialize..."
Start-Sleep -Seconds 5

if ($backendJob.State -eq "Running") {
    Write-Host "   ✓ Backend API started (Job ID: $($backendJob.Id))" -ForegroundColor Green
}
else {
    Write-Host "   ❌ Backend failed to start" -ForegroundColor Red
    exit 1
}

# Start dashboard
Write-Host ""
Write-Host "🎨 Starting Dashboard..." -ForegroundColor Yellow
$dashboardJob = Start-Job -ScriptBlock {
    Set-Location $using:PWD
    python frontend\dashboard.py
}

Start-Sleep -Seconds 3

if ($dashboardJob.State -eq "Running") {
    Write-Host "   ✓ Dashboard started (Job ID: $($dashboardJob.Id))" -ForegroundColor Green
}
else {
    Write-Host "   ❌ Dashboard failed to start" -ForegroundColor Red
    Stop-Job $backendJob
    Remove-Job $backendJob
    exit 1
}

Write-Host ""
Write-Host "=" * 50 -ForegroundColor Cyan
Write-Host "✅ System is running!" -ForegroundColor Green
Write-Host "=" * 50 -ForegroundColor Cyan
Write-Host ""
Write-Host "📊 Dashboard: http://localhost:8050" -ForegroundColor Yellow
Write-Host "🔌 API: http://localhost:5000" -ForegroundColor Yellow
Write-Host ""
Write-Host "Press Ctrl+C to stop all services" -ForegroundColor Gray
Write-Host ""

# Keep script running and monitor jobs
try {
    while ($true) {
        Start-Sleep -Seconds 2
        
        # Check if jobs are still running
        if ($backendJob.State -ne "Running" -or $dashboardJob.State -ne "Running") {
            Write-Host ""
            Write-Host "⚠️  One or more services stopped unexpectedly" -ForegroundColor Red
            break
        }
    }
}
finally {
    Write-Host ""
    Write-Host "Stopping services..." -ForegroundColor Yellow
    Stop-Job $backendJob, $dashboardJob
    Remove-Job $backendJob, $dashboardJob
    Write-Host "✓ All services stopped" -ForegroundColor Green
}
