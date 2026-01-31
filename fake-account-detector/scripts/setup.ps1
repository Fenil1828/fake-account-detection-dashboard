# Fake Account Detection - Setup Script for Windows
# Run this script in PowerShell

Write-Host "🚀 Setting up Fake Account Detector..." -ForegroundColor Cyan
Write-Host ""

# Create directories if they don't exist
Write-Host "📁 Creating directories..." -ForegroundColor Yellow
New-Item -ItemType Directory -Force -Path "data\raw" | Out-Null
New-Item -ItemType Directory -Force -Path "data\processed" | Out-Null
New-Item -ItemType Directory -Force -Path "models" | Out-Null
Write-Host "   ✓ Directories created" -ForegroundColor Green

# Download sample dataset
Write-Host ""
Write-Host "📥 Downloading sample dataset..." -ForegroundColor Yellow
$datasetPath = "data\raw\twitter_bots.csv"

if (-Not (Test-Path $datasetPath)) {
    Write-Host "   Downloading Twitter bot dataset..."
    try {
        $url = "https://raw.githubusercontent.com/jubins/Twitter-Bot-Detection/master/datasets/twitter_human_bots_dataset.csv"
        Invoke-WebRequest -Uri $url -OutFile $datasetPath
        Write-Host "   ✓ Dataset downloaded successfully" -ForegroundColor Green
    }
    catch {
        Write-Host "   ⚠️  Download failed. You may need to download manually." -ForegroundColor Red
        Write-Host "   URL: $url"
    }
}
else {
    Write-Host "   ✓ Dataset already exists" -ForegroundColor Green
}

# Install requirements
Write-Host ""
Write-Host "📦 Installing Python packages..." -ForegroundColor Yellow
try {
    pip install -r requirements.txt
    Write-Host "   ✓ Packages installed successfully" -ForegroundColor Green
}
catch {
    Write-Host "   ⚠️  Some packages may have failed to install" -ForegroundColor Red
}

Write-Host ""
Write-Host "=" * 50 -ForegroundColor Cyan
Write-Host "✅ Setup complete!" -ForegroundColor Green
Write-Host "=" * 50 -ForegroundColor Cyan
Write-Host ""
Write-Host "Next steps:"
Write-Host "1. Train the model: python backend\model_training.py"
Write-Host "2. Start the API: python backend\app.py"
Write-Host "3. Start the dashboard: python frontend\dashboard.py"
Write-Host ""
