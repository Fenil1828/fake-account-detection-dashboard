#!/bin/bash

echo "🚀 Setting up Fake Account Detector..."
echo ""

# Create directories if they don't exist
echo "📁 Creating directories..."
mkdir -p data/raw data/processed models

# Download sample dataset
echo ""
echo "📥 Downloading sample dataset..."
cd data/raw

if [ ! -f "twitter_bots.csv" ]; then
    echo "   Downloading Twitter bot dataset..."
    curl -L -o twitter_bots.csv "https://raw.githubusercontent.com/jubins/Twitter-Bot-Detection/master/datasets/twitter_human_bots_dataset.csv"
    
    if [ $? -eq 0 ]; then
        echo "   ✓ Dataset downloaded successfully"
    else
        echo "   ⚠️  Download failed. You may need to download manually."
    fi
else
    echo "   ✓ Dataset already exists"
fi

cd ../..

# Install requirements
echo ""
echo "📦 Installing Python packages..."
pip install -r requirements.txt

if [ $? -eq 0 ]; then
    echo "   ✓ Packages installed successfully"
else
    echo "   ⚠️  Some packages may have failed to install"
fi

echo ""
echo "="*50
echo "✅ Setup complete!"
echo "="*50
echo ""
echo "Next steps:"
echo "1. Train the model: python backend/model_training.py"
echo "2. Start the API: python backend/app.py"
echo "3. Start the dashboard: python frontend/dashboard.py"
echo ""
