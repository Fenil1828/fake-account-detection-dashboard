# 🚀 Quick Start Guide - Fake Account Detection System

## For Windows Users

### Step 1: Navigate to Project Directory
```powershell
cd C:\Users\VRUSHTI\Documents\GitHub\HKTN\fake-account-detector
```

### Step 2: Run Setup (One-Time)
```powershell
# Run PowerShell as Administrator (if needed)
powershell -ExecutionPolicy Bypass -File scripts\setup.ps1
```

### Step 3: Train the Model
```powershell
python backend\model_training.py
```

### Step 4: Start the System

**Option A: Start All Services Automatically**
```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_all.ps1
```

**Option B: Start Services Manually**

Terminal 1 - Start API:
```powershell
python backend\app.py
```

Terminal 2 - Start Dashboard:
```powershell
python frontend\dashboard.py
```

### Step 5: Access the Dashboard
Open your browser and go to:
- **Dashboard:** http://localhost:8050
- **API:** http://localhost:5000

---

## For Linux/Mac Users

### Step 1: Navigate to Project Directory
```bash
cd ~/path/to/fake-account-detector
```

### Step 2: Run Setup (One-Time)
```bash
chmod +x scripts/setup.sh
./scripts/setup.sh
```

### Step 3: Train the Model
```bash
python backend/model_training.py
```

### Step 4: Start the System

**Option A: Start All Services Automatically**
```bash
chmod +x scripts/run_all.sh
./scripts/run_all.sh
```

**Option B: Start Services Manually**

Terminal 1 - Start API:
```bash
python backend/app.py
```

Terminal 2 - Start Dashboard:
```bash
python frontend/dashboard.py
```

### Step 5: Access the Dashboard
Open your browser and go to:
- **Dashboard:** http://localhost:8050
- **API:** http://localhost:5000

---

## 📊 Generate Performance Report

After training the model, generate a comprehensive performance report:

```powershell
# Windows
python notebooks\evaluate_model.py

# Linux/Mac
python notebooks/evaluate_model.py
```

This will create:
- `notebooks/performance_report.json` - Detailed metrics in JSON
- `notebooks/PERFORMANCE_REPORT.md` - Human-readable report
- `notebooks/performance_plots.png` - Visualization charts

---

## 🧪 Test the API

### Using curl (Windows PowerShell)
```powershell
$body = @{
    username = "test_bot123"
    followers_count = 5
    friends_count = 5000
    statuses_count = 10000
    account_age_days = 30
    has_profile_image = $false
    bio = ""
    verified = $false
    favourites_count = 100
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://localhost:5000/api/analyze" -Method Post -Body $body -ContentType "application/json"
```

### Using curl (Linux/Mac)
```bash
curl -X POST http://localhost:5000/api/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "username": "test_bot123",
    "followers_count": 5,
    "friends_count": 5000,
    "statuses_count": 10000,
    "account_age_days": 30,
    "has_profile_image": false,
    "bio": "",
    "verified": false,
    "favourites_count": 100
  }'
```

---

## 🎯 Using the Dashboard

1. **Enter Account Details:**
   - Fill in the form on the left side
   - All fields are required for accurate analysis

2. **Click "Analyze Account":**
   - Wait for the analysis to complete (1-2 seconds)

3. **Review Results:**
   - **Risk Level:** Critical/High/Medium/Low
   - **Confidence Score:** Model's confidence in prediction
   - **Behavioral Analysis:** Posting patterns and activity
   - **Network Analysis:** Follower/following metrics
   - **Risk Factors:** Specific suspicious indicators
   - **Explanation:** Why the model made this decision

---

## ⚠️ Troubleshooting

### "Model not found" Error
```powershell
# Train the model first
python backend\model_training.py
```

### "Cannot connect to API" Error
```powershell
# Make sure API is running
python backend\app.py
```

### Port Already in Use
```powershell
# Windows - Find and kill process on port 5000
netstat -ano | findstr :5000
taskkill /PID <PID> /F

# Linux/Mac
lsof -ti:5000 | xargs kill -9
```

### Package Installation Errors
```powershell
# Upgrade pip first
python -m pip install --upgrade pip

# Install packages one by one
pip install pandas numpy scikit-learn
pip install flask flask-cors
pip install dash plotly
```

---

## 📁 Project Structure Quick Reference

```
fake-account-detector/
├── backend/              # API and ML components
│   ├── app.py           # Flask API server
│   ├── model_training.py # Train the model
│   └── feature_extraction.py # Feature engineering
├── frontend/            # Dashboard
│   └── dashboard.py     # Dash web interface
├── data/               # Datasets
│   └── raw/            # Raw data files
├── models/             # Trained models
│   └── detector.pkl    # Saved model
├── notebooks/          # Analysis and reports
│   └── evaluate_model.py # Performance evaluation
└── scripts/            # Automation scripts
    ├── setup.ps1       # Windows setup
    └── run_all.ps1     # Windows run all
```

---

## 🎓 Next Steps

1. ✅ **Explore the Dashboard** - Try different account profiles
2. ✅ **Read the Performance Report** - Understand model accuracy
3. ✅ **Test the API** - Integrate with your applications
4. ✅ **Customize Features** - Add your own detection rules
5. ✅ **Deploy** - Put the system into production

---

## 📞 Need Help?

- Check the full README.md for detailed documentation
- Review the performance report for model insights
- Test with the provided sample accounts
- Examine the code comments for implementation details

---

**Happy Detecting! 🔍**
