# 🔍 Fake Account Detection & Risk Analysis Dashboard

Advanced ML-powered system for detecting fake social media accounts with comprehensive risk analysis and explainability.

## 🎯 Project Overview

This project implements a complete fake account detection system using machine learning, featuring:
- **25+ behavioral and profile features** for comprehensive analysis
- **Gradient Boosting Classifier** with high accuracy
- **Interactive web dashboard** with real-time analysis
- **RESTful API** for integration
- **Explainable AI** with detailed risk factor identification

## 📋 Deliverables

### ✅ 1. Working Dashboard
- Real-time account analysis interface
- Behavioral pattern visualization
- Follower/following anomaly detection
- Posting frequency timelines
- Network connection graphs
- Risk level indicators with color-coded alerts

### ✅ 2. Trained Classification Model
- Gradient Boosting Classifier (100 estimators)
- 25+ engineered features
- Behavioral, network, profile, and content analysis
- Cross-validated performance metrics
- Saved model for instant predictions

### ✅ 3. Backend API Service
- Flask-based RESTful API
- `/api/analyze` - Single account analysis
- `/api/batch` - Bulk account processing
- `/api/health` - System health check
- `/api/metrics` - Model performance metrics
- Structured JSON responses with confidence scores

### ✅ 4. Explainability Component
- Feature importance analysis
- Top 5 influential factors per prediction
- Human-readable interpretations
- Risk factor categorization (Critical/High/Medium/Low)
- Detailed behavioral and network assessments

### ✅ 5. Performance Report
- Accuracy, Precision, Recall, F1-Score
- ROC AUC Score
- Confusion Matrix
- Classification Report
- Sample case studies

## 🏗️ Project Structure

```
fake-account-detector/
│
├── backend/
│   ├── app.py                    # Flask API server
│   ├── model_training.py         # ML model training
│   ├── feature_extraction.py     # Feature engineering (25+ features)
│   └── utils.py                  # Helper functions
│
├── frontend/
│   ├── dashboard.py              # Dash web interface
│   └── visualizations.py         # Chart components
│
├── data/
│   ├── raw/                      # Raw datasets
│   └── processed/                # Processed data
│
├── models/
│   └── detector.pkl              # Trained ML model
│
├── scripts/
│   ├── setup.sh                  # Automated setup
│   └── run_all.sh                # Run all services
│
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- pip package manager
- 2GB free disk space

### Installation

#### Option 1: Automated Setup (Recommended)
```bash
# Navigate to project directory
cd fake-account-detector

# Run setup script
bash scripts/setup.sh

# Train model and start all services
bash scripts/run_all.sh
```

#### Option 2: Manual Setup
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download dataset (or use your own)
cd data/raw
curl -L -o twitter_bots.csv "https://raw.githubusercontent.com/jubins/Twitter-Bot-Detection/master/datasets/twitter_human_bots_dataset.csv"
cd ../..

# 3. Train the model
python backend/model_training.py

# 4. Start the API (in one terminal)
python backend/app.py

# 5. Start the dashboard (in another terminal)
python frontend/dashboard.py
```

### Access the System
- **Dashboard**: http://localhost:8050
- **API**: http://localhost:5000
- **API Docs**: http://localhost:5000 (root endpoint)

## 📊 Features Extracted

### Profile Features (10)
- Username length and patterns
- Profile picture presence
- Bio completeness
- Location and URL presence
- Verification status
- Account age

### Behavioral Features (6)
- Posting frequency (tweets per day)
- Total status count
- Engagement metrics (likes per day)
- Activity ratios

### Network Features (6)
- Follower count
- Following count
- Follower/following ratio
- Suspicious patterns detection

### Content Features (3)
- Average tweet length
- URL sharing rate
- Hashtag usage frequency

## 🔌 API Usage

### Analyze Single Account
```bash
curl -X POST http://localhost:5000/api/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "username": "suspicious_user123",
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

### Response Format
```json
{
  "username": "suspicious_user123",
  "prediction": {
    "is_fake": true,
    "fake_probability": 0.89,
    "real_probability": 0.11,
    "confidence": 0.89,
    "risk_level": "CRITICAL"
  },
  "explanation": [
    {
      "feature": "suspicious_ff_ratio",
      "value": 1,
      "importance": 0.15,
      "interpretation": "Suspicious follower/following ratio"
    }
  ],
  "behavioral_analysis": {
    "posting_frequency": "very_high",
    "tweets_per_day": 333.33,
    "account_activity": "active"
  },
  "network_analysis": {
    "follower_count": 5,
    "following_count": 5000,
    "ratio": 0.001,
    "assessment": "suspicious - follows many, few followers",
    "risk_indicator": "high"
  },
  "risk_factors": [
    {
      "factor": "No profile picture",
      "severity": "medium",
      "description": "Account lacks a profile image"
    }
  ]
}
```

### Batch Analysis
```bash
curl -X POST http://localhost:5000/api/batch \
  -H "Content-Type: application/json" \
  -d '{
    "accounts": [
      {"username": "user1", "followers_count": 100, ...},
      {"username": "user2", "followers_count": 50, ...}
    ]
  }'
```

## 🧪 Model Performance

The model is trained on the Twitter Bot Detection dataset with the following metrics:

| Metric | Score |
|--------|-------|
| Accuracy | ~85-90% |
| Precision | ~87% |
| Recall | ~83% |
| F1-Score | ~85% |
| ROC AUC | ~0.92 |

*Note: Actual metrics depend on dataset and training parameters*

## 🎨 Dashboard Features

### 1. Account Analysis Form
- Username input
- Follower/following counts
- Tweet statistics
- Account age
- Profile completeness indicators

### 2. Risk Assessment Display
- Color-coded risk levels (Critical/High/Medium/Low)
- Confidence percentage
- Real vs Fake classification
- Risk probability gauge

### 3. Behavioral Analysis
- Posting frequency metrics
- Account activity level
- Engagement patterns
- Timeline visualizations

### 4. Network Analysis
- Follower/following distribution
- Network ratio assessment
- Connection graphs
- Anomaly detection

### 5. Risk Factors Panel
- Categorized risk factors
- Severity indicators
- Detailed descriptions
- Actionable insights

### 6. Explainability Section
- Top 5 influential features
- Feature importance bars
- Human-readable interpretations
- Transparency in decision-making

## 🔧 Configuration

### Model Parameters
Edit `backend/model_training.py`:
```python
self.model = GradientBoostingClassifier(
    n_estimators=100,      # Number of boosting stages
    learning_rate=0.1,     # Learning rate
    max_depth=5,           # Maximum tree depth
    random_state=42
)
```

### API Settings
Edit `backend/app.py`:
```python
app.run(debug=True, host='0.0.0.0', port=5000)
```

### Dashboard Settings
Edit `frontend/dashboard.py`:
```python
app.run_server(debug=True, host='0.0.0.0', port=8050)
```

## 📈 Use Cases

1. **Social Media Platforms**: Automated bot detection
2. **Marketing Teams**: Influencer verification
3. **Security Teams**: Spam account identification
4. **Research**: Social media behavior analysis
5. **Compliance**: Fake account monitoring

## 🛡️ Risk Levels

- **CRITICAL** (≥80%): Immediate action required
- **HIGH** (60-79%): High probability of fake account
- **MEDIUM** (40-59%): Suspicious patterns detected
- **LOW** (<40%): Likely legitimate account

## 🔍 Detection Patterns

### Common Fake Account Indicators
- Default username patterns (user123)
- No profile picture
- Empty bio
- Following >> Followers (ratio > 10)
- Extremely high posting frequency (>50 tweets/day)
- Account age < 30 days with high activity
- Excessive number following (>2000)

## 📝 Sample Case Studies

### Case 1: Obvious Bot
```
Username: user98765
Followers: 3
Following: 8000
Tweets: 15000
Age: 20 days
Result: CRITICAL (95% fake probability)
```

### Case 2: Legitimate Account
```
Username: john_smith
Followers: 500
Following: 300
Tweets: 1200
Age: 1825 days (5 years)
Result: LOW (8% fake probability)
```

### Case 3: Suspicious Account
```
Username: promo_deals2024
Followers: 50
Following: 3000
Tweets: 5000
Age: 90 days
Result: HIGH (72% fake probability)
```

## 🐛 Troubleshooting

### Model Not Found
```bash
# Train the model first
python backend/model_training.py
```

### API Connection Error
```bash
# Make sure API is running
python backend/app.py

# Check if port 5000 is available
netstat -an | grep 5000
```

### Dashboard Not Loading
```bash
# Ensure all dependencies are installed
pip install -r requirements.txt

# Check if port 8050 is available
netstat -an | grep 8050
```

### Dataset Download Failed
Download manually from:
https://github.com/jubins/Twitter-Bot-Detection/tree/master/datasets

## 🚀 Future Enhancements

- [ ] Deep learning models (LSTM, Transformers)
- [ ] Real-time Twitter API integration
- [ ] Multi-platform support (Instagram, Facebook)
- [ ] Advanced NLP for content analysis
- [ ] User authentication and history
- [ ] Export reports to PDF
- [ ] Batch processing from CSV
- [ ] Model retraining interface

## 📄 License

This project is created for educational and research purposes.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues.

## 📧 Contact

For questions or support, please open an issue in the repository.

---

**Built with ❤️ using Python, Flask, Dash, and Scikit-learn**
