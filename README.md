# Fake Account Detection & Risk Analysis Dashboard

A comprehensive web-based dashboard for detecting fake social media accounts using machine learning. This system analyzes behavioral patterns, network characteristics, and content signals to identify bot accounts, spam profiles, and suspicious behavior.

![Dashboard Preview](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Flask](https://img.shields.io/badge/Flask-3.0-green.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4-orange.svg)

## Features

### 🔍 Account Analysis
- **Behavioral Pattern Analysis**: Login times, session duration, posting activity
- **Follower/Following Anomalies**: Extreme ratios, unusual growth patterns
- **Posting Frequency**: Daily/weekly activity timelines
- **Network Visualization**: Social graph visualization with NetworkX

### 🤖 Machine Learning Classification
- **Ensemble Model**: Combines Random Forest, Gradient Boosting, and Logistic Regression
- **Risk Score**: 0-1 probability of being fake
- **Confidence Level**: Model certainty in classification
- **Feature Importance**: Understand which factors drive predictions

### 📊 Explainability
- Human-readable explanations for each classification
- Suspicious factors highlighted with severity levels
- Positive indicators that support authenticity
- Rule-based reasoning combined with ML insights

### 📈 Dashboard Features
- Single account analysis with detailed visualizations
- Batch processing via CSV upload
- Model performance report with metrics
- Export reports to CSV

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup

1. **Clone or navigate to the project directory**:
```bash
cd "c:\Users\HP\OneDrive\Desktop\Fake Account Detector"
```

2. **Create a virtual environment** (recommended):
```bash
python -m venv venv

# On Windows:
venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

4. **Train the model** (first time only):
```bash
python model.py
```

5. **Run the application**:
```bash
python app.py
```

6. **Open your browser** and navigate to:
```
http://localhost:5000
```

## Usage

### Single Account Analysis

1. Go to **Dashboard** or **Single Analysis** page
2. Enter account details:
   - Username
   - Followers/Following counts
   - Posts count
   - Account age
   - Profile completeness indicators
   - Engagement metrics
3. Click **Analyze Account**
4. View risk score, classification, and detailed explanation

### Batch Analysis (CSV Upload)

1. Go to **Batch Analysis** page
2. Prepare a CSV file with the following columns:
   ```
   username,followers_count,following_count,posts_count,account_age_days,has_profile_pic,bio
   ```
3. Upload the CSV file
4. View aggregate results and individual account details
5. Export results to CSV

### Model Report

View detailed model performance metrics:
- Accuracy, Precision, Recall, F1-Score
- Confusion Matrix
- Feature Importance
- Cross-validation results

## Project Structure

```
Fake Account Detector/
├── app.py                 # Flask application
├── model.py               # ML model training and inference
├── utils.py               # Helper functions and visualizations
├── requirements.txt       # Python dependencies
├── README.md              # This file
│
├── templates/             # HTML templates
│   ├── index.html         # Main dashboard
│   ├── analysis.html      # Single account analysis
│   ├── batch.html         # Batch processing
│   ├── report.html        # Model report
│   └── 404.html           # Error page
│
├── static/
│   ├── css/
│   │   └── style.css      # Custom styles
│   └── js/
│       └── main.js        # Frontend JavaScript
│
├── data/                  # Generated data
│   ├── synthetic_dataset.csv
│   └── model_metrics.json
│
└── models/                # Saved models
    └── fake_detector_model.pkl
```

## API Endpoints

### POST `/api/analyze`
Analyze a single account.

**Request Body (JSON or Form):**
```json
{
  "username": "example_user",
  "followers_count": 1000,
  "following_count": 500,
  "posts_count": 100,
  "account_age_days": 365,
  "has_profile_pic": true,
  "bio": "Example bio text"
}
```

**Response:**
```json
{
  "success": true,
  "result": {
    "is_fake": false,
    "risk_score": 0.15,
    "confidence": 0.92,
    "classification": "Genuine Account",
    "risk_level": "Very Low",
    "explanation": {...},
    "suspicious_attributes": [...],
    "positive_attributes": [...]
  }
}
```

### POST `/api/analyze-batch`
Analyze multiple accounts from CSV upload.

### GET `/api/sample-accounts`
Get sample account data for testing.

### GET `/api/model-metrics`
Get model performance metrics.

### POST `/api/export-report`
Export analysis results to CSV.

## Model Details

### Features Used (25 total)

| Category | Features |
|----------|----------|
| **Basic Stats** | followers_count, following_count, follower_following_ratio, posts_count |
| **Account Info** | account_age_days, avg_posts_per_day, profile_completeness |
| **Profile** | has_profile_pic, has_bio, bio_length, username_length, username_has_numbers |
| **Engagement** | avg_likes_per_post, avg_comments_per_post, engagement_rate |
| **Behavior** | posting_regularity, session_duration_avg, login_frequency, burst_posting_score |
| **Content** | avg_caption_length, hashtag_density, spam_word_count, duplicate_content_ratio |
| **Metadata** | verified_status, external_url |

### Model Architecture

- **Random Forest**: 100 trees, max depth 10 (weight: 2x)
- **Gradient Boosting**: 100 estimators, learning rate 0.1 (weight: 2x)
- **Logistic Regression**: L2 regularization (weight: 1x)
- **Ensemble**: Soft voting classifier

### Expected Performance

- Accuracy: ~92%
- Precision: ~90%
- Recall: ~91%
- F1-Score: ~90%

## Customization

### Retraining the Model

You can retrain the model with different parameters:

```python
from model import FakeAccountDetector, generate_synthetic_dataset

# Generate new dataset
df = generate_synthetic_dataset(n_samples=5000, fake_ratio=0.4)

# Train model
detector = FakeAccountDetector()
feature_columns = [col for col in df.columns if col not in ['label', 'username']]
X = df[feature_columns].values
y = df['label'].values

metrics = detector.train(X, y)
detector.save_model()
```

### Using Real Data

Replace the synthetic data generation with your own dataset:

1. Prepare a CSV with the required columns
2. Modify `model.py` to load your data
3. Retrain the model

## Technologies Used

- **Backend**: Flask, Flask-CORS
- **ML**: scikit-learn, NumPy, Pandas
- **Visualization**: Plotly, NetworkX
- **Frontend**: Bootstrap 5, Bootstrap Icons
- **Charts**: Plotly.js

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

This project is for educational purposes. Feel free to use and modify for your own projects.

## References

Based on research paper: "Fake Social Media Profile Detection And Reporting" (Metallurgical and Materials Engineering, Vol 31, 2025)

## Support

For issues or questions, please open an issue on the repository.
