# Fake Account Detection Dashboard

> A comprehensive ML-powered fake account detection system with real-time analysis, beautiful UI, and advanced security monitoring.

[![Version](https://img.shields.io/badge/version-1.0.0-blue.svg)](https://github.com/Fenil1828/fake-account-detection-dashboard)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![Node.js](https://img.shields.io/badge/node.js-16%2B-green.svg)](https://nodejs.org/)

## 🎯 Project Overview

This is a full-stack application designed to detect and analyze fake/bot Twitter accounts using machine learning. The system combines advanced ML models with an intuitive, beautiful user interface to provide real-time threat assessment and comprehensive account analysis.

### Key Capabilities
- **ML-Powered Detection** - Trained bot detection model with ~92% accuracy
- **Real-time Analysis** - Instant risk scoring for individual or batch accounts
- **Interactive Visualizations** - Network graphs and threat charts
- **Comprehensive Metrics** - Calculate 6+ account behavior metrics
- **Security Monitoring** - Real-time threat tracking and event logging
- **Responsive Design** - Works seamlessly on desktop, tablet, and mobile

## 📁 Project Structure

```
fake-account-detector/
├── backend/                          # Flask ML API Server
│   ├── app.py                       # Main Flask application
│   ├── feature_extraction.py        # Feature extraction pipeline
│   ├── ml_security.py               # ML model wrapper
│   ├── model_training.py            # Training pipeline
│   └── utils.py                     # Utilities
│
├── frontend/                         # React.js Dashboard
│   ├── react-dashboard/
│   │   ├── src/
│   │   │   ├── components/          # React components
│   │   │   ├── styles/              # CSS stylesheets
│   │   │   ├── utils/               # Helper functions
│   │   │   ├── hooks/               # Custom hooks
│   │   │   └── main.jsx             # Entry point
│   │   ├── package.json
│   │   ├── vite.config.js
│   │   └── index.html
│   └── dashboard.py                 # Legacy Streamlit dashboard
│
├── data/                            # Dataset directory
│   ├── raw/                         # Raw data files
│   └── processed/                   # Processed data
│
├── models/                          # Trained ML models
│
├── scripts/                         # Utility scripts
│   ├── generate_sample_data.py
│   ├── setup.sh / setup.ps1
│   └── run_all.sh / run_all.ps1
│
├── requirements.txt                 # Python dependencies
├── QUICKSTART.md
└── README.md                        # This file
```

## 🚀 Quick Start

### Prerequisites
- **Python** 3.8 or higher
- **Node.js** 16 or higher
- **npm** or **yarn**

### Automated Setup (Windows)
```bash
cd fake-account-detector
.\START.bat
```

### Manual Setup

#### 1. Backend Setup
```bash
# Create virtual environment
python -m venv .venv

# Activate (Windows)
.\.venv\Scripts\Activate.ps1

# Activate (macOS/Linux)
source .venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt
```

#### 2. Frontend Setup
```bash
cd frontend/react-dashboard

# Install Node dependencies
npm install

# Start development server
npm run dev
```

#### 3. Start Backend
```bash
cd backend
python app.py
```

### Access the Application
- **Frontend Dashboard:** http://localhost:5173
- **Backend API:** http://localhost:5000

## ✨ Features

### 🎨 Frontend Features
| Feature | Description |
|---------|-------------|
| **Manual Account Entry** | Add accounts with detailed metrics via intuitive form |
| **Sample Generation** | Quickly generate 50 or 100 realistic test accounts |
| **CSV Bulk Import** | Upload multiple accounts from CSV files (max 10MB) |
| **Network Visualization** | Interactive graph showing account relationships |
| **Real-time Analysis** | Instant threat assessment for submitted accounts |
| **Advanced Filtering** | Filter by account type, risk level, and custom metrics |
| **Batch Operations** | Perform actions on multiple accounts simultaneously |
| **Security Monitor** | Real-time threat tracking and historical analytics |
| **Detailed Results** | Comprehensive results table with sorting capabilities |
| **Account Details Modal** | Deep dive into individual account information |
| **Beautiful UI** | Glass-morphism design with smooth animations |

### 🤖 Backend Features
| Feature | Description |
|---------|-------------|
| **ML Model** | Scikit-learn based classifier trained on bot dataset |
| **Feature Extraction** | Automatic calculation of account behavior metrics |
| **Risk Scoring** | Real-time account risk assessment (0-1 scale) |
| **Pattern Recognition** | Detect suspicious bot behavior patterns |
| **Input Validation** | Comprehensive validation of account data |
| **REST API** | Clean, documented API endpoints |
| **Rate Limiting** | API rate limiting for security |
| **Event Logging** | Security events logged in real-time |

### 🎭 Account Classification
The system classifies accounts into four categories:

| Type | Characteristics | Risk Level |
|------|-----------------|-----------|
| 👑 **Celebrity** | High followers, low following | Low |
| ⭐ **Influencer** | Moderate followers, high engagement | Low |
| 👤 **Regular** | Normal user profile | Low/Medium |
| 🤖 **Bot** | Suspicious patterns detected | High |

## 🔌 API Endpoints

### Analyze Accounts
```
POST /api/analyze
Content-Type: application/json
```

**Request Body:**
```json
{
  "accounts": [
    {
      "username": "user123",
      "followers_count": 1000,
      "friends_count": 500,
      "statuses_count": 5000,
      "account_age_days": 365,
      "verified": false,
      "has_profile_image": true,
      "bio": "Tech enthusiast",
      "location": "San Francisco"
    }
  ]
}
```

**Response:**
```json
{
  "results": [
    {
      "username": "user123",
      "risk_score": 0.25,
      "risk_level": "LOW",
      "account_type": "Regular",
      "metrics": {
        "posts_per_day": 13.7,
        "follow_ratio": 0.5,
        "engagement_rate": 0.68,
        "profile_completeness": 0.85,
        "account_age_score": 0.92,
        "average_likes": 42
      }
    }
  ]
}
```

## 📊 Metrics Calculated

The system calculates the following metrics for each account:

| Metric | Formula | Interpretation |
|--------|---------|-----------------|
| **Posts Per Day** | statuses_count / account_age_days | Account activity level |
| **Follow Ratio** | friends_count / followers_count | Following/followers balance |
| **Engagement Rate** | average_likes / followers_count | Content engagement level |
| **Profile Completeness** | 1.0 if bio + location + image | Profile information %age |
| **Account Age** | account_age_days | Days since creation |
| **Average Likes** | Estimated from followers | Typical engagement per post |

## 🎯 Risk Assessment Logic

Risk scores are determined by analyzing:

- **Account Age** - Newer accounts score higher (bot creation is recent)
- **Follow Ratio** - Extreme ratios (very high following or very low) indicate bots
- **Posts Per Day** - Abnormally high posting rates suggest automation
- **Profile Completeness** - Incomplete profiles (no bio, picture, location) are suspicious
- **Verification Status** - Verified accounts have lower baseline risk
- **Engagement Patterns** - Unusual engagement metrics can indicate bot behavior

**Risk Levels:**
- 🟢 **LOW** (0.0 - 0.3)
- 🟡 **MEDIUM** (0.3 - 0.7)
- 🔴 **HIGH** (0.7 - 1.0)

## 💾 Storage & Data

- **Trained Models:** `models/` directory
- **Training Dataset:** `data/raw/twitter_bots.csv`
- **Processed Data:** `data/processed/`
- **Logs:** Security events logged in real-time
- **Database:** In-memory (ready for MongoDB/PostgreSQL integration)

## 🔐 Security Features

- ✅ **Input Validation** - All user inputs are validated
- ✅ **Rate Limiting** - API endpoints have rate limiting
- ✅ **CORS Protection** - Secure cross-origin request handling
- ✅ **Security Logging** - All actions logged with timestamps
- ✅ **Data Privacy** - No personally identifiable information stored
- ✅ **Error Handling** - Comprehensive error handling without exposing internals

## 📱 Responsive Design

The application is fully responsive and optimized for:
- ✅ **Desktop** (1200px and above)
- ✅ **Tablet** (768px - 1024px)
- ✅ **Mobile** (below 768px)

## 🎨 Design System

- **Color Palette:** Purple gradients (#667eea → #764ba2) with modern accents
- **Typography:** Inter font family for clean, modern appearance
- **Effects:** Glass-morphism design with smooth animations
- **Architecture:** Modular, reusable component structure

## 📝 Component Guide

### ManualEntryPage
Full-page form for manually entering account data
- Real-time input validation
- Auto-calculated metrics
- Automatic account type detection
- Live preview cards
- Quick "Generate 50/100" buttons

### SampleGenerator
Modal for rapid account generation
- Bulk generation (50 or 100 accounts)
- Quick entry form
- Automatic redirect to manual entry

### NetworkGraph
Interactive visualization of account relationships
- Force-directed graph layout
- Real-time node/edge updates
- Clickable nodes for details

### SecurityMonitor
Real-time threat tracking dashboard
- Event timeline with timestamps
- Threat level gauge visualization
- Historical data and trends

### ResultsTable
Analysis results display with advanced features
- Sortable columns (click headers)
- Filterable data rows
- Batch selection and actions
- Export capabilities

## 🚦 Running Utility Scripts

### Generate Sample Data
```bash
python scripts/generate_sample_data.py
```
Creates realistic test accounts in `data/processed/`.

### Evaluate Model
```bash
python notebooks/evaluate_model.py
```
Runs model evaluation and generates metrics.

### Test API
```bash
python notebooks/test_api.py
```
Tests all API endpoints with sample data.

## 📦 Dependencies

### Backend (Python)
```
flask==2.3.0
scikit-learn==1.2.0
pandas==1.5.0
numpy==1.24.0
joblib==1.2.0
```

### Frontend (Node.js)
```
react@19.2.0
vite@7.3.1
lucide-react@latest
axios@latest
```

## 🐛 Troubleshooting

### Frontend Won't Load
- Clear browser cache: `Ctrl+Shift+Delete` (Windows) or `Cmd+Shift+Delete` (Mac)
- Restart dev server: `npm run dev`
- Ensure port 5173 is available: `netstat -ano | findstr :5173`
- Check Node.js version: `node --version`

### Backend Errors
- Verify Python version: `python --version` (should be 3.8+)
- Reinstall dependencies: `pip install --upgrade -r requirements.txt`
- Check port 5000 availability: `lsof -i :5000`
- Review Flask logs for detailed error messages

### CSV Import Fails
- Verify CSV headers match expected format (username, followers_count, etc.)
- Check file encoding (must be UTF-8)
- Verify file size (max 10MB)
- Ensure no special characters in usernames

### Model Not Loading
- Confirm `models/` directory exists
- Verify model files are present: `ls models/`
- Retrain if needed: `python notebooks/model_training.py`
- Check model file permissions

## 📚 File Descriptions

### Backend Files

**app.py**
Main Flask application entry point. Handles API routes, CORS configuration, and request/response processing.

**feature_extraction.py**
Calculates account metrics and performs feature engineering. Normalizes data for ML model input.

**ml_security.py**
ML model wrapper. Handles model loading, predictions, and risk scoring.

**model_training.py**
Complete model training pipeline including data preparation and evaluation.

**utils.py**
Helper functions for data processing and input validation.

### Key Frontend Components

**App.jsx**
Main application component with state management and routing logic.

**ManualEntryPage.jsx**
Full-page form interface with real-time validation and metrics calculation.

**SampleGenerator.jsx**
Account generation modal supporting bulk and quick entry modes.

**NetworkGraph.jsx**
Interactive graph visualization using force-directed layout.

**SecurityMonitor.jsx**
Real-time event tracking with threat level display.

**ResultsTable.jsx**
Analysis results display with sorting, filtering, and batch operations.

## 🔄 Workflow

### Standard Analysis Flow

1. **Data Input**
   - Manual entry via ManualEntryPage, or
   - CSV upload via CSVUploader, or
   - Sample generation via SampleGenerator

2. **Processing**
   - Frontend validates data
   - Submits to backend API
   - Backend extracts features

3. **ML Analysis**
   - Model scores each account
   - Risk level calculated
   - Account type determined

4. **Results Display**
   - Results shown in table
   - Network graph updated
   - Security monitor refreshed

5. **User Actions**
   - Filter and sort results
   - View account details
   - Perform batch operations
   - Export data

## 📈 Performance Metrics

- **Model Accuracy:** ~92%
- **API Response Time:** <500ms per request
- **Max Batch Size:** 1000 accounts per request
- **UI Frame Rate:** 60 FPS
- **Frontend Load Time:** <2 seconds

## 🎓 Technical Stack

| Layer | Technology |
|-------|-----------|
| **Frontend** | React 19.2, Vite 7.3, Tailwind CSS |
| **Backend** | Flask, Scikit-learn, Pandas |
| **ML Model** | Random Forest Classifier |
| **Visualization** | D3.js (via custom React wrapper) |
| **API** | RESTful HTTP with JSON |

## 👨‍💻 Development Guide

### Code Style
- **React:** Functional components with hooks
- **Python:** PEP 8 compliant
- **CSS:** BEM naming convention

### Best Practices
- Component memoization for performance
- Lazy loading for large datasets
- Optimized re-renders using React keys
- Efficient API call batching

## 📄 License

MIT License - See LICENSE file for details

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Make your changes and test thoroughly
4. Commit with clear messages: `git commit -m 'Add amazing feature'`
5. Push to the branch: `git push origin feature/amazing-feature`
6. Open a Pull Request

## 📞 Support & Contact

For issues, questions, or feedback:
1. Check the Troubleshooting section above
2. Review logs in browser console and backend terminal
3. Check the network tab in browser DevTools
4. Verify both frontend and backend services are running

## 🚀 Deployment Guide

### Production Build
```bash
cd frontend/react-dashboard
npm run build
```
Outputs optimized bundle to `dist/` directory.

### Deploy Frontend
- Build outputs to `dist/`
- Deploy to Vercel, Netlify, GitHub Pages, or your hosting provider
- Set `VITE_API_URL` environment variable to production backend

### Deploy Backend
- Use Gunicorn with Nginx reverse proxy
- Set environment variables (API keys, secrets)
- Configure CORS for production domain
- Enable HTTPS
- Set up proper logging and monitoring

Example Gunicorn command:
```bash
gunicorn --workers 4 --bind 0.0.0.0:5000 app:app
```

## 📅 Roadmap

### Current Version (v1.0.0)
- ✅ Manual account entry with Generate 50/100 buttons
- ✅ Sample account generator
- ✅ ML-based fake account detection
- ✅ Real-time security monitoring
- ✅ Beautiful responsive UI
- ✅ CSV bulk import

### Planned Features (v1.1.0+)
- [ ] Database integration (MongoDB/PostgreSQL)
- [ ] User authentication and accounts
- [ ] Advanced ML models (XGBoost, Neural Networks)
- [ ] Email notifications for threats
- [ ] Account tracking over time
- [ ] Report generation and export
- [ ] Team collaboration features
- [ ] API key management
- [ ] Webhook integrations
- [ ] Data export to multiple formats

## 📊 Project Statistics

- **Files:** 50+
- **Components:** 12+
- **API Endpoints:** 5+
- **Test Cases:** Comprehensive
- **Documentation:** Complete
- **Code Coverage:** Improving

## 🙏 Acknowledgments

Built with ❤️ for the HKTN hackathon.

Special thanks to:
- The open-source community
- Scikit-learn and Flask teams
- React and Vite communities

## 📚 Additional Resources

- [Flask Documentation](https://flask.palletsprojects.com/)
- [React Documentation](https://react.dev/)
- [Scikit-learn Guide](https://scikit-learn.org/)
- [Vite Guide](https://vitejs.dev/)

## 🔗 Links

- **Repository:** https://github.com/Fenil1828/fake-account-detection-dashboard
- **Issues:** https://github.com/Fenil1828/fake-account-detection-dashboard/issues
- **Discussions:** https://github.com/Fenil1828/fake-account-detection-dashboard/discussions

---

<div align="center">

**Made with ❤️ for Fake Account Detection**

[⬆ Back to top](#fake-account-detection-dashboard)

</div>
