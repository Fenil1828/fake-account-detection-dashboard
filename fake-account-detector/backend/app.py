from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import os
import sys
from datetime import datetime
import traceback

# Add backend to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import classes needed for pickle loading
from model_training import FakeAccountDetector
from feature_extraction import FeatureExtractor

app = Flask(__name__)
CORS(app)

# Load model
MODEL_PATH = 'models/detector.pkl'
detector = None

def load_detector():
    """Load the trained model"""
    global detector
    if os.path.exists(MODEL_PATH):
        try:
            detector = joblib.load(MODEL_PATH)
            print(f"✓ Model loaded from {MODEL_PATH}")
            return True
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    else:
        print(f"⚠️  Model not found at {MODEL_PATH}")
        print("   Please train the model first: python backend/model_training.py")
        return False

# Try to load model on startup
load_detector()

@app.route('/')
def home():
    return jsonify({
        'message': 'Fake Account Detection API',
        'status': 'running',
        'version': '1.0.0',
        'endpoints': {
            'analyze': 'POST /api/analyze - Analyze single account',
            'batch': 'POST /api/batch - Analyze multiple accounts',
            'health': 'GET /api/health - Check API health',
            'metrics': 'GET /api/metrics - Get model performance metrics'
        }
    })

@app.route('/api/health')
def health():
    return jsonify({
        'status': 'healthy',
        'model_loaded': detector is not None,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/metrics')
def get_metrics():
    """Get model performance metrics"""
    if detector is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    if hasattr(detector, 'training_metrics'):
        return jsonify({
            'metrics': detector.training_metrics,
            'timestamp': datetime.now().isoformat()
        })
    else:
        return jsonify({'error': 'Training metrics not available'}), 404

@app.route('/api/analyze', methods=['POST'])
def analyze_account():
    """Analyze a single social media account"""
    try:
        if detector is None:
            return jsonify({
                'error': 'Model not loaded',
                'message': 'Please train the model first: python backend/model_training.py'
            }), 500
        
        account_data = request.json
        
        if not account_data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Validate required fields
        required_fields = ['username']
        missing_fields = [f for f in required_fields if f not in account_data]
        if missing_fields:
            return jsonify({
                'error': 'Missing required fields',
                'missing': missing_fields
            }), 400
        
        # Make prediction
        prediction = detector.predict(account_data)
        
        # Generate explanation
        explanation = detector.explain_prediction(account_data)
        
        # Analyze behavioral patterns
        behavioral_analysis = analyze_behavior(account_data)
        
        # Network analysis
        network_analysis = analyze_network(account_data)
        
        # Identify risk factors
        risk_factors = identify_risk_factors(account_data, prediction)
        
        response = {
            'username': account_data.get('username'),
            'prediction': prediction,
            'explanation': explanation,
            'behavioral_analysis': behavioral_analysis,
            'network_analysis': network_analysis,
            'risk_factors': risk_factors,
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(response)
    
    except Exception as e:
        print(f"Error in analyze_account: {e}")
        traceback.print_exc()
        return jsonify({
            'error': str(e),
            'type': type(e).__name__
        }), 500

@app.route('/api/batch', methods=['POST'])
def batch_analyze():
    """Analyze multiple accounts"""
    try:
        if detector is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        data = request.json
        accounts = data.get('accounts', [])
        
        if not accounts:
            return jsonify({'error': 'No accounts provided'}), 400
        
        results = []
        fake_count = 0
        
        for account in accounts:
            try:
                prediction = detector.predict(account)
                if prediction['is_fake']:
                    fake_count += 1
                    
                results.append({
                    'username': account.get('username', 'unknown'),
                    'prediction': prediction,
                    'risk_level': prediction['risk_level']
                })
            except Exception as e:
                results.append({
                    'username': account.get('username', 'unknown'),
                    'error': str(e)
                })
        
        return jsonify({
            'total_analyzed': len(results),
            'fake_accounts_detected': fake_count,
            'real_accounts': len(results) - fake_count,
            'results': results,
            'timestamp': datetime.now().isoformat()
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def analyze_behavior(account_data):
    """Analyze behavioral patterns"""
    followers = account_data.get('followers_count', 0)
    following = account_data.get('friends_count', 0)
    tweets = account_data.get('statuses_count', 0)
    account_age = account_data.get('account_age_days', 1)
    
    tweets_per_day = tweets / max(account_age, 1)
    
    patterns = {
        'posting_frequency': 'very_high' if tweets_per_day > 50 else 'high' if tweets_per_day > 10 else 'normal',
        'tweets_per_day': round(tweets_per_day, 2),
        'total_tweets': tweets,
        'follower_pattern': 'suspicious' if following > followers * 10 else 'normal',
        'engagement_level': 'low' if tweets > 100 and followers < 10 else 'normal',
        'account_activity': 'active' if tweets_per_day > 1 else 'moderate' if tweets_per_day > 0.1 else 'inactive'
    }
    
    return patterns

def analyze_network(account_data):
    """Analyze network characteristics"""
    followers = account_data.get('followers_count', 0)
    following = account_data.get('friends_count', 0)
    
    ratio = followers / max(following, 1)
    
    analysis = {
        'follower_count': followers,
        'following_count': following,
        'ratio': round(ratio, 2),
        'assessment': 'normal'
    }
    
    # Determine network assessment
    if following > 2000 and followers < 100:
        analysis['assessment'] = 'suspicious - follows many, few followers'
        analysis['risk_indicator'] = 'high'
    elif followers > 10000 and following < 10:
        analysis['assessment'] = 'celebrity/influencer pattern'
        analysis['risk_indicator'] = 'low'
    elif ratio < 0.1 and following > 500:
        analysis['assessment'] = 'potential bot - low follower ratio'
        analysis['risk_indicator'] = 'high'
    elif ratio > 10 and followers > 1000:
        analysis['assessment'] = 'popular account'
        analysis['risk_indicator'] = 'low'
    else:
        analysis['risk_indicator'] = 'medium'
    
    return analysis

def identify_risk_factors(account_data, prediction):
    """Identify specific risk factors"""
    risk_factors = []
    
    # Profile completeness
    if not account_data.get('has_profile_image', True):
        risk_factors.append({
            'factor': 'No profile picture',
            'severity': 'medium',
            'description': 'Account lacks a profile image'
        })
    
    if not account_data.get('bio', ''):
        risk_factors.append({
            'factor': 'Empty bio',
            'severity': 'low',
            'description': 'No bio description provided'
        })
    
    # Network anomalies
    following = account_data.get('friends_count', 0)
    followers = account_data.get('followers_count', 0)
    
    if following > 2000:
        risk_factors.append({
            'factor': 'Following too many accounts',
            'severity': 'high',
            'description': f'Following {following} accounts (threshold: 2000)'
        })
    
    if followers < 10 and account_data.get('account_age_days', 0) > 30:
        risk_factors.append({
            'factor': 'Very low follower count',
            'severity': 'medium',
            'description': f'Only {followers} followers despite account age'
        })
    
    # Username patterns
    username = account_data.get('username', '')
    if any(char.isdigit() for char in username):
        if sum(char.isdigit() for char in username) > 3:
            risk_factors.append({
                'factor': 'Username contains many numbers',
                'severity': 'low',
                'description': 'Usernames with many numbers often indicate auto-generated accounts'
            })
    
    # Behavioral patterns
    tweets = account_data.get('statuses_count', 0)
    age = account_data.get('account_age_days', 1)
    tweets_per_day = tweets / max(age, 1)
    
    if tweets_per_day > 50:
        risk_factors.append({
            'factor': 'Extremely high posting frequency',
            'severity': 'high',
            'description': f'{tweets_per_day:.1f} tweets per day (possible automation)'
        })
    
    # ML model confidence
    if prediction['fake_probability'] > 0.7:
        risk_factors.append({
            'factor': 'High ML model confidence for fake account',
            'severity': 'critical',
            'description': f'{prediction["fake_probability"]*100:.1f}% probability of being fake'
        })
    
    return risk_factors

if __name__ == '__main__':
    print("\n" + "="*50)
    print("FAKE ACCOUNT DETECTION API")
    print("="*50)
    print(f"\n🚀 Starting API server...")
    print(f"📍 API will be available at http://localhost:5000")
    print(f"📊 Model status: {'Loaded ✓' if detector else 'Not loaded ⚠️'}")
    print("\n" + "="*50 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
