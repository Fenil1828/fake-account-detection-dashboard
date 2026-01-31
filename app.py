"""
Fake Account Detection & Risk Analysis Dashboard
=================================================
Flask-based web application for detecting fake social media accounts.
Provides manual input and CSV batch processing capabilities.
"""

from flask import Flask, render_template, request, jsonify, send_file, Response
from flask_cors import CORS
import os
import json
import pandas as pd
import io
from datetime import datetime

from model import FakeAccountDetector, train_model, generate_synthetic_dataset
from utils import (
    parse_csv_upload, clean_account_data, get_sample_accounts,
    generate_network_graph, generate_behavior_timeline, generate_engagement_chart,
    generate_follower_analysis, generate_risk_distribution_chart,
    generate_feature_importance_chart, generate_confusion_matrix_chart,
    generate_metrics_comparison_chart, export_report_csv
)

# Determine absolute paths
template_dir = os.path.abspath('templates')
static_dir = os.path.abspath('static')
app = Flask(__name__, template_folder=template_dir, static_folder=static_dir)
CORS(app)

# Initialize detector
detector = FakeAccountDetector()

# Load model if exists, otherwise train
if not detector.load_model():
    print("No existing model found. Training new model...")
    train_model()
    detector.load_model()


# ==================== Web Routes ====================

@app.route('/')
def index():
    """Main dashboard page."""
    return render_template('index.html')
    # return "Hello World - Server is running!"


@app.route('/analysis')
def analysis():
    """Detailed analysis page."""
    return render_template('analysis.html')


@app.route('/batch')
def batch():
    """Batch processing page."""
    return render_template('batch.html')


@app.route('/report')
def report():
    """Model performance report page."""
    return render_template('report.html')


# ==================== API Routes ====================

@app.route('/api/analyze', methods=['POST'])
def analyze_account():
    """
    Analyze a single account for fake detection.
    
    Accepts JSON with account data or form data.
    Returns risk assessment and explanation.
    """
    try:
        if request.is_json:
            account_data = request.get_json()
        else:
            # Parse form data
            account_data = {
                'username': request.form.get('username', ''),
                'followers_count': int(request.form.get('followers_count', 0)),
                'following_count': int(request.form.get('following_count', 0)),
                'posts_count': int(request.form.get('posts_count', 0)),
                'account_age_days': int(request.form.get('account_age_days', 30)),
                'has_profile_pic': request.form.get('has_profile_pic', 'true').lower() == 'true',
                'bio': request.form.get('bio', ''),
                'avg_likes_per_post': float(request.form.get('avg_likes_per_post', 0)),
                'avg_comments_per_post': float(request.form.get('avg_comments_per_post', 0)),
                'posting_regularity': float(request.form.get('posting_regularity', 0.5)),
                'session_duration_avg': float(request.form.get('session_duration_avg', 30)),
                'login_frequency': float(request.form.get('login_frequency', 1)),
                'burst_posting_score': float(request.form.get('burst_posting_score', 0.2)),
                'is_verified': request.form.get('is_verified', 'false').lower() == 'true',
                'external_url': request.form.get('external_url', ''),
                'avg_caption_length': float(request.form.get('avg_caption_length', 50)),
                'hashtag_density': float(request.form.get('hashtag_density', 0.1)),
                'spam_word_count': int(request.form.get('spam_word_count', 0)),
                'duplicate_content_ratio': float(request.form.get('duplicate_content_ratio', 0))
            }
        
        # Clean the data
        account_data = clean_account_data(account_data)
        
        # Get prediction
        result = detector.predict(account_data)
        result['username'] = account_data.get('username', 'unknown')
        result['account_data'] = account_data
        
        # Generate visualizations
        result['charts'] = {
            'timeline': generate_behavior_timeline(account_data),
            'engagement': generate_engagement_chart(account_data),
            'followers': generate_follower_analysis(account_data)
        }
        
        return jsonify({
            'success': True,
            'result': result
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400


@app.route('/api/analyze-batch', methods=['POST'])
def analyze_batch():
    """
    Analyze multiple accounts from CSV upload.
    
    Accepts CSV file with account data.
    Returns batch risk assessment.
    """
    try:
        if 'file' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No file uploaded'
            }), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({
                'success': False,
                'error': 'No file selected'
            }), 400
        
        if not file.filename.endswith('.csv'):
            return jsonify({
                'success': False,
                'error': 'Only CSV files are supported'
            }), 400
        
        # Parse CSV
        content = file.read()
        accounts = parse_csv_upload(content, file.filename)
        
        if not accounts:
            return jsonify({
                'success': False,
                'error': 'No valid accounts found in CSV'
            }), 400
        
        # Analyze each account
        results = []
        risk_scores = []
        
        for account in accounts:
            result = detector.predict(account)
            result['username'] = account.get('username', 'unknown')
            results.append(result)
            risk_scores.append(result['risk_score'])
        
        # Generate batch visualizations
        network_chart = generate_network_graph(accounts, risk_scores)
        risk_distribution = generate_risk_distribution_chart(risk_scores)
        
        # Summary statistics
        summary = {
            'total_accounts': len(results),
            'fake_count': sum(1 for r in results if r['is_fake']),
            'genuine_count': sum(1 for r in results if not r['is_fake']),
            'avg_risk_score': sum(risk_scores) / len(risk_scores),
            'high_risk_count': sum(1 for s in risk_scores if s >= 0.7),
            'medium_risk_count': sum(1 for s in risk_scores if 0.4 <= s < 0.7),
            'low_risk_count': sum(1 for s in risk_scores if s < 0.4)
        }
        
        return jsonify({
            'success': True,
            'results': results,
            'summary': summary,
            'charts': {
                'network': network_chart,
                'risk_distribution': risk_distribution
            }
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400


@app.route('/api/sample-accounts', methods=['GET'])
def get_samples():
    """Get sample account data for demonstration."""
    samples = get_sample_accounts()
    return jsonify({
        'success': True,
        'samples': samples
    })


@app.route('/api/model-metrics', methods=['GET'])
def get_model_metrics():
    """Get model performance metrics."""
    try:
        metrics = detector.metrics
        feature_importance = detector.get_feature_importance()
        
        # Generate visualization charts
        charts = {
            'metrics': generate_metrics_comparison_chart(metrics),
            'feature_importance': generate_feature_importance_chart(feature_importance)
        }
        
        if 'confusion_matrix' in metrics:
            charts['confusion_matrix'] = generate_confusion_matrix_chart(metrics['confusion_matrix'])
        
        return jsonify({
            'success': True,
            'metrics': {
                'accuracy': metrics.get('accuracy', 0),
                'precision': metrics.get('precision', 0),
                'recall': metrics.get('recall', 0),
                'f1_score': metrics.get('f1_score', 0),
                'cv_mean': metrics.get('cv_mean', 0),
                'cv_std': metrics.get('cv_std', 0)
            },
            'feature_importance': feature_importance,
            'charts': charts
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/export-report', methods=['POST'])
def export_report():
    """Export analysis results as CSV."""
    try:
        data = request.get_json()
        results = data.get('results', [])
        
        csv_content = export_report_csv(results)
        
        return Response(
            csv_content,
            mimetype='text/csv',
            headers={
                'Content-Disposition': f'attachment; filename=fake_account_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
            }
        )
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400


@app.route('/api/retrain', methods=['POST'])
def retrain_model():
    """Retrain the model with new parameters."""
    try:
        data = request.get_json() or {}
        n_samples = data.get('n_samples', 3000)
        fake_ratio = data.get('fake_ratio', 0.4)
        
        # Generate new dataset
        df = generate_synthetic_dataset(n_samples=n_samples, fake_ratio=fake_ratio)
        
        # Prepare features
        feature_columns = [col for col in df.columns if col not in ['label', 'username']]
        X = df[feature_columns].values
        y = df['label'].values
        
        # Train
        global detector
        detector = FakeAccountDetector()
        metrics = detector.train(X, y)
        detector.save_model()
        
        return jsonify({
            'success': True,
            'message': 'Model retrained successfully',
            'metrics': {
                'accuracy': metrics['accuracy'],
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'f1_score': metrics['f1_score']
            }
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/dataset-info', methods=['GET'])
def get_dataset_info():
    """Get information about the training dataset."""
    try:
        dataset_path = 'data/synthetic_dataset.csv'
        
        if os.path.exists(dataset_path):
            df = pd.read_csv(dataset_path)
            
            info = {
                'total_samples': len(df),
                'fake_samples': int(df['label'].sum()),
                'genuine_samples': int(len(df) - df['label'].sum()),
                'features': list(df.columns),
                'feature_count': len(df.columns) - 2  # Exclude label and username
            }
            
            return jsonify({
                'success': True,
                'dataset_info': info
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Dataset not found'
            }), 404
            
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


# ==================== Error Handlers ====================

@app.errorhandler(404)
def not_found(e):
    return render_template('404.html'), 404


@app.errorhandler(500)
def server_error(e):
    return jsonify({
        'success': False,
        'error': 'Internal server error'
    }), 500


# ==================== Main ====================

if __name__ == '__main__':
    # Ensure directories exist
    os.makedirs('data', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    # Run the app
    print(f"Current working directory: {os.getcwd()}")
    print(f"Template folder: {app.template_folder}")
    print(f"Static folder: {app.static_folder}")
    app.run(debug=False, host='0.0.0.0', port=5000)
