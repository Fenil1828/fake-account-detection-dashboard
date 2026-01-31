"""
Fake Account Detection Model
============================
Machine learning model for detecting fake/bot social media accounts.
Based on behavioral features, linguistic patterns, and metadata signals.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import joblib
import os
from datetime import datetime, timedelta
import random
import json


class FakeAccountDetector:
    """
    Machine Learning model for fake account detection using ensemble methods.
    Combines Random Forest, Gradient Boosting, and Logistic Regression.
    """
    
    def __init__(self, model_path='models/fake_detector_model.pkl'):
        self.model_path = model_path
        self.scaler = StandardScaler()
        self.model = None
        self.feature_names = [
            'followers_count', 'following_count', 'follower_following_ratio',
            'posts_count', 'account_age_days', 'avg_posts_per_day',
            'has_profile_pic', 'has_bio', 'bio_length', 'username_length',
            'username_has_numbers', 'avg_likes_per_post', 'avg_comments_per_post',
            'engagement_rate', 'posting_regularity', 'session_duration_avg',
            'login_frequency', 'burst_posting_score', 'profile_completeness',
            'verified_status', 'external_url', 'avg_caption_length',
            'hashtag_density', 'spam_word_count', 'duplicate_content_ratio'
        ]
        self.feature_importance = {}
        self.metrics = {}
        
    def create_model(self):
        """Create an ensemble model using majority voting."""
        rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1
        )
        
        gb = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42
        )
        
        lr = LogisticRegression(
            max_iter=1000,
            random_state=42,
            C=1.0
        )
        
        # Ensemble with soft voting for probability outputs
        self.model = VotingClassifier(
            estimators=[
                ('rf', rf),
                ('gb', gb),
                ('lr', lr)
            ],
            voting='soft',
            weights=[2, 2, 1]  # Give more weight to tree-based models
        )
        
        return self.model
    
    def extract_features(self, account_data):
        """
        Extract features from raw account data.
        
        Parameters:
        -----------
        account_data : dict
            Raw account information
            
        Returns:
        --------
        np.array : Feature vector
        """
        features = {}
        
        # Basic counts
        followers = account_data.get('followers_count', 0)
        following = account_data.get('following_count', 0)
        posts = account_data.get('posts_count', 0)
        
        features['followers_count'] = followers
        features['following_count'] = following
        features['posts_count'] = posts
        
        # Follower/following ratio (capped to avoid infinity)
        if following > 0:
            features['follower_following_ratio'] = min(followers / following, 100)
        else:
            features['follower_following_ratio'] = followers if followers > 0 else 0
            
        # Account age
        created_date = account_data.get('created_date')
        if created_date:
            if isinstance(created_date, str):
                try:
                    created_date = datetime.fromisoformat(created_date.replace('Z', '+00:00'))
                except:
                    created_date = datetime.now() - timedelta(days=30)
            account_age = (datetime.now() - created_date.replace(tzinfo=None)).days
        else:
            account_age = account_data.get('account_age_days', 30)
        features['account_age_days'] = max(account_age, 1)
        
        # Posting frequency
        features['avg_posts_per_day'] = posts / max(features['account_age_days'], 1)
        
        # Profile completeness indicators
        features['has_profile_pic'] = 1 if account_data.get('has_profile_pic', False) else 0
        features['has_bio'] = 1 if account_data.get('bio', '') else 0
        features['bio_length'] = len(account_data.get('bio', ''))
        
        # Username analysis
        username = account_data.get('username', '')
        features['username_length'] = len(username)
        features['username_has_numbers'] = 1 if any(c.isdigit() for c in username) else 0
        
        # Engagement metrics
        features['avg_likes_per_post'] = account_data.get('avg_likes_per_post', 0)
        features['avg_comments_per_post'] = account_data.get('avg_comments_per_post', 0)
        
        # Engagement rate
        if followers > 0 and posts > 0:
            total_engagement = features['avg_likes_per_post'] + features['avg_comments_per_post']
            features['engagement_rate'] = (total_engagement / followers) * 100
        else:
            features['engagement_rate'] = 0
            
        # Behavioral features
        features['posting_regularity'] = account_data.get('posting_regularity', 0.5)
        features['session_duration_avg'] = account_data.get('session_duration_avg', 30)
        features['login_frequency'] = account_data.get('login_frequency', 1)
        features['burst_posting_score'] = account_data.get('burst_posting_score', 0)
        
        # Profile completeness score (0-1)
        completeness_factors = [
            features['has_profile_pic'],
            features['has_bio'],
            1 if features['bio_length'] > 20 else 0,
            1 if posts > 0 else 0,
            1 if followers > 10 else 0
        ]
        features['profile_completeness'] = sum(completeness_factors) / len(completeness_factors)
        
        # Additional metadata
        features['verified_status'] = 1 if account_data.get('is_verified', False) else 0
        features['external_url'] = 1 if account_data.get('external_url', '') else 0
        
        # Content analysis features
        features['avg_caption_length'] = account_data.get('avg_caption_length', 50)
        features['hashtag_density'] = account_data.get('hashtag_density', 0.1)
        features['spam_word_count'] = account_data.get('spam_word_count', 0)
        features['duplicate_content_ratio'] = account_data.get('duplicate_content_ratio', 0)
        
        # Return as ordered array matching feature_names
        return np.array([features.get(name, 0) for name in self.feature_names]).reshape(1, -1)
    
    def train(self, X, y):
        """
        Train the model on provided data.
        
        Parameters:
        -----------
        X : np.array or pd.DataFrame
            Training features
        y : np.array
            Labels (0=genuine, 1=fake)
        """
        if self.model is None:
            self.create_model()
            
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Split for validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train the model
        self.model.fit(X_train, y_train)
        
        # Calculate metrics
        y_pred = self.model.predict(X_val)
        y_prob = self.model.predict_proba(X_val)[:, 1]
        
        self.metrics = {
            'accuracy': accuracy_score(y_val, y_pred),
            'precision': precision_score(y_val, y_pred),
            'recall': recall_score(y_val, y_pred),
            'f1_score': f1_score(y_val, y_pred),
            'confusion_matrix': confusion_matrix(y_val, y_pred).tolist(),
            'classification_report': classification_report(y_val, y_pred, output_dict=True)
        }
        
        # Cross-validation scores
        cv_scores = cross_val_score(self.model, X_scaled, y, cv=5)
        self.metrics['cv_mean'] = cv_scores.mean()
        self.metrics['cv_std'] = cv_scores.std()
        
        # Extract feature importance from Random Forest
        rf_model = self.model.named_estimators_['rf']
        self.feature_importance = dict(zip(self.feature_names, rf_model.feature_importances_))
        
        return self.metrics
    
    def predict(self, account_data):
        """
        Predict if an account is fake.
        
        Parameters:
        -----------
        account_data : dict
            Account information
            
        Returns:
        --------
        dict : Prediction results with risk score, confidence, and explanation
        """
        if self.model is None:
            self.load_model()
            
        # Extract features
        features = self.extract_features(account_data)
        features_scaled = self.scaler.transform(features)
        
        # Get prediction and probability
        prediction = self.model.predict(features_scaled)[0]
        probabilities = self.model.predict_proba(features_scaled)[0]
        
        risk_score = probabilities[1]  # Probability of being fake
        confidence = max(probabilities)
        
        # Generate explanation
        explanation = self.generate_explanation(account_data, features[0], risk_score)
        
        return {
            'is_fake': bool(prediction),
            'risk_score': float(risk_score),
            'confidence': float(confidence),
            'classification': 'Fake/Bot Account' if prediction else 'Genuine Account',
            'risk_level': self.get_risk_level(risk_score),
            'explanation': explanation,
            'suspicious_attributes': explanation['suspicious_factors'],
            'positive_attributes': explanation['positive_factors']
        }
    
    def get_risk_level(self, risk_score):
        """Convert risk score to categorical risk level."""
        if risk_score < 0.2:
            return 'Very Low'
        elif risk_score < 0.4:
            return 'Low'
        elif risk_score < 0.6:
            return 'Medium'
        elif risk_score < 0.8:
            return 'High'
        else:
            return 'Very High'
    
    def generate_explanation(self, account_data, features, risk_score):
        """
        Generate human-readable explanation for the classification.
        
        Returns detailed reasons why an account is flagged or considered genuine.
        """
        suspicious_factors = []
        positive_factors = []
        
        followers = account_data.get('followers_count', 0)
        following = account_data.get('following_count', 0)
        posts = account_data.get('posts_count', 0)
        
        # Follower/Following ratio analysis
        if following > 0:
            ratio = followers / following
            if ratio < 0.1:
                suspicious_factors.append({
                    'factor': 'Abnormal follower/following ratio',
                    'detail': f'Ratio of {ratio:.2f} is suspiciously low (following many, few followers)',
                    'weight': 'high'
                })
            elif ratio > 50 and not account_data.get('is_verified', False):
                suspicious_factors.append({
                    'factor': 'Unusually high follower ratio',
                    'detail': f'Ratio of {ratio:.2f} without verification may indicate purchased followers',
                    'weight': 'medium'
                })
            elif 0.5 <= ratio <= 10:
                positive_factors.append({
                    'factor': 'Healthy follower/following balance',
                    'detail': f'Ratio of {ratio:.2f} indicates organic growth',
                    'weight': 'medium'
                })
        elif following == 0 and followers > 100:
            suspicious_factors.append({
                'factor': 'Zero following count',
                'detail': 'Account follows no one but has followers - unusual pattern',
                'weight': 'medium'
            })
            
        # Profile completeness
        if not account_data.get('has_profile_pic', False):
            suspicious_factors.append({
                'factor': 'No profile picture',
                'detail': 'Missing profile picture is common in fake accounts',
                'weight': 'medium'
            })
        else:
            positive_factors.append({
                'factor': 'Has profile picture',
                'detail': 'Profile picture present',
                'weight': 'low'
            })
            
        bio = account_data.get('bio', '')
        if not bio:
            suspicious_factors.append({
                'factor': 'Empty bio',
                'detail': 'No biographical information provided',
                'weight': 'low'
            })
        elif len(bio) > 20:
            positive_factors.append({
                'factor': 'Detailed bio',
                'detail': f'Bio contains {len(bio)} characters',
                'weight': 'low'
            })
            
        # Account age vs activity
        account_age = account_data.get('account_age_days', 30)
        if account_age < 30 and posts > 100:
            suspicious_factors.append({
                'factor': 'Rapid posting on new account',
                'detail': f'{posts} posts in {account_age} days is unusually high',
                'weight': 'high'
            })
        elif account_age > 365 and posts < 5:
            suspicious_factors.append({
                'factor': 'Old account with minimal activity',
                'detail': f'Account is {account_age} days old but only {posts} posts',
                'weight': 'medium'
            })
        elif account_age > 180:
            positive_factors.append({
                'factor': 'Established account',
                'detail': f'Account has been active for {account_age} days',
                'weight': 'medium'
            })
            
        # Engagement analysis
        engagement_rate = account_data.get('engagement_rate', 0)
        if followers > 1000 and engagement_rate < 0.5:
            suspicious_factors.append({
                'factor': 'Very low engagement rate',
                'detail': f'Only {engagement_rate:.2f}% engagement despite {followers} followers',
                'weight': 'high'
            })
        elif engagement_rate > 1 and engagement_rate < 20:
            positive_factors.append({
                'factor': 'Healthy engagement rate',
                'detail': f'{engagement_rate:.2f}% engagement indicates real audience',
                'weight': 'high'
            })
            
        # Posting patterns
        burst_score = account_data.get('burst_posting_score', 0)
        if burst_score > 0.7:
            suspicious_factors.append({
                'factor': 'Burst posting pattern',
                'detail': 'Unnatural posting intervals suggest automation',
                'weight': 'high'
            })
        
        posting_regularity = account_data.get('posting_regularity', 0.5)
        if posting_regularity > 0.8:
            positive_factors.append({
                'factor': 'Consistent posting schedule',
                'detail': 'Regular posting pattern indicates genuine user',
                'weight': 'medium'
            })
            
        # Spam indicators
        spam_count = account_data.get('spam_word_count', 0)
        if spam_count > 5:
            suspicious_factors.append({
                'factor': 'Spam-like content detected',
                'detail': f'{spam_count} spam indicators found in content',
                'weight': 'high'
            })
            
        duplicate_ratio = account_data.get('duplicate_content_ratio', 0)
        if duplicate_ratio > 0.5:
            suspicious_factors.append({
                'factor': 'High duplicate content',
                'detail': f'{duplicate_ratio*100:.0f}% of posts contain repeated content',
                'weight': 'high'
            })
            
        # Verification status
        if account_data.get('is_verified', False):
            positive_factors.append({
                'factor': 'Verified account',
                'detail': 'Platform has verified this account',
                'weight': 'high'
            })
            
        # Username analysis
        username = account_data.get('username', '')
        if len(username) > 15 and sum(c.isdigit() for c in username) > 4:
            suspicious_factors.append({
                'factor': 'Suspicious username pattern',
                'detail': 'Username contains many numbers - common in auto-generated accounts',
                'weight': 'medium'
            })
            
        return {
            'suspicious_factors': suspicious_factors,
            'positive_factors': positive_factors,
            'summary': self.generate_summary(suspicious_factors, positive_factors, risk_score)
        }
    
    def generate_summary(self, suspicious, positive, risk_score):
        """Generate a summary statement about the account."""
        if risk_score < 0.3:
            return f"This account appears genuine with {len(positive)} positive indicators and only {len(suspicious)} minor concerns."
        elif risk_score < 0.6:
            return f"This account shows mixed signals with {len(suspicious)} suspicious patterns and {len(positive)} positive attributes. Manual review recommended."
        else:
            return f"This account is likely fake/bot with {len(suspicious)} suspicious indicators. High risk of fraudulent activity."
    
    def save_model(self):
        """Save the trained model and scaler to disk."""
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'feature_importance': self.feature_importance,
            'metrics': self.metrics
        }
        joblib.dump(model_data, self.model_path)
        print(f"Model saved to {self.model_path}")
        
    def load_model(self):
        """Load a trained model from disk."""
        if os.path.exists(self.model_path):
            model_data = joblib.load(self.model_path)
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.feature_names = model_data['feature_names']
            self.feature_importance = model_data.get('feature_importance', {})
            self.metrics = model_data.get('metrics', {})
            print(f"Model loaded from {self.model_path}")
            return True
        return False
    
    def get_feature_importance(self):
        """Return feature importance sorted by importance."""
        if not self.feature_importance:
            return {}
        return dict(sorted(
            self.feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        ))


def generate_synthetic_dataset(n_samples=2000, fake_ratio=0.4):
    """
    Generate synthetic dataset for training.
    Creates realistic fake and genuine account profiles.
    
    Parameters:
    -----------
    n_samples : int
        Total number of samples to generate
    fake_ratio : float
        Proportion of fake accounts (0-1)
        
    Returns:
    --------
    pd.DataFrame : Dataset with features and labels
    """
    np.random.seed(42)
    random.seed(42)
    
    n_fake = int(n_samples * fake_ratio)
    n_genuine = n_samples - n_fake
    
    data = []
    
    # Generate genuine accounts
    for i in range(n_genuine):
        account = generate_genuine_account(i)
        account['label'] = 0
        data.append(account)
    
    # Generate fake accounts
    for i in range(n_fake):
        account = generate_fake_account(i)
        account['label'] = 1
        data.append(account)
    
    df = pd.DataFrame(data)
    return df.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle


def generate_genuine_account(idx):
    """Generate a realistic genuine account profile with all 25 features."""
    account_age = np.random.randint(90, 2000)
    followers = int(np.random.lognormal(5, 1.5))  # Log-normal distribution
    following = int(followers * np.random.uniform(0.3, 2.0))
    followers = max(followers, 10)
    following = max(following, 5)
    posts = max(int(account_age * np.random.uniform(0.05, 0.3)), 1)
    
    avg_likes = int(followers * np.random.uniform(0.02, 0.15))
    avg_comments = int(avg_likes * np.random.uniform(0.05, 0.2))
    
    # Calculate derived features
    follower_following_ratio = followers / max(following, 1)
    avg_posts_per_day = posts / max(account_age, 1)
    engagement_rate = ((avg_likes + avg_comments) / max(followers, 1)) * 100
    
    username = f'user_{idx}_{random.choice(["art", "photo", "life", "travel", "food"])}'
    username_length = len(username)
    username_has_numbers = 1 if any(c.isdigit() for c in username) else 0
    
    has_profile_pic = np.random.choice([1, 1, 1, 1, 0], p=[0.25, 0.25, 0.25, 0.20, 0.05])
    has_bio = np.random.choice([1, 1, 1, 0], p=[0.35, 0.35, 0.20, 0.10])
    bio_length = np.random.randint(20, 150) if has_bio else 0
    
    # Profile completeness (0-1)
    profile_completeness = (has_profile_pic + has_bio + (1 if bio_length > 20 else 0) + (1 if posts > 0 else 0) + (1 if followers > 10 else 0)) / 5
    
    return {
        'username': username,
        'followers_count': followers,
        'following_count': following,
        'follower_following_ratio': min(follower_following_ratio, 100),
        'posts_count': posts,
        'account_age_days': account_age,
        'avg_posts_per_day': avg_posts_per_day,
        'has_profile_pic': has_profile_pic,
        'has_bio': has_bio,
        'bio_length': bio_length,
        'username_length': username_length,
        'username_has_numbers': username_has_numbers,
        'avg_likes_per_post': avg_likes,
        'avg_comments_per_post': avg_comments,
        'engagement_rate': engagement_rate,
        'posting_regularity': np.random.uniform(0.5, 0.95),
        'session_duration_avg': np.random.uniform(15, 120),
        'login_frequency': np.random.uniform(0.5, 3),
        'burst_posting_score': np.random.uniform(0, 0.3),
        'profile_completeness': profile_completeness,
        'verified_status': np.random.choice([0, 0, 0, 0, 1], p=[0.25, 0.25, 0.25, 0.24, 0.01]),
        'external_url': np.random.choice([0, 1], p=[0.6, 0.4]),
        'avg_caption_length': np.random.randint(30, 200),
        'hashtag_density': np.random.uniform(0.05, 0.3),
        'spam_word_count': np.random.randint(0, 2),
        'duplicate_content_ratio': np.random.uniform(0, 0.2)
    }


def generate_fake_account(idx):
    """Generate a realistic fake/bot account profile with all 25 features."""
    account_type = np.random.choice(['bot', 'spam', 'purchased', 'impersonator'])
    
    if account_type == 'bot':
        # Automated bot account
        account_age = np.random.randint(10, 200)
        followers = np.random.randint(0, 100)
        following = np.random.randint(500, 5000)
        posts = np.random.randint(0, 50)
        burst_score = np.random.uniform(0.6, 1.0)
        spam_words = np.random.randint(3, 10)
        
    elif account_type == 'spam':
        # Spam account
        account_age = np.random.randint(5, 100)
        followers = np.random.randint(10, 500)
        following = np.random.randint(1000, 7500)
        posts = np.random.randint(50, 500)
        burst_score = np.random.uniform(0.5, 0.9)
        spam_words = np.random.randint(5, 15)
        
    elif account_type == 'purchased':
        # Account with purchased followers
        account_age = np.random.randint(30, 500)
        followers = np.random.randint(5000, 100000)
        following = np.random.randint(100, 1000)
        posts = np.random.randint(10, 100)
        burst_score = np.random.uniform(0.2, 0.5)
        spam_words = np.random.randint(0, 3)
        
    else:
        # Impersonator account
        account_age = np.random.randint(10, 180)
        followers = np.random.randint(100, 5000)
        following = np.random.randint(50, 500)
        posts = np.random.randint(5, 50)
        burst_score = np.random.uniform(0.3, 0.7)
        spam_words = np.random.randint(1, 5)
    
    # Ensure minimum values
    followers = max(followers, 1)
    following = max(following, 1)
    posts = max(posts, 0)
    account_age = max(account_age, 1)
    
    # Low engagement is common in fake accounts
    avg_likes = int(followers * np.random.uniform(0.001, 0.02))
    avg_comments = int(avg_likes * np.random.uniform(0.01, 0.1))
    
    # Calculate derived features
    follower_following_ratio = followers / max(following, 1)
    avg_posts_per_day = posts / max(account_age, 1)
    engagement_rate = ((avg_likes + avg_comments) / max(followers, 1)) * 100
    
    username = f'user{idx}{"".join([str(np.random.randint(0,9)) for _ in range(np.random.randint(3,8))])}'
    username_length = len(username)
    username_has_numbers = 1 if any(c.isdigit() for c in username) else 0
    
    has_profile_pic = np.random.choice([0, 0, 1], p=[0.4, 0.3, 0.3])
    has_bio = np.random.choice([0, 0, 1], p=[0.5, 0.3, 0.2])
    bio_length = np.random.randint(0, 50) if has_bio else 0
    
    # Profile completeness (0-1) - typically low for fake accounts
    profile_completeness = (has_profile_pic + has_bio + (1 if bio_length > 20 else 0) + (1 if posts > 0 else 0) + (1 if followers > 10 else 0)) / 5
    
    return {
        'username': username,
        'followers_count': followers,
        'following_count': following,
        'follower_following_ratio': min(follower_following_ratio, 100),
        'posts_count': posts,
        'account_age_days': account_age,
        'avg_posts_per_day': avg_posts_per_day,
        'has_profile_pic': has_profile_pic,
        'has_bio': has_bio,
        'bio_length': bio_length,
        'username_length': username_length,
        'username_has_numbers': username_has_numbers,
        'avg_likes_per_post': avg_likes,
        'avg_comments_per_post': avg_comments,
        'engagement_rate': engagement_rate,
        'posting_regularity': np.random.uniform(0.1, 0.5),
        'session_duration_avg': np.random.uniform(1, 30),
        'login_frequency': np.random.uniform(0.1, 1),
        'burst_posting_score': burst_score,
        'profile_completeness': profile_completeness,
        'verified_status': 0,
        'external_url': np.random.choice([0, 1], p=[0.3, 0.7]),
        'avg_caption_length': np.random.randint(5, 50),
        'hashtag_density': np.random.uniform(0.3, 0.8),
        'spam_word_count': spam_words,
        'duplicate_content_ratio': np.random.uniform(0.3, 0.8)
    }


def train_model():
    """Main function to train and save the model."""
    print("Generating synthetic dataset...")
    df = generate_synthetic_dataset(n_samples=3000, fake_ratio=0.4)
    
    print(f"Dataset shape: {df.shape}")
    print(f"Label distribution:\n{df['label'].value_counts()}")
    
    # Initialize detector to get feature names
    detector = FakeAccountDetector()
    
    # Prepare features - use exact feature names in correct order
    print(f"Using {len(detector.feature_names)} features: {detector.feature_names}")
    X = df[detector.feature_names].values
    y = df['label'].values
    
    print("\nTraining model...")
    metrics = detector.train(X, y)
    
    print("\n=== Model Performance ===")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1-Score: {metrics['f1_score']:.4f}")
    print(f"Cross-validation: {metrics['cv_mean']:.4f} (+/- {metrics['cv_std']:.4f})")
    
    print("\n=== Top 10 Feature Importance ===")
    importance = detector.get_feature_importance()
    for i, (feature, score) in enumerate(list(importance.items())[:10]):
        print(f"{i+1}. {feature}: {score:.4f}")
    
    # Save model
    detector.save_model()
    
    # Save dataset for reference
    df.to_csv('data/synthetic_dataset.csv', index=False)
    print("\nDataset saved to data/synthetic_dataset.csv")
    
    # Save metrics
    with open('data/model_metrics.json', 'w') as f:
        serializable_metrics = {
            k: v for k, v in metrics.items() 
            if k != 'classification_report'
        }
        serializable_metrics['classification_report'] = metrics.get('classification_report', {})
        json.dump(serializable_metrics, f, indent=2)
    print("Metrics saved to data/model_metrics.json")
    
    return detector


if __name__ == '__main__':
    train_model()
