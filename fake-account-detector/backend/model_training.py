import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score, precision_recall_fscore_support
import joblib
import os
import sys
from feature_extraction import FeatureExtractor

class FakeAccountDetector:
    """ML model for detecting fake social media accounts"""
    
    def __init__(self):
        self.model = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
        self.feature_extractor = FeatureExtractor()
        self.feature_names = []
        self.training_metrics = {}
        
    def prepare_data(self, df):
        """Prepare dataset for training"""
        X = []
        y = []
        
        print(f"Processing {len(df)} accounts...")
        for idx, row in df.iterrows():
            try:
                account_data = row.to_dict()
                features = self.feature_extractor.extract_all_features(account_data)
                feature_vector = self.feature_extractor.prepare_features_for_model(features)
                
                X.append(feature_vector)
                # Handle different label column names
                label = row.get('account_type', row.get('is_bot', row.get('label', 0)))
                # Convert 'bot' to 1, 'human' to 0
                if isinstance(label, str):
                    label = 1 if label.lower() == 'bot' else 0
                y.append(label)
                
                if (idx + 1) % 100 == 0:
                    print(f"Processed {idx + 1} accounts...")
            except Exception as e:
                print(f"Error processing row {idx}: {e}")
                continue
        
        return np.array(X), np.array(y)
    
    def train(self, X_train, y_train):
        """Train the model"""
        print("\n" + "="*50)
        print("Training Gradient Boosting Classifier...")
        print("="*50)
        self.model.fit(X_train, y_train)
        print("✓ Training complete!")
        
    def evaluate(self, X_test, y_test):
        """Evaluate model performance"""
        y_pred = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
        roc_auc = roc_auc_score(y_test, y_proba)
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        print("\n" + "="*50)
        print("MODEL EVALUATION RESULTS")
        print("="*50)
        print(f"\n📊 Overall Metrics:")
        print(f"   Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"   Precision: {precision:.4f}")
        print(f"   Recall:    {recall:.4f}")
        print(f"   F1-Score:  {f1:.4f}")
        print(f"   ROC AUC:   {roc_auc:.4f}")
        
        print(f"\n📈 Confusion Matrix:")
        print(f"   True Negatives:  {conf_matrix[0][0]}")
        print(f"   False Positives: {conf_matrix[0][1]}")
        print(f"   False Negatives: {conf_matrix[1][0]}")
        print(f"   True Positives:  {conf_matrix[1][1]}")
        
        print("\n" + classification_report(y_test, y_pred, target_names=['Real Account', 'Fake Account']))
        
        # Store metrics
        self.training_metrics = {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'roc_auc': float(roc_auc),
            'confusion_matrix': conf_matrix.tolist(),
            'classification_report': classification_report(y_test, y_pred, output_dict=True)
        }
        
        return self.training_metrics
    
    def predict(self, account_data):
        """Predict if account is fake"""
        features = self.feature_extractor.extract_all_features(account_data)
        feature_vector = self.feature_extractor.prepare_features_for_model(features)
        
        prediction = self.model.predict([feature_vector])[0]
        probability = self.model.predict_proba([feature_vector])[0]
        
        return {
            'is_fake': bool(prediction),
            'fake_probability': float(probability[1]),
            'real_probability': float(probability[0]),
            'confidence': float(max(probability)),
            'risk_level': self.get_risk_level(probability[1])
        }
    
    def get_risk_level(self, fake_prob):
        """Determine risk level based on probability"""
        if fake_prob >= 0.8:
            return 'CRITICAL'
        elif fake_prob >= 0.6:
            return 'HIGH'
        elif fake_prob >= 0.4:
            return 'MEDIUM'
        else:
            return 'LOW'
    
    def explain_prediction(self, account_data):
        """Generate explanation for prediction"""
        features = self.feature_extractor.extract_all_features(account_data)
        feature_vector = self.feature_extractor.prepare_features_for_model(features)
        
        # Get feature importances
        importances = self.model.feature_importances_
        
        # Create explanation
        explanations = []
        feature_list = list(features.keys())
        
        # Sort by importance
        importance_indices = np.argsort(importances)[::-1][:5]
        
        for idx in importance_indices:
            if idx < len(feature_list):
                feature_name = feature_list[idx]
                explanations.append({
                    'feature': feature_name,
                    'value': features[feature_name],
                    'importance': float(importances[idx]),
                    'interpretation': self.interpret_feature(feature_name, features[feature_name])
                })
        
        return explanations
    
    def interpret_feature(self, feature_name, value):
        """Interpret what a feature value means"""
        interpretations = {
            'default_pattern': 'Username follows default pattern (user123)' if value else 'Custom username',
            'has_profile_pic': 'Has profile picture' if value else 'Missing profile picture',
            'low_followers': 'Very low follower count' if value else 'Normal follower count',
            'suspicious_ff_ratio': 'Suspicious follower/following ratio' if value else 'Normal ratio',
            'tweets_per_day': f'{value:.2f} tweets per day',
            'account_age_days': f'Account age: {int(value)} days',
            'follower_following_ratio': f'Follower/Following ratio: {value:.2f}',
            'has_bio': 'Has bio description' if value else 'No bio description',
            'is_verified': 'Verified account' if value else 'Not verified'
        }
        
        return interpretations.get(feature_name, f'Value: {value}')
    
    def save_model(self, filepath='models/detector.pkl'):
        """Save trained model"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self, filepath)
        print(f"\n✓ Model saved to {filepath}")
    
    @staticmethod
    def load_model(filepath='models/detector.pkl'):
        """Load trained model"""
        return joblib.load(filepath)


if __name__ == "__main__":
    # Train the model
    print("\n" + "="*50)
    print("FAKE ACCOUNT DETECTION - MODEL TRAINING")
    print("="*50)
    
    # Check if dataset exists
    dataset_path = 'data/raw/twitter_bots.csv'
    if not os.path.exists(dataset_path):
        print(f"\n❌ Dataset not found at {dataset_path}")
        print("Please download the dataset first.")
        sys.exit(1)
    
    print(f"\n📂 Loading dataset from {dataset_path}...")
    df = pd.read_csv(dataset_path)
    print(f"✓ Loaded {len(df)} records")
    
    # Display dataset info
    print(f"\n📋 Dataset columns: {list(df.columns)}")
    if 'account_type' in df.columns:
        print(f"   Account types: {df['account_type'].value_counts().to_dict()}")
    
    detector = FakeAccountDetector()
    
    print("\n🔧 Preparing features...")
    # Use a subset for faster training (adjust as needed)
    sample_size = min(5000, len(df))
    df_sample = df.sample(n=sample_size, random_state=42)
    
    X, y = detector.prepare_data(df_sample)
    print(f"✓ Extracted features from {len(X)} accounts")
    print(f"   Feature vector size: {len(X[0])} features")
    print(f"   Class distribution: {np.bincount(y)}")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\n📊 Train/Test split:")
    print(f"   Training set: {len(X_train)} samples")
    print(f"   Test set: {len(X_test)} samples")
    
    detector.train(X_train, y_train)
    metrics = detector.evaluate(X_test, y_test)
    
    detector.save_model()
    
    print("\n" + "="*50)
    print("✅ MODEL TRAINING COMPLETE!")
    print("="*50)
    print("\nNext steps:")
    print("1. Start the API: python backend/app.py")
    print("2. Start the dashboard: python frontend/dashboard.py")
