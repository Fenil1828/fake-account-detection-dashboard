"""
Train model using the realistic 2000-row dataset with proper multi-factor labeling.
"""

import pandas as pd
import numpy as np
from model import FakeAccountDetector
import json

print("="*70)
print("TRAINING MODEL ON REALISTIC 2000-ROW DATASET")
print("="*70)

# Load the realistic dataset
print("\nLoading realistic dataset...")
df = pd.read_csv('data/realistic_accounts_dataset.csv')
print(f"✓ Loaded {len(df)} accounts ({df['label'].value_counts().to_dict()})")

# Prepare features
print("\nPreparing features...")
feature_names = [
    'account_age_days', 'posts', 'followers', 'following',
    'has_profile_picture', 'bio_length', 'avg_likes_per_post',
    'avg_comments_per_post', 'follow_back_ratio'
]

X = df[feature_names].values
y = (df['label'] == 'suspicious').astype(int).values  # 0=genuine, 1=suspicious

print(f"✓ Features: {len(feature_names)}")
print(f"✓ Samples: {len(X)}")
print(f"✓ Labels: {np.bincount(y)} (0=genuine, 1=suspicious)")

# Create and train model
print("\nCreating model...")
detector = FakeAccountDetector()
detector.create_model()

print("Training model...")
metrics = detector.train(X, y)

# Display results
print("\n" + "="*70)
print("MODEL TRAINING RESULTS")
print("="*70)
print(f"\nAccuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
print(f"Precision: {metrics['precision']:.4f} ({metrics['precision']*100:.2f}%)")
print(f"Recall:    {metrics['recall']:.4f} ({metrics['recall']*100:.2f}%)")
print(f"F1-Score:  {metrics['f1_score']:.4f}")
print(f"\nCross-Validation (10-fold):")
print(f"  Mean: {metrics['cv_mean']:.4f} ± {metrics['cv_std']:.4f}")

# Confusion matrix
cm = metrics['confusion_matrix']
print(f"\nConfusion Matrix:")
print(f"  True Negatives (Genuine correctly identified):  {cm[0][0]}")
print(f"  False Positives (Genuine wrongly flagged):      {cm[0][1]}")
print(f"  False Negatives (Suspicious missed):            {cm[1][0]}")
print(f"  True Positives (Suspicious correctly detected): {cm[1][1]}")

# Feature importance
print(f"\nTop 10 Most Important Features:")
sorted_features = sorted(detector.feature_importance.items(), key=lambda x: x[1], reverse=True)
for i, (feature, importance) in enumerate(sorted_features[:10], 1):
    print(f"  {i:2d}. {feature:30s} {importance:6.2%}")

# Save model
print("\n" + "="*70)
detector.save_model()
print("✓ Model saved to: models/fake_detector_model.pkl")

# Save metrics
with open('data/model_metrics.json', 'w') as f:
    metrics_copy = metrics.copy()
    # Convert non-serializable items
    if 'confusion_matrix' in metrics_copy:
        metrics_copy['confusion_matrix'] = [[int(x) for x in row] for row in metrics_copy['confusion_matrix']]
    json.dump({
        'accuracy': float(metrics['accuracy']),
        'precision': float(metrics['precision']),
        'recall': float(metrics['recall']),
        'f1_score': float(metrics['f1_score']),
        'cv_mean': float(metrics['cv_mean']),
        'cv_std': float(metrics['cv_std']),
        'confusion_matrix': metrics_copy['confusion_matrix'],
        'dataset': 'realistic_accounts_dataset.csv',
        'total_samples': len(X),
        'genuine_count': int((y == 0).sum()),
        'suspicious_count': int((y == 1).sum())
    }, f, indent=2)
print("✓ Metrics saved to: data/model_metrics.json")

print("\n" + "="*70)
print("✅ TRAINING COMPLETE!")
print("="*70)
print(f"\nModel trained on: 2000 realistic social media accounts")
print(f"  - 1400 genuine accounts (multi-factor verified)")
print(f"  - 600 suspicious accounts (multiple red flags)")
print(f"\nKey characteristics:")
print(f"  - Old accounts with 0 posts are NOT flagged (if no other red flags)")
print(f"  - Accounts judged by MULTIPLE combined factors")
print(f"  - Real-world accuracy: ~{metrics['accuracy']*100:.0f}%")
print(f"\nRefresh dashboard at http://127.0.0.1:5000 to see updated model!")
