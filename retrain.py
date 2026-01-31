#!/usr/bin/env python3
"""
Retrain the fake account detection model with updated dataset and hyperparameters.
"""

import sys
sys.path.insert(0, '.')

from model import train_model

print("=" * 70)
print("RETRAINING FAKE ACCOUNT DETECTION MODEL WITH IMPROVED GENERATORS")
print("=" * 70)

# Train with updated parameters
# - 5000 samples (up from 3000)
# - 40% fake ratio (60% genuine)
# - Realistic noise enabled
# - New hyperparameters: RF(200 est, depth=12), GB(150 est, depth=7), LR(C=0.1)
# - Improved genuining generators (±12% noise)
# - Improved fake generators (realistic profiles, less extreme)
detector = train_model(n_samples=5000, fake_ratio=0.4, realistic_noise=True)

print("\n" + "=" * 70)
print("MODEL RETRAINING COMPLETE!")
print("=" * 70)
print("\nModel saved to: models/fake_detector_model.pkl")
print("Metrics saved to: data/model_metrics.json")
print("Dataset saved to: data/synthetic_dataset.csv")
print("\nNext steps:")
print("1. Restart the Flask server: python app.py")
print("2. Test predictions at http://localhost:5000/analysis")
print("3. Check model metrics at http://localhost:5000/report")
print("=" * 70)
