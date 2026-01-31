#!/usr/bin/env python3
"""
Test script to verify fake account detection on CSV data
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model_training import FakeAccountDetector
from feature_extraction import FeatureExtractor

# Test accounts - mix of fake and real
test_accounts = [
    # Fake accounts (should be detected)
    {
        'username': 'bot_user123',
        'followers_count': 5,
        'friends_count': 8000,
        'statuses_count': 15000,
        'account_age_days': 20,
        'has_profile_image': False,
        'verified': False,
        'bio': ''
    },
    {
        'username': 'spam99',
        'followers_count': 2,
        'friends_count': 5000,
        'statuses_count': 20000,
        'account_age_days': 15,
        'has_profile_image': False,
        'verified': False,
        'bio': ''
    },
    {
        'username': 'user_bot456',
        'followers_count': 10,
        'friends_count': 2500,
        'statuses_count': 5000,
        'account_age_days': 25,
        'has_profile_image': False,
        'verified': False,
        'bio': 'Automated'
    },
    # Real accounts (should be detected as real)
    {
        'username': 'john_smith',
        'followers_count': 500,
        'friends_count': 300,
        'statuses_count': 1200,
        'account_age_days': 1825,
        'has_profile_image': True,
        'verified': False,
        'bio': 'Software developer'
    },
    {
        'username': 'real_person',
        'followers_count': 1200,
        'friends_count': 400,
        'statuses_count': 2500,
        'account_age_days': 2000,
        'has_profile_image': True,
        'verified': True,
        'bio': 'Photographer | Traveler'
    },
    {
        'username': 'legit_user',
        'followers_count': 850,
        'friends_count': 420,
        'statuses_count': 3200,
        'account_age_days': 1200,
        'has_profile_image': True,
        'verified': False,
        'bio': 'Tech enthusiast'
    },
]

def main():
    print("\n" + "="*60)
    print("FAKE ACCOUNT DETECTION - CSV UPLOAD TEST")
    print("="*60)
    
    detector = FakeAccountDetector()
    
    print("\nTesting detection on sample accounts...")
    print("-" * 60)
    
    fake_detected = 0
    real_detected = 0
    
    for account in test_accounts:
        try:
            prediction = detector.predict(account)
            
            status = "🔴 FAKE" if prediction['is_fake'] else "✅ REAL"
            fake_prob = prediction['fake_probability'] * 100
            risk = prediction['risk_level']
            
            print(f"\n{status} | {account['username']}")
            print(f"   Fake Probability: {fake_prob:.1f}%")
            print(f"   Risk Level: {risk}")
            print(f"   Followers: {account['followers_count']} | Following: {account['friends_count']}")
            print(f"   Posts: {account['statuses_count']} | Age: {account['account_age_days']} days")
            
            if prediction['is_fake']:
                fake_detected += 1
            else:
                real_detected += 1
                
        except Exception as e:
            print(f"\n❌ Error analyzing {account['username']}: {e}")
    
    print("\n" + "="*60)
    print("TEST RESULTS")
    print("="*60)
    print(f"Total Accounts: {len(test_accounts)}")
    print(f"🔴 Fake Detected: {fake_detected}")
    print(f"✅ Real Detected: {real_detected}")
    print(f"Accuracy: {((fake_detected + real_detected) / len(test_accounts)) * 100:.1f}%")
    print("="*60)
    
    if fake_detected >= 3 and real_detected >= 3:
        print("\n✅ TEST PASSED! Hybrid detection is working correctly!")
        return 0
    else:
        print("\n⚠️  TEST WARNING! May need model retraining.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
