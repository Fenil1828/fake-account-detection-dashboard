"""
Sample test cases for the Fake Account Detection system
Run this to test the API with various account profiles
"""

import requests
import json

API_URL = 'http://localhost:5000'

# Test cases
test_accounts = [
    {
        'name': 'Obvious Bot Account',
        'data': {
            'username': 'user98765',
            'followers_count': 3,
            'friends_count': 8000,
            'statuses_count': 15000,
            'account_age_days': 20,
            'has_profile_image': False,
            'bio': '',
            'verified': False,
            'favourites_count': 100,
            'location': '',
            'url': ''
        },
        'expected': 'FAKE'
    },
    {
        'name': 'Legitimate User Account',
        'data': {
            'username': 'john_smith',
            'followers_count': 500,
            'friends_count': 300,
            'statuses_count': 1200,
            'account_age_days': 1825,  # 5 years
            'has_profile_image': True,
            'bio': 'Software engineer, coffee lover, and tech enthusiast',
            'verified': False,
            'favourites_count': 800,
            'location': 'San Francisco, CA',
            'url': 'https://johnsmith.com'
        },
        'expected': 'REAL'
    },
    {
        'name': 'Suspicious Promotional Account',
        'data': {
            'username': 'promo_deals2024',
            'followers_count': 50,
            'friends_count': 3000,
            'statuses_count': 5000,
            'account_age_days': 90,
            'has_profile_image': True,
            'bio': 'Best deals! Follow for discounts!',
            'verified': False,
            'favourites_count': 200,
            'location': '',
            'url': 'http://deals.com'
        },
        'expected': 'FAKE'
    },
    {
        'name': 'Verified Celebrity Account',
        'data': {
            'username': 'celebrity_official',
            'followers_count': 1000000,
            'friends_count': 100,
            'statuses_count': 5000,
            'account_age_days': 3650,  # 10 years
            'has_profile_image': True,
            'bio': 'Official account | Actor | Producer',
            'verified': True,
            'favourites_count': 2000,
            'location': 'Los Angeles, CA',
            'url': 'https://official-site.com'
        },
        'expected': 'REAL'
    },
    {
        'name': 'New Account with Suspicious Activity',
        'data': {
            'username': 'newuser12345',
            'followers_count': 2,
            'friends_count': 500,
            'statuses_count': 2000,
            'account_age_days': 7,
            'has_profile_image': False,
            'bio': '',
            'verified': False,
            'favourites_count': 50,
            'location': '',
            'url': ''
        },
        'expected': 'FAKE'
    }
]

def test_api_health():
    """Test if API is running"""
    try:
        response = requests.get(f'{API_URL}/api/health', timeout=5)
        if response.status_code == 200:
            print("✓ API is running")
            return True
        else:
            print("✗ API returned error")
            return False
    except requests.exceptions.ConnectionError:
        print("✗ Cannot connect to API. Make sure it's running:")
        print("  python backend/app.py")
        return False

def test_single_account(test_case):
    """Test single account analysis"""
    print(f"\n{'='*60}")
    print(f"Test Case: {test_case['name']}")
    print(f"{'='*60}")
    print(f"Expected: {test_case['expected']}")
    
    try:
        response = requests.post(
            f'{API_URL}/api/analyze',
            json=test_case['data'],
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            prediction = result['prediction']
            
            print(f"\n📊 Results:")
            print(f"   Classification: {'FAKE' if prediction['is_fake'] else 'REAL'}")
            print(f"   Risk Level: {prediction['risk_level']}")
            print(f"   Confidence: {prediction['confidence']*100:.1f}%")
            print(f"   Fake Probability: {prediction['fake_probability']*100:.1f}%")
            
            # Check if prediction matches expected
            predicted = 'FAKE' if prediction['is_fake'] else 'REAL'
            if predicted == test_case['expected']:
                print(f"\n✓ Test PASSED - Correct prediction")
            else:
                print(f"\n⚠ Test WARNING - Unexpected prediction")
            
            # Show top risk factors
            if result.get('risk_factors'):
                print(f"\n🚩 Top Risk Factors:")
                for rf in result['risk_factors'][:3]:
                    print(f"   - {rf['factor']} ({rf['severity']})")
            
            return True
        else:
            print(f"✗ API Error: {response.status_code}")
            print(response.text)
            return False
            
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def test_batch_analysis():
    """Test batch analysis"""
    print(f"\n{'='*60}")
    print(f"Batch Analysis Test")
    print(f"{'='*60}")
    
    accounts = [tc['data'] for tc in test_accounts[:3]]
    
    try:
        response = requests.post(
            f'{API_URL}/api/batch',
            json={'accounts': accounts},
            timeout=15
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"\n📊 Batch Results:")
            print(f"   Total Analyzed: {result['total_analyzed']}")
            print(f"   Fake Accounts: {result['fake_accounts_detected']}")
            print(f"   Real Accounts: {result['real_accounts']}")
            print(f"\n✓ Batch analysis PASSED")
            return True
        else:
            print(f"✗ Batch analysis FAILED: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("FAKE ACCOUNT DETECTION - API TESTS")
    print("="*60)
    
    # Check API health
    if not test_api_health():
        return
    
    print("\n" + "="*60)
    print("Running Individual Test Cases")
    print("="*60)
    
    # Test individual accounts
    passed = 0
    for test_case in test_accounts:
        if test_single_account(test_case):
            passed += 1
    
    # Test batch analysis
    print("\n" + "="*60)
    print("Testing Batch Analysis")
    print("="*60)
    test_batch_analysis()
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"Individual Tests: {passed}/{len(test_accounts)} passed")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
