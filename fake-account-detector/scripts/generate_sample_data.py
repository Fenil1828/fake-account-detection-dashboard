"""
Generate sample dataset for training if real dataset is not available
This creates synthetic data that mimics real Twitter bot detection datasets
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

def generate_sample_dataset(n_samples=5000):
    """Generate synthetic account data for training"""
    
    print(f"Generating {n_samples} sample accounts...")
    
    data = []
    
    for i in range(n_samples):
        # Randomly decide if account is bot (40% bots, 60% real)
        is_bot = random.random() < 0.4
        
        if is_bot:
            # Bot account characteristics
            account = {
                'id': f'bot_{i}',
                'username': f'user{random.randint(1000, 99999)}' if random.random() < 0.7 else f'bot_{i}',
                'followers_count': random.randint(0, 50),
                'friends_count': random.randint(1000, 10000),
                'statuses_count': random.randint(5000, 50000),
                'favourites_count': random.randint(0, 500),
                'listed_count': random.randint(0, 5),
                'created_at': (datetime.now() - timedelta(days=random.randint(1, 90))).isoformat(),
                'verified': False,
                'default_profile': random.random() < 0.8,
                'default_profile_image': random.random() < 0.6,
                'has_extended_profile': False,
                'description': '' if random.random() < 0.7 else 'Follow for deals!',
                'location': '',
                'url': '',
                'account_type': 'bot'
            }
        else:
            # Real account characteristics
            account = {
                'id': f'real_{i}',
                'username': f'{random.choice(["john", "sarah", "mike", "emma", "alex"])}_{random.choice(["smith", "jones", "brown", "davis"])}',
                'followers_count': random.randint(50, 2000),
                'friends_count': random.randint(50, 1000),
                'statuses_count': random.randint(100, 5000),
                'favourites_count': random.randint(100, 3000),
                'listed_count': random.randint(0, 50),
                'created_at': (datetime.now() - timedelta(days=random.randint(365, 3650))).isoformat(),
                'verified': random.random() < 0.05,
                'default_profile': random.random() < 0.2,
                'default_profile_image': random.random() < 0.1,
                'has_extended_profile': random.random() < 0.7,
                'description': random.choice([
                    'Software engineer and tech enthusiast',
                    'Love coffee, books, and travel',
                    'Marketing professional | Dog lover',
                    'Photographer | Nature lover',
                    ''
                ]),
                'location': random.choice(['New York', 'San Francisco', 'London', 'Tokyo', '']),
                'url': 'https://example.com' if random.random() < 0.3 else '',
                'account_type': 'human'
            }
        
        # Add computed fields
        account_age = (datetime.now() - pd.to_datetime(account['created_at'])).days
        account['account_age_days'] = max(account_age, 1)
        account['has_profile_image'] = not account['default_profile_image']
        account['bio'] = account['description']
        
        data.append(account)
    
    df = pd.DataFrame(data)
    
    print(f"✓ Generated {len(df)} accounts")
    print(f"  - Bots: {len(df[df['account_type'] == 'bot'])}")
    print(f"  - Humans: {len(df[df['account_type'] == 'human'])}")
    
    return df

if __name__ == "__main__":
    import os
    
    # Create directory
    os.makedirs('data/raw', exist_ok=True)
    
    # Generate dataset
    df = generate_sample_dataset(5000)
    
    # Save to CSV
    output_path = 'data/raw/twitter_bots.csv'
    df.to_csv(output_path, index=False)
    
    print(f"\n✓ Dataset saved to {output_path}")
    print(f"\nDataset info:")
    print(df.info())
    print(f"\nSample records:")
    print(df.head())
