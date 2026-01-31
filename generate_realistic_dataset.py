"""
Generate a highly realistic social media dataset with 2,000+ accounts.
Includes both genuine and suspicious accounts with nuanced labeling.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# Set seed for reproducibility
np.random.seed(42)
random.seed(42)

# Dataset parameters
TOTAL_ACCOUNTS = 2000
GENUINE_RATIO = 0.70  # 70% genuine
SUSPICIOUS_RATIO = 0.30  # 30% suspicious

genuine_count = int(TOTAL_ACCOUNTS * GENUINE_RATIO)
suspicious_count = int(TOTAL_ACCOUNTS * SUSPICIOUS_RATIO)

# Username patterns
normal_usernames = [
    'alex_smith', 'sarah_jones', 'mike_travel', 'emma_tech', 'john_photo',
    'jessica_reads', 'david_gym', 'olivia_art', 'chris_music', 'lily_food',
    'james_sports', 'sophia_nature', 'mark_coffee', 'isabella_yoga', 'tom_books'
]

weird_usernames = [
    'freecrypto123', 'win_big_now', 'get_rich_quick', 'lucky88888', 'bitcoin_to_moon',
    'follower_boost', 'buy_likes_here', 'click_here_now', 'earn_fast_2026', 'xxx123xxx',
    'crypto_pump', 'money_maker99', 'admin_access', 'hacker_pro', 'free_money_2026'
]

def generate_account_id():
    """Generate unique account ID"""
    return f"acc_{random.randint(100000, 999999)}"

def generate_username(pattern='normal'):
    """Generate username based on pattern"""
    if pattern == 'normal':
        base = random.choice(normal_usernames)
        # Sometimes add numbers
        if random.random() < 0.3:
            base += f"_{random.randint(1, 999)}"
        return base
    elif pattern == 'weird_numbers':
        return f"user_{random.randint(111, 999)}{'8'*random.randint(2, 5)}"
    elif pattern == 'crypto':
        return random.choice(weird_usernames)
    elif pattern == 'spam':
        prefixes = ['buy_', 'get_', 'free_', 'click_', 'follow_']
        suffixes = ['now', 'here', 'fast', 'easy', '2026']
        return random.choice(prefixes) + random.choice(suffixes)
    return base

def generate_bio(spam=False):
    """Generate bio text"""
    if spam:
        spam_bios = [
            'FOLLOW FOR FOLLOW! DM FOR SHOUTOUT',
            'Get free followers click link in bio!!!',
            'Earn money fast crypto trading',
            'BUY FOLLOWERS AND LIKES HERE',
            'FREE MONEY METHOD - Link in bio'
        ]
        return random.choice(spam_bios)
    else:
        real_bios = [
            'Travel enthusiast | Photography lover',
            'Software developer | Coffee addict ☕',
            'Yoga instructor | Nature lover 🌿',
            'Food blogger | Recipe collector',
            'Book lover | Reader | Writer',
            'Fitness | Gym | Healthy living',
            'Artist | Creative | Dreamer',
            'Photographer | Visual storyteller',
            'Music lover | Concert goer',
            '',  # Many real users have no bio
        ]
        return random.choice(real_bios)

def generate_genuine_account():
    """Generate a genuine account with realistic patterns"""
    account = {
        'account_id': generate_account_id(),
        'username': generate_username('normal'),
        'account_age_days': random.randint(1, 2500),
        'posts': random.choices(
            [0, random.randint(1, 50), random.randint(51, 500), random.randint(501, 5000)],
            weights=[0.3, 0.4, 0.2, 0.1]  # Many inactive users
        )[0],
        'followers': random.choices(
            [0, random.randint(1, 100), random.randint(101, 1000), random.randint(1001, 10000)],
            weights=[0.3, 0.4, 0.2, 0.1]
        )[0],
        'following': random.choices(
            [0, random.randint(1, 200), random.randint(201, 1000), random.randint(1001, 5000)],
            weights=[0.3, 0.3, 0.25, 0.15]
        )[0],
        'has_profile_picture': random.choices([0, 1], weights=[0.3, 0.7])[0],
        'bio': generate_bio(spam=False),
        'avg_likes_per_post': 0,
        'avg_comments_per_post': 0,
        'follow_back_ratio': 0,
        'username_pattern': 'normal',
        'activity_pattern': 'normal',
        'network_pattern': 'normal'
    }
    
    # Calculate engagement if has posts
    if account['posts'] > 0:
        # Genuine users have varied but reasonable engagement
        account['avg_likes_per_post'] = random.randint(0, max(20, account['followers'] // 10))
        account['avg_comments_per_post'] = random.randint(0, max(5, account['followers'] // 50))
    
    # Calculate follow back ratio
    if account['following'] > 0:
        account['follow_back_ratio'] = min(account['followers'] / account['following'], 10)
    else:
        account['follow_back_ratio'] = 0
    
    account['label'] = 'genuine'
    return account

def generate_suspicious_account():
    """Generate a suspicious account with multiple red flags"""
    suspicious_type = random.choices(
        ['engagement_fraud', 'fake_followers', 'mass_following', 'spam_bot', 'sudden_spike', 'inactive_spam'],
        weights=[0.2, 0.2, 0.2, 0.2, 0.1, 0.1]
    )[0]
    
    account = {
        'account_id': generate_account_id(),
        'username': '',
        'account_age_days': random.randint(1, 2500),
        'posts': 0,
        'followers': 0,
        'following': 0,
        'has_profile_picture': 0,
        'bio': '',
        'avg_likes_per_post': 0,
        'avg_comments_per_post': 0,
        'follow_back_ratio': 0,
        'username_pattern': 'normal',
        'activity_pattern': 'normal',
        'network_pattern': 'normal',
        'label': 'suspicious'
    }
    
    if suspicious_type == 'engagement_fraud':
        # High posts but very low engagement
        account['posts'] = random.randint(500, 3000)
        account['followers'] = random.randint(100, 500)
        account['following'] = random.randint(200, 800)
        account['avg_likes_per_post'] = random.randint(1, 5)  # Very low for so many posts
        account['avg_comments_per_post'] = random.randint(0, 2)
        account['username'] = generate_username('normal')
        account['has_profile_picture'] = random.choices([0, 1], weights=[0.6, 0.4])[0]
        account['bio'] = generate_bio(spam=random.random() < 0.3)
        account['activity_pattern'] = 'spammy'
        
    elif suspicious_type == 'fake_followers':
        # High followers but low following and engagement
        account['posts'] = random.randint(50, 200)
        account['followers'] = random.randint(5000, 50000)
        account['following'] = random.randint(50, 500)
        account['avg_likes_per_post'] = random.randint(10, 100)  # Disproportionately low
        account['avg_comments_per_post'] = random.randint(1, 20)
        account['username'] = generate_username('normal')
        account['has_profile_picture'] = random.choices([0, 1], weights=[0.4, 0.6])[0]
        account['bio'] = generate_bio(spam=False)
        account['network_pattern'] = 'fake_follower_spike'
        
    elif suspicious_type == 'mass_following':
        # Following way more than followers (bot-like)
        account['posts'] = random.randint(0, 100)
        account['followers'] = random.randint(10, 500)
        account['following'] = random.randint(2000, 10000)
        account['avg_likes_per_post'] = random.randint(0, 10) if account['posts'] > 0 else 0
        account['avg_comments_per_post'] = random.randint(0, 3) if account['posts'] > 0 else 0
        account['username'] = generate_username(random.choice(['normal', 'weird_numbers']))
        account['has_profile_picture'] = random.choices([0, 1], weights=[0.7, 0.3])[0]
        account['bio'] = generate_bio(spam=random.random() < 0.5)
        account['network_pattern'] = 'mass_following'
        account['activity_pattern'] = 'burst'
        
    elif suspicious_type == 'spam_bot':
        # Repetitive posting with spam indicators
        account['posts'] = random.randint(500, 5000)
        account['followers'] = random.randint(50, 500)
        account['following'] = random.randint(100, 1000)
        account['avg_likes_per_post'] = random.randint(1, 10)
        account['avg_comments_per_post'] = random.randint(0, 3)
        account['username'] = generate_username(random.choice(['crypto', 'spam', 'weird_numbers']))
        account['has_profile_picture'] = random.choices([0, 1], weights=[0.8, 0.2])[0]
        account['bio'] = generate_bio(spam=True)
        account['activity_pattern'] = 'repetitive'
        
    elif suspicious_type == 'sudden_spike':
        # Very new but suddenly has many followers (bought followers)
        account['account_age_days'] = random.randint(1, 30)  # Very new
        account['posts'] = random.randint(10, 100)
        account['followers'] = random.randint(1000, 10000)  # Too many for new account
        account['following'] = random.randint(100, 500)
        account['avg_likes_per_post'] = random.randint(5, 50)
        account['avg_comments_per_post'] = random.randint(1, 10)
        account['username'] = generate_username('normal')
        account['has_profile_picture'] = random.choices([0, 1], weights=[0.5, 0.5])[0]
        account['bio'] = generate_bio(spam=random.random() < 0.4)
        account['network_pattern'] = 'fake_follower_spike'
        
    elif suspicious_type == 'inactive_spam':
        # Inactive but all spam indicators
        account['posts'] = 0
        account['followers'] = random.randint(10, 100)
        account['following'] = random.randint(500, 2000)
        account['avg_likes_per_post'] = 0
        account['avg_comments_per_post'] = 0
        account['username'] = generate_username(random.choice(['crypto', 'spam']))
        account['has_profile_picture'] = 0
        account['bio'] = generate_bio(spam=True)
        account['activity_pattern'] = 'spammy'
        account['network_pattern'] = 'mass_following'
    
    # Calculate follow back ratio
    if account['following'] > 0:
        account['follow_back_ratio'] = min(account['followers'] / account['following'], 10)
    else:
        account['follow_back_ratio'] = 0
    
    return account

def generate_dataset():
    """Generate complete dataset"""
    accounts = []
    
    print(f"Generating {genuine_count} genuine accounts...")
    for _ in range(genuine_count):
        accounts.append(generate_genuine_account())
    
    print(f"Generating {suspicious_count} suspicious accounts...")
    for _ in range(suspicious_count):
        accounts.append(generate_suspicious_account())
    
    # Shuffle to mix genuine and suspicious
    random.shuffle(accounts)
    
    # Create DataFrame
    df = pd.DataFrame(accounts)
    
    # Reorder columns
    columns = [
        'account_id', 'username', 'account_age_days', 'posts', 'followers', 'following',
        'has_profile_picture', 'bio_length', 'avg_likes_per_post', 'avg_comments_per_post',
        'follow_back_ratio', 'username_pattern', 'activity_pattern', 'network_pattern', 'label'
    ]
    
    # Add bio_length column
    df['bio_length'] = df['bio'].apply(len)
    
    # Drop bio column (we have bio_length)
    df = df.drop('bio', axis=1)
    
    # Select and reorder columns
    df = df[columns]
    
    return df

def print_statistics(df):
    """Print dataset statistics"""
    print("\n" + "="*70)
    print("DATASET STATISTICS")
    print("="*70)
    
    print(f"\nTotal accounts: {len(df)}")
    print(f"Genuine accounts: {len(df[df['label'] == 'genuine'])} ({len(df[df['label'] == 'genuine'])/len(df)*100:.1f}%)")
    print(f"Suspicious accounts: {len(df[df['label'] == 'suspicious'])} ({len(df[df['label'] == 'suspicious'])/len(df)*100:.1f}%)")
    
    print("\n" + "-"*70)
    print("GENUINE ACCOUNTS - STATISTICS")
    print("-"*70)
    genuine = df[df['label'] == 'genuine']
    print(f"Account age: {genuine['account_age_days'].min()}-{genuine['account_age_days'].max()} days (avg: {genuine['account_age_days'].mean():.0f})")
    print(f"Posts: {genuine['posts'].min()}-{genuine['posts'].max()} (avg: {genuine['posts'].mean():.0f})")
    print(f"Followers: {genuine['followers'].min()}-{genuine['followers'].max()} (avg: {genuine['followers'].mean():.0f})")
    print(f"Following: {genuine['following'].min()}-{genuine['following'].max()} (avg: {genuine['following'].mean():.0f})")
    print(f"Has profile picture: {genuine['has_profile_picture'].sum()} ({genuine['has_profile_picture'].mean()*100:.1f}%)")
    print(f"Engagement (likes/post): {genuine['avg_likes_per_post'].mean():.1f}")
    print(f"Engagement (comments/post): {genuine['avg_comments_per_post'].mean():.1f}")
    
    print("\n" + "-"*70)
    print("SUSPICIOUS ACCOUNTS - STATISTICS")
    print("-"*70)
    suspicious = df[df['label'] == 'suspicious']
    print(f"Account age: {suspicious['account_age_days'].min()}-{suspicious['account_age_days'].max()} days (avg: {suspicious['account_age_days'].mean():.0f})")
    print(f"Posts: {suspicious['posts'].min()}-{suspicious['posts'].max()} (avg: {suspicious['posts'].mean():.0f})")
    print(f"Followers: {suspicious['followers'].min()}-{suspicious['followers'].max()} (avg: {suspicious['followers'].mean():.0f})")
    print(f"Following: {suspicious['following'].min()}-{suspicious['following'].max()} (avg: {suspicious['following'].mean():.0f})")
    print(f"Has profile picture: {suspicious['has_profile_picture'].sum()} ({suspicious['has_profile_picture'].mean()*100:.1f}%)")
    print(f"Engagement (likes/post): {suspicious['avg_likes_per_post'].mean():.1f}")
    print(f"Engagement (comments/post): {suspicious['avg_comments_per_post'].mean():.1f}")
    
    print("\n" + "-"*70)
    print("SUSPICIOUS ACCOUNT PATTERNS")
    print("-"*70)
    print(f"Username patterns: {suspicious['username_pattern'].value_counts().to_dict()}")
    print(f"Activity patterns: {suspicious['activity_pattern'].value_counts().to_dict()}")
    print(f"Network patterns: {suspicious['network_pattern'].value_counts().to_dict()}")
    
    print("\n" + "-"*70)
    print("KEY EXAMPLES")
    print("-"*70)
    print("\nGENUINE ACCOUNTS WITH 0 POSTS:")
    zero_posts_genuine = genuine[genuine['posts'] == 0].head(3)
    for idx, row in zero_posts_genuine.iterrows():
        print(f"  - {row['username']}: Age {row['account_age_days']} days, {row['followers']} followers, DP: {row['has_profile_picture']}")
    
    print("\nGENUINE ACCOUNTS WITH 0 FOLLOWERS:")
    zero_followers_genuine = genuine[genuine['followers'] == 0].head(3)
    for idx, row in zero_followers_genuine.iterrows():
        print(f"  - {row['username']}: Age {row['account_age_days']} days, {row['posts']} posts, DP: {row['has_profile_picture']}")
    
    print("\nSUSPICIOUS - ENGAGEMENT FRAUD (High posts, low engagement):")
    fraud = suspicious[(suspicious['posts'] > 500) & (suspicious['avg_likes_per_post'] < 5)].head(3)
    for idx, row in fraud.iterrows():
        print(f"  - {row['username']}: {row['posts']} posts, {row['avg_likes_per_post']:.1f} avg likes, {row['followers']} followers")
    
    print("\nSUSPICIOUS - MASS FOLLOWING (Following >> Followers):")
    mass_follow = suspicious[suspicious['following'] > suspicious['followers'] * 10].head(3)
    for idx, row in mass_follow.iterrows():
        print(f"  - {row['username']}: {row['following']} following, {row['followers']} followers, ratio: {row['follow_back_ratio']:.2f}")
    
    print("\n" + "="*70)

if __name__ == "__main__":
    print("Generating realistic social media dataset...")
    df = generate_dataset()
    print_statistics(df)
    
    # Save to CSV
    output_file = 'data/realistic_accounts_dataset.csv'
    df.to_csv(output_file, index=False)
    print(f"\n✅ Dataset saved to: {output_file}")
    print(f"Total rows: {len(df)}")
    print(f"Total columns: {len(df.columns)}")
