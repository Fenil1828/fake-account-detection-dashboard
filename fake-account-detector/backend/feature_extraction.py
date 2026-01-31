import pandas as pd
import numpy as np
from datetime import datetime
import re

class FeatureExtractor:
    """Extract features from social media account data"""
    
    def extract_all_features(self, account_data):
        """Main method to extract all features"""
        features = {}
        
        # Profile features
        features.update(self.extract_profile_features(account_data))
        
        # Behavioral features
        features.update(self.extract_behavioral_features(account_data))
        
        # Network features
        features.update(self.extract_network_features(account_data))
        
        # Content features
        features.update(self.extract_content_features(account_data))
        
        return features
    
    def extract_profile_features(self, data):
        """Extract profile-based features"""
        features = {}
        
        username = data.get('username', '')
        bio = data.get('bio', '')
        
        # Username analysis
        features['username_length'] = len(username)
        features['has_numbers_in_username'] = int(bool(re.search(r'\d', username)))
        features['has_special_chars'] = int(bool(re.search(r'[^a-zA-Z0-9_]', username)))
        features['default_pattern'] = int(bool(re.search(r'user\d+', username.lower())))
        
        # Profile completeness
        features['has_profile_pic'] = int(data.get('has_profile_image', False))
        features['has_bio'] = int(len(bio) > 0)
        features['bio_length'] = len(bio)
        features['has_location'] = int(bool(data.get('location', '')))
        features['has_url'] = int(bool(data.get('url', '')))
        features['is_verified'] = int(data.get('verified', False))
        
        # Account age - ensure it's at least 1
        created_at = data.get('created_at')
        if created_at:
            if isinstance(created_at, str):
                created_at = pd.to_datetime(created_at)
            account_age = (datetime.now() - created_at).days
            features['account_age_days'] = max(account_age, 1)
        else:
            account_age = data.get('account_age_days', 0)
            # If account_age is 0 or missing, assume it's a new account (1 day)
            features['account_age_days'] = max(int(account_age), 1)
        
        return features
    
    def extract_behavioral_features(self, data):
        """Extract behavior-based features"""
        features = {}
        
        # Posting frequency
        statuses_count = data.get('statuses_count', 0)
        account_age = max(data.get('account_age_days', 1), 1)
        
        features['statuses_count'] = statuses_count
        features['tweets_per_day'] = statuses_count / account_age
        
        # Engagement metrics
        favourites_count = data.get('favourites_count', 0)
        features['favourites_count'] = favourites_count
        features['likes_per_day'] = favourites_count / account_age
        
        # Activity ratio
        features['activity_ratio'] = statuses_count / max(favourites_count, 1)
        
        return features
    
    def extract_network_features(self, data):
        """Extract network-based features"""
        features = {}
        
        followers = data.get('followers_count', 0)
        following = data.get('friends_count', 0)
        
        features['followers_count'] = followers
        features['following_count'] = following
        
        # Follower/Following ratio
        if following > 0:
            features['follower_following_ratio'] = followers / following
        else:
            features['follower_following_ratio'] = followers
        
        # Suspicious patterns
        features['follows_too_many'] = int(following > 2000)
        features['low_followers'] = int(followers < 10)
        features['suspicious_ff_ratio'] = int(following > followers * 10)
        
        return features
    
    def extract_content_features(self, data):
        """Extract content-based features"""
        features = {}
        
        # Get recent tweets if available
        tweets = data.get('recent_tweets', [])
        
        if tweets:
            # Calculate average tweet length
            tweet_lengths = [len(t.get('text', '')) for t in tweets]
            features['avg_tweet_length'] = np.mean(tweet_lengths) if tweet_lengths else 0
            
            # URL sharing frequency
            urls_count = sum(1 for t in tweets if 'http' in t.get('text', ''))
            features['url_sharing_rate'] = urls_count / len(tweets)
            
            # Hashtag usage
            hashtags_count = sum(t.get('text', '').count('#') for t in tweets)
            features['avg_hashtags_per_tweet'] = hashtags_count / len(tweets)
        else:
            features['avg_tweet_length'] = 0
            features['url_sharing_rate'] = 0
            features['avg_hashtags_per_tweet'] = 0
        
        return features

    def prepare_features_for_model(self, features):
        """Convert features dict to array for model prediction"""
        feature_order = [
            'username_length', 'has_numbers_in_username', 'has_special_chars',
            'default_pattern', 'has_profile_pic', 'has_bio', 'bio_length',
            'has_location', 'has_url', 'is_verified', 'account_age_days',
            'statuses_count', 'tweets_per_day', 'favourites_count',
            'likes_per_day', 'activity_ratio', 'followers_count',
            'following_count', 'follower_following_ratio', 'follows_too_many',
            'low_followers', 'suspicious_ff_ratio', 'avg_tweet_length',
            'url_sharing_rate', 'avg_hashtags_per_tweet'
        ]
        
        return [features.get(f, 0) for f in feature_order]
