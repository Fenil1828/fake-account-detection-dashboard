"""Utility functions for the fake account detector"""

import pandas as pd
import numpy as np
from datetime import datetime
import json

def load_dataset(filepath):
    """Load and preprocess dataset"""
    try:
        df = pd.read_csv(filepath)
        print(f"Loaded {len(df)} records from {filepath}")
        return df
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

def save_results(results, filepath):
    """Save analysis results to file"""
    try:
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {filepath}")
    except Exception as e:
        print(f"Error saving results: {e}")

def calculate_account_age(created_at):
    """Calculate account age in days"""
    if isinstance(created_at, str):
        created_at = pd.to_datetime(created_at)
    return (datetime.now() - created_at).days

def format_timestamp():
    """Get formatted timestamp"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
