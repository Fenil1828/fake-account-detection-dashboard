"""
Utility Functions for Fake Account Detection Dashboard
======================================================
Helper functions for data processing, visualization, and analysis.
"""

import pandas as pd
import numpy as np
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import random
import json
import io
import base64


def parse_csv_upload(file_content, filename):
    """
    Parse uploaded CSV file and extract account data.
    
    Parameters:
    -----------
    file_content : bytes
        Raw file content
    filename : str
        Name of the uploaded file
        
    Returns:
    --------
    list : List of account dictionaries
    """
    try:
        # Read CSV
        df = pd.read_csv(io.BytesIO(file_content))
        
        # Normalize column names
        df.columns = df.columns.str.lower().str.strip().str.replace(' ', '_')
        
        # Map common column name variations
        column_mapping = {
            'follower': 'followers_count',
            'followers': 'followers_count',
            'follower_count': 'followers_count',
            'following': 'following_count',
            'followings': 'following_count',
            'following_count': 'following_count',
            'posts': 'posts_count',
            'post_count': 'posts_count',
            'post': 'posts_count',
            'media_count': 'posts_count',
            'age': 'account_age_days',
            'account_age': 'account_age_days',
            'profile_pic': 'has_profile_pic',
            'has_pic': 'has_profile_pic',
            'profile_picture': 'has_profile_pic',
            'biography': 'bio',
            'description': 'bio',
            'user': 'username',
            'name': 'username',
            'handle': 'username',
            'verified': 'is_verified',
            'is_verified_account': 'is_verified',
            'likes': 'avg_likes_per_post',
            'avg_likes': 'avg_likes_per_post',
            'comments': 'avg_comments_per_post',
            'avg_comments': 'avg_comments_per_post'
        }
        
        for old_name, new_name in column_mapping.items():
            if old_name in df.columns and new_name not in df.columns:
                df[new_name] = df[old_name]
        
        # Convert to list of dictionaries
        accounts = df.to_dict('records')
        
        # Clean and validate each account
        cleaned_accounts = []
        for account in accounts:
            cleaned = clean_account_data(account)
            if cleaned:
                cleaned_accounts.append(cleaned)
        
        return cleaned_accounts
        
    except Exception as e:
        raise ValueError(f"Error parsing CSV: {str(e)}")


def clean_account_data(account):
    """
    Clean and validate account data dictionary.
    Auto-infers suspicious patterns from basic data when advanced fields are missing.
    
    Parameters:
    -----------
    account : dict
        Raw account data
        
    Returns:
    --------
    dict : Cleaned account data
    """
    cleaned = {}
    
    # Required fields with defaults
    cleaned['username'] = str(account.get('username', f'user_{random.randint(1000, 9999)}'))
    cleaned['followers_count'] = int(float(account.get('followers_count', 0)))
    cleaned['following_count'] = int(float(account.get('following_count', 0)))
    cleaned['posts_count'] = int(float(account.get('posts_count', 0)))
    
    # Optional fields
    cleaned['account_age_days'] = int(float(account.get('account_age_days', 30)))
    
    # Boolean fields
    has_pic = account.get('has_profile_pic', True)
    if isinstance(has_pic, str):
        has_pic = has_pic.lower() in ['true', '1', 'yes']
    cleaned['has_profile_pic'] = bool(has_pic)
    
    # Bio
    bio = account.get('bio', '')
    cleaned['bio'] = str(bio) if bio and str(bio).lower() != 'nan' else ''
    
    # Engagement metrics
    cleaned['avg_likes_per_post'] = float(account.get('avg_likes_per_post', 0))
    cleaned['avg_comments_per_post'] = float(account.get('avg_comments_per_post', 0))
    
    # Calculate engagement rate
    if cleaned['followers_count'] > 0 and cleaned['posts_count'] > 0:
        total_engagement = cleaned['avg_likes_per_post'] + cleaned['avg_comments_per_post']
        cleaned['engagement_rate'] = (total_engagement / cleaned['followers_count']) * 100
    else:
        cleaned['engagement_rate'] = 0
    
    # Verification
    is_verified = account.get('is_verified', False)
    if isinstance(is_verified, str):
        is_verified = is_verified.lower() in ['true', '1', 'yes']
    cleaned['is_verified'] = bool(is_verified)
    
    # Content features
    cleaned['external_url'] = str(account.get('external_url', ''))
    cleaned['avg_caption_length'] = float(account.get('avg_caption_length', 50))
    
    # ============================================
    # AUTO-INFER SUSPICIOUS PATTERNS FROM BASIC DATA
    # ============================================
    
    followers = cleaned['followers_count']
    following = cleaned['following_count']
    posts = cleaned['posts_count']
    account_age = cleaned['account_age_days']
    has_bio = len(cleaned['bio']) > 0
    
    # --- Infer burst_posting_score ---
    if 'burst_posting_score' in account:
        cleaned['burst_posting_score'] = float(account.get('burst_posting_score', 0.2))
    else:
        # High posting rate on new account suggests bot behavior
        if account_age > 0:
            posts_per_day = posts / account_age
            if posts_per_day > 10:
                cleaned['burst_posting_score'] = 0.9
            elif posts_per_day > 5:
                cleaned['burst_posting_score'] = 0.7
            elif posts_per_day > 2:
                cleaned['burst_posting_score'] = 0.4
            else:
                cleaned['burst_posting_score'] = 0.2
        else:
            cleaned['burst_posting_score'] = 0.5
    
    # --- Infer spam_word_count ---
    if 'spam_word_count' in account:
        cleaned['spam_word_count'] = int(float(account.get('spam_word_count', 0)))
    else:
        # Check bio for spam indicators
        spam_words = ['buy', 'cheap', 'free', 'click', 'follow', 'dm', 'promo', 'discount', 'sale', 'money', 'win', 'prize']
        bio_lower = cleaned['bio'].lower()
        spam_count = sum(1 for word in spam_words if word in bio_lower)
        cleaned['spam_word_count'] = spam_count
    
    # --- Infer hashtag_density ---
    if 'hashtag_density' in account:
        cleaned['hashtag_density'] = float(account.get('hashtag_density', 0.1))
    else:
        # Count hashtags in bio
        hashtag_count = cleaned['bio'].count('#')
        bio_words = len(cleaned['bio'].split()) if cleaned['bio'] else 1
        cleaned['hashtag_density'] = min(hashtag_count / max(bio_words, 1), 1.0)
    
    # --- Infer duplicate_content_ratio ---
    if 'duplicate_content_ratio' in account:
        cleaned['duplicate_content_ratio'] = float(account.get('duplicate_content_ratio', 0))
    else:
        suspicious_score = 0.0
        
        # Pattern 1: Following >> Followers (bot behavior)
        if following > 0 and followers / following < 0.1:
            suspicious_score += 0.4
        
        # Pattern 2: No bio is suspicious
        if not has_bio:
            suspicious_score += 0.2
        
        # Pattern 3: High followers but very low engagement (bought followers)
        if followers > 5000 and posts < 20:
            suspicious_score += 0.4
        
        # Pattern 4: Extremely high follower/following ratio without verification (bought followers)
        if following > 0 and followers / following > 100 and not is_verified and posts < 50:
            suspicious_score += 0.6
        elif following == 0 or following < 50:
            if followers > 1000 and posts < 30:
                suspicious_score += 0.6
        
        # Pattern 4b: Very high followers with almost no following (likely bought)
        if followers > 10000 and following < 100:
            suspicious_score += 0.5
        
        # Pattern 5: New account with lots of followers (suspicious growth)
        if account_age < 60 and followers > 5000:
            suspicious_score += 0.3
        
        cleaned['duplicate_content_ratio'] = min(suspicious_score, 0.9)
    
    # --- Infer posting_regularity ---
    if 'posting_regularity' in account:
        cleaned['posting_regularity'] = float(account.get('posting_regularity', 0.5))
    else:
        # Genuine users have moderate regularity
        # Bots often have very low or very high
        if cleaned['burst_posting_score'] > 0.6:
            cleaned['posting_regularity'] = 0.2  # Irregular bursts = suspicious
        elif followers > 10000 and following < 100 and posts < 30:
            cleaned['posting_regularity'] = 0.15  # Bought followers = suspicious
        elif account_age > 180 and posts > 50:
            cleaned['posting_regularity'] = 0.7  # Established account = genuine
        else:
            cleaned['posting_regularity'] = 0.5
    
    # --- Behavioral features ---
    cleaned['session_duration_avg'] = float(account.get('session_duration_avg', 30))
    cleaned['login_frequency'] = float(account.get('login_frequency', 1))
    
    # Adjust session duration based on suspicious patterns
    if 'session_duration_avg' not in account:
        if following > followers * 10 and followers < 100:
            cleaned['session_duration_avg'] = 5  # Bots have short sessions
        elif not cleaned['has_profile_pic'] and not has_bio:
            cleaned['session_duration_avg'] = 10
    
    return cleaned


def generate_network_graph(accounts_data, risk_scores):
    """
    Generate a network visualization showing account relationships.
    
    Parameters:
    -----------
    accounts_data : list
        List of account dictionaries
    risk_scores : list
        List of risk scores for each account
        
    Returns:
    --------
    str : Plotly figure as JSON
    """
    G = nx.Graph()
    
    # Add nodes
    for i, (account, risk) in enumerate(zip(accounts_data, risk_scores)):
        G.add_node(
            account.get('username', f'user_{i}'),
            risk_score=risk,
            followers=account.get('followers_count', 0),
            following=account.get('following_count', 0)
        )
    
    # Generate synthetic edges based on similarity
    nodes = list(G.nodes())
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            # Connect accounts with similar characteristics
            risk_i = G.nodes[nodes[i]]['risk_score']
            risk_j = G.nodes[nodes[j]]['risk_score']
            
            # Higher chance of connection if similar risk scores
            if abs(risk_i - risk_j) < 0.2 and random.random() < 0.3:
                G.add_edge(nodes[i], nodes[j])
    
    # Layout
    pos = nx.spring_layout(G, k=2, iterations=50)
    
    # Create edge trace
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines'
    )
    
    # Create node trace
    node_x = []
    node_y = []
    node_colors = []
    node_sizes = []
    node_text = []
    
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        
        risk = G.nodes[node]['risk_score']
        node_colors.append(risk)
        node_sizes.append(20 + G.nodes[node]['followers'] / 100)
        
        followers = G.nodes[node]['followers']
        following = G.nodes[node]['following']
        node_text.append(
            f"<b>{node}</b><br>"
            f"Risk Score: {risk:.2f}<br>"
            f"Followers: {followers:,}<br>"
            f"Following: {following:,}"
        )
    
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        text=[n[:10] for n in G.nodes()],
        textposition='top center',
        hovertext=node_text,
        marker=dict(
            showscale=True,
            colorscale='RdYlGn_r',
            color=node_colors,
            size=[max(15, min(s, 50)) for s in node_sizes],  # Clamp sizes between 15 and 50
            colorbar=dict(
                thickness=15,
                title=dict(text='Risk Score', side='right'),
                xanchor='left'
            ),
            line=dict(width=2, color='white')
        )
    )
    
    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title=dict(text='Account Network Analysis', font=dict(size=16)),
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
    )
    
    return fig.to_json()


def generate_behavior_timeline(account_data):
    """
    Generate a timeline visualization of posting behavior.
    
    Parameters:
    -----------
    account_data : dict
        Account information
        
    Returns:
    --------
    str : Plotly figure as JSON
    """
    # Generate synthetic posting data based on account characteristics
    account_age = account_data.get('account_age_days', 90)
    posts_count = account_data.get('posts_count', 30)
    burst_score = account_data.get('burst_posting_score', 0.2)
    regularity = account_data.get('posting_regularity', 0.5)
    
    # Generate posting timestamps
    end_date = datetime.now()
    start_date = end_date - timedelta(days=min(account_age, 180))
    
    dates = []
    counts = []
    
    current_date = start_date
    while current_date <= end_date:
        # Base posting probability
        base_prob = posts_count / max(account_age, 1)
        
        # Add burst behavior
        if burst_score > 0.5 and random.random() < burst_score:
            daily_posts = int(np.random.exponential(5))
        else:
            daily_posts = np.random.poisson(base_prob * regularity * 3)
        
        dates.append(current_date)
        counts.append(daily_posts)
        current_date += timedelta(days=1)
    
    # Create figure with subplots
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Daily Posting Activity', 'Cumulative Posts'),
        vertical_spacing=0.15
    )
    
    # Daily activity bar chart
    fig.add_trace(
        go.Bar(
            x=dates,
            y=counts,
            name='Daily Posts',
            marker_color='steelblue'
        ),
        row=1, col=1
    )
    
    # Cumulative line chart
    cumulative = np.cumsum(counts)
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=cumulative,
            mode='lines',
            name='Cumulative',
            line=dict(color='coral', width=2)
        ),
        row=2, col=1
    )
    
    fig.update_layout(
        height=500,
        showlegend=True,
        title_text=f"Posting Timeline for @{account_data.get('username', 'unknown')}",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig.to_json()


def generate_engagement_chart(account_data):
    """
    Generate engagement analysis visualization.
    Shows account health metrics based on available data.
    
    Parameters:
    -----------
    account_data : dict
        Account information
        
    Returns:
    --------
    str : Plotly figure as JSON
    """
    followers = account_data.get('followers_count', 100)
    following = account_data.get('following_count', 100)
    posts = account_data.get('posts_count', 20)
    account_age = account_data.get('account_age_days', 30)
    avg_likes = account_data.get('avg_likes_per_post', 0)
    avg_comments = account_data.get('avg_comments_per_post', 0)
    has_bio = len(str(account_data.get('bio', ''))) > 0
    has_profile_pic = account_data.get('has_profile_pic', True)
    is_verified = account_data.get('is_verified', False)
    
    # Calculate various account health metrics
    
    # 1. Follower/Following Ratio Score (0-100)
    if following > 0:
        ratio = followers / following
        if ratio > 10:
            ratio_score = 100  # Influencer-like
        elif ratio > 1:
            ratio_score = 70 + (ratio - 1) * 3  # Good ratio
        elif ratio > 0.5:
            ratio_score = 50 + ratio * 40  # Moderate
        elif ratio > 0.1:
            ratio_score = 20 + ratio * 300  # Low
        else:
            ratio_score = 10  # Very suspicious
    else:
        ratio_score = 50 if followers > 0 else 20
    
    # 2. Activity Score (0-100) - based on posts per day of account age
    if account_age > 0:
        posts_per_day = posts / account_age
        if 0.1 <= posts_per_day <= 3:
            activity_score = 80 + min(posts_per_day * 10, 20)  # Healthy activity
        elif posts_per_day < 0.1:
            activity_score = max(20, posts_per_day * 200)  # Low activity
        else:
            activity_score = max(30, 100 - (posts_per_day - 3) * 10)  # Too much activity (bot-like)
    else:
        activity_score = 50
    
    # 3. Profile Completeness Score (0-100)
    completeness_score = 0
    if has_profile_pic:
        completeness_score += 35
    if has_bio:
        completeness_score += 35
    if is_verified:
        completeness_score += 30
    elif followers > 100:  # Not verified but has some following
        completeness_score += 15
    if posts > 10:
        completeness_score += 10
    completeness_score = min(completeness_score, 100)
    
    # 4. Growth Health Score (0-100) - followers relative to account age
    if account_age > 0:
        followers_per_day = followers / account_age
        if 1 <= followers_per_day <= 100:
            growth_score = 80  # Healthy organic growth
        elif followers_per_day < 1:
            growth_score = 40 + followers_per_day * 40  # Slow but okay
        elif followers_per_day > 1000:
            growth_score = 30  # Suspiciously fast
        else:
            growth_score = max(40, 80 - (followers_per_day - 100) * 0.5)
    else:
        growth_score = 50
    
    # 5. Engagement Score - if we have likes/comments data, otherwise estimate
    if avg_likes > 0 or avg_comments > 0:
        engagement_rate = ((avg_likes + avg_comments) / max(followers, 1)) * 100
        if 1 <= engagement_rate <= 10:
            engagement_score = 80 + min(engagement_rate * 2, 20)
        elif engagement_rate < 1:
            engagement_score = max(20, engagement_rate * 80)
        else:
            engagement_score = max(50, 100 - (engagement_rate - 10) * 3)
    else:
        # Estimate engagement based on other factors
        engagement_score = (ratio_score * 0.4 + activity_score * 0.3 + completeness_score * 0.3)
    
    # Calculate overall health
    overall_health = (ratio_score * 0.25 + activity_score * 0.2 + 
                      completeness_score * 0.2 + growth_score * 0.15 + 
                      engagement_score * 0.2)
    
    # Create the chart with radar/spider chart and bar chart
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{'type': 'scatterpolar'}, {'type': 'indicator'}]],
        column_widths=[0.6, 0.4]
    )
    
    # Radar chart for different metrics
    categories = ['Follower Ratio', 'Activity', 'Profile', 'Growth', 'Engagement']
    values = [ratio_score, activity_score, completeness_score, growth_score, engagement_score]
    
    # Determine color based on overall health
    if overall_health >= 70:
        line_color = '#28a745'  # Green
        fill_color = 'rgba(40, 167, 69, 0.3)'
    elif overall_health >= 40:
        line_color = '#ffc107'  # Yellow
        fill_color = 'rgba(255, 193, 7, 0.3)'
    else:
        line_color = '#dc3545'  # Red
        fill_color = 'rgba(220, 53, 69, 0.3)'
    
    fig.add_trace(
        go.Scatterpolar(
            r=values + [values[0]],  # Close the polygon
            theta=categories + [categories[0]],
            fill='toself',
            fillcolor=fill_color,
            line=dict(color=line_color, width=2),
            name='Account Metrics'
        ),
        row=1, col=1
    )
    
    # Overall health indicator
    fig.add_trace(
        go.Indicator(
            mode="gauge+number",
            value=overall_health,
            title={'text': "Account Health Score", 'font': {'size': 14}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1},
                'bar': {'color': line_color},
                'bgcolor': 'white',
                'borderwidth': 2,
                'bordercolor': '#ddd',
                'steps': [
                    {'range': [0, 40], 'color': 'rgba(220, 53, 69, 0.2)'},
                    {'range': [40, 70], 'color': 'rgba(255, 193, 7, 0.2)'},
                    {'range': [70, 100], 'color': 'rgba(40, 167, 69, 0.2)'}
                ],
                'threshold': {
                    'line': {'color': "black", 'width': 2},
                    'thickness': 0.75,
                    'value': overall_health
                }
            }
        ),
        row=1, col=2
    )
    
    fig.update_layout(
        height=350,
        title=dict(text="Account Health Analysis", font=dict(size=16)),
        showlegend=False,
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                tickfont=dict(size=10)
            ),
            angularaxis=dict(tickfont=dict(size=11))
        ),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=60, r=60, t=50, b=40)
    )
    
    return fig.to_json()


def generate_follower_analysis(account_data):
    """
    Generate follower/following analysis visualization.
    
    Parameters:
    -----------
    account_data : dict
        Account information
        
    Returns:
    --------
    str : Plotly figure as JSON
    """
    followers = account_data.get('followers_count', 100)
    following = account_data.get('following_count', 100)
    
    # Calculate ratio
    ratio = followers / max(following, 1)
    
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{'type': 'pie'}, {'type': 'indicator'}]],
        column_widths=[0.5, 0.5]
    )
    
    # Pie chart for followers vs following
    fig.add_trace(
        go.Pie(
            labels=['Followers', 'Following'],
            values=[followers, following],
            hole=0.4,
            marker_colors=['#3498db', '#e74c3c'],
            textinfo='label+percent',
            textposition='outside'
        ),
        row=1, col=1
    )
    
    # Ratio indicator
    fig.add_trace(
        go.Indicator(
            mode="number+delta",
            value=ratio,
            title={'text': "Follower/Following Ratio"},
            delta={'reference': 1.0, 'relative': True},
            number={'suffix': 'x', 'valueformat': '.2f'}
        ),
        row=1, col=2
    )
    
    fig.update_layout(
        height=350,
        title_text="Follower/Following Analysis",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig.to_json()


def generate_risk_distribution_chart(risk_scores):
    """
    Generate risk score distribution visualization for batch analysis.
    
    Parameters:
    -----------
    risk_scores : list
        List of risk scores
        
    Returns:
    --------
    str : Plotly figure as JSON
    """
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{'type': 'histogram'}, {'type': 'pie'}]],
        subplot_titles=('Risk Score Distribution', 'Classification Breakdown')
    )
    
    # Histogram
    fig.add_trace(
        go.Histogram(
            x=risk_scores,
            nbinsx=20,
            marker_color='steelblue',
            name='Risk Scores'
        ),
        row=1, col=1
    )
    
    # Classification breakdown
    low_risk = sum(1 for s in risk_scores if s < 0.4)
    medium_risk = sum(1 for s in risk_scores if 0.4 <= s < 0.7)
    high_risk = sum(1 for s in risk_scores if s >= 0.7)
    
    fig.add_trace(
        go.Pie(
            labels=['Low Risk', 'Medium Risk', 'High Risk'],
            values=[low_risk, medium_risk, high_risk],
            marker_colors=['#2ecc71', '#f39c12', '#e74c3c'],
            textinfo='label+percent+value'
        ),
        row=1, col=2
    )
    
    fig.update_layout(
        height=400,
        title_text="Batch Analysis - Risk Distribution",
        showlegend=True,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig.to_json()


def generate_feature_importance_chart(feature_importance):
    """
    Generate feature importance visualization.
    
    Parameters:
    -----------
    feature_importance : dict
        Feature importance scores
        
    Returns:
    --------
    str : Plotly figure as JSON
    """
    if not feature_importance:
        return None
    
    # Sort by importance
    sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:15]
    features, importance = zip(*sorted_features)
    
    # Clean feature names for display
    clean_names = [f.replace('_', ' ').title() for f in features]
    
    fig = go.Figure(go.Bar(
        x=list(importance),
        y=clean_names,
        orientation='h',
        marker_color='steelblue',
        text=[f'{v:.3f}' for v in importance],
        textposition='auto'
    ))
    
    fig.update_layout(
        height=500,
        title_text="Top 15 Feature Importance",
        xaxis_title="Importance Score",
        yaxis_title="Feature",
        yaxis=dict(autorange="reversed"),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig.to_json()


def generate_confusion_matrix_chart(confusion_matrix):
    """
    Generate confusion matrix visualization.
    
    Parameters:
    -----------
    confusion_matrix : list
        2x2 confusion matrix
        
    Returns:
    --------
    str : Plotly figure as JSON
    """
    cm = np.array(confusion_matrix)
    
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=['Predicted Genuine', 'Predicted Fake'],
        y=['Actual Genuine', 'Actual Fake'],
        text=cm,
        texttemplate='%{text}',
        textfont={"size": 20},
        colorscale='Blues',
        showscale=True
    ))
    
    fig.update_layout(
        title='Confusion Matrix',
        height=400,
        xaxis_title='Predicted Label',
        yaxis_title='Actual Label'
    )
    
    return fig.to_json()


def generate_metrics_comparison_chart(metrics):
    """
    Generate metrics comparison visualization.
    
    Parameters:
    -----------
    metrics : dict
        Model performance metrics
        
    Returns:
    --------
    str : Plotly figure as JSON
    """
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    metric_values = [
        metrics.get('accuracy', 0),
        metrics.get('precision', 0),
        metrics.get('recall', 0),
        metrics.get('f1_score', 0)
    ]
    
    # Color based on value
    colors = ['#2ecc71' if v >= 0.8 else '#f39c12' if v >= 0.6 else '#e74c3c' for v in metric_values]
    
    fig = go.Figure(data=[
        go.Bar(
            x=metric_names,
            y=metric_values,
            marker_color=colors,
            text=[f'{v:.2%}' for v in metric_values],
            textposition='auto'
        )
    ])
    
    fig.update_layout(
        title='Model Performance Metrics',
        yaxis_title='Score',
        yaxis=dict(range=[0, 1]),
        height=400,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig.to_json()


def export_report_csv(results):
    """
    Export analysis results to CSV format.
    
    Parameters:
    -----------
    results : list
        List of analysis result dictionaries
        
    Returns:
    --------
    str : CSV content
    """
    if not results:
        return ""
    
    df = pd.DataFrame([{
        'Username': r.get('username', ''),
        'Risk Score': r.get('risk_score', 0),
        'Confidence': r.get('confidence', 0),
        'Risk Level': r.get('risk_level', ''),
        'Classification': r.get('classification', ''),
        'Suspicious Factors Count': len(r.get('suspicious_attributes', [])),
        'Positive Factors Count': len(r.get('positive_attributes', [])),
        'Summary': r.get('explanation', {}).get('summary', '')
    } for r in results])
    
    return df.to_csv(index=False)


def get_sample_accounts():
    """
    Generate sample account data for demonstration.
    
    Returns:
    --------
    list : List of sample account dictionaries
    """
    return [
        {
            'username': 'genuine_user_sarah',
            'followers_count': 1250,
            'following_count': 890,
            'posts_count': 156,
            'account_age_days': 730,
            'has_profile_pic': True,
            'bio': 'Travel enthusiast | Photography lover | Based in NYC',
            'avg_likes_per_post': 85,
            'avg_comments_per_post': 12,
            'posting_regularity': 0.75,
            'session_duration_avg': 45,
            'login_frequency': 2.5,
            'burst_posting_score': 0.15,
            'is_verified': False,
            'external_url': 'https://myblog.com',
            'avg_caption_length': 120,
            'hashtag_density': 0.15,
            'spam_word_count': 0,
            'duplicate_content_ratio': 0.05
        },
        {
            'username': 'bot_account_92847',
            'followers_count': 45,
            'following_count': 2500,
            'posts_count': 12,
            'account_age_days': 15,
            'has_profile_pic': False,
            'bio': '',
            'avg_likes_per_post': 2,
            'avg_comments_per_post': 0,
            'posting_regularity': 0.2,
            'session_duration_avg': 5,
            'login_frequency': 0.5,
            'burst_posting_score': 0.85,
            'is_verified': False,
            'external_url': 'http://spam-link.xyz',
            'avg_caption_length': 15,
            'hashtag_density': 0.7,
            'spam_word_count': 8,
            'duplicate_content_ratio': 0.65
        },
        {
            'username': 'influencer_mike',
            'followers_count': 50000,
            'following_count': 500,
            'posts_count': 890,
            'account_age_days': 1825,
            'has_profile_pic': True,
            'bio': 'Fitness Coach | 10 Years Experience | DM for coaching',
            'avg_likes_per_post': 2500,
            'avg_comments_per_post': 150,
            'posting_regularity': 0.9,
            'session_duration_avg': 60,
            'login_frequency': 4,
            'burst_posting_score': 0.1,
            'is_verified': True,
            'external_url': 'https://fitnesscoach.com',
            'avg_caption_length': 200,
            'hashtag_density': 0.2,
            'spam_word_count': 1,
            'duplicate_content_ratio': 0.1
        },
        {
            'username': 'suspicious_seller_xyz123',
            'followers_count': 15000,
            'following_count': 50,
            'posts_count': 45,
            'account_age_days': 60,
            'has_profile_pic': True,
            'bio': 'CHEAP PRODUCTS HERE!!! DM NOW!!! Best prices!!!',
            'avg_likes_per_post': 20,
            'avg_comments_per_post': 3,
            'posting_regularity': 0.3,
            'session_duration_avg': 10,
            'login_frequency': 1,
            'burst_posting_score': 0.6,
            'is_verified': False,
            'external_url': 'http://buy-cheap-stuff.net',
            'avg_caption_length': 30,
            'hashtag_density': 0.5,
            'spam_word_count': 6,
            'duplicate_content_ratio': 0.4
        }
    ]
