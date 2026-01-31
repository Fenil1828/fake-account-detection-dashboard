import dash
from dash import dcc, html, Input, Output, State, dash_table
import plotly.graph_objs as go
import plotly.express as px
import requests
import pandas as pd
from datetime import datetime
import networkx as nx

# Initialize Dash app
app = dash.Dash(__name__)
app.title = "Fake Account Detection Dashboard"

# API endpoint
API_URL = 'http://localhost:5000'

# Define color scheme
COLORS = {
    'critical': '#e74c3c',
    'high': '#e67e22',
    'medium': '#f39c12',
    'low': '#27ae60',
    'background': '#f5f6fa',
    'card': '#ffffff',
    'primary': '#3498db',
    'text': '#2c3e50'
}

# Define layout
app.layout = html.Div([
    # Header
    html.Div([
        html.H1("🔍 Fake Account Detection & Risk Analysis Dashboard", 
                style={'textAlign': 'center', 'color': COLORS['text'], 'marginBottom': 10}),
        html.P("Advanced ML-powered social media account analysis", 
               style={'textAlign': 'center', 'color': '#7f8c8d', 'fontSize': 16})
    ], style={'padding': '20px 0'}),
    
    # Main container
    html.Div([
        # Left panel - Input Section
        html.Div([
            html.Div([
                html.H3("📝 Enter Account Details", style={'color': COLORS['text'], 'marginBottom': 20}),
                
                html.Label("Username:", style={'fontWeight': 'bold'}),
                dcc.Input(id='username', type='text', placeholder='e.g., john_doe123', 
                         style={'width': '100%', 'padding': 10, 'marginBottom': 15, 'borderRadius': 5, 'border': '1px solid #ddd'}),
                
                html.Label("Followers Count:", style={'fontWeight': 'bold'}),
                dcc.Input(id='followers', type='number', placeholder='Number of followers', value=100,
                         style={'width': '100%', 'padding': 10, 'marginBottom': 15, 'borderRadius': 5, 'border': '1px solid #ddd'}),
                
                html.Label("Following Count:", style={'fontWeight': 'bold'}),
                dcc.Input(id='following', type='number', placeholder='Number following', value=150,
                         style={'width': '100%', 'padding': 10, 'marginBottom': 15, 'borderRadius': 5, 'border': '1px solid #ddd'}),
                
                html.Label("Total Tweets:", style={'fontWeight': 'bold'}),
                dcc.Input(id='tweets', type='number', placeholder='Total tweets', value=500,
                         style={'width': '100%', 'padding': 10, 'marginBottom': 15, 'borderRadius': 5, 'border': '1px solid #ddd'}),
                
                html.Label("Account Age (days):", style={'fontWeight': 'bold'}),
                dcc.Input(id='age', type='number', placeholder='Days since creation', value=365,
                         style={'width': '100%', 'padding': 10, 'marginBottom': 15, 'borderRadius': 5, 'border': '1px solid #ddd'}),
                
                html.Label("Has Profile Picture:", style={'fontWeight': 'bold'}),
                dcc.Dropdown(
                    id='has_profile_pic',
                    options=[
                        {'label': 'Yes', 'value': 1},
                        {'label': 'No', 'value': 0}
                    ],
                    value=1,
                    style={'marginBottom': 15}
                ),
                
                html.Label("Is Verified:", style={'fontWeight': 'bold'}),
                dcc.Dropdown(
                    id='is_verified',
                    options=[
                        {'label': 'Yes', 'value': 1},
                        {'label': 'No', 'value': 0}
                    ],
                    value=0,
                    style={'marginBottom': 20}
                ),
                
                html.Button('🔍 Analyze Account', id='analyze-btn', n_clicks=0,
                           style={
                               'width': '100%', 
                               'padding': 15, 
                               'backgroundColor': COLORS['primary'], 
                               'color': 'white', 
                               'border': 'none', 
                               'borderRadius': 5,
                               'fontSize': 16, 
                               'fontWeight': 'bold',
                               'cursor': 'pointer',
                               'transition': 'all 0.3s'
                           }),
                
                html.Div(id='api-status', style={'marginTop': 15, 'textAlign': 'center'})
                
            ], style={
                'backgroundColor': COLORS['card'], 
                'padding': 25, 
                'borderRadius': 10,
                'boxShadow': '0 2px 8px rgba(0,0,0,0.1)'
            }),
        ], style={'width': '30%', 'display': 'inline-block', 'verticalAlign': 'top'}),
        
        # Right panel - Results Section
        html.Div([
            html.Div(id='results-container', children=[
                html.Div([
                    html.H3("👋 Welcome!", style={'color': COLORS['text']}),
                    html.P("Enter account details on the left and click 'Analyze Account' to get started.", 
                           style={'color': '#7f8c8d', 'fontSize': 16}),
                    html.Ul([
                        html.Li("Behavioral pattern analysis"),
                        html.Li("Network anomaly detection"),
                        html.Li("ML-powered risk assessment"),
                        html.Li("Detailed explainability reports")
                    ], style={'color': '#7f8c8d'})
                ], style={
                    'backgroundColor': COLORS['card'],
                    'padding': 30,
                    'borderRadius': 10,
                    'boxShadow': '0 2px 8px rgba(0,0,0,0.1)',
                    'textAlign': 'center'
                })
            ])
        ], style={'width': '68%', 'display': 'inline-block', 'verticalAlign': 'top', 'marginLeft': '2%'}),
    ], style={'padding': '0 20px 20px 20px'}),
    
    # Footer
    html.Div([
        html.P("Powered by Machine Learning | Built with Dash & Flask", 
               style={'textAlign': 'center', 'color': '#95a5a6', 'fontSize': 14})
    ], style={'padding': 20})
    
], style={'fontFamily': 'Arial, sans-serif', 'backgroundColor': COLORS['background'], 'minHeight': '100vh'})

@app.callback(
    [Output('results-container', 'children'),
     Output('api-status', 'children')],
    Input('analyze-btn', 'n_clicks'),
    State('username', 'value'),
    State('followers', 'value'),
    State('following', 'value'),
    State('tweets', 'value'),
    State('age', 'value'),
    State('has_profile_pic', 'value'),
    State('is_verified', 'value'),
    prevent_initial_call=True
)
def analyze_account(n_clicks, username, followers, following, tweets, age, has_profile_pic, is_verified):
    if not username:
        return html.Div("⚠️ Please enter a username", 
                       style={'color': COLORS['high'], 'fontSize': 18, 'textAlign': 'center', 'padding': 20}), ""
    
    # Prepare account data
    account_data = {
        'username': username,
        'followers_count': int(followers or 0),
        'friends_count': int(following or 0),
        'statuses_count': int(tweets or 0),
        'account_age_days': int(age or 1),
        'has_profile_image': bool(has_profile_pic),
        'bio': 'Sample bio',  # Default
        'verified': bool(is_verified),
        'favourites_count': int(tweets or 0) // 2,  # Estimate
        'location': '',
        'url': ''
    }
    
    try:
        # Check API health first
        health_response = requests.get(f'{API_URL}/api/health', timeout=2)
        
        if health_response.status_code != 200:
            return create_error_display("API is not responding properly"), "🔴 API Error"
        
        # Call API
        response = requests.post(f'{API_URL}/api/analyze', 
                                json=account_data, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            return create_results_display(data, username), "🟢 Connected"
        else:
            return create_error_display(f"API Error: {response.text}"), "🔴 API Error"
    
    except requests.exceptions.ConnectionError:
        return create_error_display(
            "Cannot connect to API",
            "Make sure the backend is running: python backend/app.py"
        ), "🔴 Disconnected"
    except requests.exceptions.Timeout:
        return create_error_display("Request timed out"), "🟡 Timeout"
    except Exception as e:
        return create_error_display(f"Error: {str(e)}"), "🔴 Error"

def create_error_display(message, details=""):
    return html.Div([
        html.H3("❌ Error", style={'color': COLORS['critical']}),
        html.P(message, style={'fontSize': 18, 'color': COLORS['text']}),
        html.P(details, style={'fontSize': 14, 'color': '#7f8c8d'}) if details else None
    ], style={
        'backgroundColor': COLORS['card'],
        'padding': 30,
        'borderRadius': 10,
        'boxShadow': '0 2px 8px rgba(0,0,0,0.1)',
        'textAlign': 'center'
    })

def create_results_display(data, username):
    prediction = data['prediction']
    explanation = data.get('explanation', [])
    risk_factors = data.get('risk_factors', [])
    behavioral = data.get('behavioral_analysis', {})
    network = data.get('network_analysis', {})
    
    # Determine color based on risk
    risk_colors = {
        'CRITICAL': COLORS['critical'],
        'HIGH': COLORS['high'],
        'MEDIUM': COLORS['medium'],
        'LOW': COLORS['low']
    }
    risk_color = risk_colors.get(prediction['risk_level'], COLORS['medium'])
    
    return html.Div([
        # Risk Assessment Card
        html.Div([
            html.H2(f"@{username}", style={'marginBottom': 15, 'color': COLORS['text']}),
            html.Div([
                html.Div([
                    html.H3(prediction['risk_level'], 
                           style={'color': risk_color, 'margin': 0, 'fontSize': 32}),
                    html.P("Risk Level", style={'color': '#7f8c8d', 'margin': 0})
                ], style={'display': 'inline-block', 'width': '33%', 'textAlign': 'center'}),
                html.Div([
                    html.H3(f"{prediction['confidence']*100:.1f}%", 
                           style={'color': COLORS['text'], 'margin': 0, 'fontSize': 32}),
                    html.P("Confidence", style={'color': '#7f8c8d', 'margin': 0})
                ], style={'display': 'inline-block', 'width': '33%', 'textAlign': 'center'}),
                html.Div([
                    html.H3('🚨 FAKE' if prediction['is_fake'] else '✅ REAL', 
                           style={'color': risk_color, 'margin': 0, 'fontSize': 28}),
                    html.P("Classification", style={'color': '#7f8c8d', 'margin': 0})
                ], style={'display': 'inline-block', 'width': '33%', 'textAlign': 'center'}),
            ]),
        ], style={
            'backgroundColor': COLORS['card'], 
            'padding': 25, 
            'borderRadius': 10,
            'boxShadow': '0 2px 8px rgba(0,0,0,0.1)', 
            'marginBottom': 20
        }),
        
        # Risk Gauge and Probability Chart
        html.Div([
            html.Div([
                dcc.Graph(figure=create_risk_gauge(prediction['fake_probability']), 
                         config={'displayModeBar': False})
            ], style={'width': '48%', 'display': 'inline-block'}),
            html.Div([
                dcc.Graph(figure=create_probability_chart(prediction), 
                         config={'displayModeBar': False})
            ], style={'width': '48%', 'display': 'inline-block', 'marginLeft': '4%'}),
        ], style={'marginBottom': 20}),
        
        # Behavioral Analysis
        html.Div([
            html.H3("📊 Behavioral Analysis", style={'color': COLORS['text'], 'marginBottom': 15}),
            html.Div([
                create_metric_card("Posting Frequency", behavioral.get('posting_frequency', 'N/A').upper(), 
                                  f"{behavioral.get('tweets_per_day', 0):.2f} tweets/day"),
                create_metric_card("Account Activity", behavioral.get('account_activity', 'N/A').upper(), 
                                  f"{behavioral.get('total_tweets', 0)} total tweets"),
                create_metric_card("Engagement Level", behavioral.get('engagement_level', 'N/A').upper(), 
                                  behavioral.get('follower_pattern', 'N/A')),
            ], style={'display': 'flex', 'justifyContent': 'space-between'})
        ], style={
            'backgroundColor': COLORS['card'], 
            'padding': 25, 
            'borderRadius': 10,
            'boxShadow': '0 2px 8px rgba(0,0,0,0.1)', 
            'marginBottom': 20
        }),
        
        # Network Analysis
        html.Div([
            html.H3("🌐 Network Analysis", style={'color': COLORS['text'], 'marginBottom': 15}),
            html.Div([
                html.Div([
                    html.P(f"Followers: {network.get('follower_count', 0):,}", 
                          style={'fontSize': 18, 'fontWeight': 'bold', 'color': COLORS['text']}),
                    html.P(f"Following: {network.get('following_count', 0):,}", 
                          style={'fontSize': 18, 'fontWeight': 'bold', 'color': COLORS['text']}),
                    html.P(f"Ratio: {network.get('ratio', 0):.2f}", 
                          style={'fontSize': 18, 'fontWeight': 'bold', 'color': COLORS['text']}),
                ], style={'width': '30%', 'display': 'inline-block', 'verticalAlign': 'top'}),
                html.Div([
                    html.P(f"Assessment: {network.get('assessment', 'N/A')}", 
                          style={'fontSize': 16, 'color': '#7f8c8d'}),
                    html.P(f"Risk Indicator: {network.get('risk_indicator', 'N/A').upper()}", 
                          style={
                              'fontSize': 16, 
                              'fontWeight': 'bold',
                              'color': COLORS['critical'] if network.get('risk_indicator') == 'high' 
                                      else COLORS['medium'] if network.get('risk_indicator') == 'medium'
                                      else COLORS['low']
                          }),
                ], style={'width': '65%', 'display': 'inline-block', 'verticalAlign': 'top', 'marginLeft': '5%'}),
            ]),
            dcc.Graph(figure=create_network_chart(network), config={'displayModeBar': False})
        ], style={
            'backgroundColor': COLORS['card'], 
            'padding': 25, 
            'borderRadius': 10,
            'boxShadow': '0 2px 8px rgba(0,0,0,0.1)', 
            'marginBottom': 20
        }),
        
        # Risk Factors
        html.Div([
            html.H3("🚩 Risk Factors", style={'color': COLORS['critical'], 'marginBottom': 15}),
            create_risk_factors_table(risk_factors) if risk_factors 
            else html.P("✅ No major risk factors detected", 
                       style={'color': COLORS['low'], 'fontSize': 16, 'textAlign': 'center', 'padding': 20})
        ], style={
            'backgroundColor': COLORS['card'], 
            'padding': 25, 
            'borderRadius': 10,
            'boxShadow': '0 2px 8px rgba(0,0,0,0.1)', 
            'marginBottom': 20
        }),
        
        # Explanation
        html.Div([
            html.H3("🔍 Detection Explanation", style={'color': COLORS['text'], 'marginBottom': 15}),
            html.P("Top factors influencing this decision:", style={'color': '#7f8c8d', 'marginBottom': 15}),
            *[create_explanation_item(exp) for exp in explanation[:5]]
        ], style={
            'backgroundColor': COLORS['card'], 
            'padding': 25, 
            'borderRadius': 10,
            'boxShadow': '0 2px 8px rgba(0,0,0,0.1)'
        }),
    ])

def create_metric_card(title, value, subtitle):
    return html.Div([
        html.P(title, style={'fontSize': 14, 'color': '#7f8c8d', 'margin': 0}),
        html.H4(value, style={'fontSize': 20, 'color': COLORS['text'], 'margin': '5px 0'}),
        html.P(subtitle, style={'fontSize': 12, 'color': '#95a5a6', 'margin': 0})
    ], style={
        'backgroundColor': '#ecf0f1',
        'padding': 15,
        'borderRadius': 8,
        'width': '30%',
        'textAlign': 'center'
    })

def create_explanation_item(exp):
    importance_pct = exp['importance'] * 100
    return html.Div([
        html.Div([
            html.Strong(f"{exp['feature'].replace('_', ' ').title()}", 
                       style={'fontSize': 16, 'color': COLORS['text']}),
            html.Div(style={
                'width': f'{importance_pct}%',
                'height': 8,
                'backgroundColor': COLORS['primary'],
                'borderRadius': 4,
                'marginTop': 5
            }),
            html.P(exp['interpretation'], 
                  style={'fontSize': 14, 'color': '#7f8c8d', 'marginTop': 5}),
            html.P(f"Importance: {importance_pct:.1f}%", 
                  style={'fontSize': 12, 'color': '#95a5a6'})
        ])
    ], style={'marginBottom': 15, 'paddingBottom': 15, 'borderBottom': '1px solid #ecf0f1'})

def create_risk_factors_table(risk_factors):
    severity_colors = {
        'critical': COLORS['critical'],
        'high': COLORS['high'],
        'medium': COLORS['medium'],
        'low': COLORS['low']
    }
    
    return html.Div([
        *[html.Div([
            html.Div([
                html.Span("●", style={
                    'color': severity_colors.get(rf.get('severity', 'medium'), COLORS['medium']),
                    'fontSize': 24,
                    'marginRight': 10
                }),
                html.Span(rf.get('factor', 'Unknown'), 
                         style={'fontSize': 16, 'fontWeight': 'bold', 'color': COLORS['text']})
            ]),
            html.P(rf.get('description', ''), 
                  style={'fontSize': 14, 'color': '#7f8c8d', 'marginLeft': 34, 'marginTop': 5})
        ], style={'marginBottom': 15, 'paddingBottom': 15, 'borderBottom': '1px solid #ecf0f1'}) 
        for rf in risk_factors]
    ])

def create_risk_gauge(fake_probability):
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = fake_probability * 100,
        title = {'text': "Fake Account Probability (%)", 'font': {'size': 18}},
        number = {'suffix': "%", 'font': {'size': 32}},
        gauge = {
            'axis': {'range': [None, 100], 'tickwidth': 1},
            'bar': {'color': "darkred", 'thickness': 0.75},
            'steps': [
                {'range': [0, 40], 'color': COLORS['low']},
                {'range': [40, 70], 'color': COLORS['medium']},
                {'range': [70, 100], 'color': COLORS['critical']}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 80
            }
        }
    ))
    
    fig.update_layout(
        height=300, 
        margin=dict(l=20, r=20, t=60, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        font={'family': 'Arial'}
    )
    return fig

def create_probability_chart(prediction):
    fig = go.Figure(data=[
        go.Bar(
            x=['Real', 'Fake'],
            y=[prediction['real_probability'] * 100, prediction['fake_probability'] * 100],
            marker_color=[COLORS['low'], COLORS['critical']],
            text=[f"{prediction['real_probability']*100:.1f}%", f"{prediction['fake_probability']*100:.1f}%"],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title="Classification Probabilities",
        yaxis_title="Probability (%)",
        height=300,
        margin=dict(l=20, r=20, t=60, b=40),
        paper_bgcolor='rgba(0,0,0,0)',
        font={'family': 'Arial'}
    )
    return fig

def create_network_chart(network):
    followers = network.get('follower_count', 0)
    following = network.get('following_count', 0)
    
    fig = go.Figure(data=[
        go.Bar(
            x=['Followers', 'Following'],
            y=[followers, following],
            marker_color=[COLORS['primary'], COLORS['high']],
            text=[f"{followers:,}", f"{following:,}"],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title="Follower vs Following Distribution",
        yaxis_title="Count",
        height=250,
        margin=dict(l=20, r=20, t=60, b=40),
        paper_bgcolor='rgba(0,0,0,0)',
        font={'family': 'Arial'}
    )
    return fig

if __name__ == '__main__':
    print("\n" + "="*50)
    print("FAKE ACCOUNT DETECTION DASHBOARD")
    print("="*50)
    print(f"\n🎨 Starting Dashboard...")
    print(f"📍 Dashboard will be available at http://localhost:8050")
    print(f"🔌 Make sure API is running at http://localhost:5000")
    print("\n" + "="*50 + "\n")
    
    app.run_server(debug=True, host='0.0.0.0', port=8050)
