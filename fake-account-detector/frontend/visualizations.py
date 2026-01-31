"""Visualization components for the dashboard"""

import plotly.graph_objs as go
import plotly.express as px
import networkx as nx
import numpy as np

def create_timeline_chart(account_data):
    """Create posting timeline visualization"""
    # Simulated timeline data
    days = list(range(30))
    posts = np.random.poisson(account_data.get('tweets_per_day', 1), 30)
    
    fig = go.Figure(data=go.Scatter(
        x=days,
        y=posts,
        mode='lines+markers',
        line=dict(color='#3498db', width=2),
        marker=dict(size=6)
    ))
    
    fig.update_layout(
        title="Posting Activity (Last 30 Days)",
        xaxis_title="Days Ago",
        yaxis_title="Number of Posts",
        height=300,
        margin=dict(l=40, r=20, t=60, b=40)
    )
    
    return fig

def create_network_graph(followers, following):
    """Create network visualization"""
    G = nx.Graph()
    
    # Add central node (the account)
    G.add_node("Account", node_type="central")
    
    # Add follower nodes (sample)
    num_follower_nodes = min(followers, 10)
    for i in range(num_follower_nodes):
        G.add_node(f"F{i}", node_type="follower")
        G.add_edge("Account", f"F{i}")
    
    # Add following nodes (sample)
    num_following_nodes = min(following, 10)
    for i in range(num_following_nodes):
        G.add_node(f"Fw{i}", node_type="following")
        G.add_edge("Account", f"Fw{i}")
    
    pos = nx.spring_layout(G)
    
    edge_trace = go.Scatter(
        x=[],
        y=[],
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')
    
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_trace['x'] += tuple([x0, x1, None])
        edge_trace['y'] += tuple([y0, y1, None])
    
    node_trace = go.Scatter(
        x=[],
        y=[],
        text=[],
        mode='markers+text',
        hoverinfo='text',
        marker=dict(
            showscale=False,
            color=[],
            size=10,
            line_width=2))
    
    for node in G.nodes():
        x, y = pos[node]
        node_trace['x'] += tuple([x])
        node_trace['y'] += tuple([y])
        node_trace['text'] += tuple([node])
    
    fig = go.Figure(data=[edge_trace, node_trace],
                   layout=go.Layout(
                       title='Network Connections',
                       showlegend=False,
                       hovermode='closest',
                       margin=dict(b=0,l=0,r=0,t=40),
                       xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                       yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                   )
    
    return fig

def create_feature_importance_chart(explanation):
    """Create feature importance visualization"""
    features = [exp['feature'].replace('_', ' ').title() for exp in explanation]
    importances = [exp['importance'] * 100 for exp in explanation]
    
    fig = go.Figure(data=[
        go.Bar(
            y=features,
            x=importances,
            orientation='h',
            marker_color='#3498db'
        )
    ])
    
    fig.update_layout(
        title="Top Feature Importances",
        xaxis_title="Importance (%)",
        height=300,
        margin=dict(l=150, r=20, t=60, b=40)
    )
    
    return fig
