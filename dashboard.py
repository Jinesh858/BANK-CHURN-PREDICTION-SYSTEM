import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd
import os

def create_dashboard(server):
    """Creates a Dash app and integrates it with the Flask server."""
    
    dash_app = dash.Dash(__name__, server=server, url_base_pathname='/dashboard/')

    # Load dataset safely
    data_path = 'churn_with_clusters.csv'
    if os.path.exists(data_path):
        data = pd.read_csv(data_path)
    else:
        data = pd.DataFrame(columns=['Age', 'Balance', 'Cluster'])  # Empty fallback dataset

    # Define layout
    dash_app.layout = html.Div([
        html.H1('📊 Bank Customer Churn Dashboard', style={'textAlign': 'center'}),
        
        dcc.Tabs(id='tabs', value='tab-1', children=[
            dcc.Tab(label='📋 Customer Data', value='tab-1'),
            dcc.Tab(label='📈 Model Performance', value='tab-2'),
            dcc.Tab(label='🔍 Customer Segmentation', value='tab-3')
        ]),
        
        html.Div(id='tabs-content', style={'padding': '20px'})
    ])

    @dash_app.callback(
        Output('tabs-content', 'children'),
        [Input('tabs', 'value')]
    )
    def render_content(tab):
        """Renders content based on selected tab."""
        
        if tab == 'tab-1':
            fig = px.histogram(data, x='Age', title='Customer Age Distribution', 
                               color_discrete_sequence=['#007bff'])
            return html.Div([
                html.H3('📋 Customer Age Distribution'),
                dcc.Graph(figure=fig)
            ])
        
        elif tab == 'tab-2':
            model_performance = pd.DataFrame({
                'Model': ['Random Forest', 'KNN', 'SVM', 'Logistic Regression'],
                'Accuracy': [0.86, 0.82, 0.85, 0.80]  # Placeholder values
            })
            fig = px.bar(model_performance, x='Model', y='Accuracy', title='Model Performance Comparison',
                         color='Model', color_discrete_sequence=px.colors.qualitative.Set2)
            return html.Div([
                html.H3('📊 Model Accuracy Comparison'),
                dcc.Graph(figure=fig)
            ])
        
        elif tab == 'tab-3':
            if 'Cluster' in data.columns:
                fig = px.scatter(data, x='Age', y='Balance', color='Cluster', 
                                 title='Customer Segmentation',
                                 color_discrete_sequence=px.colors.qualitative.Pastel)
                return html.Div([
                    html.H3('🔍 Segmentation of Customers'),
                    dcc.Graph(figure=fig)
                ])
            else:
                return html.Div([
                    html.H3('❌ No segmentation data available.'),
                    html.P('Please ensure your dataset contains a "Cluster" column.')
                ])

    return dash_app
