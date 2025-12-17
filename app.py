"""
AutoML System - A Comprehensive Machine Learning Platform
Developed for ML Lab Project
Features: Data Upload, EDA, Preprocessing, Model Training, Evaluation, Comparison & Prediction
"""

import streamlit as st
import pandas as pd
import numpy as np
import io
import base64
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Scikit-learn imports
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score,
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve
)

# Classification Models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

# Rule-based Classifier
from sklearn.tree import DecisionTreeClassifier as RuleBasedClassifier

# Regression Models
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

# Clustering Models
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering

# Visualization with Plotly
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# PDF Generation - with fallback if not installed
try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image, PageBreak
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib.enums import TA_CENTER, TA_LEFT
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False

# Configure page
st.set_page_config(
    page_title="AutoML System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    /* Main container */
    .main {
        padding: 0rem 1rem;
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.4);
    }
    
    .main-header h1 {
        color: white;
        font-size: 3rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }
    
    .main-header p {
        color: rgba(255,255,255,0.9);
        font-size: 1.2rem;
    }
    
    /* Card styling */
    .metric-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        text-align: center;
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        color: #667eea;
    }
    
    .metric-label {
        color: #666;
        font-size: 1rem;
        margin-top: 0.5rem;
    }
    
    /* Success/Info/Warning boxes */
    .success-box {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    .info-box {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    .warning-box {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 25px;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
    }
    
    /* Select box styling */
    .stSelectbox > div > div {
        border-radius: 10px;
    }
    
    /* Dataframe styling */
    .dataframe {
        border-radius: 10px;
        overflow: hidden;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        border-radius: 10px 10px 0 0;
        padding: 10px 20px;
        font-weight: 600;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    /* Progress bar */
    .stProgress > div > div {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, #f5f7fa 0%, #e8ecf3 100%);
        border-radius: 10px;
        font-weight: 600;
    }
    
    /* Feature importance chart area */
    .plot-container {
        background: white;
        padding: 1rem;
        border-radius: 15px;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
    }
    
    /* EDA cards */
    .eda-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.2rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .eda-card h3 {
        font-size: 2rem;
        margin: 0;
        font-weight: 700;
    }
    
    .eda-card p {
        margin: 0.3rem 0 0 0;
        opacity: 0.9;
    }
    
    .outlier-card {
        background: linear-gradient(135deg, #f5576c 0%, #f093fb 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    
    .missing-card {
        background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None
if 'target_column' not in st.session_state:
    st.session_state.target_column = None
if 'feature_columns' not in st.session_state:
    st.session_state.feature_columns = []
if 'trained_models' not in st.session_state:
    st.session_state.trained_models = {}
if 'model_results' not in st.session_state:
    st.session_state.model_results = {}
if 'problem_type' not in st.session_state:
    st.session_state.problem_type = 'classification'
if 'label_encoders' not in st.session_state:
    st.session_state.label_encoders = {}
if 'scaler' not in st.session_state:
    st.session_state.scaler = None
if 'X_train' not in st.session_state:
    st.session_state.X_train = None
if 'X_test' not in st.session_state:
    st.session_state.X_test = None
if 'y_train' not in st.session_state:
    st.session_state.y_train = None
if 'y_test' not in st.session_state:
    st.session_state.y_test = None
if 'eda_report' not in st.session_state:
    st.session_state.eda_report = {}
if 'preprocessing_log' not in st.session_state:
    st.session_state.preprocessing_log = []
if 'outliers_info' not in st.session_state:
    st.session_state.outliers_info = {}
if 'eda_figures' not in st.session_state:
    st.session_state.eda_figures = {}
if 'original_features' not in st.session_state:
    st.session_state.original_features = []
if 'original_data' not in st.session_state:
    st.session_state.original_data = None
if 'encoding_method' not in st.session_state:
    st.session_state.encoding_method = "Label Encoding"


# ============== SAMPLE DATA FUNCTIONS ==============

def load_sample_data(dataset_name):
    """Load sample datasets for demonstration"""
    if dataset_name == "Iris (Classification)":
        from sklearn.datasets import load_iris
        data = load_iris()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['target'] = data.target
        return df
    elif dataset_name == "Wine (Classification)":
        from sklearn.datasets import load_wine
        data = load_wine()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['target'] = data.target
        return df
    elif dataset_name == "Breast Cancer (Classification)":
        from sklearn.datasets import load_breast_cancer
        data = load_breast_cancer()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['target'] = data.target
        return df
    elif dataset_name == "Diabetes (Regression)":
        from sklearn.datasets import load_diabetes
        data = load_diabetes()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['target'] = data.target
        return df
    elif dataset_name == "Boston Housing (Regression)":
        # Create synthetic boston-like dataset
        from sklearn.datasets import make_regression
        X, y = make_regression(n_samples=506, n_features=13, noise=10, random_state=42)
        feature_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']
        df = pd.DataFrame(X, columns=feature_names)
        df['target'] = y
        return df
    return None


# ============== EDA FUNCTIONS ==============

def detect_outliers_iqr(df, column):
    """Detect outliers using IQR method"""
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    return len(outliers), lower_bound, upper_bound, outliers.index.tolist()


def detect_outliers_zscore(df, column, threshold=3):
    """Detect outliers using Z-score method"""
    mean = df[column].mean()
    std = df[column].std()
    z_scores = np.abs((df[column] - mean) / std)
    outliers = df[z_scores > threshold]
    return len(outliers), outliers.index.tolist()


def perform_eda(df):
    """Perform comprehensive EDA on the dataset"""
    eda_report = {}
    
    # Basic info
    eda_report['shape'] = df.shape
    eda_report['columns'] = list(df.columns)
    eda_report['dtypes'] = df.dtypes.astype(str).to_dict()
    
    # Missing values analysis
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    eda_report['missing_values'] = missing.to_dict()
    eda_report['missing_pct'] = missing_pct.to_dict()
    eda_report['total_missing'] = missing.sum()
    eda_report['total_missing_pct'] = (missing.sum() / (df.shape[0] * df.shape[1]) * 100).round(2)
    
    # Numeric columns analysis
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    eda_report['numeric_columns'] = numeric_cols
    eda_report['categorical_columns'] = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Statistics for numeric columns
    if numeric_cols:
        eda_report['statistics'] = df[numeric_cols].describe().to_dict()
    
    # Outlier detection for numeric columns
    outliers_info = {}
    for col in numeric_cols:
        if df[col].notna().sum() > 0:
            count, lower, upper, indices = detect_outliers_iqr(df, col)
            outliers_info[col] = {
                'count': count,
                'percentage': round(count / len(df) * 100, 2),
                'lower_bound': round(lower, 4),
                'upper_bound': round(upper, 4),
                'indices': indices[:10]  # Store first 10 indices
            }
    eda_report['outliers'] = outliers_info
    
    # Duplicate rows
    eda_report['duplicates'] = df.duplicated().sum()
    eda_report['duplicates_pct'] = round(df.duplicated().sum() / len(df) * 100, 2)
    
    # Unique values per column
    eda_report['unique_values'] = {col: df[col].nunique() for col in df.columns}
    
    return eda_report


def plot_missing_values(df):
    """Create missing values visualization"""
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=True)
    
    if len(missing) == 0:
        return None
    
    colors = ['#EF4444' if v > len(df)*0.3 else '#F59E0B' if v > len(df)*0.1 else '#10B981' 
              for v in missing.values]
    
    fig = go.Figure(go.Bar(
        x=missing.values,
        y=missing.index,
        orientation='h',
        marker=dict(color=colors, line=dict(color='white', width=2)),
        text=[f'{v} ({v/len(df)*100:.1f}%)' for v in missing.values],
        textposition='outside',
        hovertemplate='<b>%{y}</b><br>Missing: %{x}<br>Percentage: %{text}<extra></extra>'
    ))
    
    fig.update_layout(
        title=dict(text='🔍 Missing Values Analysis', font=dict(size=20, color='#1a1a2e', family='Arial Black'), x=0.5),
        xaxis=dict(title='Number of Missing Values', gridcolor='rgba(0,0,0,0.08)'),
        yaxis=dict(title=''),
        height=max(300, len(missing) * 40),
        template='plotly_white',
        paper_bgcolor='#fafbfc',
        plot_bgcolor='#fafbfc',
        margin=dict(l=20, r=100, t=60, b=40)
    )
    
    return fig


def plot_outliers_boxplot(df, columns):
    """Create box plots to show outliers"""
    if not columns:
        return None
    
    n_cols = min(len(columns), 6)
    fig = make_subplots(rows=1, cols=n_cols, subplot_titles=[f'📦 {col[:15]}...' if len(col) > 15 else f'📦 {col}' for col in columns[:n_cols]])
    
    colors = ['#6366F1', '#10B981', '#F59E0B', '#EF4444', '#8B5CF6', '#06B6D4']
    
    for i, col in enumerate(columns[:n_cols], 1):
        fig.add_trace(go.Box(
            y=df[col].dropna(),
            name=col,
            marker=dict(color=colors[i-1], size=6),
            line=dict(color=colors[i-1], width=2),
            fillcolor=f'rgba{tuple(list(int(colors[i-1].lstrip("#")[j:j+2], 16) for j in (0, 2, 4)) + [0.3])}',
            boxpoints='outliers',
            hovertemplate=f'<b>{col}</b><br>Value: %{{y:.3f}}<extra></extra>'
        ), row=1, col=i)
    
    fig.update_layout(
        title=dict(text='📊 Outlier Detection (Box Plots)', font=dict(size=20, color='#1a1a2e', family='Arial Black'), x=0.5),
        height=450,
        showlegend=False,
        template='plotly_white',
        paper_bgcolor='#fafbfc',
        plot_bgcolor='#fafbfc'
    )
    
    return fig


def plot_distribution_grid(df, columns):
    """Create distribution plots for numeric columns"""
    if not columns:
        return None
    
    n_cols = min(len(columns), 6)
    n_rows = (n_cols + 2) // 3
    cols_per_row = min(3, n_cols)
    
    fig = make_subplots(rows=n_rows, cols=cols_per_row, 
                        subplot_titles=[f'📈 {col[:12]}...' if len(col) > 12 else f'📈 {col}' for col in columns[:n_cols]])
    
    colors = ['#6366F1', '#10B981', '#F59E0B', '#EF4444', '#8B5CF6', '#06B6D4']
    
    for i, col in enumerate(columns[:n_cols]):
        row = i // cols_per_row + 1
        col_idx = i % cols_per_row + 1
        
        fig.add_trace(go.Histogram(
            x=df[col].dropna(),
            name=col,
            marker=dict(color=colors[i % len(colors)], line=dict(color='white', width=1)),
            opacity=0.85,
            hovertemplate=f'<b>{col}</b><br>Range: %{{x}}<br>Count: %{{y}}<extra></extra>'
        ), row=row, col=col_idx)
    
    fig.update_layout(
        title=dict(text='📊 Feature Distributions', font=dict(size=20, color='#1a1a2e', family='Arial Black'), x=0.5),
        height=300 * n_rows,
        showlegend=False,
        template='plotly_white',
        paper_bgcolor='#fafbfc',
        plot_bgcolor='#fafbfc'
    )
    
    return fig


def plot_correlation_heatmap(df):
    """Create correlation heatmap"""
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.shape[1] < 2:
        return None
    
    corr = numeric_df.corr()
    
    # Create hover text
    hover_text = []
    for i in range(len(corr)):
        row = []
        for j in range(len(corr.columns)):
            row.append(f'<b>{corr.index[i]}</b> vs <b>{corr.columns[j]}</b><br>Correlation: {corr.iloc[i, j]:.4f}')
        hover_text.append(row)
    
    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=corr.columns,
        y=corr.index,
        colorscale=[
            [0, '#EF4444'], [0.25, '#FCA5A5'], [0.5, '#FAFAFA'],
            [0.75, '#93C5FD'], [1, '#3B82F6']
        ],
        zmin=-1, zmax=1,
        hovertemplate='%{text}<extra></extra>',
        text=hover_text,
        colorbar=dict(title='Correlation', thickness=15)
    ))
    
    # Add annotations
    for i in range(len(corr)):
        for j in range(len(corr.columns)):
            val = corr.iloc[i, j]
            fig.add_annotation(
                x=corr.columns[j], y=corr.index[i],
                text=f'{val:.2f}',
                font=dict(color='white' if abs(val) > 0.5 else '#333', size=9),
                showarrow=False
            )
    
    fig.update_layout(
        title=dict(text='🔗 Feature Correlation Matrix', font=dict(size=20, color='#1a1a2e', family='Arial Black'), x=0.5),
        height=max(500, len(corr.columns) * 35),
        template='plotly_white',
        paper_bgcolor='#fafbfc',
        plot_bgcolor='#fafbfc',
        xaxis=dict(tickangle=45)
    )
    
    return fig


def plot_target_distribution(df, target_col):
    """Plot target variable distribution"""
    fig = go.Figure()
    
    if df[target_col].dtype in ['object', 'category'] or df[target_col].nunique() <= 10:
        # Categorical - use bar chart
        value_counts = df[target_col].value_counts()
        colors = px.colors.qualitative.Set2[:len(value_counts)]
        
        fig.add_trace(go.Bar(
            x=value_counts.index.astype(str),
            y=value_counts.values,
            marker=dict(color=colors, line=dict(color='white', width=2)),
            text=[f'{v} ({v/len(df)*100:.1f}%)' for v in value_counts.values],
            textposition='outside',
            hovertemplate='<b>Class: %{x}</b><br>Count: %{y}<br>%{text}<extra></extra>'
        ))
        
        fig.update_layout(
            title=dict(text=f'🎯 Target Distribution: {target_col}', font=dict(size=20, color='#1a1a2e', family='Arial Black'), x=0.5),
            xaxis=dict(title='Class'),
            yaxis=dict(title='Count')
        )
    else:
        # Continuous - use histogram
        fig.add_trace(go.Histogram(
            x=df[target_col],
            marker=dict(color='#6366F1', line=dict(color='white', width=1)),
            opacity=0.85,
            hovertemplate='<b>Range:</b> %{x}<br><b>Count:</b> %{y}<extra></extra>'
        ))
        
        fig.update_layout(
            title=dict(text=f'🎯 Target Distribution: {target_col}', font=dict(size=20, color='#1a1a2e', family='Arial Black'), x=0.5),
            xaxis=dict(title=target_col),
            yaxis=dict(title='Frequency')
        )
    
    fig.update_layout(
        height=400,
        template='plotly_white',
        paper_bgcolor='#fafbfc',
        plot_bgcolor='#fafbfc'
    )
    
    return fig


def plot_categorical_distribution(df, columns):
    """Create bar charts for categorical columns"""
    if not columns:
        return None
    
    n_cols = min(len(columns), 6)
    n_rows = (n_cols + 1) // 2
    cols_per_row = min(2, n_cols)
    
    fig = make_subplots(rows=n_rows, cols=cols_per_row, 
                        subplot_titles=[f'📊 {col[:15]}...' if len(col) > 15 else f'📊 {col}' for col in columns[:n_cols]])
    
    colors = px.colors.qualitative.Set2
    
    for i, col in enumerate(columns[:n_cols]):
        row = i // cols_per_row + 1
        col_idx = i % cols_per_row + 1
        
        value_counts = df[col].value_counts().head(10)  # Top 10 categories
        
        fig.add_trace(go.Bar(
            x=value_counts.index.astype(str),
            y=value_counts.values,
            marker=dict(color=colors[i % len(colors)], line=dict(color='white', width=1)),
            text=value_counts.values,
            textposition='outside',
            name=col,
            hovertemplate=f'<b>{col}</b><br>Category: %{{x}}<br>Count: %{{y}}<extra></extra>'
        ), row=row, col=col_idx)
    
    fig.update_layout(
        title=dict(text='📊 Categorical Feature Distributions', font=dict(size=20, color='#1a1a2e', family='Arial Black'), x=0.5),
        height=300 * n_rows,
        showlegend=False,
        template='plotly_white',
        paper_bgcolor='#fafbfc',
        plot_bgcolor='#fafbfc'
    )
    
    return fig


def plot_pairwise_scatter(df, columns, target_col=None):
    """Create scatter plot matrix for selected columns"""
    if len(columns) < 2:
        return None
    
    cols_to_plot = columns[:4]  # Max 4 columns for readability
    
    if target_col and target_col in df.columns:
        fig = px.scatter_matrix(
            df,
            dimensions=cols_to_plot,
            color=target_col,
            title='🔗 Pairwise Feature Relationships',
            opacity=0.6
        )
    else:
        fig = px.scatter_matrix(
            df,
            dimensions=cols_to_plot,
            title='🔗 Pairwise Feature Relationships',
            opacity=0.6
        )
    
    fig.update_layout(
        height=700,
        template='plotly_white',
        paper_bgcolor='#fafbfc',
        title=dict(font=dict(size=20, color='#1a1a2e', family='Arial Black'), x=0.5)
    )
    fig.update_traces(diagonal_visible=False, showupperhalf=False)
    
    return fig


def plot_feature_vs_target(df, feature_col, target_col):
    """Create scatter or violin plot of feature vs target"""
    fig = go.Figure()
    
    is_target_categorical = df[target_col].dtype == 'object' or df[target_col].nunique() <= 10
    is_feature_categorical = df[feature_col].dtype == 'object' or df[feature_col].nunique() <= 10
    
    if is_target_categorical and not is_feature_categorical:
        # Violin plot: numeric feature by categorical target
        for i, target_val in enumerate(df[target_col].unique()):
            fig.add_trace(go.Violin(
                x=df[df[target_col] == target_val][target_col].astype(str),
                y=df[df[target_col] == target_val][feature_col],
                name=str(target_val),
                box_visible=True,
                meanline_visible=True
            ))
        fig.update_layout(
            title=f'📈 {feature_col} by {target_col}',
            xaxis_title=target_col,
            yaxis_title=feature_col
        )
    elif not is_target_categorical and not is_feature_categorical:
        # Scatter plot: both numeric
        fig.add_trace(go.Scatter(
            x=df[feature_col],
            y=df[target_col],
            mode='markers',
            marker=dict(color='#6366F1', size=6, opacity=0.6),
            hovertemplate=f'<b>{feature_col}:</b> %{{x:.3f}}<br><b>{target_col}:</b> %{{y:.3f}}<extra></extra>'
        ))
        
        # Add trendline
        from scipy import stats
        slope, intercept, r_value, p_value, std_err = stats.linregress(df[feature_col].dropna(), df[target_col].dropna())
        x_line = np.linspace(df[feature_col].min(), df[feature_col].max(), 100)
        y_line = slope * x_line + intercept
        
        fig.add_trace(go.Scatter(
            x=x_line, y=y_line,
            mode='lines',
            name=f'Trend (R²={r_value**2:.3f})',
            line=dict(color='#EF4444', dash='dash')
        ))
        
        fig.update_layout(
            title=f'📈 {feature_col} vs {target_col}',
            xaxis_title=feature_col,
            yaxis_title=target_col
        )
    else:
        # Both categorical - use heatmap
        cross_tab = pd.crosstab(df[feature_col], df[target_col])
        fig = go.Figure(go.Heatmap(
            z=cross_tab.values,
            x=cross_tab.columns.astype(str),
            y=cross_tab.index.astype(str),
            colorscale='Viridis'
        ))
        fig.update_layout(
            title=f'📊 {feature_col} vs {target_col} (Cross-tabulation)',
            xaxis_title=target_col,
            yaxis_title=feature_col
        )
    
    fig.update_layout(
        height=400,
        template='plotly_white',
        paper_bgcolor='#fafbfc',
        plot_bgcolor='#fafbfc',
        title=dict(font=dict(size=18, color='#1a1a2e', family='Arial Black'), x=0.5)
    )
    
    return fig


def plot_data_types_pie(df):
    """Create pie chart showing data types distribution"""
    dtype_counts = df.dtypes.astype(str).value_counts()
    
    fig = go.Figure(go.Pie(
        labels=dtype_counts.index,
        values=dtype_counts.values,
        hole=0.4,
        marker=dict(colors=px.colors.qualitative.Set2),
        textinfo='label+percent',
        hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
    ))
    
    fig.update_layout(
        title=dict(text='📊 Data Types Distribution', font=dict(size=18, color='#1a1a2e', family='Arial Black'), x=0.5),
        height=350,
        template='plotly_white',
        paper_bgcolor='#fafbfc'
    )
    
    return fig


def plot_unique_values_bar(df):
    """Create bar chart showing unique values per column"""
    unique_counts = df.nunique().sort_values(ascending=True)
    
    # Color code: high cardinality = red, low = green
    colors = ['#EF4444' if v > len(df)*0.5 else '#F59E0B' if v > len(df)*0.1 else '#10B981' 
              for v in unique_counts.values]
    
    fig = go.Figure(go.Bar(
        x=unique_counts.values,
        y=unique_counts.index,
        orientation='h',
        marker=dict(color=colors, line=dict(color='white', width=1)),
        text=unique_counts.values,
        textposition='outside',
        hovertemplate='<b>%{y}</b><br>Unique Values: %{x}<extra></extra>'
    ))
    
    fig.update_layout(
        title=dict(text='📊 Unique Values per Column', font=dict(size=18, color='#1a1a2e', family='Arial Black'), x=0.5),
        xaxis=dict(title='Number of Unique Values'),
        yaxis=dict(title=''),
        height=max(350, len(unique_counts) * 25),
        template='plotly_white',
        paper_bgcolor='#fafbfc',
        plot_bgcolor='#fafbfc',
        margin=dict(l=20, r=80, t=60, b=40)
    )
    
    return fig


def load_sample_data(dataset_name):
    if dataset_name == "Iris (Classification)":
        from sklearn.datasets import load_iris
        data = load_iris()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['target'] = data.target
        return df
    elif dataset_name == "Wine (Classification)":
        from sklearn.datasets import load_wine
        data = load_wine()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['target'] = data.target
        return df
    elif dataset_name == "Breast Cancer (Classification)":
        from sklearn.datasets import load_breast_cancer
        data = load_breast_cancer()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['target'] = data.target
        return df
    elif dataset_name == "Diabetes (Regression)":
        from sklearn.datasets import load_diabetes
        data = load_diabetes()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['target'] = data.target
        return df
    elif dataset_name == "Boston Housing (Regression)":
        # Create synthetic housing data since boston is deprecated
        np.random.seed(42)
        n_samples = 506
        df = pd.DataFrame({
            'CRIM': np.random.exponential(3, n_samples),
            'ZN': np.random.uniform(0, 100, n_samples),
            'INDUS': np.random.uniform(0, 30, n_samples),
            'CHAS': np.random.binomial(1, 0.07, n_samples),
            'NOX': np.random.uniform(0.3, 0.9, n_samples),
            'RM': np.random.normal(6.3, 0.7, n_samples),
            'AGE': np.random.uniform(0, 100, n_samples),
            'DIS': np.random.exponential(3, n_samples),
            'RAD': np.random.randint(1, 25, n_samples),
            'TAX': np.random.uniform(180, 720, n_samples),
            'PTRATIO': np.random.uniform(12, 22, n_samples),
            'B': np.random.uniform(0, 400, n_samples),
            'LSTAT': np.random.uniform(1, 40, n_samples),
        })
        df['target'] = 22 + 5*df['RM'] - 0.5*df['LSTAT'] + np.random.normal(0, 3, n_samples)
        return df
    return None


def get_data_summary(df):
    """Generate comprehensive data summary"""
    summary = {
        'rows': len(df),
        'columns': len(df.columns),
        'numeric_cols': len(df.select_dtypes(include=[np.number]).columns),
        'categorical_cols': len(df.select_dtypes(include=['object', 'category']).columns),
        'missing_values': df.isnull().sum().sum(),
        'duplicate_rows': df.duplicated().sum(),
        'memory_usage': df.memory_usage(deep=True).sum() / 1024**2  # MB
    }
    return summary


def preprocess_data(df, target_col, feature_cols, handle_missing, scaling_method, encode_categorical):
    """Preprocess the data"""
    processed_df = df[feature_cols + [target_col]].copy()
    label_encoders = {}
    
    # Handle missing values
    if handle_missing == "Drop rows":
        processed_df = processed_df.dropna()
    elif handle_missing == "Mean imputation":
        numeric_cols = processed_df.select_dtypes(include=[np.number]).columns
        imputer = SimpleImputer(strategy='mean')
        processed_df[numeric_cols] = imputer.fit_transform(processed_df[numeric_cols])
    elif handle_missing == "Median imputation":
        numeric_cols = processed_df.select_dtypes(include=[np.number]).columns
        imputer = SimpleImputer(strategy='median')
        processed_df[numeric_cols] = imputer.fit_transform(processed_df[numeric_cols])
    elif handle_missing == "Mode imputation":
        imputer = SimpleImputer(strategy='most_frequent')
        processed_df = pd.DataFrame(imputer.fit_transform(processed_df), columns=processed_df.columns)
    
    # Encode categorical variables
    if encode_categorical:
        cat_cols = processed_df.select_dtypes(include=['object', 'category']).columns
        for col in cat_cols:
            if col != target_col:
                le = LabelEncoder()
                processed_df[col] = le.fit_transform(processed_df[col].astype(str))
                label_encoders[col] = le
    
    # Encode target if categorical
    if processed_df[target_col].dtype == 'object' or processed_df[target_col].dtype.name == 'category':
        le = LabelEncoder()
        processed_df[target_col] = le.fit_transform(processed_df[target_col].astype(str))
        label_encoders[target_col] = le
    
    # Scale features
    feature_data = processed_df[feature_cols]
    scaler = None
    
    if scaling_method == "StandardScaler":
        scaler = StandardScaler()
        feature_data = pd.DataFrame(scaler.fit_transform(feature_data), columns=feature_cols)
    elif scaling_method == "MinMaxScaler":
        scaler = MinMaxScaler()
        feature_data = pd.DataFrame(scaler.fit_transform(feature_data), columns=feature_cols)
    
    processed_df[feature_cols] = feature_data.values
    
    return processed_df, label_encoders, scaler


def train_model(model_name, X_train, y_train, problem_type, hyperparams=None):
    """Train a machine learning model"""
    models = {
        'classification': {
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
            'Decision Tree': DecisionTreeClassifier(random_state=42),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(random_state=42),
            'SVM': SVC(probability=True, random_state=42),
            'KNN': KNeighborsClassifier(),
            'Naive Bayes': GaussianNB(),
            'AdaBoost': AdaBoostClassifier(random_state=42),
            'Rule-Based Classifier': DecisionTreeClassifier(max_depth=5, min_samples_leaf=10, random_state=42)  # Simple interpretable rules
        },
        'regression': {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(random_state=42),
            'Lasso Regression': Lasso(random_state=42),
            'ElasticNet': ElasticNet(random_state=42),
            'Decision Tree': DecisionTreeRegressor(random_state=42),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(random_state=42),
            'SVR': SVR(),
            'KNN': KNeighborsRegressor(),
            'AdaBoost': AdaBoostRegressor(random_state=42)
        }
    }
    
    model = models[problem_type][model_name]
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test, problem_type):
    """Evaluate model performance"""
    y_pred = model.predict(X_test)
    
    if problem_type == 'classification':
        metrics = {
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
            'Recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
            'F1-Score': f1_score(y_test, y_pred, average='weighted', zero_division=0)
        }
        
        # Get confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        metrics['Confusion Matrix'] = cm
        
        # Get classification report
        metrics['Classification Report'] = classification_report(y_test, y_pred, zero_division=0)
        
        # ROC AUC for binary classification
        if len(np.unique(y_test)) == 2:
            if hasattr(model, 'predict_proba'):
                y_prob = model.predict_proba(X_test)[:, 1]
                fpr, tpr, _ = roc_curve(y_test, y_prob)
                metrics['ROC AUC'] = auc(fpr, tpr)
                metrics['FPR'] = fpr
                metrics['TPR'] = tpr
    else:
        metrics = {
            'MAE': mean_absolute_error(y_test, y_pred),
            'MSE': mean_squared_error(y_test, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
            'R² Score': r2_score(y_test, y_pred)
        }
        metrics['y_pred'] = y_pred
    
    return metrics


def plot_confusion_matrix(cm, title='Confusion Matrix'):
    """Plot confusion matrix with Plotly - Enhanced colors and tooltips"""
    # Create labels for axes
    labels = [f'Class {i}' for i in range(cm.shape[0])]
    
    # Create custom hover text
    hover_text = []
    for i in range(cm.shape[0]):
        row = []
        for j in range(cm.shape[1]):
            row.append(f'True: {labels[i]}<br>Predicted: {labels[j]}<br>Count: {cm[i, j]}')
        hover_text.append(row)
    
    # Create annotation text
    annotations = []
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annotations.append(
                dict(
                    x=j, y=i,
                    text=f'<b>{cm[i, j]}</b>',
                    font=dict(color='white' if cm[i, j] > cm.max()/2 else '#333', size=18, family='Arial Black'),
                    showarrow=False
                )
            )
    
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=labels,
        y=labels,
        colorscale=[
            [0, '#E8F4FD'],
            [0.25, '#81D4FA'],
            [0.5, '#29B6F6'],
            [0.75, '#0288D1'],
            [1, '#01579B']
        ],
        showscale=True,
        colorbar=dict(
            title=dict(text='Count', font=dict(size=14, color='#333')),
            tickfont=dict(size=12, color='#333'),
            thickness=20,
            len=0.8
        ),
        hovertemplate='%{text}<extra></extra>',
        text=hover_text
    ))
    
    fig.update_layout(
        title=dict(text=f'📊 {title}', font=dict(size=20, color='#1a1a2e', family='Arial Black'), x=0.5),
        xaxis=dict(
            title=dict(text='Predicted Label', font=dict(size=14, color='#333')),
            tickfont=dict(size=12, color='#333'),
            tickmode='linear',
            showgrid=False
        ),
        yaxis=dict(
            title=dict(text='True Label', font=dict(size=14, color='#333')),
            tickfont=dict(size=12, color='#333'),
            tickmode='linear',
            autorange='reversed',
            showgrid=False
        ),
        annotations=annotations,
        height=520,
        template='plotly_white',
        paper_bgcolor='#fafbfc',
        plot_bgcolor='#fafbfc',
        margin=dict(l=80, r=80, t=80, b=80)
    )
    
    return fig


def plot_roc_curve(fpr, tpr, roc_auc):
    """Plot ROC curve with Plotly - Enhanced colors and tooltips"""
    fig = go.Figure()
    
    # Add filled area under ROC curve
    fig.add_trace(go.Scatter(
        x=fpr, y=tpr,
        fill='tozeroy',
        fillcolor='rgba(99, 102, 241, 0.2)',
        line=dict(color='#6366F1', width=4),
        name=f'ROC Curve (AUC = {roc_auc:.3f})',
        hovertemplate='<b>ROC Curve</b><br>FPR: %{x:.3f}<br>TPR: %{y:.3f}<extra></extra>'
    ))
    
    # Add diagonal line
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        line=dict(color='#EF4444', width=2, dash='dash'),
        name='Random Classifier',
        hovertemplate='<b>Random Classifier</b><br>FPR: %{x:.3f}<br>TPR: %{y:.3f}<extra></extra>'
    ))
    
    # Add AUC annotation
    fig.add_annotation(
        x=0.6, y=0.3,
        text=f'<b>AUC = {roc_auc:.4f}</b>',
        font=dict(size=18, color='#6366F1', family='Arial Black'),
        showarrow=False,
        bgcolor='rgba(255,255,255,0.9)',
        bordercolor='#6366F1',
        borderwidth=2,
        borderpad=8
    )
    
    fig.update_layout(
        title=dict(text='📈 ROC Curve Analysis', font=dict(size=20, color='#1a1a2e', family='Arial Black'), x=0.5),
        xaxis=dict(
            title=dict(text='False Positive Rate', font=dict(size=14, color='#333')),
            range=[0, 1],
            gridcolor='rgba(0,0,0,0.08)',
            tickfont=dict(size=12, color='#333'),
            showline=True,
            linecolor='#e0e0e0'
        ),
        yaxis=dict(
            title=dict(text='True Positive Rate', font=dict(size=14, color='#333')),
            range=[0, 1.05],
            gridcolor='rgba(0,0,0,0.08)',
            tickfont=dict(size=12, color='#333'),
            showline=True,
            linecolor='#e0e0e0'
        ),
        legend=dict(
            x=0.55, y=0.12,
            bgcolor='rgba(255,255,255,0.95)',
            bordercolor='#e0e0e0',
            borderwidth=1,
            font=dict(size=12, color='#333')
        ),
        height=520,
        template='plotly_white',
        paper_bgcolor='#fafbfc',
        plot_bgcolor='#fafbfc',
        margin=dict(l=80, r=40, t=80, b=80)
    )
    
    return fig


def plot_feature_importance(model, feature_names, title='Feature Importance'):
    """Plot feature importance with Plotly - Enhanced colors and tooltips"""
    try:
        if hasattr(model, 'feature_importances_'):
            importance = np.array(model.feature_importances_).flatten()
        elif hasattr(model, 'coef_'):
            coef = np.array(model.coef_)
            if coef.ndim > 1:
                importance = np.abs(coef).mean(axis=0).flatten()
            else:
                importance = np.abs(coef).flatten()
        else:
            return None
        
        # Ensure feature_names is a list
        feature_names = list(feature_names)
        
        # Handle dimension mismatch
        if len(importance) != len(feature_names):
            min_len = min(len(importance), len(feature_names))
            importance = importance[:min_len]
            feature_names = feature_names[:min_len]
        
        # Sort by importance - limit to top 15
        n_features = min(15, len(importance))
        indices = np.argsort(importance)[-n_features:]  # Get top N indices
        
        sorted_features = [feature_names[i] for i in indices]
        sorted_importance = [importance[i] for i in indices]
        
        # Create beautiful gradient colors (purple to cyan)
        colors = [f'rgba({int(99 + (i/n_features)*50)}, {int(102 + (i/n_features)*100)}, {int(241 - (i/n_features)*40)}, 0.9)' for i in range(n_features)]
        
        fig = go.Figure(go.Bar(
            x=sorted_importance,
            y=sorted_features,
            orientation='h',
            marker=dict(
                color=colors,
                line=dict(color='white', width=2)
            ),
            text=[f'<b>{v:.4f}</b>' for v in sorted_importance],
            textposition='outside',
            textfont=dict(size=11, color='#333'),
            hovertemplate='<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>'
        ))
        
        fig.update_layout(
            title=dict(text=f'🎯 {title}', font=dict(size=20, color='#1a1a2e', family='Arial Black'), x=0.5),
            xaxis=dict(
                title=dict(text='Importance Score', font=dict(size=14, color='#333')),
                gridcolor='rgba(0,0,0,0.08)',
                tickfont=dict(size=12, color='#333'),
                showline=True,
                linecolor='#e0e0e0'
            ),
            yaxis=dict(
                title='',
                tickfont=dict(size=12, color='#333'),
                showline=True,
                linecolor='#e0e0e0'
            ),
            height=max(450, n_features * 40),
            template='plotly_white',
            paper_bgcolor='#fafbfc',
            plot_bgcolor='#fafbfc',
            margin=dict(l=20, r=100, t=80, b=50),
            bargap=0.3
        )
        
        return fig
    except Exception as e:
        return None


def plot_actual_vs_predicted(y_test, y_pred):
    """Plot actual vs predicted values for regression with Plotly - Enhanced"""
    y_test_arr = np.array(y_test)
    y_pred_arr = np.array(y_pred)
    min_val = min(y_test_arr.min(), y_pred_arr.min())
    max_val = max(y_test_arr.max(), y_pred_arr.max())
    
    # Calculate R² for annotation
    ss_res = np.sum((y_test_arr - y_pred_arr) ** 2)
    ss_tot = np.sum((y_test_arr - np.mean(y_test_arr)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    fig = go.Figure()
    
    # Scatter plot with gradient colors based on error
    errors = np.abs(y_test_arr - y_pred_arr)
    
    fig.add_trace(go.Scatter(
        x=y_test_arr, y=y_pred_arr,
        mode='markers',
        marker=dict(
            color=errors,
            colorscale='RdYlGn_r',
            size=12,
            opacity=0.8,
            line=dict(color='white', width=1),
            colorbar=dict(
                title=dict(text='Error', font=dict(size=12)),
                thickness=15,
                len=0.6
            )
        ),
        name='Predictions',
        hovertemplate='<b>Prediction</b><br>Actual: %{x:.3f}<br>Predicted: %{y:.3f}<br>Error: %{marker.color:.3f}<extra></extra>'
    ))
    
    # Perfect prediction line
    fig.add_trace(go.Scatter(
        x=[min_val, max_val], y=[min_val, max_val],
        mode='lines',
        line=dict(color='#10B981', width=3, dash='dash'),
        name='Perfect Prediction',
        hovertemplate='<b>Perfect Prediction Line</b><extra></extra>'
    ))
    
    # Add R² annotation
    fig.add_annotation(
        x=0.05, y=0.95,
        xref='paper', yref='paper',
        text=f'<b>R² = {r2:.4f}</b>',
        font=dict(size=16, color='#10B981', family='Arial Black'),
        showarrow=False,
        bgcolor='rgba(255,255,255,0.9)',
        bordercolor='#10B981',
        borderwidth=2,
        borderpad=8
    )
    
    fig.update_layout(
        title=dict(text='📊 Actual vs Predicted Values', font=dict(size=20, color='#1a1a2e', family='Arial Black'), x=0.5),
        xaxis=dict(
            title=dict(text='Actual Values', font=dict(size=14, color='#333')),
            gridcolor='rgba(0,0,0,0.08)',
            tickfont=dict(size=12, color='#333'),
            showline=True,
            linecolor='#e0e0e0'
        ),
        yaxis=dict(
            title=dict(text='Predicted Values', font=dict(size=14, color='#333')),
            gridcolor='rgba(0,0,0,0.08)',
            tickfont=dict(size=12, color='#333'),
            showline=True,
            linecolor='#e0e0e0'
        ),
        legend=dict(
            x=0.02, y=0.88,
            bgcolor='rgba(255,255,255,0.95)',
            bordercolor='#e0e0e0',
            borderwidth=1,
            font=dict(size=12, color='#333')
        ),
        height=550,
        template='plotly_white',
        paper_bgcolor='#fafbfc',
        plot_bgcolor='#fafbfc',
        margin=dict(l=80, r=80, t=80, b=80)
    )
    
    return fig


def plot_residuals(y_test, y_pred):
    """Plot residuals for regression with Plotly - Enhanced"""
    y_test_arr = np.array(y_test)
    y_pred_arr = np.array(y_pred)
    residuals = y_test_arr - y_pred_arr
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('<b>📉 Residuals vs Predicted</b>', '<b>📊 Residuals Distribution</b>'),
        horizontal_spacing=0.12
    )
    
    # Residuals scatter plot - color by absolute residual
    abs_residuals = np.abs(residuals)
    
    fig.add_trace(go.Scatter(
        x=y_pred_arr, y=residuals,
        mode='markers',
        marker=dict(
            color=abs_residuals,
            colorscale='Turbo',
            size=10,
            opacity=0.75,
            line=dict(color='white', width=1),
            showscale=True,
            colorbar=dict(
                title=dict(text='|Error|', font=dict(size=11)),
                x=0.45,
                thickness=12,
                len=0.8
            )
        ),
        name='Residuals',
        hovertemplate='<b>Residual</b><br>Predicted: %{x:.3f}<br>Residual: %{y:.3f}<extra></extra>'
    ), row=1, col=1)
    
    # Zero line
    fig.add_hline(y=0, line_dash='dash', line_color='#EF4444', line_width=3, row=1, col=1)
    
    # Residuals histogram with better colors
    fig.add_trace(go.Histogram(
        x=residuals,
        nbinsx=30,
        marker=dict(
            color='#8B5CF6',
            line=dict(color='white', width=1.5)
        ),
        name='Distribution',
        opacity=0.85,
        hovertemplate='<b>Distribution</b><br>Range: %{x}<br>Count: %{y}<extra></extra>'
    ), row=1, col=2)
    
    # Zero line for histogram
    fig.add_vline(x=0, line_dash='dash', line_color='#EF4444', line_width=3, row=1, col=2)
    
    fig.update_layout(
        title=dict(text='🔍 Residual Analysis', font=dict(size=20, color='#1a1a2e', family='Arial Black'), x=0.5),
        height=500,
        template='plotly_white',
        paper_bgcolor='#fafbfc',
        plot_bgcolor='#fafbfc',
        showlegend=False,
        margin=dict(l=60, r=60, t=100, b=60)
    )
    
    fig.update_xaxes(title_text='Predicted Values', gridcolor='rgba(0,0,0,0.08)', row=1, col=1)
    fig.update_yaxes(title_text='Residuals', gridcolor='rgba(0,0,0,0.08)', row=1, col=1)
    fig.update_xaxes(title_text='Residual Value', gridcolor='rgba(0,0,0,0.08)', row=1, col=2)
    fig.update_yaxes(title_text='Frequency', gridcolor='rgba(0,0,0,0.08)', row=1, col=2)
    
    return fig


def plot_correlation_matrix(df):
    """Plot correlation matrix with Plotly - Enhanced"""
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()
    
    # Create custom hover text
    hover_text = []
    for i in range(len(corr.columns)):
        row = []
        for j in range(len(corr.columns)):
            row.append(f'<b>{corr.columns[i]}</b> vs <b>{corr.columns[j]}</b><br>Correlation: {corr.iloc[i, j]:.4f}')
        hover_text.append(row)
    
    # Create annotation text
    annotations = []
    for i in range(len(corr.columns)):
        for j in range(len(corr.columns)):
            val = corr.iloc[i, j]
            text_color = 'white' if abs(val) > 0.5 else '#333'
            annotations.append(
                dict(
                    x=corr.columns[j], y=corr.columns[i],
                    text=f'{val:.2f}',
                    font=dict(color=text_color, size=10, family='Arial Bold'),
                    showarrow=False
                )
            )
    
    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=corr.columns,
        y=corr.columns,
        colorscale=[
            [0, '#EF4444'],      # Strong negative - Red
            [0.25, '#FCA5A5'],   # Weak negative - Light Red
            [0.5, '#FAFAFA'],    # No correlation - White
            [0.75, '#93C5FD'],   # Weak positive - Light Blue
            [1, '#3B82F6']       # Strong positive - Blue
        ],
        zmin=-1, zmax=1,
        colorbar=dict(
            title=dict(text='Correlation', font=dict(size=13, color='#333')),
            tickfont=dict(size=11, color='#333'),
            thickness=18,
            len=0.85
        ),
        hovertemplate='%{text}<extra></extra>',
        text=hover_text
    ))
    
    fig.update_layout(
        title=dict(text='📋 Feature Correlation Matrix', font=dict(size=20, color='#1a1a2e', family='Arial Black'), x=0.5),
        annotations=annotations,
        height=max(550, len(corr.columns) * 35),
        template='plotly_white',
        paper_bgcolor='#fafbfc',
        plot_bgcolor='#fafbfc',
        xaxis=dict(tickangle=45, tickfont=dict(size=10, color='#333'), showgrid=False),
        yaxis=dict(tickfont=dict(size=10, color='#333'), showgrid=False),
        margin=dict(l=100, r=80, t=100, b=100)
    )
    
    return fig


def plot_distribution(df, column):
    """Plot distribution of a column with Plotly - Enhanced"""
    data = df[column].dropna()
    
    # Get stats for annotation
    mean_val = data.mean()
    median_val = data.median()
    std_val = data.std()
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(f'<b>📊 Distribution of {column}</b>', f'<b>📦 Box Plot of {column}</b>'),
        horizontal_spacing=0.12
    )
    
    # Histogram with better colors
    fig.add_trace(go.Histogram(
        x=data,
        nbinsx=35,
        marker=dict(
            color='#06B6D4',
            line=dict(color='white', width=1.5)
        ),
        opacity=0.85,
        name='Distribution',
        hovertemplate='<b>Range:</b> %{x}<br><b>Count:</b> %{y}<extra></extra>'
    ), row=1, col=1)
    
    # Add mean line
    fig.add_vline(
        x=mean_val, line_width=3, line_dash='dash', line_color='#F59E0B',
        annotation_text=f'Mean: {mean_val:.2f}', annotation_position='top',
        row=1, col=1
    )
    
    # Box plot with better colors
    fig.add_trace(go.Box(
        y=data,
        marker=dict(color='#8B5CF6', size=6, opacity=0.7),
        fillcolor='rgba(139, 92, 246, 0.4)',
        line=dict(color='#7C3AED', width=2),
        name=column,
        boxpoints='outliers',
        hovertemplate='<b>Value:</b> %{y:.3f}<extra></extra>'
    ), row=1, col=2)
    
    fig.update_xaxes(title_text=column, row=1, col=1, gridcolor='rgba(0,0,0,0.08)')
    fig.update_yaxes(title_text='Frequency', row=1, col=1, gridcolor='rgba(0,0,0,0.08)')
    fig.update_yaxes(title_text=column, row=1, col=2, gridcolor='rgba(0,0,0,0.08)')
    
    fig.update_layout(
        height=480,
        showlegend=False,
        template='plotly_white',
        paper_bgcolor='#fafbfc',
        plot_bgcolor='#fafbfc',
        margin=dict(l=60, r=60, t=100, b=60)
    )
    
    # Add stats annotation
    fig.add_annotation(
        x=0.98, y=0.98,
        xref='paper', yref='paper',
        text=f'<b>Stats</b><br>Mean: {mean_val:.3f}<br>Median: {median_val:.3f}<br>Std: {std_val:.3f}',
        font=dict(size=11, color='#333'),
        align='right',
        showarrow=False,
        bgcolor='rgba(255,255,255,0.95)',
        bordercolor='#e0e0e0',
        borderwidth=1,
        borderpad=8
    )
    
    return fig


def plot_model_comparison(results_df, metric_col, title):
    """Plot model comparison bar chart with Plotly - Enhanced"""
    # Sort for better visualization
    sorted_df = results_df.sort_values(metric_col, ascending=True)
    n_models = len(sorted_df)
    
    # Beautiful color palette - vibrant and distinct
    colors = [
        '#6366F1',  # Indigo
        '#10B981',  # Emerald
        '#F59E0B',  # Amber
        '#EF4444',  # Red
        '#8B5CF6',  # Violet
        '#06B6D4',  # Cyan
        '#EC4899',  # Pink
        '#84CC16',  # Lime
        '#F97316',  # Orange
        '#14B8A6',  # Teal
    ]
    bar_colors = [colors[i % len(colors)] for i in range(n_models)]
    
    fig = go.Figure(go.Bar(
        x=sorted_df[metric_col],
        y=sorted_df['Model'],
        orientation='h',
        marker=dict(
            color=bar_colors,
            line=dict(color='white', width=2),
            opacity=0.9
        ),
        text=[f'<b>{v:.4f}</b>' for v in sorted_df[metric_col]],
        textposition='outside',
        textfont=dict(size=12, color='#333'),
        hovertemplate='<b>%{y}</b><br>' + metric_col + ': %{x:.4f}<extra></extra>'
    ))
    
    fig.update_layout(
        title=dict(text=f'🏆 {title}', font=dict(size=20, color='#1a1a2e', family='Arial Black'), x=0.5),
        xaxis=dict(
            title=dict(text=metric_col, font=dict(size=14, color='#333')),
            gridcolor='rgba(0,0,0,0.08)',
            tickfont=dict(size=12, color='#333')
        ),
        yaxis=dict(
            title='',
            tickfont=dict(size=12, color='#333')
        ),
        height=max(450, n_models * 50),
        template='plotly_white',
        paper_bgcolor='#fafbfc',
        plot_bgcolor='#fafbfc',
        margin=dict(l=20, r=100, t=80, b=50)
    )
    
    return fig


def download_model_results(results_df):
    """Create downloadable CSV of results"""
    csv = results_df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    return f'<a href="data:file/csv;base64,{b64}" download="model_results.csv">📥 Download Results CSV</a>'


# Main App
def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🤖 AutoML System</h1>
        <p>Your Intelligent Machine Learning Platform</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/clouds/200/000000/artificial-intelligence.png", width=150)
        st.markdown("---")
        st.markdown("### 📊 Navigation")
        
        page = st.radio(
            "",
            ["🏠 Home", "📁 Data Upload", "� EDA & Analysis", "🔧 Preprocessing", "🎯 Model Training", 
             "📈 Evaluation", "⚖️ Model Comparison", "🔮 Predictions", "📄 Generate Report", "📚 About"],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        
        # Quick stats if data is loaded
        if st.session_state.data is not None:
            st.markdown("### 📊 Data Overview")
            st.metric("Rows", len(st.session_state.data))
            st.metric("Columns", len(st.session_state.data.columns))
            
            if st.session_state.target_column:
                st.metric("Target", st.session_state.target_column)
            
            if st.session_state.trained_models:
                st.metric("Trained Models", len(st.session_state.trained_models))
    
    # Page content
    if page == "🏠 Home":
        show_home_page()
    elif page == "📁 Data Upload":
        show_data_upload_page()
    elif page == "🔍 EDA & Analysis":
        show_eda_page()
    elif page == "🔧 Preprocessing":
        show_preprocessing_page()
    elif page == "🎯 Model Training":
        show_training_page()
    elif page == "📈 Evaluation":
        show_evaluation_page()
    elif page == "⚖️ Model Comparison":
        show_comparison_page()
    elif page == "🔮 Predictions":
        show_prediction_page()
    elif page == "📄 Generate Report":
        show_report_page()
    elif page == "📚 About":
        show_about_page()


def show_home_page():
    """Display home page"""
    st.markdown("## Welcome to AutoML System! 👋")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">📁</div>
            <div class="metric-label">Upload Data</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">🔧</div>
            <div class="metric-label">Preprocess</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">🎯</div>
            <div class="metric-label">Train Models</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col4:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">🔮</div>
            <div class="metric-label">Predict</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("""
    ### 🚀 Getting Started
    
    Follow these simple steps to build your machine learning model:
    
    1. **📁 Upload Data**: Upload your CSV file or use sample datasets
    2. **🔧 Preprocessing**: Handle missing values, encode categories, and scale features
    3. **🎯 Model Training**: Select and train multiple ML models
    4. **📈 Evaluation**: View detailed performance metrics and visualizations
    5. **⚖️ Model Comparison**: Compare all trained models side by side
    6. **🔮 Predictions**: Make predictions on new data
    """)
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 📊 Supported Problem Types
        
        - **Classification**: Logistic Regression, Decision Tree, Random Forest, 
          Gradient Boosting, SVM, KNN, Naive Bayes, AdaBoost
        - **Regression**: Linear Regression, Ridge, Lasso, ElasticNet, 
          Decision Tree, Random Forest, Gradient Boosting, SVR, KNN, AdaBoost
        """)
    
    with col2:
        st.markdown("""
        ### ✨ Key Features
        
        - 📤 Easy data upload (CSV support)
        - 🧹 Automatic data preprocessing
        - 🤖 Multiple ML algorithms
        - 📊 Comprehensive visualizations
        - 📈 Detailed model evaluation
        - ⬇️ Export results and predictions
        """)


def show_data_upload_page():
    """Display data upload page"""
    st.markdown("## 📁 Data Upload")
    
    tab1, tab2 = st.tabs(["📤 Upload File", "📊 Sample Datasets"])
    
    with tab1:
        uploaded_file = st.file_uploader(
            "Upload your CSV file",
            type=['csv'],
            help="Upload a CSV file with your data"
        )
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.session_state.data = df
                st.success(f"✅ Successfully loaded {uploaded_file.name}")
            except Exception as e:
                st.error(f"Error loading file: {e}")
    
    with tab2:
        st.markdown("### Choose a sample dataset to explore:")
        
        sample_dataset = st.selectbox(
            "Select Dataset",
            ["", "Iris (Classification)", "Wine (Classification)", 
             "Breast Cancer (Classification)", "Diabetes (Regression)",
             "Boston Housing (Regression)"]
        )
        
        if sample_dataset and st.button("Load Dataset", type="primary"):
            df = load_sample_data(sample_dataset)
            if df is not None:
                st.session_state.data = df
                st.success(f"✅ Successfully loaded {sample_dataset} dataset")
    
    # Display data if loaded
    if st.session_state.data is not None:
        st.markdown("---")
        st.markdown("### 📋 Data Preview")
        
        df = st.session_state.data
        
        # Summary metrics
        summary = get_data_summary(df)
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("📊 Rows", summary['rows'])
        col2.metric("📋 Columns", summary['columns'])
        col3.metric("🔢 Numeric", summary['numeric_cols'])
        col4.metric("📝 Categorical", summary['categorical_cols'])
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("❓ Missing Values", summary['missing_values'])
        col2.metric("🔄 Duplicates", summary['duplicate_rows'])
        col3.metric("💾 Memory (MB)", f"{summary['memory_usage']:.2f}")
        col4.metric("", "")
        
        # Data preview
        st.dataframe(df.head(20), use_container_width=True)
        
        # Column info
        with st.expander("📊 Column Information"):
            col_info = pd.DataFrame({
                'Column': df.columns,
                'Type': df.dtypes.values,
                'Non-Null': df.count().values,
                'Null': df.isnull().sum().values,
                'Unique': df.nunique().values
            })
            st.dataframe(col_info, use_container_width=True)
        
        # Descriptive statistics
        with st.expander("📈 Descriptive Statistics"):
            st.dataframe(df.describe(), use_container_width=True)


def show_eda_page():
    """Display comprehensive EDA page with meaningful visualizations"""
    st.markdown("## 🔍 Exploratory Data Analysis")
    
    if st.session_state.data is None:
        st.warning("⚠️ Please upload data first!")
        return
    
    df = st.session_state.data
    
    # Perform EDA if not already done
    if not st.session_state.eda_report:
        st.session_state.eda_report = perform_eda(df)
    
    eda = st.session_state.eda_report
    
    # Overview Cards
    st.markdown("### 📊 Dataset Overview")
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    with col1:
        st.markdown(f"""
        <div class="eda-card">
            <h3>{eda['shape'][0]:,}</h3>
            <p>Total Rows</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="eda-card">
            <h3>{eda['shape'][1]}</h3>
            <p>Total Columns</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="eda-card">
            <h3>{len(eda['numeric_columns'])}</h3>
            <p>Numeric</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="eda-card">
            <h3>{len(eda['categorical_columns'])}</h3>
            <p>Categorical</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col5:
        st.markdown(f"""
        <div class="eda-card">
            <h3>{eda['total_missing']}</h3>
            <p>Missing</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col6:
        st.markdown(f"""
        <div class="eda-card">
            <h3>{eda['duplicates']}</h3>
            <p>Duplicates</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Quick Data Quality Summary
    col1, col2 = st.columns(2)
    with col1:
        fig_dtype = plot_data_types_pie(df)
        st.plotly_chart(fig_dtype, use_container_width=True)
    
    with col2:
        fig_unique = plot_unique_values_bar(df)
        st.plotly_chart(fig_unique, use_container_width=True)
    
    st.markdown("---")
    
    # Tabs for different EDA sections
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "❓ Missing Values", "📊 Outliers", "📈 Numeric Distributions", 
        "📋 Categorical", "🔗 Correlations", "🎯 Target Analysis"
    ])
    
    with tab1:
        st.markdown("### ❓ Missing Values Analysis")
        
        total_missing = eda['total_missing']
        if total_missing > 0:
            st.markdown(f"""
            <div class="warning-box">
                ⚠️ Dataset has <strong>{total_missing:,}</strong> missing values (<strong>{eda['total_missing_pct']:.2f}%</strong> of total data)
            </div>
            """, unsafe_allow_html=True)
            
            # Missing values table
            missing_df = pd.DataFrame({
                'Column': list(eda['missing_values'].keys()),
                'Missing Count': list(eda['missing_values'].values()),
                'Missing %': [f"{v:.2f}%" for v in eda['missing_pct'].values()]
            })
            missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values('Missing Count', ascending=False)
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.markdown("#### 📋 Missing Values Summary")
                st.dataframe(missing_df, use_container_width=True, height=400)
            
            with col2:
                fig = plot_missing_values(df)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
            
            # Recommendations
            st.markdown("#### 💡 Recommendations")
            high_missing = [col for col, val in eda['missing_values'].items() if val > len(df)*0.5]
            low_missing = [col for col, val in eda['missing_values'].items() if 0 < val <= len(df)*0.05]
            
            if high_missing:
                st.warning(f"🔴 Consider dropping columns with >50% missing: {', '.join(high_missing)}")
            if low_missing:
                st.info(f"🟢 For columns with <5% missing, simple imputation should work: {', '.join(low_missing[:5])}")
        else:
            st.success("✅ Excellent! No missing values found in the dataset!")
            st.balloons()
    
    with tab2:
        st.markdown("### 📊 Outlier Detection (IQR Method)")
        
        if eda['outliers']:
            total_outliers = sum(info['count'] for info in eda['outliers'].values())
            
            if total_outliers > 0:
                st.markdown(f"""
                <div class="warning-box">
                    🔍 Detected <strong>{total_outliers:,}</strong> potential outliers across <strong>{len(eda['outliers'])}</strong> numeric columns
                </div>
                """, unsafe_allow_html=True)
                
                # Outlier details table
                outlier_data = []
                for col, info in eda['outliers'].items():
                    if info['count'] > 0:
                        outlier_data.append({
                            'Column': col,
                            'Outliers': info['count'],
                            'Percentage': f"{info['percentage']:.2f}%",
                            'Lower Bound': f"{info['lower_bound']:.2f}",
                            'Upper Bound': f"{info['upper_bound']:.2f}"
                        })
                
                if outlier_data:
                    col1, col2 = st.columns([1, 1])
                    with col1:
                        st.markdown("#### 📋 Outlier Summary")
                        outlier_df = pd.DataFrame(outlier_data).sort_values('Outliers', ascending=False)
                        st.dataframe(outlier_df, use_container_width=True)
                    
                    with col2:
                        # Outlier percentage bar chart
                        fig_outlier_pct = go.Figure(go.Bar(
                            x=[d['Outliers'] for d in outlier_data],
                            y=[d['Column'] for d in outlier_data],
                            orientation='h',
                            marker=dict(color='#EF4444', line=dict(color='white', width=1)),
                            text=[d['Percentage'] for d in outlier_data],
                            textposition='outside'
                        ))
                        fig_outlier_pct.update_layout(
                            title='📊 Outliers by Column',
                            height=max(300, len(outlier_data) * 30),
                            template='plotly_white',
                            xaxis_title='Number of Outliers',
                            yaxis_title=''
                        )
                        st.plotly_chart(fig_outlier_pct, use_container_width=True)
                
                # Box plots for outlier visualization
                st.markdown("#### 📦 Box Plot Visualization")
                numeric_cols = eda['numeric_columns'][:6]
                
                fig = plot_outliers_boxplot(df, numeric_cols)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                
                if len(eda['numeric_columns']) > 6:
                    with st.expander("📊 View More Columns"):
                        remaining_cols = eda['numeric_columns'][6:12]
                        fig2 = plot_outliers_boxplot(df, remaining_cols)
                        if fig2:
                            st.plotly_chart(fig2, use_container_width=True)
            else:
                st.success("✅ No significant outliers detected in numeric columns!")
        else:
            st.info("No numeric columns found for outlier detection.")
    
    with tab3:
        st.markdown("### 📈 Numeric Feature Distributions")
        
        numeric_cols = eda['numeric_columns']
        
        if numeric_cols:
            # Histograms
            st.markdown("#### 📊 Histograms")
            fig = plot_distribution_grid(df, numeric_cols[:6])
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            
            if len(numeric_cols) > 6:
                with st.expander("📊 View More Distributions"):
                    fig2 = plot_distribution_grid(df, numeric_cols[6:12])
                    if fig2:
                        st.plotly_chart(fig2, use_container_width=True)
            
            # Statistics summary
            st.markdown("#### 📋 Descriptive Statistics")
            stats_df = df[numeric_cols].describe().T
            stats_df['skewness'] = df[numeric_cols].skew()
            stats_df['kurtosis'] = df[numeric_cols].kurtosis()
            st.dataframe(stats_df.round(4), use_container_width=True)
            
            # Skewness interpretation
            st.markdown("#### 💡 Skewness Interpretation")
            highly_skewed = stats_df[abs(stats_df['skewness']) > 1].index.tolist()
            if highly_skewed:
                st.warning(f"⚠️ Highly skewed features (|skewness| > 1): {', '.join(highly_skewed[:5])}")
                st.info("💡 Consider log transformation or Box-Cox transformation for these features.")
        else:
            st.info("No numeric columns found for distribution analysis.")
    
    with tab4:
        st.markdown("### 📋 Categorical Feature Analysis")
        
        categorical_cols = eda['categorical_columns']
        
        if categorical_cols:
            # Categorical distributions
            fig_cat = plot_categorical_distribution(df, categorical_cols[:6])
            if fig_cat:
                st.plotly_chart(fig_cat, use_container_width=True)
            
            # Value counts for each categorical
            st.markdown("#### 📊 Category Details")
            
            for col in categorical_cols[:4]:
                with st.expander(f"📋 {col} - {df[col].nunique()} unique values"):
                    value_counts = df[col].value_counts()
                    
                    col1, col2 = st.columns([1, 2])
                    with col1:
                        st.dataframe(pd.DataFrame({
                            'Value': value_counts.index[:10],
                            'Count': value_counts.values[:10],
                            'Percentage': [f"{v/len(df)*100:.1f}%" for v in value_counts.values[:10]]
                        }), use_container_width=True)
                    
                    with col2:
                        fig_pie = go.Figure(go.Pie(
                            labels=value_counts.index[:8],
                            values=value_counts.values[:8],
                            hole=0.4,
                            marker=dict(colors=px.colors.qualitative.Set2)
                        ))
                        fig_pie.update_layout(
                            title=f'{col} Distribution',
                            height=300,
                            template='plotly_white'
                        )
                        st.plotly_chart(fig_pie, use_container_width=True)
            
            # High cardinality warning
            high_card = [col for col in categorical_cols if df[col].nunique() > 50]
            if high_card:
                st.warning(f"⚠️ High cardinality features (>50 unique values): {', '.join(high_card)}")
                st.info("💡 Consider grouping rare categories or using target encoding.")
        else:
            st.info("No categorical columns found in the dataset.")
    
    with tab5:
        st.markdown("### 🔗 Correlation Analysis")
        
        numeric_df = df.select_dtypes(include=[np.number])
        
        if numeric_df.shape[1] >= 2:
            # Correlation heatmap
            fig = plot_correlation_heatmap(df)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            
            # High correlations table
            corr = numeric_df.corr()
            high_corr = []
            for i in range(len(corr.columns)):
                for j in range(i+1, len(corr.columns)):
                    if abs(corr.iloc[i, j]) > 0.7:
                        high_corr.append({
                            'Feature 1': corr.columns[i],
                            'Feature 2': corr.columns[j],
                            'Correlation': round(corr.iloc[i, j], 4),
                            'Strength': 'Very Strong' if abs(corr.iloc[i, j]) > 0.9 else 'Strong'
                        })
            
            if high_corr:
                st.markdown("#### ⚠️ Highly Correlated Feature Pairs (|r| > 0.7)")
                st.dataframe(pd.DataFrame(high_corr).sort_values('Correlation', key=abs, ascending=False), 
                            use_container_width=True)
                st.info("💡 **Tip:** Consider removing one feature from highly correlated pairs to reduce multicollinearity.")
            else:
                st.success("✅ No highly correlated feature pairs found (|r| > 0.7)")
            
            # Pairwise scatter plots
            st.markdown("#### 🔗 Pairwise Scatter Plots")
            if len(numeric_df.columns) >= 2:
                cols_for_scatter = st.multiselect(
                    "Select columns for pairwise plot (max 4):",
                    numeric_df.columns.tolist(),
                    default=numeric_df.columns.tolist()[:4]
                )
                if len(cols_for_scatter) >= 2:
                    fig_scatter = plot_pairwise_scatter(df, cols_for_scatter[:4], st.session_state.target_column)
                    if fig_scatter:
                        st.plotly_chart(fig_scatter, use_container_width=True)
        else:
            st.info("Need at least 2 numeric columns for correlation analysis.")
    
    with tab6:
        st.markdown("### 🎯 Target Variable Analysis")
        
        if st.session_state.target_column:
            target_col = st.session_state.target_column
            
            # Target distribution
            fig = plot_target_distribution(df, target_col)
            st.plotly_chart(fig, use_container_width=True)
            
            # Target statistics
            st.markdown("#### 📊 Target Statistics")
            
            if df[target_col].dtype in ['object', 'category'] or df[target_col].nunique() <= 10:
                # Classification target
                value_counts = df[target_col].value_counts()
                
                col1, col2 = st.columns(2)
                with col1:
                    st.dataframe(pd.DataFrame({
                        'Class': value_counts.index,
                        'Count': value_counts.values,
                        'Percentage': [f"{v/len(df)*100:.2f}%" for v in value_counts.values]
                    }), use_container_width=True)
                
                with col2:
                    # Class balance visualization
                    fig_balance = go.Figure(go.Bar(
                        x=value_counts.index.astype(str),
                        y=value_counts.values,
                        marker=dict(color=px.colors.qualitative.Set2[:len(value_counts)]),
                        text=[f'{v/len(df)*100:.1f}%' for v in value_counts.values],
                        textposition='outside'
                    ))
                    fig_balance.update_layout(
                        title='Class Distribution',
                        height=350,
                        template='plotly_white',
                        xaxis_title='Class',
                        yaxis_title='Count'
                    )
                    st.plotly_chart(fig_balance, use_container_width=True)
                
                # Class imbalance warning
                max_class = value_counts.max()
                min_class = value_counts.min()
                imbalance_ratio = max_class / min_class
                
                if imbalance_ratio > 3:
                    st.warning(f"⚠️ Class imbalance detected! Ratio: {imbalance_ratio:.1f}:1")
                    st.info("💡 Consider using: SMOTE, class weights, or stratified sampling.")
                else:
                    st.success(f"✅ Classes are relatively balanced. Ratio: {imbalance_ratio:.1f}:1")
            else:
                # Regression target
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Mean", f"{df[target_col].mean():.4f}")
                col2.metric("Median", f"{df[target_col].median():.4f}")
                col3.metric("Std Dev", f"{df[target_col].std():.4f}")
                col4.metric("Range", f"{df[target_col].max() - df[target_col].min():.4f}")
                
                # Distribution shape
                skewness = df[target_col].skew()
                if abs(skewness) > 1:
                    st.warning(f"⚠️ Target is highly skewed (skewness: {skewness:.3f}). Consider log transformation.")
            
            # Feature vs Target analysis
            st.markdown("#### 📈 Feature vs Target Relationship")
            numeric_features = [col for col in eda['numeric_columns'] if col != target_col]
            
            if numeric_features:
                selected_feature = st.selectbox("Select a feature to analyze against target:", numeric_features)
                if selected_feature:
                    fig_vs = plot_feature_vs_target(df, selected_feature, target_col)
                    st.plotly_chart(fig_vs, use_container_width=True)
        else:
            st.info("👆 Please select a target column in the Preprocessing page first to see target analysis.")
            st.markdown("**Quick tip:** Go to the Preprocessing page and select your target variable.")
    
    # Store EDA figures for report
    st.session_state.eda_figures = {
        'missing': plot_missing_values(df),
        'outliers': plot_outliers_boxplot(df, eda['numeric_columns'][:6]) if eda['numeric_columns'] else None,
        'distributions': plot_distribution_grid(df, eda['numeric_columns'][:6]) if eda['numeric_columns'] else None,
        'correlation': plot_correlation_heatmap(df),
        'target': plot_target_distribution(df, st.session_state.target_column) if st.session_state.target_column else None
    }


def show_preprocessing_page():
    """Display preprocessing page"""
    st.markdown("## 🔧 Data Preprocessing")
    
    if st.session_state.data is None:
        st.warning("⚠️ Please upload data first!")
        return
    
    df = st.session_state.data
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🎯 Target Selection")
        target_column = st.selectbox(
            "Select Target Column",
            options=[""] + list(df.columns),
            help="Select the column you want to predict"
        )
        
        if target_column:
            st.session_state.target_column = target_column
            
            # Determine problem type
            unique_values = df[target_column].nunique()
            if df[target_column].dtype in ['object', 'category'] or unique_values <= 10:
                st.session_state.problem_type = 'classification'
            else:
                st.session_state.problem_type = 'regression'
            
            st.info(f"🎯 Detected Problem Type: **{st.session_state.problem_type.title()}**")
    
    with col2:
        st.markdown("### 📊 Feature Selection")
        available_features = [col for col in df.columns if col != target_column]
        feature_columns = st.multiselect(
            "Select Feature Columns",
            options=available_features,
            default=available_features,
            help="Select columns to use as features"
        )
        st.session_state.feature_columns = feature_columns
    
    st.markdown("---")
    
    st.markdown("### ⚙️ Preprocessing Options")
    
    # Show current data issues
    if st.session_state.eda_report:
        eda = st.session_state.eda_report
        
        with st.expander("📊 Current Data Issues Detected", expanded=True):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                missing_total = eda['total_missing']
                st.markdown(f"""
                <div class="{'warning-box' if missing_total > 0 else 'success-box'}">
                    <strong>❓ Missing Values:</strong> {missing_total}
                    <br>({eda['total_missing_pct']:.2f}% of data)
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                outlier_total = sum(info['count'] for info in eda['outliers'].values()) if eda['outliers'] else 0
                st.markdown(f"""
                <div class="{'warning-box' if outlier_total > 0 else 'success-box'}">
                    <strong>📊 Outliers:</strong> {outlier_total}
                    <br>(across {len(eda['numeric_columns'])} numeric cols)
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="info-box">
                    <strong>🔄 Duplicates:</strong> {eda['duplicates']}
                    <br>({eda['duplicates_pct']:.2f}% of rows)
                </div>
                """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("#### 🧹 Data Cleaning Options")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Handle Missing Values**")
        handle_missing = st.selectbox(
            "Select method for missing values:",
            ["Drop rows with missing", "Mean imputation (numeric)", "Median imputation (numeric)", 
             "Mode imputation (categorical)", "Fill with 0", "Forward fill", "None - Keep as is"],
            help="Choose how to handle missing values in your dataset"
        )
        
        if handle_missing != "None - Keep as is":
            st.info(f"💡 **Action:** {handle_missing}")
    
    with col2:
        st.markdown("**Handle Outliers**")
        handle_outliers = st.selectbox(
            "Select method for outliers:",
            ["None - Keep outliers", "Remove outliers (IQR)", "Cap outliers (IQR)", 
             "Remove outliers (Z-score)", "Log transform"],
            help="Choose how to handle outliers in numeric columns"
        )
        
        if handle_outliers != "None - Keep outliers":
            st.info(f"💡 **Action:** {handle_outliers}")
    
    st.markdown("---")
    st.markdown("#### 🔢 Feature Engineering")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        scaling_method = st.selectbox(
            "Feature Scaling",
            ["None", "StandardScaler (Z-score)", "MinMaxScaler (0-1)", "RobustScaler (Median)"],
            help="Standardize or normalize your features"
        )
    
    with col2:
        encode_categorical = st.selectbox(
            "Encode Categorical",
            ["Label Encoding", "One-Hot Encoding", "None"],
            help="Convert categorical variables to numeric"
        )
    
    with col3:
        remove_duplicates = st.checkbox("Remove Duplicate Rows", value=True)
    
    st.markdown("---")
    st.markdown("#### 📊 Train/Test Split Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        test_size = st.slider("Test Set Size (%)", 10, 40, 20, help="Percentage of data for testing") / 100
    
    with col2:
        random_state = st.number_input("Random State", value=42, min_value=0, help="Set seed for reproducibility")
    
    # Apply Preprocessing Button
    if st.button("🚀 Apply Preprocessing", type="primary", use_container_width=True):
        if not target_column or not feature_columns:
            st.error("Please select target and feature columns!")
            return
        
        with st.spinner("Processing data..."):
            try:
                processing_log = []
                processed_df = df.copy()
                
                # Store original features and data BEFORE any preprocessing
                st.session_state.original_features = feature_columns.copy()
                st.session_state.original_data = df.copy()
                
                # Remove duplicates
                if remove_duplicates and processed_df.duplicated().sum() > 0:
                    before = len(processed_df)
                    processed_df = processed_df.drop_duplicates()
                    after = len(processed_df)
                    processing_log.append(f"Removed {before - after} duplicate rows")
                
                # Handle missing values
                if handle_missing != "None - Keep as is":
                    before_missing = processed_df.isnull().sum().sum()
                    
                    if handle_missing == "Drop rows with missing":
                        processed_df = processed_df.dropna()
                    elif handle_missing == "Mean imputation (numeric)":
                        numeric_cols = processed_df.select_dtypes(include=[np.number]).columns
                        processed_df[numeric_cols] = processed_df[numeric_cols].fillna(processed_df[numeric_cols].mean())
                    elif handle_missing == "Median imputation (numeric)":
                        numeric_cols = processed_df.select_dtypes(include=[np.number]).columns
                        processed_df[numeric_cols] = processed_df[numeric_cols].fillna(processed_df[numeric_cols].median())
                    elif handle_missing == "Mode imputation (categorical)":
                        cat_cols = processed_df.select_dtypes(include=['object', 'category']).columns
                        for col in cat_cols:
                            processed_df[col] = processed_df[col].fillna(processed_df[col].mode()[0] if len(processed_df[col].mode()) > 0 else 'Unknown')
                    elif handle_missing == "Fill with 0":
                        processed_df = processed_df.fillna(0)
                    elif handle_missing == "Forward fill":
                        processed_df = processed_df.fillna(method='ffill').fillna(method='bfill')
                    
                    after_missing = processed_df.isnull().sum().sum()
                    processing_log.append(f"Handled missing values: {before_missing} → {after_missing}")
                
                # Handle outliers
                if handle_outliers != "None - Keep outliers":
                    numeric_cols = processed_df[feature_columns].select_dtypes(include=[np.number]).columns
                    outliers_removed = 0
                    
                    for col in numeric_cols:
                        if handle_outliers in ["Remove outliers (IQR)", "Cap outliers (IQR)"]:
                            Q1 = processed_df[col].quantile(0.25)
                            Q3 = processed_df[col].quantile(0.75)
                            IQR = Q3 - Q1
                            lower = Q1 - 1.5 * IQR
                            upper = Q3 + 1.5 * IQR
                            
                            if handle_outliers == "Remove outliers (IQR)":
                                before = len(processed_df)
                                processed_df = processed_df[(processed_df[col] >= lower) & (processed_df[col] <= upper)]
                                outliers_removed += before - len(processed_df)
                            else:  # Cap
                                processed_df[col] = processed_df[col].clip(lower, upper)
                        
                        elif handle_outliers == "Remove outliers (Z-score)":
                            mean = processed_df[col].mean()
                            std = processed_df[col].std()
                            before = len(processed_df)
                            processed_df = processed_df[np.abs((processed_df[col] - mean) / std) <= 3]
                            outliers_removed += before - len(processed_df)
                        
                        elif handle_outliers == "Log transform":
                            if (processed_df[col] > 0).all():
                                processed_df[col] = np.log1p(processed_df[col])
                    
                    if outliers_removed > 0:
                        processing_log.append(f"Removed {outliers_removed} outlier rows")
                    elif handle_outliers == "Cap outliers (IQR)":
                        processing_log.append("Capped outliers to IQR bounds")
                    elif handle_outliers == "Log transform":
                        processing_log.append("Applied log transformation to numeric features")
                
                # Encode categorical variables
                label_encoders = {}
                if encode_categorical != "None":
                    cat_cols = processed_df[feature_columns].select_dtypes(include=['object', 'category']).columns
                    
                    if encode_categorical == "Label Encoding":
                        for col in cat_cols:
                            le = LabelEncoder()
                            processed_df[col] = le.fit_transform(processed_df[col].astype(str))
                            label_encoders[col] = le
                        if len(cat_cols) > 0:
                            processing_log.append(f"Label encoded {len(cat_cols)} categorical columns")
                    
                    elif encode_categorical == "One-Hot Encoding":
                        if len(cat_cols) > 0:
                            processed_df = pd.get_dummies(processed_df, columns=cat_cols, drop_first=True)
                            # Update feature columns
                            feature_columns = [col for col in processed_df.columns if col != target_column]
                            st.session_state.feature_columns = feature_columns
                            processing_log.append(f"One-hot encoded {len(cat_cols)} categorical columns")
                
                # Encode target if categorical
                if processed_df[target_column].dtype == 'object':
                    le = LabelEncoder()
                    processed_df[target_column] = le.fit_transform(processed_df[target_column])
                    label_encoders['target'] = le
                    processing_log.append("Label encoded target variable")
                
                # Scale features
                scaler = None
                if scaling_method != "None":
                    numeric_features = processed_df[feature_columns].select_dtypes(include=[np.number]).columns
                    
                    if scaling_method == "StandardScaler (Z-score)":
                        scaler = StandardScaler()
                    elif scaling_method == "MinMaxScaler (0-1)":
                        scaler = MinMaxScaler()
                    elif scaling_method == "RobustScaler (Median)":
                        scaler = RobustScaler()
                    
                    if scaler and len(numeric_features) > 0:
                        processed_df[numeric_features] = scaler.fit_transform(processed_df[numeric_features])
                        processing_log.append(f"Applied {scaling_method.split()[0]} to {len(numeric_features)} features")
                
                # Save preprocessed data
                st.session_state.processed_data = processed_df
                st.session_state.label_encoders = label_encoders
                st.session_state.scaler = scaler
                st.session_state.preprocessing_log = processing_log
                st.session_state.encoding_method = encode_categorical
                
                # Update feature columns in case of one-hot encoding
                feature_columns = st.session_state.feature_columns
                
                # Split data
                X = processed_df[feature_columns]
                y = processed_df[target_column]
                
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=random_state
                )
                
                st.session_state.X_train = X_train
                st.session_state.X_test = X_test
                st.session_state.y_train = y_train
                st.session_state.y_test = y_test
                
                st.success("✅ Preprocessing completed successfully!")
                
                # Show Preprocessing Log
                st.markdown("### 📋 Preprocessing Steps Applied")
                
                if processing_log:
                    for i, step in enumerate(processing_log, 1):
                        st.markdown(f"""
                        <div style="background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); 
                                    color: white; padding: 0.5rem 1rem; border-radius: 8px; margin: 0.3rem 0;">
                            ✅ <strong>Step {i}:</strong> {step}
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.info("No preprocessing steps were applied.")
                
                st.markdown("---")
                
                # Show results
                st.markdown("### 📊 Data Split Summary")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("🏋️ Training Samples", len(X_train))
                
                with col2:
                    st.metric("🧪 Test Samples", len(X_test))
                
                with col3:
                    st.metric("📊 Features Used", len(feature_columns))
                
                with col4:
                    st.metric("🔄 Encoded Columns", len(label_encoders))
                
                # Visual split representation
                train_pct = len(X_train) / (len(X_train) + len(X_test)) * 100
                test_pct = len(X_test) / (len(X_train) + len(X_test)) * 100
                
                # Create a proper stacked bar chart
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=[train_pct],
                    y=['Data Split'],
                    orientation='h',
                    name='Training',
                    text=[f'Training: {train_pct:.1f}% ({len(X_train):,} samples)'],
                    textposition='inside',
                    marker=dict(color='#10B981'),
                    width=0.5
                ))
                fig.add_trace(go.Bar(
                    x=[test_pct],
                    y=['Data Split'],
                    orientation='h',
                    name='Test',
                    text=[f'Test: {test_pct:.1f}% ({len(X_test):,} samples)'],
                    textposition='inside',
                    marker=dict(color='#6366F1'),
                    width=0.5
                ))
                fig.update_layout(
                    title='📊 Train/Test Split Visualization',
                    barmode='stack',
                    height=200,
                    showlegend=True,
                    legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5),
                    xaxis=dict(title='Percentage', range=[0, 100], ticksuffix='%'),
                    yaxis=dict(showticklabels=False),
                    template='plotly_white',
                    margin=dict(l=20, r=20, t=60, b=40)
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Processed data preview
                with st.expander("📋 View Processed Data Sample"):
                    st.dataframe(processed_df.head(10), use_container_width=True)
                
            except Exception as e:
                st.error(f"Error during preprocessing: {e}")
                import traceback
                st.code(traceback.format_exc())


def show_training_page():
    """Display model training page"""
    st.markdown("## 🎯 Model Training")
    
    if st.session_state.X_train is None:
        st.warning("⚠️ Please preprocess your data first!")
        return
    
    problem_type = st.session_state.problem_type
    
    classification_models = [
        'Logistic Regression', 'Decision Tree', 'Random Forest',
        'Gradient Boosting', 'SVM', 'KNN', 'Naive Bayes', 'AdaBoost', 'Rule-Based Classifier'
    ]
    
    regression_models = [
        'Linear Regression', 'Ridge Regression', 'Lasso Regression',
        'ElasticNet', 'Decision Tree', 'Random Forest',
        'Gradient Boosting', 'SVR', 'KNN', 'AdaBoost'
    ]
    
    available_models = classification_models if problem_type == 'classification' else regression_models
    
    st.markdown(f"### 🤖 Select Models ({problem_type.title()})")
    
    selected_models = st.multiselect(
        "Choose models to train",
        available_models,
        default=available_models[:3]
    )
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        train_all = st.checkbox("Train All Models", value=False)
    
    with col2:
        if train_all:
            selected_models = available_models
    
    if st.button("🚀 Train Models", type="primary"):
        if not selected_models:
            st.error("Please select at least one model!")
            return
        
        X_train = st.session_state.X_train
        y_train = st.session_state.y_train
        X_test = st.session_state.X_test
        y_test = st.session_state.y_test
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        results = []
        
        for i, model_name in enumerate(selected_models):
            status_text.text(f"Training {model_name}...")
            
            try:
                # Train model
                model = train_model(model_name, X_train, y_train, problem_type)
                
                # Evaluate model
                metrics = evaluate_model(model, X_test, y_test, problem_type)
                
                # Store model and results
                st.session_state.trained_models[model_name] = model
                st.session_state.model_results[model_name] = metrics
                
                # Create result entry
                if problem_type == 'classification':
                    result = {
                        'Model': model_name,
                        'Accuracy': metrics['Accuracy'],
                        'Precision': metrics['Precision'],
                        'Recall': metrics['Recall'],
                        'F1-Score': metrics['F1-Score']
                    }
                else:
                    result = {
                        'Model': model_name,
                        'MAE': metrics['MAE'],
                        'MSE': metrics['MSE'],
                        'RMSE': metrics['RMSE'],
                        'R² Score': metrics['R² Score']
                    }
                
                results.append(result)
                
            except Exception as e:
                st.warning(f"⚠️ Error training {model_name}: {e}")
            
            progress_bar.progress((i + 1) / len(selected_models))
        
        status_text.text("Training completed!")
        
        if results:
            st.success(f"✅ Successfully trained {len(results)} models!")
            
            # Display results
            st.markdown("### 📊 Training Results")
            results_df = pd.DataFrame(results)
            
            # Sort by best metric
            if problem_type == 'classification':
                results_df = results_df.sort_values('Accuracy', ascending=False)
            else:
                results_df = results_df.sort_values('R² Score', ascending=False)
            
            st.dataframe(results_df, use_container_width=True)
            
            # Best model highlight
            best_model = results_df.iloc[0]['Model']
            st.markdown(f"""
            <div class="success-box">
                🏆 <strong>Best Model:</strong> {best_model}
            </div>
            """, unsafe_allow_html=True)


def show_evaluation_page():
    """Display model evaluation page"""
    st.markdown("## 📈 Model Evaluation")
    
    if not st.session_state.trained_models:
        st.warning("⚠️ Please train models first!")
        return
    
    model_names = list(st.session_state.trained_models.keys())
    selected_model = st.selectbox("Select Model to Evaluate", model_names)
    
    if selected_model:
        model = st.session_state.trained_models[selected_model]
        metrics = st.session_state.model_results[selected_model]
        
        # Determine problem type from actual metrics (more reliable)
        problem_type = 'classification' if 'Accuracy' in metrics else 'regression'
        
        st.markdown(f"### 📊 Performance Metrics - {selected_model}")
        
        if problem_type == 'classification':
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Accuracy", f"{metrics['Accuracy']:.4f}")
            col2.metric("Precision", f"{metrics['Precision']:.4f}")
            col3.metric("Recall", f"{metrics['Recall']:.4f}")
            col4.metric("F1-Score", f"{metrics['F1-Score']:.4f}")
            
            st.markdown("---")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 📊 Confusion Matrix")
                fig = plot_confusion_matrix(metrics['Confusion Matrix'], f'Confusion Matrix - {selected_model}')
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                if 'ROC AUC' in metrics:
                    st.markdown("### 📈 ROC Curve")
                    fig = plot_roc_curve(metrics['FPR'], metrics['TPR'], metrics['ROC AUC'])
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.markdown("### 📋 Classification Report")
                    st.text(metrics['Classification Report'])
            
            # Classification Report
            with st.expander("📋 Detailed Classification Report"):
                st.text(metrics['Classification Report'])
        
        else:  # Regression
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("MAE", f"{metrics['MAE']:.4f}")
            col2.metric("MSE", f"{metrics['MSE']:.4f}")
            col3.metric("RMSE", f"{metrics['RMSE']:.4f}")
            col4.metric("R² Score", f"{metrics['R² Score']:.4f}")
            
            st.markdown("---")
            
            y_test = st.session_state.y_test
            y_pred = metrics['y_pred']
            
            # Actual vs Predicted
            st.markdown("### 📊 Actual vs Predicted Values")
            fig = plot_actual_vs_predicted(y_test, y_pred)
            st.plotly_chart(fig, use_container_width=True)
            
            # Residuals
            st.markdown("### 📉 Residual Analysis")
            fig = plot_residuals(y_test, y_pred)
            st.plotly_chart(fig, use_container_width=True)
        
        # Feature Importance
        st.markdown("---")
        st.markdown("### 🎯 Feature Importance")
        
        feature_names = st.session_state.feature_columns
        fig = plot_feature_importance(model, feature_names, f'Feature Importance - {selected_model}')
        
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Feature importance not available for this model type.")


def show_comparison_page():
    """Display model comparison page"""
    st.markdown("## ⚖️ Model Comparison")
    
    if not st.session_state.trained_models or len(st.session_state.trained_models) < 2:
        st.warning("⚠️ Please train at least 2 models to compare!")
        return
    
    results = st.session_state.model_results
    
    # Determine problem type from actual metrics (more reliable)
    first_metrics = list(results.values())[0]
    problem_type = 'classification' if 'Accuracy' in first_metrics else 'regression'
    
    # Create comparison dataframe
    comparison_data = []
    
    for model_name, metrics in results.items():
        if 'Accuracy' in metrics:  # Classification model
            row = {
                'Model': model_name,
                'Accuracy': metrics['Accuracy'],
                'Precision': metrics['Precision'],
                'Recall': metrics['Recall'],
                'F1-Score': metrics['F1-Score']
            }
        else:  # Regression model
            row = {
                'Model': model_name,
                'MAE': metrics['MAE'],
                'MSE': metrics['MSE'],
                'RMSE': metrics['RMSE'],
                'R² Score': metrics['R² Score']
            }
        comparison_data.append(row)
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Sort by best metric
    if problem_type == 'classification':
        comparison_df = comparison_df.sort_values('Accuracy', ascending=False)
        primary_metric = 'Accuracy'
    else:
        comparison_df = comparison_df.sort_values('R² Score', ascending=False)
        primary_metric = 'R² Score'
    
    st.markdown("### 📊 Model Performance Comparison")
    
    # Display comparison table with highlighting
    st.dataframe(
        comparison_df.style.background_gradient(cmap='viridis', subset=[primary_metric]),
        use_container_width=True
    )
    
    st.markdown("---")
    
    # Download link
    st.markdown(download_model_results(comparison_df), unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Visualization
    st.markdown("### 📈 Visual Comparison")
    
    if problem_type == 'classification':
        metric_options = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    else:
        metric_options = ['R² Score', 'MAE', 'RMSE']
    
    selected_metric = st.selectbox("Select Metric to Visualize", metric_options)
    
    fig = plot_model_comparison(comparison_df, selected_metric, f'Model Comparison - {selected_metric}')
    st.plotly_chart(fig, use_container_width=True)
    
    # Best model summary
    best_model = comparison_df.iloc[0]['Model']
    best_score = comparison_df.iloc[0][primary_metric]
    
    st.markdown(f"""
    <div class="success-box">
        🏆 <strong>Best Model:</strong> {best_model}<br>
        📊 <strong>{primary_metric}:</strong> {best_score:.4f}
    </div>
    """, unsafe_allow_html=True)


def show_prediction_page():
    """Display prediction page"""
    st.markdown("## 🔮 Make Predictions")
    
    if not st.session_state.trained_models:
        st.warning("⚠️ Please train models first!")
        return
    
    model_names = list(st.session_state.trained_models.keys())
    selected_model = st.selectbox("Select Model for Prediction", model_names)
    
    # Determine if this is classification or regression
    is_classification = any(name in selected_model.lower() for name in 
                           ['logistic', 'knn', 'decision tree', 'random forest', 'svm', 'naive bayes', 'rule'])
    
    # Also check from model results
    if st.session_state.model_results:
        first_result = list(st.session_state.model_results.values())[0]
        if 'Accuracy' in first_result:
            is_classification = True
        elif 'MSE' in first_result or 'RMSE' in first_result:
            is_classification = False
    
    st.markdown("---")
    
    tab1, tab2 = st.tabs(["📝 Manual Input", "📁 Batch Prediction"])
    
    with tab1:
        st.markdown("### Enter Feature Values")
        st.info("💡 Enter the values based on the ORIGINAL data (before preprocessing). The system will automatically transform them.")
        
        # Get ORIGINAL feature columns (before any encoding)
        original_df = st.session_state.data
        target_col = st.session_state.target_column
        original_features = [col for col in original_df.columns if col != target_col]
        
        input_data = {}
        cols = st.columns(3)
        
        for i, feature in enumerate(original_features):
            with cols[i % 3]:
                col_data = original_df[feature]
                
                # Check if column is categorical (object type or has few unique values)
                if col_data.dtype == 'object' or (col_data.nunique() <= 10 and col_data.dtype in ['int64', 'int32']):
                    # Categorical - show dropdown
                    unique_values = col_data.dropna().unique().tolist()
                    unique_values = sorted([str(v) for v in unique_values])
                    input_data[feature] = st.selectbox(
                        f"📋 {feature}",
                        options=unique_values,
                        help=f"Categorical feature with {len(unique_values)} unique values"
                    )
                else:
                    # Numeric - show number input
                    min_val = float(col_data.min())
                    max_val = float(col_data.max())
                    mean_val = float(col_data.mean())
                    
                    input_data[feature] = st.number_input(
                        f"🔢 {feature}",
                        min_value=min_val - abs(min_val)*0.5 if min_val != 0 else -100.0,
                        max_value=max_val + abs(max_val)*0.5 if max_val != 0 else 100.0,
                        value=mean_val,
                        help=f"Range: {min_val:.2f} to {max_val:.2f}"
                    )
        
        if st.button("🔮 Predict", type="primary"):
            model = st.session_state.trained_models[selected_model]
            
            try:
                # Create input dataframe with original feature names
                input_df = pd.DataFrame([input_data])
                
                # Apply the same preprocessing as training data
                processed_input = preprocess_input_for_prediction(input_df, original_features)
                
                # Make prediction
                prediction = model.predict(processed_input)[0]
                
                st.markdown("### 🎯 Prediction Result")
                
                # Handle classification vs regression output
                if is_classification:
                    # For classification, convert to integer class
                    pred_class = int(round(prediction))
                    
                    # Decode prediction if label encoder exists for target
                    if 'target' in st.session_state.label_encoders:
                        le = st.session_state.label_encoders['target']
                        try:
                            decoded_prediction = le.inverse_transform([pred_class])[0]
                            st.success(f"**🎯 Predicted Class:** {decoded_prediction}")
                        except:
                            st.success(f"**🎯 Predicted Class:** {pred_class}")
                    elif st.session_state.target_column in st.session_state.label_encoders:
                        le = st.session_state.label_encoders[st.session_state.target_column]
                        try:
                            decoded_prediction = le.inverse_transform([pred_class])[0]
                            st.success(f"**🎯 Predicted Class:** {decoded_prediction}")
                        except:
                            st.success(f"**🎯 Predicted Class:** {pred_class}")
                    else:
                        st.success(f"**🎯 Predicted Class:** {pred_class}")
                    
                    # Get prediction probability if available
                    if hasattr(model, 'predict_proba'):
                        try:
                            prob = model.predict_proba(processed_input)[0]
                            st.markdown("### 📊 Prediction Probabilities")
                            
                            # Get class labels
                            if 'target' in st.session_state.label_encoders:
                                le = st.session_state.label_encoders['target']
                                class_labels = le.classes_
                            else:
                                class_labels = [f"Class {i}" for i in range(len(prob))]
                            
                            prob_df = pd.DataFrame({
                                'Class': class_labels[:len(prob)],
                                'Probability': prob
                            }).sort_values('Probability', ascending=False)
                            
                            # Visualize probabilities
                            fig = px.bar(prob_df, x='Class', y='Probability', 
                                        title='Prediction Confidence',
                                        color='Probability',
                                        color_continuous_scale='Viridis')
                            fig.update_layout(height=300)
                            st.plotly_chart(fig, use_container_width=True)
                        except Exception as e:
                            pass
                else:
                    # For regression, show numeric value
                    st.success(f"**📈 Predicted Value:** {prediction:.4f}")
                    
            except Exception as e:
                st.error(f"❌ Prediction Error: {str(e)}")
                st.info("💡 This might happen if the model was trained with different preprocessing. Try retraining the model.")
    
    with tab2:
        st.markdown("### Upload CSV for Batch Predictions")
        st.info("💡 Upload a CSV with the same columns as your original data (before preprocessing).")
        
        uploaded_file = st.file_uploader(
            "Upload CSV file for predictions",
            type=['csv'],
            key="prediction_upload"
        )
        
        if uploaded_file is not None:
            try:
                pred_df = pd.read_csv(uploaded_file)
                st.dataframe(pred_df.head(), use_container_width=True)
                
                # Get original feature columns
                original_df = st.session_state.data
                target_col = st.session_state.target_column
                original_features = [col for col in original_df.columns if col != target_col]
                
                # Check if all original features are present
                missing_features = set(original_features) - set(pred_df.columns)
                
                if missing_features:
                    st.error(f"❌ Missing features in uploaded file: {missing_features}")
                else:
                    if st.button("🚀 Run Batch Prediction", type="primary"):
                        model = st.session_state.trained_models[selected_model]
                        
                        try:
                            # Prepare data with preprocessing
                            X_pred = pred_df[original_features]
                            processed_X = preprocess_input_for_prediction(X_pred, original_features)
                            
                            # Make predictions
                            predictions = model.predict(processed_X)
                            
                            # Add predictions to dataframe
                            result_df = pred_df.copy()
                            
                            if is_classification:
                                predictions = np.round(predictions).astype(int)
                                result_df['Prediction'] = predictions
                                
                                # Decode if needed
                                if 'target' in st.session_state.label_encoders:
                                    le = st.session_state.label_encoders['target']
                                    try:
                                        result_df['Prediction_Label'] = le.inverse_transform(predictions)
                                    except:
                                        result_df['Prediction_Label'] = predictions
                            else:
                                result_df['Prediction'] = predictions
                            
                            st.markdown("### 📊 Prediction Results")
                            st.dataframe(result_df, use_container_width=True)
                            
                            # Download link
                            csv = result_df.to_csv(index=False)
                            b64 = base64.b64encode(csv.encode()).decode()
                            st.markdown(
                                f'<a href="data:file/csv;base64,{b64}" download="predictions.csv">📥 Download Predictions</a>',
                                unsafe_allow_html=True
                            )
                        except Exception as e:
                            st.error(f"❌ Batch prediction error: {str(e)}")
                        
            except Exception as e:
                st.error(f"Error processing file: {e}")


def preprocess_input_for_prediction(input_df, original_features):
    """
    Transform input data to match the format used during training.
    This handles label encoding, one-hot encoding, and scaling.
    """
    processed = input_df.copy()
    
    # Get the feature columns that the model expects (after preprocessing)
    expected_features = st.session_state.feature_columns
    label_encoders = st.session_state.label_encoders
    scaler = st.session_state.scaler
    encoding_method = st.session_state.get('encoding_method', 'Label Encoding')
    
    # Check if one-hot encoding was used
    one_hot_used = encoding_method == "One-Hot Encoding"
    
    if one_hot_used:
        # Get original data to match one-hot encoding pattern
        original_data = st.session_state.original_data
        
        # Identify categorical columns from original data
        cat_cols = []
        for col in processed.columns:
            if col in original_data.columns:
                if original_data[col].dtype == 'object':
                    cat_cols.append(col)
        
        if cat_cols:
            processed = pd.get_dummies(processed, columns=cat_cols, drop_first=True)
        
        # Make sure all expected columns exist (fill missing with 0)
        for col in expected_features:
            if col not in processed.columns:
                processed[col] = 0
        
        # Ensure we have the right columns in right order
        missing_cols = set(expected_features) - set(processed.columns)
        for col in missing_cols:
            processed[col] = 0
        
        # Keep only expected features in correct order
        processed = processed[expected_features]
    else:
        # Label encoding was used
        for col in processed.columns:
            if col in label_encoders:
                le = label_encoders[col]
                try:
                    processed[col] = le.transform(processed[col].astype(str))
                except ValueError as e:
                    # Handle unseen labels - use the most common class (0)
                    st.warning(f"⚠️ Unknown value in '{col}'. Using default encoding.")
                    processed[col] = 0
        
        # Ensure columns match expected features
        for col in expected_features:
            if col not in processed.columns:
                processed[col] = 0
        
        # Keep only expected features in correct order
        processed = processed[[col for col in expected_features if col in processed.columns]]
    
    # Convert all columns to numeric
    for col in processed.columns:
        processed[col] = pd.to_numeric(processed[col], errors='coerce').fillna(0)
    
    # Apply scaling if it was used
    if scaler is not None:
        try:
            processed = pd.DataFrame(
                scaler.transform(processed),
                columns=processed.columns,
                index=processed.index
            )
        except Exception as e:
            st.warning(f"⚠️ Scaling adjustment applied.")
    
    return processed


def generate_pdf_report():
    """Generate comprehensive PDF report"""
    if not REPORTLAB_AVAILABLE:
        return None
    
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=50, leftMargin=50, topMargin=50, bottomMargin=50)
    
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#667eea'),
        spaceAfter=30,
        alignment=TA_CENTER
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=16,
        textColor=colors.HexColor('#764ba2'),
        spaceBefore=20,
        spaceAfter=10
    )
    
    subheading_style = ParagraphStyle(
        'CustomSubheading',
        parent=styles['Heading3'],
        fontSize=12,
        textColor=colors.HexColor('#333333'),
        spaceBefore=10,
        spaceAfter=5
    )
    
    normal_style = ParagraphStyle(
        'CustomNormal',
        parent=styles['Normal'],
        fontSize=10,
        textColor=colors.HexColor('#444444'),
        spaceAfter=8
    )
    
    elements = []
    
    # Title
    elements.append(Paragraph("AutoML System Report", title_style))
    elements.append(Paragraph(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", normal_style))
    elements.append(Spacer(1, 20))
    
    # Dataset Information
    if st.session_state.data is not None:
        df = st.session_state.data
        
        elements.append(Paragraph("1. Dataset Information", heading_style))
        
        dataset_info = [
            ['Property', 'Value'],
            ['Total Rows', str(len(df))],
            ['Total Columns', str(len(df.columns))],
            ['Numeric Columns', str(len(df.select_dtypes(include=[np.number]).columns))],
            ['Categorical Columns', str(len(df.select_dtypes(include=['object', 'category']).columns))],
            ['Missing Values', str(df.isnull().sum().sum())],
            ['Duplicate Rows', str(df.duplicated().sum())],
            ['Memory Usage (MB)', f"{df.memory_usage(deep=True).sum() / 1024**2:.2f}"]
        ]
        
        if st.session_state.target_column:
            dataset_info.append(['Target Column', st.session_state.target_column])
            dataset_info.append(['Problem Type', st.session_state.problem_type.title()])
        
        table = Table(dataset_info, colWidths=[200, 250])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#667eea')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f5f5f5')),
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#dddddd')),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f9f9f9')])
        ]))
        elements.append(table)
        elements.append(Spacer(1, 20))
        
        # Column Details
        elements.append(Paragraph("Column Details", subheading_style))
        
        col_data = [['Column Name', 'Data Type', 'Non-Null', 'Missing', 'Unique']]
        for col in df.columns[:15]:  # Limit to 15 columns
            col_data.append([
                col[:25],
                str(df[col].dtype),
                str(df[col].count()),
                str(df[col].isnull().sum()),
                str(df[col].nunique())
            ])
        
        col_table = Table(col_data, colWidths=[120, 80, 70, 70, 70])
        col_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#764ba2')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#dddddd')),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f5f5f5')])
        ]))
        elements.append(col_table)
        elements.append(Spacer(1, 15))
        
        # EDA Summary
        if st.session_state.eda_report:
            elements.append(PageBreak())
            elements.append(Paragraph("2. Exploratory Data Analysis", heading_style))
            
            eda = st.session_state.eda_report
            
            # Missing Values
            elements.append(Paragraph("Missing Values Summary", subheading_style))
            if eda['total_missing'] > 0:
                missing_data = [['Column', 'Missing Count', 'Missing %']]
                for col, count in eda['missing_values'].items():
                    if count > 0:
                        missing_data.append([col[:25], str(count), f"{eda['missing_pct'][col]:.2f}%"])
                
                if len(missing_data) > 1:
                    missing_table = Table(missing_data, colWidths=[180, 100, 100])
                    missing_table.setStyle(TableStyle([
                        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#f5576c')),
                        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                        ('FONTSIZE', (0, 0), (-1, -1), 9),
                        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#dddddd'))
                    ]))
                    elements.append(missing_table)
            else:
                elements.append(Paragraph("No missing values found in the dataset.", normal_style))
            
            elements.append(Spacer(1, 15))
            
            # Outliers
            elements.append(Paragraph("Outlier Detection Summary", subheading_style))
            if eda['outliers']:
                outlier_data = [['Column', 'Outliers', '%', 'Lower Bound', 'Upper Bound']]
                for col, info in eda['outliers'].items():
                    if info['count'] > 0:
                        outlier_data.append([
                            col[:20], str(info['count']), f"{info['percentage']:.2f}%",
                            f"{info['lower_bound']:.2f}", f"{info['upper_bound']:.2f}"
                        ])
                
                if len(outlier_data) > 1:
                    outlier_table = Table(outlier_data, colWidths=[100, 60, 50, 80, 80])
                    outlier_table.setStyle(TableStyle([
                        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#f093fb')),
                        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                        ('FONTSIZE', (0, 0), (-1, -1), 8),
                        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#dddddd'))
                    ]))
                    elements.append(outlier_table)
            
            elements.append(Spacer(1, 15))
            
            # Statistics
            elements.append(Paragraph("Descriptive Statistics", subheading_style))
            numeric_cols = df.select_dtypes(include=[np.number]).columns[:8]
            if len(numeric_cols) > 0:
                desc = df[numeric_cols].describe().round(3)
                stat_data = [['Statistic'] + [col[:12] for col in numeric_cols]]
                for idx in desc.index:
                    row = [idx] + [f"{desc.loc[idx, col]:.3f}" for col in numeric_cols]
                    stat_data.append(row)
                
                stat_table = Table(stat_data)
                stat_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#667eea')),
                    ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#667eea')),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('TEXTCOLOR', (0, 0), (0, -1), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, -1), 7),
                    ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#dddddd'))
                ]))
                elements.append(stat_table)
    
    # Model Results
    if st.session_state.model_results:
        elements.append(PageBreak())
        elements.append(Paragraph("3. Model Training Results", heading_style))
        
        results = st.session_state.model_results
        first_result = list(results.values())[0]
        is_classification = 'Accuracy' in first_result
        
        if is_classification:
            model_data = [['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score']]
            for model_name, metrics in results.items():
                model_data.append([
                    model_name,
                    f"{metrics.get('Accuracy', 0):.4f}",
                    f"{metrics.get('Precision', 0):.4f}",
                    f"{metrics.get('Recall', 0):.4f}",
                    f"{metrics.get('F1-Score', 0):.4f}"
                ])
        else:
            model_data = [['Model', 'MAE', 'MSE', 'RMSE', 'R² Score']]
            for model_name, metrics in results.items():
                model_data.append([
                    model_name,
                    f"{metrics.get('MAE', 0):.4f}",
                    f"{metrics.get('MSE', 0):.4f}",
                    f"{metrics.get('RMSE', 0):.4f}",
                    f"{metrics.get('R² Score', 0):.4f}"
                ])
        
        model_table = Table(model_data, colWidths=[120, 80, 80, 80, 80])
        model_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#11998e')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#dddddd')),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#e8f5e9')])
        ]))
        elements.append(model_table)
        elements.append(Spacer(1, 15))
        
        # Best Model
        if is_classification:
            best_model = max(results.items(), key=lambda x: x[1].get('Accuracy', 0))
            elements.append(Paragraph(f"Best Model: {best_model[0]} (Accuracy: {best_model[1]['Accuracy']:.4f})", subheading_style))
        else:
            best_model = max(results.items(), key=lambda x: x[1].get('R² Score', 0))
            elements.append(Paragraph(f"Best Model: {best_model[0]} (R² Score: {best_model[1]['R² Score']:.4f})", subheading_style))
    
    # Train/Test Split Info
    if st.session_state.X_train is not None:
        elements.append(Spacer(1, 15))
        elements.append(Paragraph("4. Train/Test Split Summary", heading_style))
        
        split_data = [
            ['Set', 'Samples', 'Percentage'],
            ['Training Set', str(len(st.session_state.X_train)), f"{len(st.session_state.X_train)/(len(st.session_state.X_train)+len(st.session_state.X_test))*100:.1f}%"],
            ['Test Set', str(len(st.session_state.X_test)), f"{len(st.session_state.X_test)/(len(st.session_state.X_train)+len(st.session_state.X_test))*100:.1f}%"],
            ['Total', str(len(st.session_state.X_train) + len(st.session_state.X_test)), '100%']
        ]
        
        split_table = Table(split_data, colWidths=[150, 100, 100])
        split_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#4facfe')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#dddddd'))
        ]))
        elements.append(split_table)
    
    # Preprocessing Log
    if st.session_state.preprocessing_log:
        elements.append(Spacer(1, 15))
        elements.append(Paragraph("5. Preprocessing Steps Applied", heading_style))
        
        for i, step in enumerate(st.session_state.preprocessing_log, 1):
            elements.append(Paragraph(f"{i}. {step}", normal_style))
    
    # Footer
    elements.append(Spacer(1, 30))
    elements.append(Paragraph("─" * 60, normal_style))
    elements.append(Paragraph("Report generated by AutoML System", 
                             ParagraphStyle('Footer', parent=normal_style, alignment=TA_CENTER, textColor=colors.gray)))
    
    doc.build(elements)
    buffer.seek(0)
    return buffer


def show_report_page():
    """Display report generation page"""
    st.markdown("## 📄 Generate Report")
    
    st.markdown("""
    <div class="info-box">
        📊 Generate a comprehensive PDF report containing all your analysis results, 
        EDA findings, model performance metrics, and visualizations.
    </div>
    """, unsafe_allow_html=True)
    
    if st.session_state.data is None:
        st.warning("⚠️ Please upload data first!")
        return
    
    # Report Preview
    st.markdown("### 📋 Report Contents Preview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📊 Dataset Information")
        df = st.session_state.data
        st.write(f"- **Rows:** {len(df):,}")
        st.write(f"- **Columns:** {len(df.columns)}")
        st.write(f"- **Missing Values:** {df.isnull().sum().sum()}")
        st.write(f"- **Duplicate Rows:** {df.duplicated().sum()}")
        
        if st.session_state.target_column:
            st.write(f"- **Target:** {st.session_state.target_column}")
            st.write(f"- **Problem Type:** {st.session_state.problem_type.title()}")
    
    with col2:
        st.markdown("#### 🤖 Model Results")
        if st.session_state.model_results:
            st.write(f"- **Models Trained:** {len(st.session_state.model_results)}")
            
            results = st.session_state.model_results
            first_result = list(results.values())[0]
            
            if 'Accuracy' in first_result:
                best = max(results.items(), key=lambda x: x[1].get('Accuracy', 0))
                st.write(f"- **Best Model:** {best[0]}")
                st.write(f"- **Best Accuracy:** {best[1]['Accuracy']:.4f}")
            else:
                best = max(results.items(), key=lambda x: x[1].get('R² Score', 0))
                st.write(f"- **Best Model:** {best[0]}")
                st.write(f"- **Best R² Score:** {best[1]['R² Score']:.4f}")
        else:
            st.write("- No models trained yet")
    
    st.markdown("---")
    
    # Report sections to include
    st.markdown("### ⚙️ Report Options")
    
    col1, col2 = st.columns(2)
    
    with col1:
        include_eda = st.checkbox("Include EDA Analysis", value=True)
        include_stats = st.checkbox("Include Statistical Summary", value=True)
        include_missing = st.checkbox("Include Missing Values Analysis", value=True)
    
    with col2:
        include_outliers = st.checkbox("Include Outlier Detection", value=True)
        include_models = st.checkbox("Include Model Results", value=True)
        include_split = st.checkbox("Include Train/Test Split Info", value=True)
    
    st.markdown("---")
    
    # Generate button
    if not REPORTLAB_AVAILABLE:
        st.warning("⚠️ PDF generation requires the 'reportlab' library. Please install it: `pip install reportlab`")
        st.info("📥 You can still download CSV reports below.")
    else:
        if st.button("📥 Generate PDF Report", type="primary", use_container_width=True):
            with st.spinner("Generating report..."):
                try:
                    # Perform EDA if not done
                    if not st.session_state.eda_report:
                        st.session_state.eda_report = perform_eda(st.session_state.data)
                    
                    pdf_buffer = generate_pdf_report()
                    
                    if pdf_buffer:
                        st.success("✅ Report generated successfully!")
                        
                        # Download button
                        st.download_button(
                            label="📥 Download PDF Report",
                            data=pdf_buffer,
                            file_name=f"automl_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                            mime="application/pdf",
                            use_container_width=True
                        )
                    else:
                        st.error("Failed to generate PDF report.")
                    
                except Exception as e:
                    st.error(f"Error generating report: {e}")
                    st.info("💡 Make sure you have the reportlab library installed: pip install reportlab")
    
    # Alternative: Download as CSV/Excel
    st.markdown("---")
    st.markdown("### 📊 Alternative Export Options")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.session_state.data is not None:
            csv = st.session_state.data.to_csv(index=False)
            st.download_button(
                label="📥 Download Data (CSV)",
                data=csv,
                file_name="dataset.csv",
                mime="text/csv"
            )
    
    with col2:
        if st.session_state.model_results:
            results = st.session_state.model_results
            first_result = list(results.values())[0]
            
            if 'Accuracy' in first_result:
                results_df = pd.DataFrame([
                    {'Model': name, 'Accuracy': m['Accuracy'], 'Precision': m['Precision'], 
                     'Recall': m['Recall'], 'F1-Score': m['F1-Score']}
                    for name, m in results.items()
                ])
            else:
                results_df = pd.DataFrame([
                    {'Model': name, 'MAE': m['MAE'], 'MSE': m['MSE'], 
                     'RMSE': m['RMSE'], 'R² Score': m['R² Score']}
                    for name, m in results.items()
                ])
            
            csv = results_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Results (CSV)",
                data=csv,
                file_name="model_results.csv",
                mime="text/csv"
            )
    
    with col3:
        if st.session_state.eda_report:
            eda = st.session_state.eda_report
            eda_summary = {
                'Total Rows': eda['shape'][0],
                'Total Columns': eda['shape'][1],
                'Numeric Columns': len(eda['numeric_columns']),
                'Categorical Columns': len(eda['categorical_columns']),
                'Total Missing Values': eda['total_missing'],
                'Missing Percentage': f"{eda['total_missing_pct']}%",
                'Duplicate Rows': eda['duplicates']
            }
            eda_df = pd.DataFrame([eda_summary])
            csv = eda_df.to_csv(index=False)
            st.download_button(
                label="📥 Download EDA Summary",
                data=csv,
                file_name="eda_summary.csv",
                mime="text/csv"
            )


def show_about_page():
    """Display about page"""
    st.markdown("## 📚 About AutoML System")
    
    st.markdown("""
    ### 🤖 What is AutoML?
    
    AutoML (Automated Machine Learning) is the process of automating the time-consuming, 
    iterative tasks of machine learning model development. This system provides an easy-to-use 
    interface for building, training, and deploying machine learning models without extensive 
    coding knowledge.
    
    ---
    
    ### ✨ Features
    
    | Feature | Description |
    |---------|-------------|
    | 📁 Data Upload | Support for CSV files and sample datasets |
    | 🧹 Preprocessing | Missing value handling, encoding, scaling |
    | 🤖 Model Training | Multiple classification and regression algorithms |
    | 📊 Evaluation | Comprehensive metrics and visualizations |
    | ⚖️ Comparison | Side-by-side model comparison |
    | 🔮 Predictions | Single and batch predictions |
    
    ---
    
    ### 🔧 Supported Algorithms
    
    **Classification:**
    - Logistic Regression
    - Decision Tree
    - Random Forest
    - Gradient Boosting
    - Support Vector Machine (SVM)
    - K-Nearest Neighbors (KNN)
    - Naive Bayes
    - AdaBoost
    
    **Regression:**
    - Linear Regression
    - Ridge Regression
    - Lasso Regression
    - ElasticNet
    - Decision Tree
    - Random Forest
    - Gradient Boosting
    - Support Vector Regression (SVR)
    - K-Nearest Neighbors (KNN)
    - AdaBoost
    
    ---
    
    ### 📈 Evaluation Metrics
    
    **Classification:**
    - Accuracy, Precision, Recall, F1-Score
    - Confusion Matrix
    - ROC Curve & AUC
    - Classification Report
    
    **Regression:**
    - Mean Absolute Error (MAE)
    - Mean Squared Error (MSE)
    - Root Mean Squared Error (RMSE)
    - R² Score
    - Residual Analysis
    
    ---
    
    ### 👨‍💻 Developed For
    
    **Machine Learning Lab Project**
    
    This comprehensive AutoML system was developed as a semester project 
    demonstrating the practical application of machine learning concepts.
    
    ---
    
    ### 📝 Usage Instructions
    
    1. **Start by uploading your data** or selecting a sample dataset
    2. **Configure preprocessing** options (handle missing values, scale features, etc.)
    3. **Select and train** multiple machine learning models
    4. **Evaluate** model performance with detailed metrics and visualizations
    5. **Compare** models to find the best one for your data
    6. **Make predictions** on new data using trained models
    
    ---
    
    ### 🛠️ Technical Stack
    
    - **Frontend:** Streamlit
    - **ML Library:** Scikit-learn
    - **Data Processing:** Pandas, NumPy
    - **Visualization:** Plotly (Interactive Charts)
    
    ---
    
    <div style="text-align: center; padding: 2rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 15px; color: white;">
        <h3>🚀 Ready to build your ML model?</h3>
        <p>Navigate to "Data Upload" from the sidebar to get started!</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
