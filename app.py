import streamlit as st
import pandas as pd
import numpy as np
import io
import base64
from datetime import datetime
import time
import warnings
warnings.filterwarnings('ignore')

try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_LEFT
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, RobustScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer

# Try to import imbalanced-learn for handling class imbalance
try:
    from imblearn.over_sampling import SMOTE, RandomOverSampler
    from imblearn.under_sampling import RandomUnderSampler
    IMBLEARN_AVAILABLE = True
except ImportError:
    IMBLEARN_AVAILABLE = False
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, roc_curve, auc
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

#page config
st.set_page_config(page_title="Classifyy", page_icon="logo.png", layout="wide", initial_sidebar_state="expanded")

#custom css
st.markdown("""
<style>
    .main {padding: 0rem 1rem;}
    .main-header {background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 2rem; border-radius: 15px; margin-bottom: 2rem; text-align: center; box-shadow: 0 10px 30px rgba(102, 126, 234, 0.4);}
    .main-header h1 {color: white; font-size: 2.5rem; font-weight: 700; margin-bottom: 0.5rem;}
    .main-header p {color: rgba(255,255,255,0.9); font-size: 1.2rem;}
    .metric-card {background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); padding: 1.5rem; border-radius: 15px; box-shadow: 0 5px 15px rgba(0,0,0,0.1); text-align: center;}
    .metric-card h3 {font-size: 1.5rem; margin-bottom: 0.5rem; color: #667eea;}
    .metric-card p {font-size: 1rem; color: #333; margin: 0;}
    .stButton > button {background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border: none; padding: 0.75rem 2rem; border-radius: 25px; font-weight: 600;}
    .workflow-step {background: #f8f9fa; padding: 1rem; border-radius: 10px; margin-bottom: 0.5rem; border-left: 4px solid #667eea; color:black;}
</style>
""", unsafe_allow_html=True)

#session state initialization
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
if 'original_features' not in st.session_state:
    st.session_state.original_features = []
if 'encoding_method' not in st.session_state:
    st.session_state.encoding_method = "Label Encoding"
if 'ordinal_encoder' not in st.session_state:
    st.session_state.ordinal_encoder = None
if 'best_params' not in st.session_state:
    st.session_state.best_params = {}
if 'training_times' not in st.session_state:
    st.session_state.training_times = {}

#sample data loading
def load_sample_data(dataset_name):
    if dataset_name == "Iris":
        from sklearn.datasets import load_iris
        data = load_iris()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['target'] = data.target
        return df
    elif dataset_name == "Wine":
        from sklearn.datasets import load_wine
        data = load_wine()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['target'] = data.target
        return df
    elif dataset_name == "Breast Cancer":
        from sklearn.datasets import load_breast_cancer
        data = load_breast_cancer()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['target'] = data.target
        return df
    return None

#eda functions
def detect_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    return len(outliers), lower_bound, upper_bound

def perform_eda(df):
    eda_report = {}
    eda_report['shape'] = df.shape
    eda_report['columns'] = list(df.columns)
    eda_report['dtypes'] = df.dtypes.astype(str).to_dict()
    
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    eda_report['missing_values'] = missing.to_dict()
    eda_report['missing_pct'] = missing_pct.to_dict()
    eda_report['total_missing'] = missing.sum()
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    eda_report['numeric_columns'] = numeric_cols
    eda_report['categorical_columns'] = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    if numeric_cols:
        eda_report['statistics'] = df[numeric_cols].describe().to_dict()
    
    outliers_info = {}
    for col in numeric_cols:
        if df[col].notna().sum() > 0:
            count, lower, upper = detect_outliers_iqr(df, col)
            outliers_info[col] = {'count': count, 'percentage': round(count / len(df) * 100, 2)}
    eda_report['outliers'] = outliers_info
    eda_report['duplicates'] = df.duplicated().sum()
    
    #detect class imbalance
    eda_report['class_imbalance'] = {}
    for col in df.columns:
        if df[col].nunique() <= 20 and df[col].dtype in ['object', 'category', 'int64', 'int32']:
            value_counts = df[col].value_counts()
            if len(value_counts) > 1:
                minority_pct = (value_counts.min() / len(df)) * 100
                if minority_pct < 20:
                    eda_report['class_imbalance'][col] = {
                        'minority_class': value_counts.idxmin(),
                        'percentage': round(minority_pct, 2)
                    }
    
    #detect high-cardinality features
    eda_report['high_cardinality'] = {}
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    for col in cat_cols:
        unique_count = df[col].nunique()
        if unique_count > 20:
            eda_report['high_cardinality'][col] = {
                'unique_values': unique_count,
                'percentage': round((unique_count / len(df)) * 100, 2)
            }
    
    #detect constant/near-constant features
    eda_report['constant_features'] = []
    for col in numeric_cols:
        if df[col].notna().sum() > 0:
            variance = df[col].var()
            if variance < 0.01:  # Near-zero variance threshold
                eda_report['constant_features'].append({
                    'feature': col,
                    'variance': round(variance, 6),
                    'unique_values': df[col].nunique()
                })
    
    return eda_report

#visualization functions
def plot_missing_values(df):
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=True)
    if len(missing) == 0:
        return None
    
    colors = ['#EF4444' if v > len(df)*0.3 else '#F59E0B' if v > len(df)*0.1 else '#10B981' for v in missing.values]
    fig = go.Figure(go.Bar(x=missing.values, y=missing.index, orientation='h', marker=dict(color=colors)))
    fig.update_layout(title='Missing Values Analysis', xaxis_title='Count', height=max(300, len(missing) * 40))
    return fig

def generate_pdf_report():
    """Generate a prettier PDF report using ReportLab with images (FR-41, FR-43)"""
    if not PDF_AVAILABLE:
        return None
    
    try:
        from reportlab.platypus import Image as RLImage
        from reportlab.lib.utils import ImageReader
        import plotly.io as pio
    except ImportError:
        st.warning("Image support in PDF requires plotly.io")
        return None
    
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=18)
    
    # Container for the 'Flowable' objects
    elements = []
    
    # Define styles
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#764ba2'),
        spaceAfter=30,
        alignment=TA_CENTER,
        fontName='Helvetica-Bold'
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=16,
        textColor=colors.HexColor('#764ba2'),
        spaceAfter=12,
        spaceBefore=12,
        fontName='Helvetica-Bold'
    )
    
    # Title
    elements.append(Paragraph("AutoML System - Analysis Report", title_style))
    elements.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
    elements.append(Spacer(1, 0.3*inch))
    
    # Dataset Information
    if st.session_state.data is not None:
        df = st.session_state.data
        elements.append(Paragraph("1. Dataset Information", heading_style))
        
        dataset_data = [
            ['Metric', 'Value'],
            ['Total Rows', str(len(df))],
            ['Total Columns', str(len(df.columns))],
            ['Missing Values', str(df.isnull().sum().sum())],
            ['Duplicate Rows', str(df.duplicated().sum())]
        ]
        
        dataset_table = Table(dataset_data, colWidths=[3*inch, 3*inch])
        dataset_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#764ba2')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        elements.append(dataset_table)
        elements.append(Spacer(1, 0.3*inch))
    
    # Preprocessing Information
    if st.session_state.preprocessing_log:
        elements.append(Paragraph("2. Preprocessing Steps", heading_style))
        
        if st.session_state.target_column:
            elements.append(Paragraph(f"<b>Target Column:</b> {st.session_state.target_column}", styles['Normal']))
        elements.append(Paragraph(f"<b>Features Used:</b> {len(st.session_state.feature_columns)}", styles['Normal']))
        elements.append(Spacer(1, 0.1*inch))
        
        for i, step in enumerate(st.session_state.preprocessing_log, 1):
            elements.append(Paragraph(f"{i}. {step}", styles['Normal']))
        
        elements.append(Spacer(1, 0.3*inch))
    
    # Model Results
    if st.session_state.model_results:
        elements.append(Paragraph("3. Model Performance Results", heading_style))
        
        # Create table for model results
        model_data = [['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score']]
        
        for model_name, metrics in st.session_state.model_results.items():
            model_data.append([
                model_name,
                f"{metrics['Accuracy']:.4f}",
                f"{metrics['Precision']:.4f}",
                f"{metrics['Recall']:.4f}",
                f"{metrics['F1-Score']:.4f}"
            ])
        
        model_table = Table(model_data, colWidths=[2*inch, 1*inch, 1*inch, 1*inch, 1*inch])
        model_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#764ba2')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 11),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.lightgrey),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey])
        ]))
        
        elements.append(model_table)
        elements.append(Spacer(1, 0.2*inch))
        
        # Best Model
        best_model = max(st.session_state.model_results.items(), key=lambda x: x[1]['Accuracy'])
        best_text = f"<b>Best Performing Model:</b> {best_model[0]} with Accuracy: {best_model[1]['Accuracy']:.4f}"
        elements.append(Paragraph(best_text, styles['Normal']))
        
        # Training Times
        if st.session_state.training_times:
            elements.append(Spacer(1, 0.2*inch))
            elements.append(Paragraph("<b>Training Times:</b>", styles['Normal']))
            for model_name, train_time in st.session_state.training_times.items():
                elements.append(Paragraph(f"• {model_name}: {train_time:.2f}s", styles['Normal']))
    
    # Add Correlation Matrix Image (FR-41)
    if st.session_state.data is not None:
        elements.append(PageBreak())
        elements.append(Paragraph("4. Correlation Matrix", heading_style))
        
        try:
            fig = plot_correlation_heatmap(st.session_state.data)
            if fig:
                img_bytes = pio.to_image(fig, format='png', width=700, height=500)
                img_buffer = io.BytesIO(img_bytes)
                img = RLImage(img_buffer, width=6*inch, height=4*inch)
                elements.append(img)
                elements.append(Spacer(1, 0.2*inch))
        except Exception as e:
            elements.append(Paragraph(f"Could not generate correlation matrix: {str(e)}", styles['Normal']))
    
    # Add Model Comparison Chart (FR-43)
    if st.session_state.model_results and len(st.session_state.model_results) >= 2:
        elements.append(PageBreak())
        elements.append(Paragraph("5. Model Comparison", heading_style))
        
        try:
            comparison_data = [{
                'Model': name, 
                'Accuracy': m['Accuracy'], 
                'Precision': m['Precision'], 
                'Recall': m['Recall'], 
                'F1-Score': m['F1-Score']
            } for name, m in st.session_state.model_results.items()]
            comparison_df = pd.DataFrame(comparison_data).sort_values('Accuracy', ascending=False)
            
            fig = plot_model_comparison(comparison_df, 'Accuracy')
            if fig:
                img_bytes = pio.to_image(fig, format='png', width=700, height=500)
                img_buffer = io.BytesIO(img_bytes)
                img = RLImage(img_buffer, width=6*inch, height=4*inch)
                elements.append(img)
        except Exception as e:
            elements.append(Paragraph(f"Could not generate model comparison chart: {str(e)}", styles['Normal']))
    
    # Build PDF
    doc.build(elements)
    
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes

def plot_correlation_heatmap(df):
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.shape[1] < 2:
        return None
    corr = numeric_df.corr()
    fig = go.Figure(data=go.Heatmap(z=corr.values, x=corr.columns, y=corr.index, colorscale='RdBu', zmin=-1, zmax=1))
    fig.update_layout(title='Correlation Matrix', height=500)
    return fig

def plot_target_distribution(df, target_col):
    fig = go.Figure()
    value_counts = df[target_col].value_counts()
    colors = px.colors.qualitative.Set2[:len(value_counts)]
    fig.add_trace(go.Bar(x=value_counts.index.astype(str), y=value_counts.values, marker=dict(color=colors)))
    fig.update_layout(title=f'Target Distribution: {target_col}', xaxis_title='Class', yaxis_title='Count', height=400)
    return fig

def plot_confusion_matrix(cm, title='Confusion Matrix'):
    #get actual class names if available
    if 'target' in st.session_state.label_encoders:
        labels = st.session_state.label_encoders['target'].classes_
    else:
        #try to get unique values from original data
        if st.session_state.data is not None and st.session_state.target_column:
            labels = sorted(st.session_state.data[st.session_state.target_column].unique())
        else:
            labels = [f'Class {i}' for i in range(cm.shape[0])]
    
    labels = [str(label) for label in labels][:cm.shape[0]]
    
    fig = go.Figure(data=go.Heatmap(z=cm, x=labels, y=labels, colorscale='Blues', showscale=True))
    
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            fig.add_annotation(x=j, y=i, text=str(cm[i, j]), font=dict(color='white' if cm[i, j] > cm.max()/2 else '#333', size=14), showarrow=False)
    
    fig.update_layout(title=title, xaxis_title='Predicted', yaxis_title='True', height=450)
    return fig

def plot_roc_curve(fpr, tpr, roc_auc):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, fill='tozeroy', name=f'ROC (AUC={roc_auc:.3f})'))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], line=dict(dash='dash'), name='Random'))
    fig.update_layout(title='ROC Curve', xaxis_title='False Positive Rate', yaxis_title='True Positive Rate', height=450)
    return fig

def plot_feature_importance(model, feature_names, title='Feature Importance'):
    try:
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importance = np.abs(model.coef_).mean(axis=0) if model.coef_.ndim > 1 else np.abs(model.coef_)
        else:
            return None
        
        indices = np.argsort(importance)[-15:]
        fig = go.Figure(go.Bar(x=importance[indices], y=[feature_names[i] for i in indices], orientation='h'))
        fig.update_layout(title=title, xaxis_title='Importance', height=max(400, len(indices) * 30))
        return fig
    except:
        return None

def plot_model_comparison(results_df, metric_col):
    sorted_df = results_df.sort_values(metric_col, ascending=True)
    colors = ['#6366F1', '#10B981', '#F59E0B', '#EF4444', '#8B5CF6', '#06B6D4', '#EC4899', '#84CC16', '#F97316']
    fig = go.Figure(go.Bar(x=sorted_df[metric_col], y=sorted_df['Model'], orientation='h', marker=dict(color=colors[:len(sorted_df)])))
    fig.update_layout(title=f'Model Comparison - {metric_col}', xaxis_title=metric_col, height=max(400, len(sorted_df) * 50))
    return fig

#model training and evaluation
def train_model(model_name, X_train, y_train, optimize_hyperparams=False):
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42),
        'SVM': SVC(probability=True, random_state=42),
        'KNN': KNeighborsClassifier(),
        'Naive Bayes': GaussianNB(),
        'AdaBoost': AdaBoostClassifier(random_state=42),
        'Rule-Based Classifier': DecisionTreeClassifier(max_depth=5, random_state=42)
    }
    
    #hyperparameter grids for optimization
    param_grids = {
        'Logistic Regression': {'C': [0.01, 0.1, 1, 10], 'penalty': ['l2'], 'solver': ['lbfgs', 'saga']},
        'Decision Tree': {'max_depth': [3, 5, 7, 10, None], 'min_samples_split': [2, 5, 10]},
        'Random Forest': {'n_estimators': [50, 100, 200], 'max_depth': [5, 10, None], 'min_samples_split': [2, 5]},
        'Gradient Boosting': {'n_estimators': [50, 100], 'learning_rate': [0.01, 0.1, 0.2], 'max_depth': [3, 5]},
        'SVM': {'C': [0.1, 1, 10], 'kernel': ['rbf', 'linear'], 'gamma': ['scale', 'auto']},
        'KNN': {'n_neighbors': [3, 5, 7, 9], 'weights': ['uniform', 'distance']},
        'AdaBoost': {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 1]},
        'Rule-Based Classifier': {'max_depth': [3, 5, 7], 'min_samples_split': [2, 5, 10]}
    }
    
    model = models[model_name]
    best_params = None
    
    if optimize_hyperparams and model_name in param_grids:
        #use randomizedsearchcv for faster optimization
        grid_search = RandomizedSearchCV(
            model, 
            param_grids[model_name], 
            cv=3, 
            n_iter=10,
            scoring='accuracy', 
            n_jobs=-1, 
            random_state=42
        )
        grid_search.fit(X_train, y_train)
        model = grid_search.best_estimator_
        best_params = grid_search.best_params_
    else:
        model.fit(X_train, y_train)
    
    return model, best_params

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    metrics = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
        'Recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
        'F1-Score': f1_score(y_test, y_pred, average='weighted', zero_division=0),
        'Confusion Matrix': confusion_matrix(y_test, y_pred),
        'Classification Report': classification_report(y_test, y_pred, zero_division=0)
    }
    
    if len(np.unique(y_test)) == 2 and hasattr(model, 'predict_proba'):
        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        metrics['ROC AUC'] = auc(fpr, tpr)
        metrics['FPR'] = fpr
        metrics['TPR'] = tpr
    
    return metrics

def preprocess_input_for_prediction(input_df, original_features):
    processed = input_df.copy()
    expected_features = st.session_state.feature_columns
    label_encoders = st.session_state.label_encoders
    ordinal_encoder = st.session_state.ordinal_encoder
    scaler = st.session_state.scaler
    encoding_method = st.session_state.encoding_method
    
    if encoding_method == "One-Hot Encoding":
        original_data = st.session_state.data
        cat_cols = [col for col in processed.columns if col in original_data.columns and original_data[col].dtype == 'object']
        if cat_cols:
            processed = pd.get_dummies(processed, columns=cat_cols, drop_first=True)
        for col in expected_features:
            if col not in processed.columns:
                processed[col] = 0
        processed = processed[expected_features]
    elif encoding_method == "Ordinal Encoding":
        original_data = st.session_state.data
        cat_cols = [col for col in processed.columns if col in original_data.columns and original_data[col].dtype == 'object']
        if cat_cols and ordinal_encoder is not None:
            try:
                processed[cat_cols] = ordinal_encoder.transform(processed[cat_cols].astype(str))
            except:
                for col in cat_cols:
                    processed[col] = 0
        for col in expected_features:
            if col not in processed.columns:
                processed[col] = 0
        processed = processed[[col for col in expected_features if col in processed.columns]]
    else:
        for col in processed.columns:
            if col in label_encoders:
                le = label_encoders[col]
                try:
                    processed[col] = le.transform(processed[col].astype(str))
                except:
                    processed[col] = 0
        for col in expected_features:
            if col not in processed.columns:
                processed[col] = 0
        processed = processed[[col for col in expected_features if col in processed.columns]]
    
    for col in processed.columns:
        processed[col] = pd.to_numeric(processed[col], errors='coerce').fillna(0)
    
    if scaler is not None:
        processed = pd.DataFrame(scaler.transform(processed), columns=processed.columns, index=processed.index)
    
    return processed

#main pages
def show_home_page():
    st.markdown("""<div class="main-header">
        <h1>Classifyy</h1>
        <p>Automated Machine Learning Classification Platform</p>
    </div>""", unsafe_allow_html=True)
    

    st.markdown("### Quick Start Guide")
    
    steps = [
        ("1. Upload Data", "Upload your CSV file or select a sample dataset to get started"),
        ("2. Exploratory Data Analysis", "Explore your data with visualizations and statistics"),
        ("3. Preprocessing", "Handle missing values, encode categorical features, and scale data"),
        ("4. Train Models", "Train up to 9 different classification algorithms"),
        ("5. Evaluate", "View detailed metrics, confusion matrix, and ROC curves"),
        ("6. Compare", "Compare performance across all trained models"),
        ("7. Predict", "Make predictions on new data and download results")
    ]
    
    for title, desc in steps:
        st.markdown(f'<div class="workflow-step"><strong>{title}</strong><br>{desc}</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### Available Models")
    models = ["Logistic Regression", "Decision Tree", "Random Forest", "Gradient Boosting", 
            "SVM", "KNN", "Naive Bayes", "AdaBoost", "Rule-Based Classifier"]
    cols = st.columns(3)
    for i, model in enumerate(models):
        cols[i % 3].markdown(f"- {model}")

    st.markdown("---")
    st.markdown("### Members")
    members = ["- Adeena Reeham — 480941", "- Fatima Ali — 470708",  "- Zain Kashif — 458822"]
    cols = st.columns(1)
    for i, member in enumerate(members):
        cols[0].markdown(f"{member}")

def show_data_upload_page():
    st.markdown("## Data Upload")
    
    tab1, tab2 = st.tabs(["Upload File", "Sample Datasets"])
    
    with tab1:
        uploaded_file = st.file_uploader(
            "Upload CSV file", 
            type=['csv'],
            help="Upload your dataset in CSV format. First row should contain column names."
        )
        if uploaded_file:
            try:
                df = pd.read_csv(uploaded_file)
                st.session_state.data = df
                st.success(f"Loaded {uploaded_file.name}")
            except Exception as e:
                st.error(f"Error: {e}")
    
    with tab2:
        sample_dataset = st.selectbox(
            "Select Dataset", 
            ["", "Iris", "Wine", "Breast Cancer"],
            help="Choose from pre-loaded sample datasets to explore the AutoML system"
        )
        if sample_dataset and st.button("Load Dataset", type="primary", help="Load the selected sample dataset"):
            df = load_sample_data(sample_dataset)
            if df is not None:
                st.session_state.data = df
                st.success(f"Loaded {sample_dataset} dataset")
    
    if st.session_state.data is not None:
        df = st.session_state.data
        st.markdown("---")
        st.markdown("### Data Preview")
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Rows", len(df))
        col2.metric("Columns", len(df.columns))
        col3.metric("Missing", df.isnull().sum().sum())
        col4.metric("Duplicates", df.duplicated().sum())
        
        st.dataframe(df.head(20), use_container_width=True)
        
        with st.expander("Column Info"):
            col_info = pd.DataFrame({'Column': df.columns, 'Type': df.dtypes.values, 'Non-Null': df.count().values, 'Unique': df.nunique().values})
            st.dataframe(col_info, use_container_width=True)

def show_eda_page():
    st.markdown("## Exploratory Data Analysis")
    
    if st.session_state.data is None:
        st.warning("Please upload data first!")
        return
    
    df = st.session_state.data
    
    if not st.session_state.eda_report:
        st.session_state.eda_report = perform_eda(df)
    
    eda = st.session_state.eda_report
    
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Rows", eda['shape'][0])
    col2.metric("Columns", eda['shape'][1])
    col3.metric("Numeric", len(eda['numeric_columns']))
    col4.metric("Missing", eda['total_missing'])
    col5.metric("Duplicates", eda['duplicates'])
    
    # Display Data Quality Issues (FR-9, FR-10, FR-11)
    issues_found = []
    if eda.get('class_imbalance'):
        issues_found.append(f"⚠️ Class Imbalance: {len(eda['class_imbalance'])} column(s)")
    if eda.get('high_cardinality'):
        issues_found.append(f"⚠️ High Cardinality: {len(eda['high_cardinality'])} column(s)")
    if eda.get('constant_features'):
        issues_found.append(f"⚠️ Constant Features: {len(eda['constant_features'])} column(s)")
    
    if issues_found:
        st.warning("**Data Quality Issues Detected:**\n\n" + "\n\n".join(issues_found))
    else:
        st.success("✓ No major data quality issues detected")
    
    st.markdown("---")
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Missing Values", "Correlations", "Distributions", "Statistics", "Data Quality Issues"])
    
    with tab1:
        if eda['total_missing'] > 0:
            st.warning(f"{eda['total_missing']} missing values found")
            fig = plot_missing_values(df)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            
            # Show missing values percentage
            st.markdown("### Missing Values Details")
            missing_df = pd.DataFrame({
                'Column': df.columns,
                'Missing Count': df.isnull().sum().values,
                'Percentage': (df.isnull().sum().values / len(df) * 100).round(2)
            })
            missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values('Percentage', ascending=False)
            st.dataframe(missing_df, use_container_width=True)
        else:
            st.success("No missing values!")
    
    with tab2:
        fig = plot_correlation_heatmap(df)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
            
            # Show highly correlated pairs
            numeric_df = df.select_dtypes(include=[np.number])
            if numeric_df.shape[1] >= 2:
                st.markdown("### Highly Correlated Features")
                corr = numeric_df.corr()
                high_corr = []
                for i in range(len(corr.columns)):
                    for j in range(i+1, len(corr.columns)):
                        if abs(corr.iloc[i, j]) > 0.7:
                            high_corr.append({
                                'Feature 1': corr.columns[i],
                                'Feature 2': corr.columns[j],
                                'Correlation': round(corr.iloc[i, j], 3)
                            })
                if high_corr:
                    high_corr_df = pd.DataFrame(high_corr).sort_values('Correlation', key=lambda x: abs(x), ascending=False)
                    st.dataframe(high_corr_df, use_container_width=True)
                else:
                    st.info("No highly correlated features found (threshold: 0.7)")
        else:
            st.info("Need at least 2 numeric columns for correlation")
    
    with tab3:
        st.markdown("### Feature Distributions")
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if numeric_cols:
            # Select feature to visualize
            selected_feature = st.selectbox("Select feature to visualize", numeric_cols)
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Histogram
                fig = px.histogram(df, x=selected_feature, nbins=30, title=f'Distribution of {selected_feature}')
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Box plot
                fig = px.box(df, y=selected_feature, title=f'Box Plot of {selected_feature}')
                st.plotly_chart(fig, use_container_width=True)
            
            # Show distribution for all numeric features
            if st.checkbox("Show all numeric distributions"):
                cols_per_row = 3
                for i in range(0, len(numeric_cols), cols_per_row):
                    cols = st.columns(cols_per_row)
                    for j, col in enumerate(cols):
                        if i + j < len(numeric_cols):
                            feature = numeric_cols[i + j]
                            with col:
                                fig = px.histogram(df, x=feature, title=feature, height=300)
                                fig.update_layout(showlegend=False, title_font_size=10)
                                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No numeric columns found for distribution analysis")
    
    with tab4:
        st.markdown("### Statistical Summary")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            st.markdown("#### Numeric Features")
            stats_df = df[numeric_cols].describe().T
            stats_df['range'] = stats_df['max'] - stats_df['min']
            stats_df['variance'] = df[numeric_cols].var()
            st.dataframe(stats_df.style.format("{:.2f}"), use_container_width=True)
        
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        if categorical_cols:
            st.markdown("#### Categorical Features")
            cat_stats = []
            for col in categorical_cols:
                cat_stats.append({
                    'Feature': col,
                    'Unique Values': df[col].nunique(),
                    'Most Common': df[col].mode()[0] if len(df[col].mode()) > 0 else 'N/A',
                    'Most Common Count': df[col].value_counts().iloc[0] if len(df[col]) > 0 else 0
                })
            st.dataframe(pd.DataFrame(cat_stats), use_container_width=True)
    
    with tab5:
        st.markdown("### Data Quality Issues")
        
        # Class Imbalance (FR-9)
        if eda.get('class_imbalance'):
            st.markdown("#### ⚠️ Class Imbalance Detected")
            st.info("Class imbalance occurs when one class has significantly fewer samples (< 20% of total). This can lead to biased models.")
            imbalance_data = []
            for col, info in eda['class_imbalance'].items():
                imbalance_data.append({
                    'Column': col,
                    'Minority Class': info['minority_class'],
                    'Percentage': info['percentage']
                })
            imbalance_df = pd.DataFrame(imbalance_data).sort_values('Percentage', ascending=True)
            imbalance_df['Percentage'] = imbalance_df['Percentage'].apply(lambda x: f"{x}%")
            st.dataframe(imbalance_df, use_container_width=True)
        else:
            st.success("✓ No class imbalance issues detected")
        
        st.markdown("---")
        
        # High Cardinality (FR-10)
        if eda.get('high_cardinality'):
            st.markdown("#### ⚠️ High Cardinality Features Detected")
            st.info("High cardinality (> 20 unique values) in categorical features can lead to overfitting and increased memory usage.")
            cardinality_data = []
            for col, info in eda['high_cardinality'].items():
                cardinality_data.append({
                    'Column': col,
                    'Unique Values': info['unique_values'],
                    'Percentage of Total': info['percentage']
                })
            cardinality_df = pd.DataFrame(cardinality_data).sort_values('Percentage of Total', ascending=False)
            cardinality_df['Percentage of Total'] = cardinality_df['Percentage of Total'].apply(lambda x: f"{x}%")
            st.dataframe(cardinality_df, use_container_width=True)
        else:
            st.success("✓ No high cardinality issues detected")
        
        st.markdown("---")
        
        # Constant Features (FR-11)
        if eda.get('constant_features'):
            st.markdown("#### ⚠️ Constant/Near-Constant Features Detected")
            st.info("Features with near-zero variance provide little to no information and should be removed.")
            constant_df = pd.DataFrame(eda['constant_features'])
            constant_df.columns = ['Feature', 'Variance', 'Unique Values']
            st.dataframe(constant_df, use_container_width=True)
        else:
            st.success("✓ No constant features detected")

def show_preprocessing_page():
    st.markdown("## Data Preprocessing")
    
    if st.session_state.data is None:
        st.warning("Please upload data first!")
        return
    
    df = st.session_state.data
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Use session state to preserve target column selection
        current_target = st.session_state.target_column if st.session_state.target_column in df.columns else ""
        target_options = [""] + list(df.columns)
        target_index = target_options.index(current_target) if current_target in target_options else 0
        target_column = st.selectbox(
            "Select Target Column", 
            target_options, 
            index=target_index, 
            key="target_select",
            help="Choose the column you want to predict (dependent variable)"
        )
        if target_column:
            st.session_state.target_column = target_column
            st.info(f"Classification - {df[target_column].nunique()} classes")
            # Show target distribution
            fig = plot_target_distribution(df, target_column)
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        available_features = [col for col in df.columns if col != target_column]
        # Preserve previously selected features if they're still valid
        default_features = available_features
        if st.session_state.feature_columns:
            valid_previous = [f for f in st.session_state.feature_columns if f in available_features]
            if valid_previous:
                default_features = valid_previous
        
        feature_columns = st.multiselect(
            "Select Features", 
            available_features, 
            default=default_features,
            help="Choose which columns to use as input features (independent variables) for training"
        )
        st.session_state.feature_columns = feature_columns
    
    st.markdown("---")
    
    #data quality issues handling section
    st.markdown("### 🔧 Data Quality Issues Handling")
    
    if st.session_state.eda_report:
        eda = st.session_state.eda_report
    else:
        eda = {}
    
    issues_col1, issues_col2, issues_col3 = st.columns(3)
    
    with issues_col1:
        #handle constant features
        st.markdown("**Constant Features**")
        has_constant = bool(eda.get('constant_features'))
        if has_constant:
            const_features = [f['feature'] for f in eda['constant_features']]
            st.caption(f"{len(const_features)} detected")
        else:
            st.caption("None detected")
        
        handle_constant = st.selectbox(
            "Action",
            ["Keep", "Remove"],
            key="handle_constant",
            disabled=not has_constant,
            help="Remove constant features as they provide no information for prediction"
        )
        
        if has_constant and handle_constant == "Remove":
            with st.expander("Features to remove"):
                st.write(const_features)
    
    with issues_col2:
        #handle class imbalance
        st.markdown("**Class Imbalance**")
        has_imbalance = bool(eda.get('class_imbalance'))
        if has_imbalance:
            imbalanced_cols = list(eda['class_imbalance'].keys())
            st.caption(f"{len(imbalanced_cols)} column(s)")
        else:
            st.caption("None detected")
        
        if IMBLEARN_AVAILABLE:
            handle_imbalance = st.selectbox(
                "Action",
                ["None", "SMOTE", "Random Oversample", "Random Undersample"],
                key="handle_imbalance",
                disabled=not has_imbalance,
                help="Balance classes to improve model performance. SMOTE creates synthetic samples."
            )
        else:
            handle_imbalance = st.selectbox(
                "Action",
                ["None", "Class Weights"],
                key="handle_imbalance",
                disabled=not has_imbalance,
                help="Install imbalanced-learn for more options: pip install imbalanced-learn"
            )
        
        if has_imbalance and handle_imbalance != "None":
            with st.expander("Affected columns"):
                st.write(imbalanced_cols)
    
    with issues_col3:
        #handle high cardinality
        st.markdown("**High Cardinality**")
        has_cardinality = bool(eda.get('high_cardinality'))
        if has_cardinality:
            high_card_cols = list(eda['high_cardinality'].keys())
            st.caption(f"{len(high_card_cols)} column(s)")
        else:
            st.caption("None detected")
        
        handle_cardinality = st.selectbox(
            "Action",
            ["Keep All", "Top 10 + Other", "Top 20 + Other", "Remove Columns"],
            key="handle_cardinality",
            disabled=not has_cardinality,
            help="Reduce cardinality by keeping only top categories or removing these columns"
        )
        
        if has_cardinality and handle_cardinality != "Keep All":
            with st.expander("Affected columns"):
                st.write(high_card_cols)
    
    st.markdown("---")
    
    #standard preprocessing options
    st.markdown("### ⚙️ Standard Preprocessing Options")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        handle_missing = st.selectbox(
            "Handle Missing", 
            ["Drop rows", "Mean imputation", "Median imputation", "Mode imputation", "None"],
            help="Choose how to handle missing values. Drop: Remove rows. Mean/Median: Fill with average. Mode: Fill with most common value."
        )
    
    with col2:
        encode_categorical = st.selectbox(
            "Encode Categorical", 
            ["Label Encoding", "One-Hot Encoding", "Ordinal Encoding", "None"],
            help="Label: Assigns unique integers to categories. One-Hot: Creates binary columns. Ordinal: Preserves order for ordinal data."
        )
    
    with col3:
        scaling_method = st.selectbox(
            "Feature Scaling", 
            ["None", "StandardScaler", "MinMaxScaler", "RobustScaler"],
            help="StandardScaler: Zero mean, unit variance. MinMaxScaler: Scale to [0,1]. RobustScaler: Robust to outliers."
        )
    
    test_size = st.slider(
        "Test Size (%)", 
        10, 40, 20,
        help="Percentage of data to use for testing. Higher = more reliable test metrics but less training data."
    ) / 100
    
    #train-test split preview
    if target_column and feature_columns and len(df) > 0:
        st.markdown("### Train-Test Split Preview")
        train_samples = int(len(df) * (1 - test_size))
        test_samples = len(df) - train_samples
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Samples", len(df))
        col2.metric("Training Samples", train_samples)
        col3.metric("Test Samples", test_samples)
    
    col1, col2 = st.columns(2)
    
    with col1:
        apply_button = st.button("Apply Preprocessing", type="primary", use_container_width=True, help="Apply selected preprocessing steps to your data")
    
    with col2:
        #reset preprocessing button
        if st.button("Reset Preprocessing", use_container_width=True, help="Revert to original dataset and clear all preprocessing"):
            st.session_state.processed_data = None
            st.session_state.label_encoders = {}
            st.session_state.scaler = None
            st.session_state.ordinal_encoder = None
            st.session_state.preprocessing_log = []
            st.session_state.X_train = None
            st.session_state.X_test = None
            st.session_state.y_train = None
            st.session_state.y_test = None
            st.session_state.trained_models = {}
            st.session_state.model_results = {}
            st.session_state.best_params = {}
            st.session_state.training_times = {}
            st.success("Preprocessing reset! Dataset reverted to original state.")
            st.rerun()
    
    if apply_button:
        if not target_column or not feature_columns:
            st.error("Select target and features!")
            return
        
        with st.spinner("Processing..."):
            try:
                processing_log = []
                processed_df = df.copy()
                
                st.session_state.original_features = feature_columns.copy()
                
                #handle constant features
                if handle_constant == "Remove" and st.session_state.eda_report.get('constant_features'):
                    const_features = [f['feature'] for f in st.session_state.eda_report['constant_features']]
                    # Only remove if they're in feature_columns
                    features_to_remove = [f for f in const_features if f in feature_columns]
                    if features_to_remove:
                        feature_columns = [f for f in feature_columns if f not in features_to_remove]
                        st.session_state.feature_columns = feature_columns
                        processing_log.append(f"Removed {len(features_to_remove)} constant feature(s)")
                
                #handle high cardinality features
                if handle_cardinality != "Keep All" and st.session_state.eda_report.get('high_cardinality'):
                    high_card_cols = list(st.session_state.eda_report['high_cardinality'].keys())
                    high_card_in_features = [col for col in high_card_cols if col in feature_columns]
                    
                    if handle_cardinality == "Remove Columns":
                        feature_columns = [f for f in feature_columns if f not in high_card_in_features]
                        st.session_state.feature_columns = feature_columns
                        processing_log.append(f"Removed {len(high_card_in_features)} high cardinality column(s)")
                    elif "Top" in handle_cardinality:
                        top_n = 10 if "Top 10" in handle_cardinality else 20
                        for col in high_card_in_features:
                            if col in processed_df.columns:
                                top_categories = processed_df[col].value_counts().nlargest(top_n).index
                                processed_df[col] = processed_df[col].apply(
                                    lambda x: x if x in top_categories else 'Other'
                                )
                        processing_log.append(f"Reduced cardinality: kept top {top_n} categories for {len(high_card_in_features)} column(s)")
                
                #handle missing
                if handle_missing != "None":
                    if handle_missing == "Drop rows":
                        processed_df = processed_df.dropna()
                    elif handle_missing == "Mean imputation":
                        numeric_cols = processed_df.select_dtypes(include=[np.number]).columns
                        processed_df[numeric_cols] = processed_df[numeric_cols].fillna(processed_df[numeric_cols].mean())
                    elif handle_missing == "Median imputation":
                        numeric_cols = processed_df.select_dtypes(include=[np.number]).columns
                        processed_df[numeric_cols] = processed_df[numeric_cols].fillna(processed_df[numeric_cols].median())
                    elif handle_missing == "Mode imputation":
                        for col in processed_df.columns:
                            processed_df[col] = processed_df[col].fillna(processed_df[col].mode()[0] if len(processed_df[col].mode()) > 0 else 'Unknown')
                    processing_log.append(f"Applied {handle_missing}")
                
                #encode categorical
                label_encoders = {}
                ordinal_encoder = None
                if encode_categorical != "None":
                    cat_cols = processed_df[feature_columns].select_dtypes(include=['object', 'category']).columns
                    if encode_categorical == "Label Encoding":
                        for col in cat_cols:
                            le = LabelEncoder()
                            processed_df[col] = le.fit_transform(processed_df[col].astype(str))
                            label_encoders[col] = le
                        processing_log.append(f"Label encoded {len(cat_cols)} columns")
                    elif encode_categorical == "One-Hot Encoding":
                        processed_df = pd.get_dummies(processed_df, columns=cat_cols, drop_first=True)
                        feature_columns = [col for col in processed_df.columns if col != target_column]
                        st.session_state.feature_columns = feature_columns
                        processing_log.append(f"One-hot encoded {len(cat_cols)} columns")
                    elif encode_categorical == "Ordinal Encoding":
                        # Ordinal Encoding (FR-16)
                        if len(cat_cols) > 0:
                            ordinal_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
                            processed_df[cat_cols] = ordinal_encoder.fit_transform(processed_df[cat_cols].astype(str))
                            st.session_state.ordinal_encoder = ordinal_encoder
                            processing_log.append(f"Ordinal encoded {len(cat_cols)} columns")
                
                #encode target
                if processed_df[target_column].dtype == 'object':
                    le = LabelEncoder()
                    processed_df[target_column] = le.fit_transform(processed_df[target_column])
                    label_encoders['target'] = le
                else:
                    processed_df[target_column] = processed_df[target_column].astype(int)
                
                #scale features
                scaler = None
                if scaling_method != "None":
                    numeric_features = processed_df[feature_columns].select_dtypes(include=[np.number]).columns
                    if scaling_method == "StandardScaler":
                        scaler = StandardScaler()
                    elif scaling_method == "MinMaxScaler":
                        scaler = MinMaxScaler()
                    elif scaling_method == "RobustScaler":
                        scaler = RobustScaler()
                    if scaler and len(numeric_features) > 0:
                        processed_df[numeric_features] = scaler.fit_transform(processed_df[numeric_features])
                        processing_log.append(f"Applied {scaling_method}")
                
                st.session_state.processed_data = processed_df
                st.session_state.label_encoders = label_encoders
                st.session_state.scaler = scaler
                st.session_state.ordinal_encoder = ordinal_encoder
                st.session_state.preprocessing_log = processing_log
                st.session_state.encoding_method = encode_categorical
                
                #split data
                X = processed_df[feature_columns]
                y = processed_df[target_column]
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
                
                #handle class imbalance (applied only to training data)
                if handle_imbalance != "None" and IMBLEARN_AVAILABLE:
                    try:
                        if handle_imbalance == "SMOTE":
                            #check if we have enough samples for smote
                            min_samples = y_train.value_counts().min()
                            k_neighbors = min(5, min_samples - 1) if min_samples > 1 else 1
                            if k_neighbors > 0:
                                smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
                                X_train, y_train = smote.fit_resample(X_train, y_train)
                                processing_log.append(f"Applied SMOTE oversampling")
                            else:
                                st.warning("Not enough samples for SMOTE. Skipping.")
                        elif handle_imbalance == "Random Oversample":
                            ros = RandomOverSampler(random_state=42)
                            X_train, y_train = ros.fit_resample(X_train, y_train)
                            processing_log.append(f"Applied random oversampling")
                        elif handle_imbalance == "Random Undersample":
                            rus = RandomUnderSampler(random_state=42)
                            X_train, y_train = rus.fit_resample(X_train, y_train)
                            processing_log.append(f"Applied random undersampling")
                    except Exception as e:
                        st.warning(f"Could not apply {handle_imbalance}: {e}")
                elif handle_imbalance == "Class Weights" and not IMBLEARN_AVAILABLE:
                    processing_log.append("Class weights will be applied during model training")
                    #note: class weights would need to be passed to model training
                
                st.session_state.X_train = X_train
                st.session_state.X_test = X_test
                st.session_state.y_train = y_train
                st.session_state.y_test = y_test
                
                st.success("Preprocessing completed!")
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Training Samples", len(X_train))
                col2.metric("Test Samples", len(X_test))
                col3.metric("Features", len(feature_columns))
                
                for i, step in enumerate(processing_log, 1):
                    st.markdown(f"Step {i}: {step}")
                
            except Exception as e:
                st.error(f"Error: {e}")

def show_training_page():
    st.markdown("## Model Training")
    
    if st.session_state.X_train is None:
        st.warning("Preprocess data first!")
        return
    
    available_models = ['Logistic Regression', 'Decision Tree', 'Random Forest', 'Gradient Boosting', 'SVM', 'KNN', 'Naive Bayes', 'AdaBoost', 'Rule-Based Classifier']
    
    #add select all checkbox
    col1, col2 = st.columns([1, 4])
    with col1:
        select_all = st.checkbox("Select All Models", help="Train all available classification models")
    
    if select_all:
        selected_models = st.multiselect(
            "Select Models", 
            available_models, 
            default=available_models,
            help="Choose which machine learning algorithms to train"
        )
    else:
        selected_models = st.multiselect(
            "Select Models", 
            available_models, 
            default=available_models[:3],
            help="Choose which machine learning algorithms to train"
        )
    
    #hyperparameter optimization toggle
    optimize_hyperparams = st.checkbox(
        "Enable Hyperparameter Optimization",
        value=False,
        help="Use RandomizedSearchCV to find optimal hyperparameters. This will increase training time but may improve model performance."
    )
    
    if st.button("Train Models", type="primary"):
        if not selected_models:
            st.error("Select at least one model!")
            return
        
        X_train = st.session_state.X_train
        y_train = st.session_state.y_train
        X_test = st.session_state.X_test
        y_test = st.session_state.y_test
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        results = []
        
        for i, model_name in enumerate(selected_models):
            try:
                status_text.text(f"Training {model_name}...")
                
                #record training time
                start_time = time.time()
                model, best_params = train_model(model_name, X_train, y_train, optimize_hyperparams)
                training_time = time.time() - start_time
                
                metrics = evaluate_model(model, X_test, y_test)
                
                st.session_state.trained_models[model_name] = model
                st.session_state.model_results[model_name] = metrics
                st.session_state.training_times[model_name] = training_time
                if best_params:
                    st.session_state.best_params[model_name] = best_params
                
                results.append({
                    'Model': model_name, 
                    'Accuracy': metrics['Accuracy'], 
                    'Precision': metrics['Precision'], 
                    'Recall': metrics['Recall'], 
                    'F1-Score': metrics['F1-Score'],
                    'Training Time (s)': round(training_time, 2)
                })
            except Exception as e:
                st.warning(f"Error training {model_name}: {e}")
            
            progress_bar.progress((i + 1) / len(selected_models))
        
        if results:
            status_text.text("Training complete!")
            st.success(f"Trained {len(results)} models!")
            results_df = pd.DataFrame(results).sort_values('Accuracy', ascending=False)
            #reset index to start from 1
            results_df.index = range(1, len(results_df) + 1)
            results_df.index.name = 'Rank'
            st.dataframe(results_df, use_container_width=True)
            
            best = results_df.iloc[0]
            st.success(f"Best Model: {best['Model']} (Accuracy: {best['Accuracy']:.4f}, Training Time: {best['Training Time (s)']}s)")
            
            #display best parameters
            if st.session_state.best_params:
                st.markdown("---")
                st.markdown("### Optimized Hyperparameters")
                for model_name, params in st.session_state.best_params.items():
                    with st.expander(f"{model_name} - Best Parameters"):
                        st.json(params)

def show_evaluation_page():
    st.markdown("## Model Evaluation")
    
    if not st.session_state.trained_models:
        st.warning("Train models first!")
        return
    
    selected_model = st.selectbox("Select Model", list(st.session_state.trained_models.keys()))
    
    if selected_model:
        model = st.session_state.trained_models[selected_model]
        metrics = st.session_state.model_results[selected_model]
        
        st.markdown(f"### {selected_model}")
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Accuracy", f"{metrics['Accuracy']:.4f}")
        col2.metric("Precision", f"{metrics['Precision']:.4f}")
        col3.metric("Recall", f"{metrics['Recall']:.4f}")
        col4.metric("F1-Score", f"{metrics['F1-Score']:.4f}")
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = plot_confusion_matrix(metrics['Confusion Matrix'], f'{selected_model} - Confusion Matrix')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if 'ROC AUC' in metrics:
                fig = plot_roc_curve(metrics['FPR'], metrics['TPR'], metrics['ROC AUC'])
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.text(metrics['Classification Report'])
        
        fig = plot_feature_importance(model, st.session_state.feature_columns, f'{selected_model} - Feature Importance')
        if fig:
            st.plotly_chart(fig, use_container_width=True)

def show_comparison_page():
    st.markdown("## Model Comparison")
    
    if len(st.session_state.trained_models) < 2:
        st.warning("Train at least 2 models!")
        return
    
    results = st.session_state.model_results
    comparison_data = [{'Model': name, 'Accuracy': m['Accuracy'], 'Precision': m['Precision'], 'Recall': m['Recall'], 'F1-Score': m['F1-Score']} for name, m in results.items()]
    comparison_df = pd.DataFrame(comparison_data).sort_values('Accuracy', ascending=False)
    
    st.dataframe(comparison_df.style.background_gradient(cmap='viridis', subset=['Accuracy']), use_container_width=True)
    
    st.markdown("---")
    
    metric = st.selectbox("Select Metric", ['Accuracy', 'Precision', 'Recall', 'F1-Score'])
    fig = plot_model_comparison(comparison_df, metric)
    st.plotly_chart(fig, use_container_width=True)
    
    best = comparison_df.iloc[0]
    st.success(f"Best Model: {best['Model']} - {metric}: {best[metric]:.4f}")

def show_report_page():
    st.markdown("## Download Analysis Report")
    
    st.markdown("""
    Generate and download a comprehensive PDF report containing all your analysis results.
    
    The report includes:
    - Dataset information and statistics
    - Preprocessing steps applied
    - Model performance metrics
    - Best model recommendation
    """)
    
    if not PDF_AVAILABLE:
        st.error("PDF generation requires the 'reportlab' library.")
        st.code("pip install reportlab", language="bash")
        return
    
    st.markdown("---")
    
    #show preview of what will be in the report
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### Dataset Info")
        if st.session_state.data is not None:
            df = st.session_state.data
            st.write(f"- Rows: {len(df)}")
            st.write(f"- Columns: {len(df.columns)}")
            st.write(f"- Missing: {df.isnull().sum().sum()}")
        else:
            st.info("No data loaded")
    
    with col2:
        st.markdown("### Preprocessing")
        if st.session_state.preprocessing_log:
            st.write(f"- Steps: {len(st.session_state.preprocessing_log)}")
            st.write(f"- Target: {st.session_state.target_column}")
            st.write(f"- Features: {len(st.session_state.feature_columns)}")
        else:
            st.info("No preprocessing done")
    
    with col3:
        st.markdown("### Models Trained")
        if st.session_state.trained_models:
            st.write(f"- Total: {len(st.session_state.trained_models)}")
            best = max(st.session_state.model_results.items(), key=lambda x: x[1]['Accuracy'])
            st.write(f"- Best: {best[0]}")
            st.write(f"- Accuracy: {best[1]['Accuracy']:.4f}")
        else:
            st.info("No models trained")
    
    st.markdown("---")
    
    if st.button("Generate PDF Report", type="primary", use_container_width=True):
        if not st.session_state.data is not None:
            st.warning("Please upload data first!")
            return
        
        with st.spinner("Generating PDF report..."):
            try:
                pdf_bytes = generate_pdf_report()
                if pdf_bytes:
                    st.success("PDF report generated successfully!")
                    st.download_button(
                        label="Download PDF Report",
                        data=pdf_bytes,
                        file_name=f"automl_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                        mime="application/pdf",
                        type="primary",
                        use_container_width=True
                    )
                else:
                    st.error("Could not generate PDF report.")
            except Exception as e:
                st.error(f"Error generating PDF: {e}")
                st.exception(e)

def show_prediction_page():
    st.markdown("## Predictions")
    
    if not st.session_state.trained_models:
        st.warning("Train models first!")
        return
    
    selected_model = st.selectbox("Select Model", list(st.session_state.trained_models.keys()))
    
    tab1, tab2 = st.tabs(["Single Prediction", "Batch Prediction"])
    
    with tab1:
        st.markdown("### Enter Values")
        original_df = st.session_state.data
        target_col = st.session_state.target_column
        original_features = [col for col in original_df.columns if col != target_col]
        
        input_data = {}
        cols = st.columns(3)
        
        for i, feature in enumerate(original_features):
            with cols[i % 3]:
                col_data = original_df[feature]
                if col_data.dtype == 'object' or (col_data.nunique() <= 10 and col_data.dtype in ['int64', 'int32']):
                    unique_values = sorted([str(v) for v in col_data.dropna().unique()])
                    input_data[feature] = st.selectbox(f"{feature}", unique_values)
                else:
                    input_data[feature] = st.number_input(f"{feature}", value=float(col_data.mean()))
        
        if st.button("Predict", type="primary"):
            model = st.session_state.trained_models[selected_model]
            try:
                input_df = pd.DataFrame([input_data])
                processed_input = preprocess_input_for_prediction(input_df, original_features)
                
                #get raw prediction
                raw_prediction = model.predict(processed_input)[0]
                
                #always decode to actual class label
                if 'target' in st.session_state.label_encoders:
                    le = st.session_state.label_encoders['target']
                    prediction_idx = int(round(raw_prediction))
                    decoded = le.inverse_transform([prediction_idx])[0]
                    st.success(f"**Predicted Class:** {decoded}")
                else:
                    #if target wasn't encoded, show the original value
                    original_target_values = original_df[target_col].unique()
                    prediction_idx = int(round(raw_prediction))
                    if prediction_idx < len(original_target_values):
                        st.success(f"**Predicted Class:** {original_target_values[prediction_idx]}")
                    else:
                        st.success(f"**Predicted Class:** {prediction_idx}")
                
                if hasattr(model, 'predict_proba'):
                    prob = model.predict_proba(processed_input)[0]
                    if 'target' in st.session_state.label_encoders:
                        class_labels = st.session_state.label_encoders['target'].classes_
                    else:
                        class_labels = [f"Class {i}" for i in range(len(prob))]
                    prob_df = pd.DataFrame({'Class': class_labels[:len(prob)], 'Probability': prob})
                    fig = px.bar(prob_df, x='Class', y='Probability', title='Confidence')
                    st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error: {e}")
    
    with tab2:
        uploaded_file = st.file_uploader("Upload CSV", type=['csv'])
        if uploaded_file:
            try:
                pred_df = pd.read_csv(uploaded_file)
                st.dataframe(pred_df.head())
                
                if st.button("Batch Predict", type="primary"):
                    model = st.session_state.trained_models[selected_model]
                    original_features = [col for col in st.session_state.data.columns if col != st.session_state.target_column]
                    X_pred = pred_df[original_features]
                    processed_X = preprocess_input_for_prediction(X_pred, original_features)
                    raw_predictions = model.predict(processed_X)
                    predictions = np.round(raw_predictions).astype(int)
                    
                    result_df = pred_df.copy()
                    
                    #always show actual class labels, not encoded values
                    if 'target' in st.session_state.label_encoders:
                        result_df['Prediction'] = st.session_state.label_encoders['target'].inverse_transform(predictions)
                    else:
                        result_df['Prediction'] = predictions
                    
                    st.dataframe(result_df)
                    csv = result_df.to_csv(index=False)
                    st.download_button("Download CSV", csv, "predictions.csv", "text/csv")
            except Exception as e:
                st.error(f"Error: {e}")

#main app
def main():
    with st.sidebar:
        st.markdown("### Classifyy")
        st.markdown("---")
        
        page = st.radio("Navigation", ["Home", "Upload", "EDA", "Preprocessing", "Training", "Evaluation", "Comparison", "Predictions", "Report"])
        
        st.markdown("---")
        
        if st.session_state.data is not None:
            st.metric("Rows", len(st.session_state.data))
            st.metric("Columns", len(st.session_state.data.columns))
            if st.session_state.trained_models:
                st.metric("Models Trained", len(st.session_state.trained_models))
    
    if page == "Home":
        show_home_page()
    elif page == "Upload":
        show_data_upload_page()
    elif page == "EDA":
        show_eda_page()
    elif page == "Preprocessing":
        show_preprocessing_page()
    elif page == "Training":
        show_training_page()
    elif page == "Evaluation":
        show_evaluation_page()
    elif page == "Comparison":
        show_comparison_page()
    elif page == "Predictions":
        show_prediction_page()
    elif page == "Report":
        show_report_page()

if __name__ == "__main__":
    main()