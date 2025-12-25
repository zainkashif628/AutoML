# Classifyy - AutoML Classification System

A web-based, no-code AutoML system for automated classification with comprehensive data analysis, preprocessing, model training, and evaluation. Built with Streamlit for an intuitive user experience.

## Overview

Classifyy automates the complete machine learning pipeline for classification tasks through an interactive web interface. It provides a production-ready solution for data scientists, students, and analysts who want to perform classification without writing code.

**Team:** Fatima Ali (470708), Adeena Reeham (480941), Zain Kashif (458822)

## Key Features

- **Interactive Web Interface** - User-friendly Streamlit dashboard with guided workflow
- **Flexible Data Upload** - Upload CSV files or use built-in sample datasets (Iris, Wine, Breast Cancer)
- **Comprehensive EDA** - Missing values, outliers, correlations, distributions with interactive visualizations
- **Smart Preprocessing** - Handle missing values, outliers, class imbalance, scaling, and encoding
- **9 ML Models** - Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, SVM, KNN, Naive Bayes, AdaBoost, Rule-Based
- **Hyperparameter Tuning** - Automated RandomizedSearchCV with cross-validation
- **Complete Evaluation** - Accuracy, Precision, Recall, F1, ROC-AUC, confusion matrices with interactive plots
- **Model Comparison** - Side-by-side performance comparison across all trained models
- **Prediction Interface** - Single and batch prediction capabilities with confidence scores
- **PDF Reports** - Downloadable comprehensive analysis reports with visualizations
- **Streamlit Cloud Ready** - Deploy instantly to Streamlit Cloud

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/your-username/automl-classification-system.git
cd AutoML

# Install dependencies
pip install -r requirements.txt
```

### Running the Application

Launch the Streamlit web application:

```bash
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`

### Usage Workflow

1. **Upload Data**: Upload your CSV file or select a sample dataset (Iris, Wine, or Breast Cancer)
2. **Explore Data**: View dataset statistics, visualizations, and data quality issues in the EDA page
3. **Preprocess**: Configure preprocessing options:
   - Handle missing values (mean, median, mode, drop)
   - Manage outliers (cap, remove, keep)
   - Address class imbalance (SMOTE, oversample, undersample)
   - Scale features (Standard, MinMax, Robust)
   - Encode categorical variables (Label, One-Hot, Ordinal)
   - Set train-test split ratio
4. **Train Models**: Select models to train and optionally enable hyperparameter optimization
5. **Evaluate**: View detailed metrics, confusion matrices, ROC curves, and feature importance
6. **Compare**: Compare all trained models side-by-side with interactive charts
7. **Predict**: Make single or batch predictions on new data
8. **Download Report**: Generate and download comprehensive PDF analysis report

## Project Structure

```
automl-classification-system/
├── app.py                        # Main Streamlit application
├── requirements.txt              # Python dependencies
├── packages.txt                  # System packages for deployment
├── README.md                     # Documentation (this file)
├── logo.png                      # Application logo
└── reports/                      # Generated PDF reports (created at runtime)
```

## Requirements

Key dependencies (see [requirements.txt](requirements.txt) for complete list):

```txt
streamlit>=1.52.0
pandas>=2.2.0
numpy>=2.2.0
scikit-learn>=1.7.0
matplotlib>=3.10.0
seaborn>=0.13.0
plotly>=6.5.0
scipy>=1.15.0
reportlab>=4.0.0        # For PDF report generation
imbalanced-learn        # For handling class imbalance (SMOTE)
```

## Application Workflow

### Navigation Pages

The application consists of 9 main pages accessible via the sidebar:

1. **Home** - Overview of features, quick start guide, and team information
2. **Upload** - Upload CSV files or select sample datasets (Iris, Wine, Breast Cancer)
3. **EDA (Exploratory Data Analysis)** - View dataset statistics, visualizations, and detect data quality issues:
   - Missing values analysis with interactive charts
   - Correlation heatmaps for numeric features
   - Distribution plots for all features
   - Statistical summaries
   - Data quality issues (class imbalance, high cardinality, constant features)
4. **Preprocessing** - Configure and apply data transformations:
   - Select target column and feature columns
   - Handle missing values (Mean, Median, Mode, Drop)
   - Manage outliers (Cap, Remove, Keep)
   - Address class imbalance (SMOTE, Random Oversample, Random Undersample)
   - Scale features (StandardScaler, MinMaxScaler, RobustScaler)
   - Encode categorical variables (Label Encoding, One-Hot Encoding, Ordinal Encoding)
   - Set train-test split ratio (10-40%)
5. **Training** - Train machine learning models:
   - Select one or all of 9 available models
   - Enable/disable hyperparameter optimization (RandomizedSearchCV)
   - View training progress and time for each model
   - Automatic best parameter selection when optimization is enabled
6. **Evaluation** - Detailed model performance analysis:
   - Comprehensive metrics (Accuracy, Precision, Recall, F1-Score, ROC-AUC)
   - Interactive confusion matrices
   - ROC curves for binary classification
   - Feature importance plots (for applicable models)
   - Classification reports
   - Hyperparameter details (when optimization was used)
7. **Comparison** - Side-by-side model comparison:
   - Sortable comparison table with all metrics
   - Interactive bar charts for metric visualization
   - Best model recommendation
8. **Predictions** - Make predictions on new data:
   - **Single Prediction**: Enter feature values manually with intelligent input types
   - **Batch Prediction**: Upload CSV file for bulk predictions
   - Confidence scores and probability distributions
   - Download predictions as CSV
9. **Report** - Generate comprehensive PDF reports:
   - Dataset information and statistics
   - All preprocessing steps applied
   - Model performance comparison table
   - Visual charts (correlation matrix, model comparison)
   - Best model recommendation

## Supported Models

The application supports 9 classification algorithms:

| Model | Use Case | Hyperparameter Tuning |
|-------|----------|----------------------|
| Logistic Regression | Linear classification, baseline model | C, penalty, solver |
| Decision Tree | Interpretable, non-linear patterns | max_depth, min_samples_split |
| Random Forest | Ensemble, robust to overfitting | n_estimators, max_depth, min_samples_split |
| Gradient Boosting | High accuracy, sequential ensemble | n_estimators, learning_rate, max_depth |
| SVM | High-dimensional data, kernel tricks | C, kernel, gamma |
| KNN | Instance-based, non-parametric | n_neighbors, weights |
| Naive Bayes | Probabilistic, fast training, independent features | N/A |
| AdaBoost | Boosting ensemble, focuses on misclassified samples | n_estimators, learning_rate |
| Rule-Based Classifier | Simple decision tree baseline | max_depth, min_samples_split |

### Hyperparameter Optimization

When enabled, the application uses `RandomizedSearchCV` with 5-fold cross-validation to find optimal hyperparameters for each model (except Naive Bayes). The best parameters are automatically applied and saved for reference.

## Features in Detail

### Data Quality Detection

Classifyy automatically detects and flags:
- **Missing Values**: Identifies columns with missing data and calculates percentages
- **Outliers**: Uses IQR method to detect outliers in numeric columns
- **Class Imbalance**: Detects when target classes are not equally distributed
- **High Cardinality**: Flags categorical features with too many unique values
- **Constant Features**: Identifies features with near-zero variance

### Preprocessing Options

**Missing Values Handling**:
- Mean imputation (numeric)
- Median imputation (numeric)
- Mode imputation (categorical)
- Drop rows with missing values

**Outlier Management**:
- Cap outliers at IQR boundaries
- Remove outlier rows
- Keep outliers as-is

**Class Imbalance Solutions** (requires imbalanced-learn):
- SMOTE (Synthetic Minority Over-sampling)
- Random Over-sampling
- Random Under-sampling

**Feature Scaling**:
- StandardScaler (zero mean, unit variance)
- MinMaxScaler (0-1 range)
- RobustScaler (using median and IQR)

**Categorical Encoding**:
- Label Encoding (ordinal mapping)
- One-Hot Encoding (binary columns)
- Ordinal Encoding (custom order)

### Visualization Features

- Interactive Plotly charts
- Missing values bar charts with color coding
- Correlation heatmaps
- Target distribution charts
- Feature distribution plots (histograms/count plots)
- Confusion matrices with annotations
- ROC curves with AUC scores
- Model comparison bar charts
- Feature importance plots
- Prediction confidence charts

### Model Evaluation Metrics

For each trained model, the application provides:
- **Accuracy**: Overall correct predictions
- **Precision**: Positive prediction accuracy (weighted average)
- **Recall**: True positive rate (weighted average)
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Interactive heatmap showing prediction vs actual
- **ROC-AUC**: For binary classification problems
- **Classification Report**: Detailed per-class metrics
- **Training Time**: Time taken to train the model
- **Best Parameters**: When hyperparameter optimization is enabled

## Example Output

### Model Comparison Table

```
╔═══════════════════════════════════════════════════════════════╗
║  Model                  Accuracy  Precision  Recall  F1-Score ║
╠═══════════════════════════════════════════════════════════════╣
║  Random Forest          0.9667    0.9677     0.9667  0.9667   ║
║  Gradient Boosting      0.9667    0.9677     0.9667  0.9667   ║
║  SVM                    0.9333    0.9355     0.9333  0.9333   ║
║  Logistic Regression    0.9333    0.9355     0.9333  0.9333   ║
║  KNN                    0.9000    0.9032     0.9000  0.9000   ║
║  Decision Tree          0.8667    0.8710     0.8667  0.8667   ║
║  AdaBoost               0.8667    0.8710     0.8667  0.8667   ║
║  Naive Bayes            0.8333    0.8387     0.8333  0.8333   ║
║  Rule-Based Classifier  0.8000    0.8065     0.8000  0.8000   ║
╚═══════════════════════════════════════════════════════════════╝
```

### Preprocessing Log Example

```
Applied Preprocessing Steps:
✓ Target column selected: species
✓ Feature columns selected: 4 features
✓ Missing value imputation: mean
✓ Outlier handling: cap
✓ Feature scaling: StandardScaler
✓ Categorical encoding: Label Encoding
✓ Train-test split: 80-20
✓ Training samples: 120, Test samples: 30
```

## Deployment

### Local Deployment

Simply run the Streamlit application:

```bash
streamlit run app.py
```

Access the application at `http://localhost:8501`

### Streamlit Cloud Deployment

1. Push your code to a GitHub repository
2. Go to [Streamlit Cloud](https://streamlit.io/cloud)
3. Sign in with your GitHub account
4. Click "New app"
5. Select your repository, branch, and `app.py` as the main file
6. Click "Deploy"

The app will be live at `https://your-app-name.streamlit.app`

**Note**: Ensure `requirements.txt` and `packages.txt` are included for proper dependency installation.

## Sample Datasets

The application includes three built-in datasets for quick testing:

1. **Iris Dataset**: 150 samples, 4 features, 3 classes (Setosa, Versicolor, Virginica)
2. **Wine Dataset**: 178 samples, 13 features, 3 classes (wine quality)
3. **Breast Cancer Dataset**: 569 samples, 30 features, 2 classes (Malignant, Benign)

These are loaded directly from scikit-learn's datasets module.

## Technical Details

### Session State Management

The application uses Streamlit's session state to maintain data across page navigation:
- `data`: Original uploaded dataset
- `processed_data`: Preprocessed dataset
- `target_column`: Selected target variable
- `feature_columns`: List of selected feature columns
- `trained_models`: Dictionary of trained model objects
- `model_results`: Dictionary of model evaluation metrics
- `label_encoders`: Encoders for categorical variables
- `scaler`: Fitted scaling transformer
- `X_train`, `X_test`, `y_train`, `y_test`: Train-test split data
- `eda_report`: Comprehensive EDA analysis results
- `preprocessing_log`: List of applied preprocessing steps
- `best_params`: Best hyperparameters for each model

### Error Handling

The application includes comprehensive error handling:
- Missing data validation before operations
- Type checking for uploaded files
- Exception handling for model training and prediction
- User-friendly error messages with suggestions

### Performance Considerations

- Models are trained sequentially with progress indicators
- Large datasets may benefit from feature selection
- Hyperparameter optimization increases training time significantly
- PDF generation may be slow with many models and large datasets

## Documentation

For additional context about the project:
- **Implementation**: `app.py` - Complete Streamlit application
- **Dependencies**: `requirements.txt` - Python package requirements
- **System Packages**: `packages.txt` - System-level dependencies for deployment

## Contributing

This is an academic project developed for CS-245 Machine Learning coursework. For suggestions or improvements:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/improvement`)
5. Submit a pull request

## Troubleshooting

### Common Issues

**PDF Generation Error**:
```bash
pip install reportlab
```

**Class Imbalance Options Not Available**:
```bash
pip install imbalanced-learn
```

**Module Import Errors**:
```bash
pip install -r requirements.txt --upgrade
```

**Streamlit Port Already in Use**:
```bash
streamlit run app.py --server.port 8502
```

## License

This project is part of CS-245 Machine Learning coursework at [Your University Name].

## Contact

- **Fatima Ali** - Student ID: 470708
- **Adeena Reeham** - Student ID: 480941  
- **Zain Kashif** - Student ID: 458822

## Acknowledgments

- Built using [Streamlit](https://streamlit.io/) for the web interface
- Machine learning models from [scikit-learn](https://scikit-learn.org/)
- Visualizations powered by [Plotly](https://plotly.com/) and [Matplotlib](https://matplotlib.org/)
- PDF reports generated with [ReportLab](https://www.reportlab.com/)

---

**Course**: CS-245 Machine Learning  
**Project**: AutoML Classification System  
**Academic Year**: 2024-2025