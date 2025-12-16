# AutoML Classification System

An intelligent, no-code AutoML system for automated classification with comprehensive data analysis, preprocessing, model training, and evaluation.

## Overview

This AutoML system automates the complete machine learning pipeline for classification tasks. Built for CS-245 Machine Learning course, it provides a production-ready solution for data scientists, students, and analysts.

**Team:** Fatima Ali (470708), Adeena Reeham (480941), Zain Kashif (458822)

## Key Features

- **Automated EDA** - Missing values, outliers, correlations, distributions
- **Smart Preprocessing** - Imputation, scaling, encoding with user control
- **7 ML Models** - Logistic Regression, KNN, Decision Tree, Naive Bayes, Random Forest, SVM, Rule-Based
- **Hyperparameter Tuning** - GridSearch/RandomizedSearch with cross-validation
- **Complete Evaluation** - Accuracy, Precision, Recall, F1, ROC-AUC, confusion matrices
- **Auto Reports** - Downloadable reports with model comparison and recommendations
- **Streamlit Ready** - Deploy to Streamlit Cloud instantly

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/your-username/automl-classification-system.git
cd automl-classification-system

# Install dependencies
pip install -r requirements.txt
```

### Usage (Jupyter Notebook)

```python
from automl_system import AutoMLPipeline

# Load your dataset
pipeline = AutoMLPipeline(filepath='data.csv', target_column='target')

# Run complete pipeline
results = pipeline.run_complete_pipeline(
    imputation_strategy='mean',
    outlier_method='cap',
    scaling_method='standard',
    encoding_method='onehot',
    test_size=0.2,
    optimize_hyperparams=True
)

# Get best model
best_model = results['best_model']
print(f"Best Model: {results['best_model_name']}")
```

### Usage (Streamlit App)

```bash
streamlit run app.py
```

## Project Structure

```
automl-classification-system/
├── automl_system.ipynb          # Complete implementation notebook
├── app.py                        # Streamlit application
├── requirements.txt              # Python dependencies
├── README.md                     # This file
├── sample_data/                  # Sample datasets
├── reports/                      # Generated reports
└── screenshots/                  # App screenshots
```

## Requirements

```txt
pandas>=1.5.0
numpy>=1.23.0
scikit-learn>=1.3.0
matplotlib>=3.6.0
seaborn>=0.12.0
streamlit>=1.28.0
scipy>=1.10.0
```

## Pipeline Workflow

1. **Upload CSV** → Validate and extract metadata
2. **Issue Detection** → Flag missing values, outliers, imbalance
3. **User Approval** → Choose preprocessing strategies
4. **Preprocessing** → Clean and transform data
5. **EDA** → Visualize distributions and correlations
6. **Model Training** → Train 7 classifiers with timing
7. **Optimization** → Tune hyperparameters automatically
8. **Evaluation** → Compare models with comprehensive metrics
9. **Report** → Generate downloadable analysis report

## Supported Models

| Model | Use Case |
|-------|----------|
| Logistic Regression | Linear classification, baseline |
| K-Nearest Neighbors | Non-linear, instance-based |
| Decision Tree | Interpretable, non-linear |
| Naive Bayes | Probabilistic, fast training |
| Random Forest | Ensemble, robust to overfitting |
| SVM | High-dimensional data |
| Rule-Based | Simple majority class baseline |

## Example Output

```
MODEL COMPARISON TABLE (Ranked by F1-Score)
═══════════════════════════════════════════════════════════
   Model                    Accuracy  Precision  Recall  F1-Score
1  Random Forest            0.9667    0.9677     0.9667  0.9667
2  SVM                      0.9667    0.9677     0.9667  0.9667
3  Logistic Regression      0.9333    0.9355     0.9333  0.9333
```

## Deployment

### Local Deployment
```bash
streamlit run app.py
```

### Streamlit Cloud
1. Push code to GitHub
2. Connect to [Streamlit Cloud](https://streamlit.io/cloud)
3. Select repository and branch
4. Deploy!

## Documentation

For detailed documentation, see:
- **Project Requirements**: `Semester Project _AutoML System for Classification.pdf`
- **Lab Work**: `ML_lab11,12[1].pdf`
- **Implementation**: `automl_system.ipynb`

## Contributing

This is an academic project. For suggestions or improvements:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## License

This project is part of CS-245 Machine Learning coursework.

## Contact

- **Fatima Ali** - 470708
- **Adeena Reeham** - 480941  
- **Zain Kashif** - 458822

---