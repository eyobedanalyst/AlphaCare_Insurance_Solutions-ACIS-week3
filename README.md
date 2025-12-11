# Task 1 & Task 2 Progress Report

Data Engineering & Version Control Workflow for Insurance Analysis Project
1. Project Overview

This project focuses on building a reproducible, auditable, and version-controlled data workflow for insurance-related analytics tasks. The work covers data collection, preprocessing, exploratory data analysis (EDA), and the setup of a fully traceable data pipeline using Git, branching strategy, and Data Version Control (DVC).

The overarching objective is to ensure that every dataset, preprocessing step, and script contributing to analysis can be reproduced at any time. This is critical for regulated industries such as finance and insurance, where audit trails and data lineage are mandatory.

2. Task 1 Summary – Data Collection, Cleaning, EDA
2.1 Activities Completed

Scraped or loaded datasets into the workspace.

Preprocessed data:

Standardized column names

Handled missing values

Converted datatypes

Univariate Analysis

Histograms for numerical fields (e.g., TotalPremium, TotalClaims)

Bar plots for categorical variables (InsuranceCoverType, AutoMake, ZipCode)

Bivariate / Multivariate Analysis

Scatter plots exploring relationships between TotalPremium and TotalClaims

Grouped analysis by ZipCode

Correlation matrices on numerical variables

Geographical-based comparisons

Compared trends in cover type, auto make, claims differences, and premiums across regions.

2.2 Tools Used

Python

pandas / matplotlib / seaborn

Jupyter Notebook

GitHub Repository with commits from task-1 branch

3. Task 2 – Reproducible Workflow with Data Version Control (DVC)
3.1 Objective

Implement a reproducible and fully traceable data pipeline using DVC, ensuring that all datasets are versioned just like code.

3.2 Steps Completed
Step 1 — Install DVC
pip install dvc

Step 2 — Initialize DVC inside the repository
dvc init


This creates:

.dvc/ directory

.dvcignore

Modifications in .gitignore

Step 3 — Create Local DVC Storage

A remote storage location is required for datasets to be pushed to.
This is not GitHub storage, it is simply a folder on your machine that DVC will use as a data warehouse.

Example:

mkdir dvc_storage


DVC remote configuration:

dvc remote add -d localstorage dvc_storage


Now DVC knows where to store and retrieve dataset versions.

Step 4 — Add Dataset to DVC

Move your dataset (e.g., insurance_data.csv) into the project folder.

Run:

dvc add insurance_data.csv


This produces:

insurance_data.csv.dvc file (tracked by Git)

Adds the actual data file to .gitignore

Step 5 — Commit Changes to Git
git add .
git commit -m "Added dataset to DVC and configured remote storage"

Step 6 — Push Data to DVC Remote
dvc push


This uploads your dataset to the local remote folder, ensuring reproducibility.

3.3 Git Branching Requirements Completed

Merged necessary task-1 work into main via Pull Request

Created new branch:

git checkout -b task-2


All DVC configuration is committed to task-2

DVC remote storage successfully configured

First version of the dataset tracked and pushed

4. Challenges Faced
4.1 Confusion About DVC Remote Storage

You originally expected a file like rating.txt.dvc to appear automatically.
You learned that .dvc files only appear after running dvc add on a dataset.

4.2 Unclear Purpose of Local Remote Storage

Initially unclear why a storage directory was needed.
You now understand:

Git tracks metadata (the .dvc file)

DVC remote stores the actual dataset

This separation enables lightweight Git repos and full data reproducibility.

4.3 First-Time DVC Workflow

You were new to:

DVC initialization

Remote configuration

Understanding .gitignore changes

Proper versioning of large files outside GitHub

These are common challenges when adopting a data versioning pipeline.

4.4 Branch Management

You faced difficulty merging, creating branches, and ensuring that task-1 code appeared correctly in main.
This has now been resolved using Pull Requests and clean branch creation.

This repository contains the implementation of Task 3: Statistical Hypothesis Testing and Task 4: Predictive Modeling for an insurance risk analysis project. The goal is to statistically validate risk drivers and build predictive models for claim severity and premium optimization.

🚀 Quick Start
Prerequisites
bash
Python 3.8+
Git
Installation
bash
# Clone the repository
git clone https://github.com/yourusername/insurance-risk-analysis.git
cd insurance-risk-analysis

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install DVC (if not already installed)
pip install dvc
Repository Setup
bash
# Initialize DVC
dvc init

# Set up remote storage (local example)
mkdir ./dvc_storage
dvc remote add -d localstorage ./dvc_storage

# Pull processed data
dvc pull
📁 Project Structure
text
insurance-risk-analysis/
│
├── data/                           # Data directory (DVC tracked)
│   ├── processed/                  # Cleaned datasets
│   ├── features/                   # Engineered features
│   └── models/                     # Saved models
│
├── notebooks/                      # Jupyter notebooks
│   ├── task3_hypothesis_testing.ipynb
│   ├── task4_modeling.ipynb
│   └── results_analysis.ipynb
│
├── src/                            # Source code
│   ├── __init__.py
│   ├── hypothesis_testing/         # Task 3 modules
│   │   ├── __init__.py
│   │   ├── statistical_tests.py    # Statistical test implementations
│   │   ├── metrics_calculator.py   # KPI calculations
│   │   └── segmentation.py         # Data segmentation functions
│   │
│   └── modeling/                   # Task 4 modules
│       ├── __init__.py
│       ├── data_preprocessor.py    # Data cleaning & preparation
│       ├── feature_engineer.py     # Feature engineering
│       ├── model_trainer.py        # Model training pipeline
│       ├── model_evaluator.py      # Model evaluation metrics
│       └── shap_analyzer.py        # SHAP analysis for interpretability
│
├── tests/                          # Unit tests
│   ├── test_hypothesis_tests.py
│   └── test_models.py
│
├── reports/                        # Generated reports
│   ├── hypothesis_test_results/
│   ├── model_performance/
│   └── business_recommendations/
│
├── config/                         # Configuration files
│   ├── model_config.yaml
│   └── test_config.yaml
│
├── .github/workflows/              # CI/CD pipelines
├── requirements.txt                # Python dependencies
├── dvc.yaml                        # DVC pipeline definition
├── README.md                       # This file
└── .gitignore
🔬 Task 3: Statistical Hypothesis Testing
Objective
Statistically validate or reject key hypotheses about risk drivers to form the basis of a new segmentation strategy.

Hypotheses Tested
Hypothesis	Test Method	Metric	Significance Level
H₁: No risk differences across provinces	ANOVA	Claim Frequency	α = 0.05
H₂: No risk differences between zip codes	Kruskal-Wallis	Claim Severity	α = 0.05
H₃: No margin differences between zip codes	Two-sample t-test	Profit Margin	α = 0.05
H₄: No risk differences between genders	Chi-square test	Claim Frequency	α = 0.05
Key Metrics
Claim Frequency: Proportion of policies with at least one claim

Claim Severity: Average claim amount (given a claim occurred)

Margin: TotalPremium - TotalClaims

Usage
python
# Run all hypothesis tests
from src.hypothesis_testing.statistical_tests import HypothesisTester

tester = HypothesisTester(data_path='data/processed/insurance_data_clean.csv')
results = tester.run_all_tests()
tester.generate_report(output_path='reports/hypothesis_test_results/')
Expected Output
Statistical test results (p-values, test statistics)

Visualizations of group comparisons

Business recommendations based on findings

CSV/Excel report with detailed analysis

🤖 Task 4: Predictive Modeling
Objective
Build and evaluate predictive models for claim severity prediction and premium optimization.

Modeling Goals
Claim Severity Prediction: Predict TotalClaims for policies with claims > 0

Premium Optimization: Develop ML model to predict appropriate premiums

Claim Probability: Binary classification for claim occurrence probability

Models Implemented
Model	Purpose	Target Variable	Evaluation Metrics
Linear Regression	Baseline	TotalClaims	RMSE, R²
Random Forest	Claim Severity	TotalClaims	RMSE, R², MAE
XGBoost	Best-performing	TotalClaims	RMSE, R², MAE
Random Forest Classifier	Claim Probability	HasClaim (0/1)	Accuracy, Precision, Recall, AUC-ROC
Feature Engineering
python
# Key engineered features:
1. VehicleAge = CurrentYear - VehicleYear
2. PremiumToValueRatio = TotalPremium / CustomValueEstimate
3. UrbanDensityScore (based on ZipCode)
4. PreviousClaimsCount (rolling window)
5. RiskScore = f(VehicleType, Province, AgeGroup)
Usage
python
# Full modeling pipeline
from src.modeling.model_trainer import ModelTrainer

# Initialize trainer
trainer = ModelTrainer(
    data_path='data/processed/insurance_data_clean.csv',
    config_path='config/model_config.yaml'
)

# Train models
models, results = trainer.train_all_models()

# Evaluate models
trainer.evaluate_models(models)

# Generate SHAP analysis
trainer.explain_models(models, output_path='reports/model_performance/')
Model Evaluation
python
# Expected performance metrics:
- RMSE: Measure of prediction error (lower is better)
- R²: Proportion of variance explained (higher is better)
- MAE: Mean absolute error
- AUC-ROC: For classification models
- Feature Importance: Top influential features
📊 Results Interpretation
For Rejected Hypotheses (Task 3)
markdown
Example Interpretation:
"We reject H₀ for provinces (p < 0.01). Gauteng exhibits 24% higher claim frequency
than Western Cape, suggesting regional risk adjustment to premiums is warranted."
For Predictive Models (Task 4)
markdown
Example Business Insight:
"SHAP analysis reveals VehicleAge increases predicted claim amount by R850 per year.
This provides quantitative evidence to refine age-based premium adjustments."
🛠️ Development Workflow
Branch Strategy
bash
# Create feature branch for task 3
git checkout -b task3-hypothesis-testing

# Create feature branch for task 4
git checkout -b task4-predictive-models

# Merge to main after completion
git checkout main
git merge task3-hypothesis-testing --no-ff
git merge task4-predictive-models --no-ff
Commit Guidelines
bash
# Descriptive commit messages
git commit -m "TASK-3: Add ANOVA test for provincial risk differences"
git commit -m "TASK-4: Implement XGBoost model with SHAP interpretability"
git commit -m "TASK-4: Feature engineering - add UrbanDensityScore"
Running Tests
bash
# Run hypothesis testing unit tests
pytest tests/test_hypothesis_tests.py -v

# Run modeling unit tests
pytest tests/test_models.py -v

# Run all tests with coverage
pytest --cov=src tests/
📈 Output Files
Task 3 Outputs
text
reports/hypothesis_test_results/
├── statistical_test_results.csv
├── p_values_summary.xlsx
├── visualizations/
│   ├── province_risk_comparison.png
│   ├── gender_risk_differences.png
│   └── zipcode_margin_heatmap.png
└── business_recommendations.md
Task 4 Outputs
text
reports/model_performance/
├── model_comparison.csv
├── best_model_performance.json
├── feature_importance/
│   ├── xgboost_feature_importance.png
│   ├── shap_summary_plot.png
│   └── top_features_analysis.md
├── predictions/
│   ├── test_set_predictions.csv
│   └── validation_results.json
└── model_cards/
    ├── xgboost_model_card.md
    └── random_forest_model_card.md
🔧 Configuration
Model Configuration (config/model_config.yaml)
yaml
data:
  train_test_split: 0.8
  random_state: 42
  target_column: 'TotalClaims'
  
models:
  linear_regression:
    fit_intercept: true
    normalize: false
    
  random_forest:
    n_estimators: 100
    max_depth: 10
    min_samples_split: 5
    
  xgboost:
    n_estimators: 200
    max_depth: 7
    learning_rate: 0.1
    subsample: 0.8
    
evaluation:
  metrics: ['rmse', 'r2', 'mae']
  cross_validation_folds: 5
  
shap:
  n_samples: 100
  plot_type: 'bar'
📝 Reporting
Generate Complete Report
bash
# Run the reporting script
python src/reporting/generate_final_report.py \
    --hypothesis_results reports/hypothesis_test_results/ \
    --model_results reports/model_performance/ \
    --output reports/final_report/
Report Structure
Executive Summary: Key findings and recommendations

Hypothesis Testing Results: Statistical validation of risk drivers

Model Performance: Comparison of predictive models

Business Implications: Actionable insights for pricing strategy

Technical Details: Methodology and implementation notes

Appendix: Code snippets, additional visualizations

🚨 Troubleshooting
Common Issues
DVC Data Not Found

bash
# Pull data from remote storage
dvc pull

# Or reproduce the pipeline
dvc repro
Missing Dependencies

bash
# Update requirements
pip install -r requirements.txt --upgrade
Memory Issues with Large Datasets

python
# Use chunking in configuration
data:
  chunk_size: 10000
  use_dask: true  # For very large datasets
Debug Mode
bash
# Run with debug logging
python -m src.modeling.model_trainer --debug

# Or set environment variable
export LOG_LEVEL=DEBUG
🤝 Contributing
Code Standards
Follow PEP 8 for Python code

Add docstrings to all functions and classes

Write unit tests for new functionality

Update documentation when changing features

Pull Request Process
Create feature branch from main

Add tests for new functionality

Ensure all tests pass

Update README if needed

Create pull request with description