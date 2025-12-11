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
```python
"""
Statistical hypothesis testing module for insurance risk analysis.
"""
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import ttest_ind, chi2_contingency, f_oneway, kruskal
from statsmodels.stats.proportion import proportions_ztest
from typing import Dict, Tuple, List, Any
import warnings
warnings.filterwarnings('ignore')

class HypothesisTester:
    """Class for conducting statistical hypothesis tests on insurance data."""
    
    def __init__(self, data: pd.DataFrame, alpha: float = 0.05):
        """
        Initialize HypothesisTester.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Cleaned insurance data
        alpha : float
            Significance level (default: 0.05)
        """
        self.data = data.copy()
        self.alpha = alpha
        self.results = {}
        
    def calculate_metrics(self) -> Dict[str, float]:
        """Calculate key risk metrics."""
        # Claim frequency
        claim_freq = self.data['HasClaim'].mean()
        
        # Claim severity (only for policies with claims)
        claim_data = self.data[self.data['TotalClaims'] > 0]
        claim_severity = claim_data['TotalClaims'].mean()
        
        # Margin
        self.data['Margin'] = self.data['TotalPremium'] - self.data['TotalClaims']
        avg_margin = self.data['Margin'].mean()
        
        # Loss ratio
        total_claims = self.data['TotalClaims'].sum()
        total_premium = self.data['TotalPremium'].sum()
        loss_ratio = total_claims / total_premium if total_premium > 0 else np.nan
        
        return {
            'claim_frequency': claim_freq,
            'claim_severity': claim_severity,
            'avg_margin': avg_margin,
            'loss_ratio': loss_ratio
        }
    
    def test_province_risk_differences(self) -> Dict[str, Any]:
        """
        Test H0: No risk differences across provinces.
        
        Returns:
        --------
        dict with test results
        """
        # Group by province
        provinces = self.data['Province'].unique()
        
        # Test 1: Claim frequency (chi-square)
        contingency_table = pd.crosstab(
            self.data['Province'], 
            self.data['HasClaim']
        )
        chi2, p_freq, dof, expected = chi2_contingency(contingency_table)
        
        # Test 2: Claim severity (ANOVA)
        severity_groups = [
            self.data[self.data['Province'] == prov]['TotalClaims'].values
            for prov in provinces
            if len(self.data[self.data['Province'] == prov]) > 30
        ]
        f_stat, p_sev = f_oneway(*severity_groups)
        
        # Calculate provincial loss ratios
        province_stats = []
        for province in provinces:
            province_data = self.data[self.data['Province'] == province]
            if len(province_data) > 0:
                loss_ratio = province_data['TotalClaims'].sum() / province_data['TotalPremium'].sum()
                claim_freq = province_data['HasClaim'].mean()
                province_stats.append({
                    'province': province,
                    'loss_ratio': loss_ratio,
                    'claim_frequency': claim_freq,
                    'n_policies': len(province_data)
                })
        
        result = {
            'test_name': 'province_risk_differences',
            'null_hypothesis': 'No risk differences across provinces',
            'p_value_frequency': p_freq,
            'p_value_severity': p_sev,
            'chi2_statistic': chi2,
            'f_statistic': f_stat,
            'reject_null': p_freq < self.alpha or p_sev < self.alpha,
            'province_statistics': pd.DataFrame(province_stats),
            'recommendation': self._generate_province_recommendation(pd.DataFrame(province_stats))
        }
        
        self.results['province_test'] = result
        return result
    
    def test_zipcode_risk_differences(self) -> Dict[str, Any]:
        """
        Test H0: No risk differences between zip codes.
        Using top 10% vs bottom 10% by claim frequency.
        """
        # Calculate claim frequency by zip code
        zip_stats = self.data.groupby('ZipCode').agg({
            'HasClaim': 'mean',
            'TotalClaims': 'mean',
            'PolicyID': 'count'
        }).rename(columns={'PolicyID': 'count', 'HasClaim': 'claim_freq'})
        
        # Filter zip codes with sufficient data
        zip_stats = zip_stats[zip_stats['count'] >= 10]
        
        if len(zip_stats) < 20:
            raise ValueError("Insufficient zip codes for comparison")
        
        # Get top and bottom 10% by claim frequency
        n_top = max(1, int(len(zip_stats) * 0.1))
        top_zipcodes = zip_stats.nlargest(n_top, 'claim_freq').index.tolist()
        bottom_zipcodes = zip_stats.nsmallest(n_top, 'claim_freq').index.tolist()
        
        # Extract data for groups
        top_data = self.data[self.data['ZipCode'].isin(top_zipcodes)]
        bottom_data = self.data[self.data['ZipCode'].isin(bottom_zipcodes)]
        
        # Test claim severity difference (Mann-Whitney U test)
        u_stat, p_severity = stats.mannwhitneyu(
            top_data[top_data['TotalClaims'] > 0]['TotalClaims'],
            bottom_data[bottom_data['TotalClaims'] > 0]['TotalClaims'],
            alternative='two-sided'
        )
        
        # Test claim frequency difference (proportion test)
        n1, n2 = len(top_data), len(bottom_data)
        x1, x2 = top_data['HasClaim'].sum(), bottom_data['HasClaim'].sum()
        
        z_stat, p_frequency = proportions_ztest(
            [x1, x2], 
            [n1, n2], 
            alternative='two-sided'
        )
        
        result = {
            'test_name': 'zipcode_risk_differences',
            'null_hypothesis': 'No risk differences between zip codes',
            'p_value_frequency': p_frequency,
            'p_value_severity': p_severity,
            'z_statistic': z_stat,
            'u_statistic': u_stat,
            'reject_null': p_frequency < self.alpha or p_severity < self.alpha,
            'top_zipcodes_avg_freq': top_data['HasClaim'].mean(),
            'bottom_zipcodes_avg_freq': bottom_data['HasClaim'].mean(),
            'top_zipcodes_avg_severity': top_data[top_data['TotalClaims'] > 0]['TotalClaims'].mean(),
            'bottom_zipcodes_avg_severity': bottom_data[bottom_data['TotalClaims'] > 0]['TotalClaims'].mean(),
            'risk_ratio': top_data['HasClaim'].mean() / bottom_data['HasClaim'].mean() if bottom_data['HasClaim'].mean() > 0 else np.inf
        }
        
        self.results['zipcode_test'] = result
        return result
    
    def test_zipcode_margin_differences(self) -> Dict[str, Any]:
        """
        Test H0: No significant margin difference between zip codes.
        """
        # Calculate average margin by zip code
        zip_margins = self.data.groupby('ZipCode').agg({
            'Margin': 'mean',
            'PolicyID': 'count'
        }).rename(columns={'PolicyID': 'count'})
        
        # Filter for zip codes with sufficient data
        zip_margins = zip_margins[zip_margins['count'] >= 15]
        
        if len(zip_margins) < 2:
            raise ValueError("Insufficient zip codes for margin comparison")
        
        # Get top and bottom quartiles by margin
        n_quartile = max(1, len(zip_margins) // 4)
        top_zips = zip_margins.nlargest(n_quartile, 'Margin').index.tolist()
        bottom_zips = zip_margins.nsmallest(n_quartile, 'Margin').index.tolist()
        
        # Extract data
        top_data = self.data[self.data['ZipCode'].isin(top_zips)]
        bottom_data = self.data[self.data['ZipCode'].isin(bottom_zips)]
        
        # T-test for margin difference
        t_stat, p_value = ttest_ind(
            top_data['Margin'],
            bottom_data['Margin'],
            equal_var=False  # Welch's t-test
        )
        
        result = {
            'test_name': 'zipcode_margin_differences',
            'null_hypothesis': 'No significant margin difference between zip codes',
            'p_value': p_value,
            't_statistic': t_stat,
            'reject_null': p_value < self.alpha,
            'avg_margin_top': top_data['Margin'].mean(),
            'avg_margin_bottom': bottom_data['Margin'].mean(),
            'margin_difference': top_data['Margin'].mean() - bottom_data['Margin'].mean(),
            'margin_ratio': top_data['Margin'].mean() / bottom_data['Margin'].mean() if bottom_data['Margin'].mean() > 0 else np.inf
        }
        
        self.results['margin_test'] = result
        return result
    
    def test_gender_risk_differences(self) -> Dict[str, Any]:
        """
        Test H0: No significant risk difference between women and men.
        """
        # Filter for known genders
        gender_data = self.data[self.data['Gender'].isin(['Male', 'Female'])]
        
        if len(gender_data) < 100:
            raise ValueError("Insufficient gender data for comparison")
        
        # Group by gender
        male_data = gender_data[gender_data['Gender'] == 'Male']
        female_data = gender_data[gender_data['Gender'] == 'Female']
        
        # Test 1: Claim frequency (proportion test)
        n_male, n_female = len(male_data), len(female_data)
        x_male, x_female = male_data['HasClaim'].sum(), female_data['HasClaim'].sum()
        
        z_stat, p_frequency = proportions_ztest(
            [x_male, x_female],
            [n_male, n_female],
            alternative='two-sided'
        )
        
        # Test 2: Claim severity (t-test)
        male_claims = male_data[male_data['TotalClaims'] > 0]['TotalClaims']
        female_claims = female_data[female_data['TotalClaims'] > 0]['TotalClaims']
        
        if len(male_claims) > 5 and len(female_claims) > 5:
            t_stat, p_severity = ttest_ind(male_claims, female_claims, equal_var=False)
        else:
            t_stat, p_severity = np.nan, np.nan
        
        # Test 3: Loss ratio
        male_loss_ratio = male_data['TotalClaims'].sum() / male_data['TotalPremium'].sum()
        female_loss_ratio = female_data['TotalClaims'].sum() / female_data['TotalPremium'].sum()
        
        result = {
            'test_name': 'gender_risk_differences',
            'null_hypothesis': 'No significant risk difference between women and men',
            'p_value_frequency': p_frequency,
            'p_value_severity': p_severity,
            'z_statistic': z_stat,
            't_statistic': t_stat,
            'reject_null': p_frequency < self.alpha,
            'male_claim_freq': male_data['HasClaim'].mean(),
            'female_claim_freq': female_data['HasClaim'].mean(),
            'male_loss_ratio': male_loss_ratio,
            'female_loss_ratio': female_loss_ratio,
            'risk_difference': male_data['HasClaim'].mean() - female_data['HasClaim'].mean(),
            'relative_risk': male_data['HasClaim'].mean() / female_data['HasClaim'].mean() if female_data['HasClaim'].mean() > 0 else np.inf
        }
        
        self.results['gender_test'] = result
        return result
    
    def run_all_tests(self) -> Dict[str, Dict]:
        """Run all hypothesis tests."""
        print("Running all hypothesis tests...")
        
        tests = [
            self.test_province_risk_differences,
            self.test_zipcode_risk_differences,
            self.test_zipcode_margin_differences,
            self.test_gender_risk_differences
        ]
        
        for test_func in tests:
            try:
                test_func()
                print(f"✓ Completed: {test_func.__name__}")
            except Exception as e:
                print(f"✗ Failed {test_func.__name__}: {str(e)}")
        
        return self.results
    
    def _generate_province_recommendation(self, province_stats: pd.DataFrame) -> str:
        """Generate business recommendation based on province analysis."""
        if len(province_stats) < 2:
            return "Insufficient province data for recommendation"
        
        # Find highest and lowest risk provinces
        highest_risk = province_stats.loc[province_stats['loss_ratio'].idxmax()]
        lowest_risk = province_stats.loc[province_stats['loss_ratio'].idxmin()]
        
        risk_ratio = highest_risk['loss_ratio'] / lowest_risk['loss_ratio']
        
        if risk_ratio > 1.3:
            return (f"Strong evidence for regional pricing: {highest_risk['province']} has "
                   f"{risk_ratio:.1f}x higher loss ratio than {lowest_risk['province']}. "
                   f"Consider premium adjustments of 15-25% for high-risk regions.")
        elif risk_ratio > 1.1:
            return (f"Moderate regional differences detected. Consider tiered pricing "
                   f"with 5-15% adjustments between regions.")
        else:
            return "Minimal regional differences. Focus on other risk factors."
    
    def generate_report(self, output_path: str = './reports/hypothesis_tests/'):
        """Generate comprehensive test report."""
        import os
        os.makedirs(output_path, exist_ok=True)
        
        # Summary DataFrame
        summary_data = []
        for test_name, result in self.results.items():
            summary_data.append({
                'Test': result['test_name'].replace('_', ' ').title(),
                'Null Hypothesis': result['null_hypothesis'],
                'Key p-value': min(result.get('p_value', 1), 
                                 result.get('p_value_frequency', 1)),
                'Reject H0?': 'Yes' if result['reject_null'] else 'No',
                'Evidence Strength': self._get_evidence_strength(result)
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(f'{output_path}/hypothesis_test_summary.csv', index=False)
        
        # Detailed results
        with open(f'{output_path}/detailed_results.txt', 'w') as f:
            f.write("HYPOTHESIS TESTING RESULTS\n")
            f.write("=" * 50 + "\n\n")
            
            for test_name, result in self.results.items():
                f.write(f"\n{result['test_name'].replace('_', ' ').upper()}\n")
                f.write("-" * 40 + "\n")
                f.write(f"Null Hypothesis: {result['null_hypothesis']}\n")
                
                # Add p-values
                for key, value in result.items():
                    if 'p_value' in key:
                        f.write(f"{key}: {value:.6f}\n")
                        f.write(f"Significant at α={self.alpha}: {'YES' if value < self.alpha else 'NO'}\n")
                
                f.write(f"Decision: {'REJECT' if result['reject_null'] else 'FAIL TO REJECT'} null hypothesis\n")
                
                # Add business implications
                if 'recommendation' in result:
                    f.write(f"\nBusiness Recommendation: {result['recommendation']}\n")
                
                f.write("\n" + "=" * 40 + "\n")
        
        print(f"Report generated at: {output_path}")
        return summary_df
    
    def _get_evidence_strength(self, result: Dict) -> str:
        """Get evidence strength description."""
        p_value = min(result.get('p_value', 1), 
                     result.get('p_value_frequency', 1))
        
        if p_value < 0.01:
            return 'Very Strong'
        elif p_value < 0.05:
            return 'Strong'
        elif p_value < 0.1:
            return 'Moderate'
        else:
            return 'Weak'

```
```python

"""
Model training and evaluation module for insurance predictive modeling.
"""
import pandas as pd
import numpy as np
from typing import Dict, Tuple, List, Any, Optional
import pickle
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Model imports
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import xgboost as xgb

# For interpretability
import shap

class InsuranceModelTrainer:
    """
    Train and evaluate models for insurance claim prediction and premium optimization.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize model trainer.
        
        Parameters:
        -----------
        config_path : str, optional
            Path to configuration file
        """
        self.config = self._load_config(config_path)
        self.models = {}
        self.results = {}
        self.feature_importance = {}
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_names = None
        
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Load configuration from file or use defaults."""
        default_config = {
            'data': {
                'test_size': 0.2,
                'random_state': 42,
                'target_regression': 'TotalClaims',
                'target_classification': 'HasClaim'
            },
            'models': {
                'linear_regression': {
                    'fit_intercept': True,
                    'normalize': False
                },
                'random_forest_reg': {
                    'n_estimators': 100,
                    'max_depth': 10,
                    'min_samples_split': 5,
                    'random_state': 42
                },
                'xgboost_reg': {
                    'n_estimators': 200,
                    'max_depth': 7,
                    'learning_rate': 0.1,
                    'subsample': 0.8,
                    'random_state': 42
                },
                'random_forest_clf': {
                    'n_estimators': 100,
                    'max_depth': 8,
                    'random_state': 42
                }
            },
            'evaluation': {
                'cv_folds': 5,
                'scoring_regression': 'neg_mean_squared_error',
                'scoring_classification': 'roc_auc'
            },
            'shap': {
                'n_samples': 100,
                'max_display': 10
            }
        }
        
        if config_path:
            import yaml
            with open(config_path, 'r') as f:
                user_config = yaml.safe_load(f)
                # Merge with defaults
                default_config.update(user_config)
        
        return default_config
    
    def prepare_data(self, data: pd.DataFrame, task: str = 'regression') -> Tuple:
        """
        Prepare data for modeling.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Input data
        task : str
            'regression' for claim severity, 'classification' for claim probability
            
        Returns:
        --------
        X_train, X_test, y_train, y_test, feature_names
        """
        # Make copy to avoid modifying original
        df = data.copy()
        
        # Feature engineering
        df = self._engineer_features(df)
        
        # Define features and target
        if task == 'regression':
            # For claim severity prediction, only use policies with claims
            df = df[df['TotalClaims'] > 0].copy()
            target = self.config['data']['target_regression']
        else:  # classification
            target = self.config['data']['target_classification']
            if target not in df.columns:
                df['HasClaim'] = (df['TotalClaims'] > 0).astype(int)
        
        # Define feature columns
        numerical_features = [
            'VehicleAge', 'CustomValueEstimate', 
            'PremiumToValueRatio', 'UrbanDensityScore',
            'PreviousClaimsCount', 'DriverAge'
        ]
        
        categorical_features = [
            'Province', 'VehicleType', 'VehicleMake',
            'CoverType', 'Gender'
        ]
        
        # Check which features exist in data
        numerical_features = [f for f in numerical_features if f in df.columns]
        categorical_features = [f for f in categorical_features if f in df.columns]
        
        all_features = numerical_features + categorical_features
        self.feature_names = all_features
        
        # Split features and target
        X = df[all_features]
        y = df[target]
        
        # Train-test split
        test_size = self.config['data']['test_size']
        random_state = self.config['data']['random_state']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, 
            stratify=y if task == 'classification' else None
        )
        
        # Preprocessing pipeline
        numeric_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))
        ])
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numerical_features),
                ('cat', categorical_transformer, categorical_features)
            ]
        )
        
        # Fit and transform
        X_train_processed = preprocessor.fit_transform(X_train)
        X_test_processed = preprocessor.transform(X_test)
        
        # Get feature names after one-hot encoding
        if categorical_features:
            ohe = preprocessor.named_transformers_['cat'].named_steps['onehot']
            cat_feature_names = ohe.get_feature_names_out(categorical_features)
            all_feature_names = numerical_features + list(cat_feature_names)
        else:
            all_feature_names = numerical_features
        
        self.preprocessor = preprocessor
        self.feature_names_processed = all_feature_names
        
        self.X_train = X_train_processed
        self.X_test = X_test_processed
        self.y_train = y_train
        self.y_test = y_test
        
        print(f"Data prepared for {task}:")
        print(f"  Training samples: {len(X_train_processed)}")
        print(f"  Test samples: {len(X_test_processed)}")
        print(f"  Features: {len(all_feature_names)}")
        
        return X_train_processed, X_test_processed, y_train, y_test, all_feature_names
    
    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer new features for modeling."""
        df_eng = df.copy()
        
        # Vehicle age
        if 'VehicleYear' in df_eng.columns:
            current_year = datetime.now().year
            df_eng['VehicleAge'] = current_year - df_eng['VehicleYear']
        
        # Premium to value ratio
        if 'TotalPremium' in df_eng.columns and 'CustomValueEstimate' in df_eng.columns:
            df_eng['PremiumToValueRatio'] = df_eng['TotalPremium'] / df_eng['CustomValueEstimate']
            df_eng['PremiumToValueRatio'].replace([np.inf, -np.inf], np.nan, inplace=True)
        
        # Urban density score (simplified)
        if 'ZipCode' in df_eng.columns:
            # Create a simple urban score based on zip code frequency
            zip_freq = df_eng['ZipCode'].value_counts(normalize=True)
            df_eng['UrbanDensityScore'] = df_eng['ZipCode'].map(zip_freq)
        
        # Previous claims count (placeholder - would need historical data)
        df_eng['PreviousClaimsCount'] = 0
        
        # Driver age if available
        if 'DateOfBirth' in df_eng.columns:
            df_eng['DriverAge'] = (pd.to_datetime('2023-01-01') - pd.to_datetime(df_eng['DateOfBirth'])).dt.days / 365.25
        
        # Fill missing values
        numeric_cols = df_eng.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df_eng[col].isnull().any():
                df_eng[col].fillna(df_eng[col].median(), inplace=True)
        
        # Fill categorical missing values
        categorical_cols = df_eng.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df_eng[col].isnull().any():
                df_eng[col].fillna('Unknown', inplace=True)
        
        return df_eng
    
    def train_linear_regression(self) -> LinearRegression:
        """Train linear regression model."""
        print("Training Linear Regression...")
        
        config = self.config['models']['linear_regression']
        model = LinearRegression(
            fit_intercept=config['fit_intercept']
        )
        
        model.fit(self.X_train, self.y_train)
        self.models['linear_regression'] = model
        
        # Evaluate
        train_pred = model.predict(self.X_train)
        test_pred = model.predict(self.X_test)
        
        metrics = self._calculate_regression_metrics(
            self.y_train, self.y_test, train_pred, test_pred
        )
        
        self.results['linear_regression'] = metrics
        print(f"  Test R²: {metrics['test_r2']:.4f}")
        print(f"  Test RMSE: {metrics['test_rmse']:.2f}")
        
        return model
    
    def train_random_forest_regressor(self) -> RandomForestRegressor:
        """Train random forest regressor."""
        print("Training Random Forest Regressor...")
        
        config = self.config['models']['random_forest_reg']
        model = RandomForestRegressor(
            n_estimators=config['n_estimators'],
            max_depth=config['max_depth'],
            min_samples_split=config['min_samples_split'],
            random_state=config['random_state'],
            n_jobs=-1
        )
        
        model.fit(self.X_train, self.y_train)
        self.models['random_forest_reg'] = model
        
        # Evaluate
        train_pred = model.predict(self.X_train)
        test_pred = model.predict(self.X_test)
        
        metrics = self._calculate_regression_metrics(
            self.y_train, self.y_test, train_pred, test_pred
        )
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': self.feature_names_processed,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        self.feature_importance['random_forest_reg'] = feature_importance
        self.results['random_forest_reg'] = metrics
        
        print(f"  Test R²: {metrics['test_r2']:.4f}")
        print(f"  Test RMSE: {metrics['test_rmse']:.2f}")
        
        return model
    
    def train_xgboost_regressor(self) -> xgb.XGBRegressor:
        """Train XGBoost regressor."""
        print("Training XGBoost Regressor...")
        
        config = self.config['models']['xgboost_reg']
        model = xgb.XGBRegressor(
            n_estimators=config['n_estimators'],
            max_depth=config['max_depth'],
            learning_rate=config['learning_rate'],
            subsample=config['subsample'],
            random_state=config['random_state'],
            n_jobs=-1
        )
        
        model.fit(self.X_train, self.y_train)
        self.models['xgboost_reg'] = model
        
        # Evaluate
        train_pred = model.predict(self.X_train)
        test_pred = model.predict(self.X_test)
        
        metrics = self._calculate_regression_metrics(
            self.y_train, self.y_test, train_pred, test_pred
        )
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': self.feature_names_processed,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        self.feature_importance['xgboost_reg'] = feature_importance
        self.results['xgboost_reg'] = metrics
        
        print(f"  Test R²: {metrics['test_r2']:.4f}")
        print(f"  Test RMSE: {metrics['test_rmse']:.2f}")
        
        return model
    
    def train_random_forest_classifier(self) -> RandomForestClassifier:
        """Train random forest classifier for claim probability."""
        print("Training Random Forest Classifier (Claim Probability)...")
        
        # Prepare data for classification
        if self.X_train is None:
            raise ValueError("Data not prepared. Call prepare_data() first with task='classification'")
        
        config = self.config['models']['random_forest_clf']
        model = RandomForestClassifier(
            n_estimators=config['n_estimators'],
            max_depth=config['max_depth'],
            random_state=config['random_state'],
            n_jobs=-1,
            class_weight='balanced'
        )
        
        model.fit(self.X_train, self.y_train)
        self.models['random_forest_clf'] = model
        
        # Evaluate
        train_pred_proba = model.predict_proba(self.X_train)[:, 1]
        test_pred_proba = model.predict_proba(self.X_test)[:, 1]
        test_pred = model.predict(self.X_test)
        
        metrics = self._calculate_classification_metrics(
            self.y_train, self.y_test, train_pred_proba, test_pred_proba, test_pred
        )
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': self.feature_names_processed,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        self.feature_importance['random_forest_clf'] = feature_importance
        self.results['random_forest_clf'] = metrics
        
        print(f"  Test Accuracy: {metrics['test_accuracy']:.4f}")
        print(f"  Test AUC-ROC: {metrics['test_auc_roc']:.4f}")
        
        return model
    
    def _calculate_regression_metrics(self, y_train, y_test, train_pred, test_pred) -> Dict:
        """Calculate regression metrics."""
       

```

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