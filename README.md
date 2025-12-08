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