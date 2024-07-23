# Development of an Enhanced Risk Scoring Model for Cybersecurity Incidents in Banks: A Case Study of the Central Bank of Nigeria

## Thesis Project

> This project involves developing three risk scoring methods (Weighted Scoring Model, Logistic Regression Model, and Random Forest Model) using private datasets from the network logs of the Central Bank of Nigeria. The goal is to enhance the risk scoring model for cybersecurity incidents.

## Table of Contents

- [Development of an Enhanced Risk Scoring Model for Cybersecurity Incidents in Banks: A Case Study of the Central Bank of Nigeria](#development-of-an-enhanced-risk-scoring-model-for-cybersecurity-incidents-in-banks-a-case-study-of-the-central-bank-of-nigeria)
  - [Thesis Project](#thesis-project)
  - [Table of Contents](#table-of-contents)
  - [Introduction](#introduction)
  - [Project Structure](#project-structure)
  - [Setup and Installation](#setup-and-installation)
  - [Data Preprocessing](#data-preprocessing)
  - [Model Development](#model-development)
  - [Model Evaluation](#model-evaluation)
  - [Plots and Results](#plots-and-results)
  - [Saved Models and Tables](#saved-models-and-tables)
  - [Acknowledgments](#acknowledgments)
  - [Contact](#contact)

## Introduction

> This project aims to improve the accuracy and reliability of risk scoring models for cybersecurity incidents in banks. The project involves:

1. Developing a Weighted Scoring Model.
2. Building and training Logistic Regression and Random Forest models.
3. Evaluating the performance of these models using various metrics.
4. Visualizing the results and comparing the models.

## Project Structure

```
├── data
│ ├── cybersecurity_incidents.csv # Dataset used for training and evaluation
├── models
│ ├── logistic_regression_model.joblib # Trained Logistic Regression model
│ ├── random_forest_model.joblib # Trained Random Forest model
│ ├── weighted_scoring_weights.joblib # Weights for Weighted Scoring model
├── plots
│ ├── feature_correlation_heatmap.png # Correlation heatmap of features
│ ├── learning_curve_logistic_regression.png # Learning curve for Logistic Regression
│ ├── learning_curve_random_forest.png # Learning curve for Random Forest
│ ├── model_comparison_plots.png # Comparison plots for models
│ ├── precision_recall_curve.png # Precision-Recall curve
│ ├── risk_score_distribution_by_category.png # Risk score distribution by category
│ ├── weighted_risk_scores_distribution.png # Distribution of weighted risk scores
│ ├── feature_importance_bar_plot.png # Feature importance bar plot
├── tables
│ ├── confusion_matrix_logistic_regression.csv # Confusion matrix for Logistic Regression
│ ├── confusion_matrix_random_forest.csv # Confusion matrix for Random Forest
│ ├── model_performance_comparison.csv # Model performance comparison table
│ ├── top_10_important_features.csv # Top 10 important features table
├── main.py # Main script for running the project
├── README.md # Project readme file
```

## Setup and Installation

1. **Clone the repository:**

   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. **Install the required packages:**

   ```bash
   pip install -r requirements.txt
   ```

3. **Ensure the dataset is placed in the `data` directory:**
   - `cybersecurity_incidents.csv`

## Data Preprocessing

The data preprocessing involves:

1. Converting the `date` column to datetime format.
2. Encoding categorical features (`incident_type`, `severity`, `patch_status`).
3. Creating a binary `high_risk` column based on `risk_score`.

## Model Development

Three models are developed:

1. **Weighted Scoring Model:**

   - A manual weighting approach where each feature is assigned a specific weight.
   - Risk scores are calculated based on these weights.

2. **Logistic Regression Model:**

   - A statistical model that uses a logistic function to model a binary dependent variable.
   - Trained using the `scikit-learn` library.

3. **Random Forest Model:**
   - An ensemble learning method using multiple decision trees.
   - Trained using the `scikit-learn` library.

## Model Evaluation

The models are evaluated based on:

1. **Confusion Matrix:** To visualize the performance.
2. **Classification Report:** To obtain precision, recall, and F1-score.
3. **ROC Curve and AUC:** To compare the models' ability to discriminate between classes.
4. **Precision-Recall Curve:** To visualize the trade-off between precision and recall.

## Plots and Results

Various plots are generated to visualize the results:

1. **Confusion Matrices:** For both Logistic Regression and Random Forest models.
2. **ROC Curve:** For both models.
3. **Feature Importance:** For the Random Forest model.
4. **Learning Curves:** To show the training and cross-validation scores.
5. **Risk Score Distributions:** To visualize the distribution of risk scores.

## Saved Models and Tables

- Trained models and their weights are saved in the `models` directory.
- Generated tables for the results and discussions are saved in the `tables` directory.

## Acknowledgments

This project was developed as part of a thesis titled 'Development of an Enhanced Risk Scoring Model for Cybersecurity Incidents in Banks: A Case Study of the Central Bank of Nigeria.' Special thanks to my advisors and the Central Bank of Nigeria for providing the dataset and valuable insights.

## Contact

For any questions or suggestions, please contact:

- Name: [Your Name]
- Email: [Your Email]
