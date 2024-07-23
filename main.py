import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
import joblib

# Create directories for plots, tables, and models
os.makedirs('plots', exist_ok=True)
os.makedirs('tables', exist_ok=True)
os.makedirs('models', exist_ok=True)

# Load the data
df = pd.read_csv('cybersecurity_incidents.csv')

# Preprocess the data
df['date'] = pd.to_datetime(df['date'])
df['incident_type'] = pd.Categorical(df['incident_type']).codes
df['severity'] = pd.Categorical(df['severity'], categories=['low', 'medium', 'high', 'critical'], ordered=True).codes
df['patch_status'] = pd.Categorical(df['patch_status']).codes

# Create binary risk category (high risk if risk_score > 50)
df['high_risk'] = (df['risk_score'] > 50).astype(int)

# Select features for modeling
features = ['incident_type', 'severity', 'data_exposed', 'system_downtime_hours', 'financial_impact', 
            'detection_time_hours', 'resolution_time_hours', 'affected_systems', 'patch_status', 
            'employee_awareness_score', 'third_party_involved', 'regulatory_compliance_violated', 'repeated_incident']

X = df[features]
y = df['high_risk']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 1. Weighted Scoring Model
def weighted_score(row, weights):
    return sum(row * weights)

# Define weights for each feature (you should adjust these based on domain knowledge)
weights = [0.1, 0.15, 0.1, 0.1, 0.15, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05]

# Calculate risk scores
df['weighted_risk_score'] = df[features].apply(weighted_score, axis=1, weights=weights)

# Classify as high risk if score is above the mean
mean_score = df['weighted_risk_score'].mean()
df['weighted_high_risk'] = (df['weighted_risk_score'] > mean_score).astype(int)

# Evaluate
weighted_accuracy = (df['weighted_high_risk'] == df['high_risk']).mean()
print(f"Weighted Scoring Model Accuracy: {weighted_accuracy:.2f}")

# Save the weights of the Weighted Scoring Model
joblib.dump(weights, 'models/weighted_scoring_weights.joblib')

# 2. Logistic Regression Model
log_reg = LogisticRegression(random_state=42)
log_reg.fit(X_train_scaled, y_train)
y_pred_log = log_reg.predict(X_test_scaled)
y_pred_proba_log = log_reg.predict_proba(X_test_scaled)[:, 1]

print("\nLogistic Regression Results:")
print(confusion_matrix(y_test, y_pred_log))
print(classification_report(y_test, y_pred_log))

# Save the Logistic Regression model
joblib.dump(log_reg, 'models/logistic_regression_model.joblib')

# 3. Random Forest Model
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
y_pred_proba_rf = rf.predict_proba(X_test)[:, 1]

print("\nRandom Forest Results:")
print(confusion_matrix(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))

# Save the Random Forest model
joblib.dump(rf, 'models/random_forest_model.joblib')

# Feature importance
feature_importance = pd.DataFrame({'feature': features, 'importance': rf.feature_importances_})
feature_importance = feature_importance.sort_values('importance', ascending=False)
print("\nFeature Importance:")
print(feature_importance)

# Additional plotting functions
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure(figsize=(10, 6))
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")

    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.grid()
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
    plt.legend(loc="best")
    return plt

# Plotting
plt.figure(figsize=(20, 15))

# 1. Confusion Matrix for Logistic Regression
plt.subplot(2, 2, 1)
sns.heatmap(confusion_matrix(y_test, y_pred_log), annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix - Logistic Regression')
plt.ylabel('True label')
plt.xlabel('Predicted label')

# 2. Confusion Matrix for Random Forest
plt.subplot(2, 2, 2)
sns.heatmap(confusion_matrix(y_test, y_pred_rf), annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix - Random Forest')
plt.ylabel('True label')
plt.xlabel('Predicted label')

# 3. ROC Curve
plt.subplot(2, 2, 3)
fpr_log, tpr_log, _ = roc_curve(y_test, y_pred_proba_log)
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_pred_proba_rf)
plt.plot(fpr_log, tpr_log, label=f'Logistic Regression (AUC = {auc(fpr_log, tpr_log):.2f})')
plt.plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC = {auc(fpr_rf, tpr_rf):.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()

# 4. Feature Importance for Random Forest
plt.subplot(2, 2, 4)
feature_importance.plot(x='feature', y='importance', kind='bar')
plt.title('Feature Importance - Random Forest')
plt.xlabel('Features')
plt.ylabel('Importance')
plt.xticks(rotation=90)

plt.tight_layout()
plt.savefig('plots/model_comparison_plots.png')
plt.close()

# Distribution of Weighted Risk Scores
plt.figure(figsize=(10, 6))
sns.histplot(df['weighted_risk_score'], kde=True)
plt.axvline(mean_score, color='r', linestyle='--', label='Mean Score')
plt.title('Distribution of Weighted Risk Scores')
plt.xlabel('Risk Score')
plt.ylabel('Frequency')
plt.legend()
plt.savefig('plots/weighted_risk_scores_distribution.png')
plt.close()

# Feature Correlation Heatmap
plt.figure(figsize=(12, 10))
correlation_matrix = X.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5, cbar_kws={'label': 'Correlation'})
plt.title('Feature Correlation Heatmap')
plt.tight_layout()
plt.savefig('plots/feature_correlation_heatmap.png')
plt.close()

# Learning Curves
plot_learning_curve(log_reg, "Learning Curve (Logistic Regression)", X_train_scaled, y_train, cv=5)
plt.savefig('plots/learning_curve_logistic_regression.png')
plt.close()

plot_learning_curve(rf, "Learning Curve (Random Forest)", X_train, y_train, cv=5)
plt.savefig('plots/learning_curve_random_forest.png')
plt.close()

# Precision-Recall Curves
plt.figure(figsize=(10, 6))

# Calculate precision and recall for Logistic Regression
precision_log, recall_log, _ = precision_recall_curve(y_test, y_pred_proba_log)

# Calculate precision and recall for Random Forest
precision_rf, recall_rf, _ = precision_recall_curve(y_test, y_pred_proba_rf)

# Plot the curves
sns.lineplot(x=recall_log, y=precision_log, label='Logistic Regression')
sns.lineplot(x=recall_rf, y=precision_rf, label='Random Forest')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.savefig('plots/precision_recall_curve.png')
plt.close()

# Feature Importance Bar Plot (Separate plot for better visibility)
plt.figure(figsize=(12, 8))
sns.barplot(x='feature', y='importance', data=feature_importance)
plt.title('Feature Importance - Random Forest')
plt.xlabel('Features')
plt.ylabel('Importance')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('plots/feature_importance_bar_plot.png')
plt.close()

# Risk Score Distribution by Actual Risk Category
plt.figure(figsize=(10, 6))
sns.kdeplot(data=df[df['high_risk'] == 0]['weighted_risk_score'], fill=True, label='Low Risk')
sns.kdeplot(data=df[df['high_risk'] == 1]['weighted_risk_score'], fill=True, label='High Risk')
plt.axvline(mean_score, color='r', linestyle='--', label='Mean Score')
plt.title('Distribution of Risk Scores by Actual Risk Category')
plt.xlabel('Risk Score')
plt.ylabel('Density')
plt.legend()
plt.savefig('plots/risk_score_distribution_by_category.png')
plt.close()

# Generate tables for Chapter Four: Results & Discussions

# Table 1: Model Performance Comparison
model_performance = pd.DataFrame({
    'Model': ['Weighted Scoring', 'Logistic Regression', 'Random Forest'],
    'Accuracy': [
        weighted_accuracy,
        float(classification_report(y_test, y_pred_log, output_dict=True)['accuracy']),
        float(classification_report(y_test, y_pred_rf, output_dict=True)['accuracy'])
    ],
    'Precision': [
        'N/A',
        classification_report(y_test, y_pred_log, output_dict=True)['weighted avg']['precision'],
        classification_report(y_test, y_pred_rf, output_dict=True)['weighted avg']['precision']
    ],
    'Recall': [
        'N/A',
        classification_report(y_test, y_pred_log, output_dict=True)['weighted avg']['recall'],
        classification_report(y_test, y_pred_rf, output_dict=True)['weighted avg']['recall']
    ],
    'F1-Score': [
        'N/A',
        classification_report(y_test, y_pred_log, output_dict=True)['weighted avg']['f1-score'],
        classification_report(y_test, y_pred_rf, output_dict=True)['weighted avg']['f1-score']
    ]
})

print("\nTable 1: Model Performance Comparison")
print(model_performance.to_string(index=False))

# Table 2: Top 10 Most Important Features
top_10_features = feature_importance.head(10)
print("\nTable 2: Top 10 Most Important Features")
print(top_10_features.to_string(index=False))

# Table 3: Confusion Matrix for Logistic Regression
cm_log = confusion_matrix(y_test, y_pred_log)
cm_log_df = pd.DataFrame(cm_log, columns=['Predicted Negative', 'Predicted Positive'], 
                         index=['Actual Negative', 'Actual Positive'])
print("\nTable 3: Confusion Matrix for Logistic Regression")
print(cm_log_df.to_string())

# Table 4: Confusion Matrix for Random Forest
cm_rf = confusion_matrix(y_test, y_pred_rf)
cm_rf_df = pd.DataFrame(cm_rf, columns=['Predicted Negative', 'Predicted Positive'], 
                        index=['Actual Negative', 'Actual Positive'])
print("\nTable 4: Confusion Matrix for Random Forest")
print(cm_rf_df.to_string())

# Save tables to CSV files
model_performance.to_csv('tables/model_performance_comparison.csv', index=False)
top_10_features.to_csv('tables/top_10_important_features.csv', index=False)
cm_log_df.to_csv('tables/confusion_matrix_logistic_regression.csv')
cm_rf_df.to_csv('tables/confusion_matrix_random_forest.csv')
