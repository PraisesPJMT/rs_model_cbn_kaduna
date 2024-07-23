import pandas as pd
from datetime import datetime

# Read the CSV file
df = pd.read_csv('cybersecurity_incidents_dataset.csv')

# Convert 'date' column to datetime format
df['date'] = pd.to_datetime(df['date'])

# Handle missing values for numerical columns
numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].mean())

# Handle missing values for categorical columns
categorical_cols = df.select_dtypes(include=['object', 'category']).columns
df[categorical_cols] = df[categorical_cols].fillna(df[categorical_cols].mode().iloc[0])

# Handle outliers
df['data_exposed'] = df['data_exposed'].clip(lower=0)
df['system_downtime_hours'] = df['system_downtime_hours'].clip(upper=72)
df['financial_impact'] = df['financial_impact'].clip(lower=0)
df['detection_time_hours'] = df['detection_time_hours'].clip(lower=0)
df['resolution_time_hours'] = df['resolution_time_hours'].clip(lower=0)
df['affected_systems'] = df['affected_systems'].clip(lower=1)
df['risk_score'] = df['risk_score'].clip(lower=0, upper=100)

# Encode categorical variables
for col in ['incident_type', 'severity', 'patch_status']:
    df[col] = pd.Categorical(df[col]).codes

# Save the cleaned dataset to a new CSV file
df.to_csv('cleaned_cybersecurity_incidents.csv', index=False)

print("Dataset has been cleaned and saved as 'cleaned_cybersecurity_incidents.csv'")
