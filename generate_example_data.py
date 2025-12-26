"""
Generate example dataset for testing AutoML Agent
Creates a customer churn prediction dataset
"""

import pandas as pd
import numpy as np

# Set seed for reproducibility
np.random.seed(42)

# Number of samples
n_samples = 1000

# Generate features
data = {
    # Numerical features
    'age': np.random.randint(18, 70, n_samples),
    'income': np.random.randint(20000, 150000, n_samples),
    'credit_score': np.random.randint(300, 850, n_samples),
    'months_customer': np.random.randint(1, 120, n_samples),
    'num_products': np.random.randint(1, 5, n_samples),
    'balance': np.random.uniform(0, 100000, n_samples),

    # Categorical features
    'country': np.random.choice(['USA', 'UK', 'Germany', 'France', 'Spain'], n_samples),
    'gender': np.random.choice(['Male', 'Female'], n_samples),
    'has_credit_card': np.random.choice(['Yes', 'No'], n_samples),
    'is_active_member': np.random.choice(['Yes', 'No'], n_samples),
}

df = pd.DataFrame(data)

# Generate target based on features with some logic
churn_probability = (
    (df['age'] < 30).astype(int) * 0.15 +
    (df['credit_score'] < 500).astype(int) * 0.25 +
    (df['num_products'] == 1).astype(int) * 0.20 +
    (df['is_active_member'] == 'No').astype(int) * 0.30 +
    (df['balance'] < 10000).astype(int) * 0.10 +
    np.random.uniform(0, 0.3, n_samples)  # Add some randomness
)

# Convert to binary classification
df['churn'] = (churn_probability > 0.5).astype(int)

# Add some missing values to test preprocessing
missing_indices_income = np.random.choice(n_samples, size=50, replace=False)
missing_indices_balance = np.random.choice(n_samples, size=30, replace=False)
missing_indices_country = np.random.choice(n_samples, size=20, replace=False)

df.loc[missing_indices_income, 'income'] = np.nan
df.loc[missing_indices_balance, 'balance'] = np.nan
df.loc[missing_indices_country, 'country'] = np.nan

# Reorder columns (target last)
columns = [col for col in df.columns if col != 'churn'] + ['churn']
df = df[columns]

# Save to CSV
output_path = 'data/customer_churn.csv'
df.to_csv(output_path, index=False)

print(f"✓ Dataset created: {output_path}")
print(f"  Samples: {len(df)}")
print(f"  Features: {len(df.columns) - 1}")
print(f"  Target: churn (binary classification)")
print(f"\nTarget distribution:")
print(df['churn'].value_counts())
print(f"\nMissing values:")
print(df.isnull().sum()[df.isnull().sum() > 0])
print(f"\nFirst 5 rows:")
print(df.head())
