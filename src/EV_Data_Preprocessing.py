
# Import Required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load Dataset
df = pd.read_csv('../data/EV_Battery_Performance.csv')
df.index = range(1, len(df) + 1)

df.head()

# Check Missing & Duplicate Values
print("\nMissing Values:\n", df.isnull().sum())
print("\nDuplicate Rows:", df.duplicated().sum())

# Drop Unnecessary Columns
df.drop(columns=['source_url', 'fast_charge_port', 'seats'], inplace=True, errors='ignore')
print("Irrelevant columns dropped")

# Handle Missing Values (Preserve Data Types)
for col in df.select_dtypes(include=['float64', 'int64']).columns:
    if df[col].isnull().sum() > 0:
        mean_val = df[col].mean()
        if df[col].dtype == 'int64':
            df[col] = df[col].fillna(int(round(mean_val)))
        else:
            df[col] = df[col].fillna(round(mean_val, 2))


# For categorical columns — fill with mode
for col in df.select_dtypes(include=['object']).columns:
    if df[col].isnull().sum() > 0:
        mode_val = df[col].mode()[0]
        df[col] = df[col].fillna(mode_val)

# Remove Duplicate Rows
initial_len = len(df)
df.drop_duplicates(inplace=True)
print(f"Duplicates removed: {initial_len - len(df)} rows dropped")

# Handle Outliers using IQR (Preserving Data Type)
for col in df.select_dtypes(include=['float64', 'int64']).columns:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)

# Replace Zero Values in Important Columns with Mean
numeric_cols = [
    'battery_capacity_kWh', 'number_of_cells', 'torque_nm',
    'acceleration_0_100_s', 'fast_charging_power_kw_dc', 'towing_capacity_kg'
]

for col in numeric_cols:
    if col in df.columns:
        mean_val = df.loc[df[col] != 0, col].mean()
        if df[col].dtype == 'int64':
            df[col] = df[col].replace(0, int(round(mean_val)))
        else:
            df[col] = df[col].replace(0, round(mean_val, 2))


# Confirm Data Types & Info
df.info()

# List Categorical Columns
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
print("Categorical Columns:", categorical_cols)

print(df[categorical_cols].head())

# Final Check for Missing Values
print("Remaining Missing Values:\n", df.isnull().sum())

# Save the Cleaned Dataset
df.to_csv('../data/EV_Battery_Performance_Clean.csv', index=False)
print("Cleaned dataset saved as: EV_Battery_Performance_Clean.csv")



