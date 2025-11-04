
# Import essential libraries
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set plot style
sns.set(style="whitegrid")


# Load dataset
df = pd.read_csv('../data/EV_Battery_Performance.csv')


# Display first few rows
df.index = range(1, len(df) + 1)
df.info()
df.head()

# Get basic info about the dataset
df.info()

# Shape of the dataset
print("Shape of dataset:", df.shape)

# Check column names
print("\nColumns in dataset:\n", df.columns.tolist())

# Descriptive statistics
df.describe().round(2)

# Missing values
print("\nMissing values in each column:\n", df.isnull().sum())

# Check for duplicate rows
print("\nNumber of duplicate rows:", df.duplicated().sum())

# Display unique counts for categorical features if any
for col in df.columns:
    if df[col].dtype == 'object':
        print(f"{col}: {df[col].nunique()} unique values")


