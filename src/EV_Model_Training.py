# Import Required Libraries

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Load the cleaned dataset

df = pd.read_csv('../data/EV_Battery_Performance_Clean.csv')
df.head()

# Select Relevant Columns

features = ['battery_capacity_kWh', 'efficiency_wh_per_km', 'motor_power_kw',
            'top_speed_kmh', 'torque_nm', 'acceleration_0_100_s',
            'battery_type', 'drivetrain', 'car_body_type', 'brand']
target = 'range_km'

data = df[features + [target]].dropna()

data.head()

# Encode categorical columns

categorical_cols = df.select_dtypes(include=['object']).columns

# Label Encoding
le = LabelEncoder()
for col in categorical_cols:
    df[col] = le.fit_transform(df[col].astype(str))

# Define features (X) and target (y)

X = df.drop(columns=['range_km'])  # Independent variables
y = df['range_km']                 # Target variable

# Train-Test Split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Feature Scaling

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Multiple Models and Save Best One

models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(),
    "Lasso Regression": Lasso(),
    "Random Forest": RandomForestRegressor(random_state=42)
}

results = []
best_model = None
best_score = -np.inf  # Initialize with a very low score

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    results.append({
        "Model": name,
        "R2 Score": r2,
        "MAE": mae,
        "RMSE": rmse
    })
    
    # Save best model based on R²
    if r2 > best_score:
        best_score = r2
        best_model = model
        best_model_name = name

# Display Model Performance

results_df = pd.DataFrame(results)
print("\nModel Evaluation Results:\n")
print(results_df)

# Visualize Model Comparison

plt.figure(figsize=(8, 5))
sns.barplot(x='Model', y='R2 Score', hue='Model', data=results_df, palette='viridis', legend=False)
plt.title('Model Comparison (R² Score)')
plt.ylim(0.9, 1)
plt.show()
plt.savefig('../visuals/model_comparison_(R² Score).png', dpi=300, bbox_inches='tight')

# Model Comparison (RMSE)

plt.figure(figsize=(8, 5))
sns.barplot(x='Model', y='RMSE', hue='Model', data=results_df, palette='coolwarm', legend=False)
plt.title('Model Comparison (RMSE)')
plt.show()
plt.savefig('../visuals/model_comparison_(RMSE).png', dpi=300, bbox_inches='tight')

# Model Comparison (MAE)

plt.figure(figsize=(8, 5))
sns.barplot(x='Model', y='MAE', hue='Model', data=results_df, palette='plasma', legend=False)
plt.title('Model Comparison (MAE)')
plt.show()
plt.savefig('../visuals/model_comparison_(MAE).png', dpi=300, bbox_inches='tight')

# feature_importance

feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': best_model.feature_importances_
}).sort_values(by='Importance', ascending=False)

feature_importance

# Save the best model
joblib.dump(best_model, f'../model/{best_model_name.replace(" ", "_").lower()}_ev_range.pkl')
print(f"\nBest model saved: {best_model_name}")

# Save results
results_df.to_csv('../model/model_results.csv', index=False)
print("Model results saved successfully.")