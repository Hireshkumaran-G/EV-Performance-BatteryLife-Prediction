# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Style settings
sns.set(style="whitegrid", palette="viridis")
plt.rcParams['figure.figsize'] = (8, 5)

# Load Cleaned Dataset
df = pd.read_csv("EV_Battery_Performance_Clean.csv")
df.index=range(1,len(df)+1)
df.head()

# A. Performance Analysis

# Range Distribution

plt.figure()
sns.histplot(
    df['range_km'], 
    kde=True, 
    bins=30, 
    color='seagreen'
)
plt.title('Distribution of EV Range (km)')
plt.xlabel('Range (km)')
plt.ylabel('Count')
plt.show()

# Battery Capacity vs Range

plt.figure(figsize=(8,6))
plt.scatter(
    df['battery_capacity_kWh'], 
    df['range_km'], 
    s=df['torque_nm']/10,
    c=df['top_speed_kmh'], 
    cmap='viridis', 
    alpha=0.7
)
plt.colorbar(label='Top Speed (km/h)')
plt.title("Battery Capacity vs Range")
plt.xlabel("Battery Capacity (kWh)")
plt.ylabel("Range (km)")
plt.show()

# Efficiency vs Range 

plt.figure(figsize=(8,6))
plt.hexbin(
    df['efficiency_wh_per_km'], 
    df['range_km'],
    gridsize=25,          
    cmap='viridis',       
    mincnt=1              
)
plt.colorbar(label='Number of Cars')
plt.title('Efficiency vs Range (Hexbin Density Plot)')
plt.xlabel('Efficiency (Wh/km)')
plt.ylabel('Range (km)')
plt.show()

# Top Speed vs Range

sns.jointplot(
    x='top_speed_kmh', 
    y='range_km', 
    data=df, 
    kind='reg', 
    color='royalblue'
)
plt.suptitle('Top Speed vs Range (Joint View)', y=1.02)
plt.show()

# Motor Power vs Battery Range

df['motor_power_kw'] = (df['torque_nm'] * df['top_speed_kmh']) / 9549

plt.figure(figsize=(8,6))
plt.hexbin(
    df['motor_power_kw'],
    df['range_km'], 
    gridsize=25, 
    cmap='plasma', 
    alpha=0.8, 
    mincnt=1
)
plt.colorbar(label='Data Density')
plt.title('Power vs Range Density')
plt.xlabel('Motor Power (kW)')
plt.ylabel('Range (km)')
plt.show()

# B. Battery Life & Efficiency Analysis

# Distribution of Efficiency

plt.figure()
sns.histplot(
    df['efficiency_wh_per_km'], 
    kde=True, 
    bins=30
)
plt.title('Distribution of Efficiency (Wh/km)')
plt.xlabel('Efficiency (Wh/km)')
plt.ylabel('Count')
plt.show()

# Battery Capacity vs Efficiency 

plt.figure(figsize=(8,6))
plt.hexbin(
    df['efficiency_wh_per_km'], 
    df['battery_capacity_kWh'], 
    gridsize=25,        
    cmap='plasma',        
    mincnt=1              
)
plt.colorbar(label='Number of Cars')
plt.title('Battery Capacity vs Efficiency (Hexbin Density Plot)')
plt.ylabel('Battery Capacity (kWh)')
plt.xlabel('Efficiency (Wh/km)')
plt.show()

# C. Technical Feature Impact

# Torque vs Range

plt.figure(figsize=(8,6))
sns.scatterplot(
    x='torque_nm', 
    y='range_km', 
    hue='segment', 
    data=df
)
plt.title("Torque vs Range by Segment")
plt.xlabel("Torque (Nm)")
plt.ylabel("Range (km)")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()

# Acceleration vs Range

plt.figure()
sns.scatterplot(
    x='acceleration_0_100_s', 
    y='range_km', 
    data=df, 
    color='darkorchid'
)
plt.title('Acceleration (0–100 km/h) vs Range')
plt.xlabel('Acceleration Time (s)')
plt.ylabel('Range (km)')
plt.show()

# Range Distribution across Segments

plt.figure(figsize=(10,6))
sns.boxplot(
    x='segment', 
    y='range_km', 
    data=df
)
plt.title("Range Distribution across Segments")
plt.xlabel("Segment")
plt.ylabel("Range (km)")
plt.xticks(rotation=45)
plt.show()

# Correlation Heatmap

numeric_cols = ['top_speed_kmh', 'battery_capacity_kWh', 'torque_nm', 
                'efficiency_wh_per_km', 'range_km', 'acceleration_0_100_s']
corr = df[numeric_cols].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Between Key Performance Metrics')
plt.show()

# D. Brand & Category Insights

# Average Range per Brand

top_brands = df['brand'].value_counts().head(10).index

plt.figure(figsize=(10,5))
sns.barplot(
    data=df[df['brand'].isin(top_brands)],
    x='brand',
    y='battery_capacity_kWh',
    estimator=np.mean,
    hue='brand',
    dodge=False,
    palette='plasma',
    legend=False
)
plt.xticks(rotation=90)
plt.title('Average Battery Capacity by Top 10 Brands')
plt.xlabel('Brand')
plt.ylabel('Battery Capacity (kWh)')
plt.tight_layout()
plt.show()

# Battery Type Distribution 

if 'battery_type' in df.columns:
    plt.figure(figsize=(6,6))
    battery_counts = df['battery_type'].value_counts()
    plt.pie(
        battery_counts,
        labels=battery_counts.index,
        autopct='%1.1f%%',
        startangle=90,
        wedgeprops=dict(width=0.4)
    )
    plt.title('Battery Type Distribution')
    plt.show()

# Car Body Type vs Energy Efficiency

plt.figure(figsize=(10,6))
sns.violinplot(
    data=df,
    x='car_body_type',
    y='efficiency_wh_per_km',
    hue='car_body_type',        
    palette='Spectral',
    legend=False,               
    dodge=False
)
plt.title('Efficiency Distribution by Car Body Type')
plt.xlabel('Car Body Type')
plt.ylabel('Efficiency (Wh/km)')
plt.tight_layout()
plt.show()