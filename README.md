# Electric Vehicle Performance and Battery Analysis

**Project Title:** AI-Based Analysis of Electric Vehicle Performance and Battery Life  
**Internship Theme:** Electric Vehicles (AI + Green Skills) — Shell × Edunet × AICTE  

## Overview
This project analyzes electric vehicle (EV) performance and battery parameters to explore factors that influence efficiency and driving range.

## Objectives
- Understand relationships between EV parameters and performance  
- Identify key predictors influencing **battery range** and **efficiency**  
- Build regression model to predict EV range
- Visualize insights with **Streamlit dashboards**  
- Explore **Generative AI tools** (OpenAI, Gemini, DeepSeek, DALL·E) for chatbot/report generation  
- Deploy final interactive dashboard using Streamlit  

## Dataset
**Source:** [Kaggle - Electric Car Performance and Battery Dataset](https://www.kaggle.com/datasets/afnansaifafnan/electric-car-performance-and-battery-dataset)

**Included Columns:**
- Brand, Model, Top Speed (km/h), Battery Capacity (kWh), Battery Type  
- Torque (Nm), Efficiency (Wh/km), Range (km), Acceleration (0–100 km/h)  
- Fast Charging Power (kW), Port Type, Cargo Volume (L), Seats, Drivetrain  
- Segment, Dimensions (mm), Car Body Type, Source URL  

## Data Cleaning  
The dataset was cleaned using **Pandas** and **NumPy** to prepare it for analysis.  

**Cleaning Steps Performed**
1. Removed redundant columns (`source_url`, `fast_charge_port`, `seats`).  
2. Filled missing values using column-wise **mean/median** based on datatype.  
3. Replaced zero values in key numeric fields with their column mean (preserving type).  
4. Detected and clipped outliers using the **IQR method**.  
5. Ensured proper formatting and datatype consistency (e.g., integer vs. float).  
6. Removed duplicate rows and standardized numerical precision (2 decimal places).  
7. Exported cleaned dataset → `data/EV_Battery_Performance_Clean.csv` 

## Data Visualization
Exploratory Data Analysis (EDA) was conducted using Matplotlib and Seaborn to uncover performance patterns and relationships between key features.

**Visualization Highlights:**
- Range Distribution across all EV models.
- Battery Capacity vs Range – identifies how capacity impacts distance.
- Efficiency vs Range – measures energy effectiveness.
- Top Speed vs Range and Motor Power vs Battery Range for performance trade-offs.
- Battery Capacity vs Efficiency – analyzes optimal energy use.
- Torque vs Range and Acceleration vs Range – impact of drivetrain and power.
- Average Range per Brand and Range Distribution across Segments for market insights.
- Battery Type Distribution and Drivetrain vs Range comparisons.
- Correlation Heatmap summarizing feature interdependence.

## Requirements
Install dependencies:
```bash
pip install -r requirements.txt