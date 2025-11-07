# Electric Vehicle Performance and Battery Analysis

**Project Title:** AI-Based Analysis of Electric Vehicle Performance and Battery Life  
**Internship Theme:** Electric Vehicles (AI + Green Skills) — Shell × Edunet × AICTE  

---

## Overview
This project analyzes electric vehicle (EV) performance and battery parameters to explore factors that influence **efficiency**, **driving range**, and **real-world usability**. Using machine learning and generative AI, it delivers **actionable insights** through an **interactive Streamlit dashboard**.

---

## Objectives
- Understand relationships between EV parameters and performance  
- Identify key predictors influencing **battery range** and **efficiency**  
- Build a **regression model** to predict EV range from technical specs  
- Visualize insights with **interactive Streamlit dashboards**  
- Integrate **Generative AI (Google Gemini)** for natural language queries and comparisons  
- Deploy a **production-ready, user-friendly web app**  

**All objectives successfully achieved**

---

## Dataset
**Source:** [Kaggle - Electric Car Performance and Battery Dataset](https://www.kaggle.com/datasets/afnansaifafnan/electric-car-performance-and-battery-dataset)  
**Size:** 474+ real-world electric vehicles

**Key Columns Used:**
|           Column            |            Description               |
|-----------------------------|--------------------------------------|
| `brand`, `model`            | Manufacturer and model name          |
| `top_speed_kmh`             | Max speed in km/h                    |
| `battery_capacity_kWh`      | Battery size                         |
| `efficiency_wh_per_km`      | Energy used per km                   |
| `range_km`                  | **Target**: Real-world driving range |
| `torque_nm`                 | Motor torque                         |
| `acceleration_0_100_s`      | 0–100 km/h time                      |
| `fast_charging_power_kw_dc` | DC fast charge speed                 |
| `drivetrain`, `segment`     | AWD/FWD/RWD, vehicle class           |

---

## Data Cleaning  
The raw dataset was cleaned using **Pandas** and **NumPy** to ensure model-ready quality.

**Cleaning Steps Performed:**
1. Removed redundant columns (`source_url`, `fast_charge_port`, `seats`)  
2. Filled missing values using **column-wise mean/median** based on data type  
3. Replaced zero values in key numeric fields with **column mean** (preserving type)  
4. Detected and **clipped outliers** using the **IQR method** (1.5× rule)  
5. Ensured **datatype consistency** (int vs float) and **2-decimal precision**  
6. Removed **duplicate rows** and standardized naming  
7. Exported cleaned dataset → `data/EV_Battery_Performance_Clean.csv`

**Result:** High-quality, analysis-ready dataset with **zero missing values**

---

## Data Visualization
Exploratory Data Analysis (EDA) was conducted using **Matplotlib**, **Seaborn**, and **Plotly** to uncover performance patterns.

**Interactive Visualization Highlights:**
- **Battery Capacity vs Range** – Strong positive correlation  
- **Efficiency vs Range** – Lower Wh/km = higher range (non-linear)  
- **Top Speed vs Range** – Diminishing returns beyond 200 km/h  
- **Torque & Acceleration Impact** – Performance vs efficiency trade-off  
- **Average Range per Brand** – Tesla, Lucid lead; mass-market brands improving  
- **Segment-wise Range Distribution** – Luxury EVs dominate long-range  
- **Drivetrain Comparison** – AWD offers balance, RWD better efficiency  
- **Correlation Heatmap** – `battery_capacity`, `efficiency` top predictors of range  

All plots are **interactive** in the live dashboard.

---

## Machine Learning Model
**Task:** Predict **real-world range (km)** from 6 input features  
**Model:** `RandomForestRegressor` (Scikit-learn)  
**Features Used:**
```python
['top_speed_kmh', 'battery_capacity_kWh', 'torque_nm', 
 'efficiency_wh_per_km', 'acceleration_0_100_s', 'fast_charging_power_kw_dc']
 ```

**Training Process**
- **Train-test split:** 80/20  
- **Hyperparameter tuning:** `via GridSearchCV`
- **Model saved as:** `models/random_forest_simple.pkl`
- **RMSE on test set:** ~18.4 km (**strong performance**)  
- **Model integrated into Streamlit for live predictions**  

---

**Features:**
- **Natural language input:** e.g. *“Lucid vs Tesla?”*  
- **Brand detection** from user query  
- **Extracts** top range & performance models per brand  
- **Generates** 2–3 sentence expert comparison  
- **Renders** cross-brand radar charts *(range vs speed models)*  
- **Integrated** with Streamlit secrets (`GEMINI_API_KEY`)  

**Example Output:**
> “Lucid Air Grand Touring offers 830 km range with a 112 kWh battery, making it the efficiency leader.  
> Tesla Model S Plaid accelerates fastest (2.1 s) but sacrifices range (637 km) for performance.”  

---

**Streamlit Dashboard (Live App)**  
**Deployed at:** [ev-performance-batterylife-prediction](https://ev-performance-batterylife-prediction.streamlit.app/)

**Interactive Features:**

|            Section             |              Functionality                     |
|--------------------------------|------------------------------------------------|
| **Range Predictor**            | Enter specs → instant ML prediction            |
| **Auto-fill from Real Models** | Select Brand → Model → specs auto-load         |
| **Live Plots**                 | 3 Plotly tabs: Battery, Efficiency, Top 10 EVs |
| **EV Comparison**              | Multi-select up to 10 models → radar chart     |
| **AI Chat Assistant**          | Ask in plain English → get insights + visuals  |

---

**Tech Stack**
- **Frontend:** Streamlit, Custom CSS, Plotly  
- **ML:** Scikit-learn, Joblib  
- **AI:** Google Gemini API  
- **Deployment:** Streamlit Community Cloud *(auto-deploy from GitHub)*  


## Requirements
Install dependencies:
```bash
pip install -r requirements.txt