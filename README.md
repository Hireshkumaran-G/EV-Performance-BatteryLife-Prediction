# EV Car Performance and Battery Analysis

## Overview
This project analyzes electric vehicle (EV) performance and battery parameters to understand factors affecting efficiency and range.

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

*Dataset cleaning (Excel preprocessing) performed externally; raw dataset uploaded here.*

## Requirements
Install dependencies:
```bash
pip install -r requirements.txt