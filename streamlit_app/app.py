import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import FancyBboxPatch
import plotly.express as px
import os
import plotly.graph_objects as go

# --- Page Config ---
st.set_page_config(
    page_title="EV Range Predictor ⚡",
    page_icon="🚗",
    layout="centered",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for Cool Look ---
st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        color: #ffffff;
    }
    .stApp {
        background: transparent;
    }
    .css-1d391kg, .css-1cpxl2t {
        color: white !important;
    }
    .sidebar .css-1d391kg {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 20px;
    }
    .metric-card {
        background: rgba(255, 255, 255, 0.15);
        padding: 1.5rem;
        border-radius: 15px;
        border: 1px solid rgba(255, 255, 255, 0.2);
        text-align: center;
        backdrop-filter: blur(8px);
    }
    .title {
        font-size: 3.5rem !important;
        font-weight: 900;
        background: linear-gradient(90deg, #00f5ff, #ff00c8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0;
    }
    .subtitle {
        text-align: center;
        color: #cccccc;
        font-size: 1.2rem;
        margin-top: -10px;
    }
    .stButton>button {
        background: linear-gradient(45deg, #ff00c8, #00f5ff);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 12px 30px;
        font-weight: bold;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        transform: translateY(-3px);
        box-shadow: 0 10px 20px rgba(0, 245, 255, 0.4);
    }
    .plot-container {
        background: rgba(255, 255, 255, 0.1);
        padding: 20px;
        border-radius: 15px;
        border: 1px solid rgba(255, 255, 255, 0.15);
        backdrop-filter: blur(10px);
    }
</style>
""", unsafe_allow_html=True)

# --- Load Model & Data ---
@st.cache_resource
def load_model():
    return joblib.load("models/random_forest_simple.pkl")

@st.cache_data
def load_data():
    return pd.read_csv("data/EV_Battery_Performance_Clean.csv")

model = load_model()
df = load_data()

# --- Title ---
st.markdown("<h1 class='title'>⚡ EV Range Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Predict real-world range using AI • Explore 474+ Electric Vehicles</p>", unsafe_allow_html=True)

# --- Sidebar Inputs ---
st.sidebar.markdown("### Electric Vehicle Predictor")
st.sidebar.header("🔧 Input EV Specifications")

col_a, col_b = st.sidebar.columns(2)

with col_a:
    battery_capacity = st.number_input("🔋 Capacity (kWh)", 10.0, 150.0, 75.0, 5.0)
    efficiency = st.number_input("⚡ Efficiency (Wh/km)", 100, 300, 180, 5)
    top_speed = st.number_input("🏎️ Top Speed (km/h)", 100, 300, 200, 10)

with col_b:
    torque = st.number_input("🔩 Torque (Nm)", 100, 1000, 350, 25)
    acceleration = st.number_input("⏱️ 0-100 km/h (s)", 2.0, 15.0, 6.0, 0.5)
    charging_power = st.number_input("🔌 Fast Charge (kW)", 20, 350, 150, 10)

# Brand & Model Filter
st.sidebar.markdown("### 🎯 Filter by Brand")
brands = ["All"] + sorted(df['brand'].unique())
selected_brand = st.sidebar.selectbox("Choose Brand", brands)

if selected_brand != "All":
    models = df[df['brand'] == selected_brand]['model'].unique()
    selected_model = st.sidebar.selectbox("Choose Model", models)
else:
    selected_model = None

# --- Auto-fill with real model ---
if selected_model:
    row = df[(df['brand'] == selected_brand) & (df['model'] == selected_model)].iloc[0]
    battery_capacity = row['battery_capacity_kWh']
    efficiency = row['efficiency_wh_per_km']
    top_speed = row['top_speed_kmh']
    torque = row['torque_nm']
    acceleration = row['acceleration_0_100_s']
    charging_power = row['fast_charging_power_kw_dc']

    st.sidebar.success(f"Loaded: **{selected_brand} {selected_model}**")

# --- Prediction ---
features = np.array([[top_speed, battery_capacity, torque, efficiency, acceleration, charging_power]])
prediction = model.predict(features)[0]

col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
    st.markdown(f"### 🚗 Predicted Range")
    st.markdown(f"## <span style='font-size:3rem; color:#00f5ff'>{prediction:.1f} km</span>", unsafe_allow_html=True)
    st.markdown(f"###### *Based on {battery_capacity} kWh • {efficiency} Wh/km efficiency*")
    st.markdown("</div>", unsafe_allow_html=True)

# --- Interactive Plots ---
st.markdown("## 📊 Live Data Insights")

tab1, tab2, tab3 = st.tabs(["🔋 Battery vs Range", "⚡ Efficiency Impact", "🏆 Top Performers"])

with tab1:
    fig = px.scatter(
        df, x='battery_capacity_kWh', y='range_km',
        color='segment', size='fast_charging_power_kw_dc',
        hover_data=['brand', 'model', 'top_speed_kmh'],
        title="Battery Capacity vs Real-World Range",
        color_discrete_sequence=px.colors.sequential.Plasma
    )
    fig.update_layout(template="plotly_dark", height=600)
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    fig = px.scatter(
        df, x='efficiency_wh_per_km', y='range_km',
        color='drivetrain', size='battery_capacity_kWh',
        hover_data=['brand', 'model'],
        title="Efficiency vs Range (Lower Wh/km = Better)",
        color_discrete_sequence=px.colors.sequential.Viridis
    )
    fig.update_layout(template="plotly_dark", height=600)
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    top10 = df.nlargest(10, 'range_km')[['brand', 'model', 'range_km', 'battery_capacity_kWh', 'efficiency_wh_per_km']]
    fig = go.Figure(data=[go.Bar(
        x=top10['brand'] + " " + top10['model'],
        y=top10['range_km'],
        text=top10['range_km'],
        textposition='auto',
        marker_color=px.colors.sequential.Magma
    )])
    fig.update_layout(title="🏆 Top 10 EVs by Range", template="plotly_dark", height=600)
    st.plotly_chart(fig, use_container_width=True)

# --- EV Comparison Section ---
st.markdown("## Compare EVs")

# Create full name column once (cached)
if 'brand_model' not in df.columns:
    df['brand_model'] = df['brand'] + " " + df['model']

# Multiselect using full names
compare_options = df['brand_model'].unique()
compare_brands = st.multiselect(
    "Select up to 10 EVs to compare",
    options=compare_options,
    max_selections=10
)

if len(compare_brands) > 0:
    # Filter selected EVs
    selected_rows = df[df['brand_model'].isin(compare_brands)].copy()

    # Prepare radar chart data
    radar_df = selected_rows.set_index('brand_model')[
        ['range_km', 'top_speed_kmh', 'battery_capacity_kWh',
         'fast_charging_power_kw_dc', 'torque_nm', 'efficiency_wh_per_km']
    ].T

    # Normalize values for better radar scaling (optional but recommended)
    radar_normalized = radar_df / radar_df.max()

    fig = go.Figure()
    colors = [
    '#00f5ff',  # Cyan (Tesla-style)
    '#ff00c8',  # Magenta
    '#00ff88',  # Lime Green
    '#ffaa00',  # Orange
    '#ff3366',  # Hot Pink
    '#33ccff',  # Sky Blue
    '#ffcc00',  # Golden Yellow
    '#cc66ff',  # Purple
    '#66ff99',  # Mint Green
    '#ff6666'   # Coral Red
]

    for i, col in enumerate(radar_normalized.columns):
        fig.add_trace(go.Scatterpolar(
            r=radar_normalized[col],
            theta=radar_normalized.index,
            fill='toself',
            name=col,
            line_color=colors[i % len(colors)]
        ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1])
        ),
        showlegend=True,
        title="EV Radar Comparison (Normalized)",
        template="plotly_dark",
        height=600
    )
    
    st.plotly_chart(fig, use_container_width=False)

    # Optional: Show table
    st.markdown("### Selected EV Specs")
    display_cols = ['brand', 'model', 'range_km', 'battery_capacity_kWh',
                    'efficiency_wh_per_km', 'top_speed_kmh', 'fast_charging_power_kw_dc']
    st.dataframe(selected_rows[display_cols].set_index(['brand', 'model']), use_container_width=True)

# --- Footer ---
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #888; padding: 20px;'>
        <p>🚀 <strong>Shell Edunet x AICTE Internship Project</strong> | 
        EV Performance & Battery Life Prediction using Machine Learning</p>
        <p>Built with ❤️ using <b>Streamlit</b> • <b>Scikit-learn</b> • <b>Plotly</b></p>
    </div>
    """,
    unsafe_allow_html=True
)
# -------------------------------------------------
# 9. AI CHAT ASSISTANT – Fixed Brand Detection + Cross‑Brand Radar
# -------------------------------------------------
st.markdown("## AI Chat Assistant")
st.markdown(
    "<p style='color:#bbb;'>Ask about brands (e.g., 'Lucid vs Tesla') – get top models + cross‑brand comparisons.</p>",
    unsafe_allow_html=True,
)

# ---- Safe stats ----
avg_range = df["range_km"].mean()
battery_capacity = locals().get("battery_capacity", df["battery_capacity_kWh"].mean())
efficiency       = locals().get("efficiency", df["efficiency_wh_per_km"].mean())

# ---- Cached: top range & top performance per brand ----
@st.cache_data
def get_brand_extremes(brand_query):
    subset = df[df["brand"].str.contains(brand_query, case=False, na=False)]
    if subset.empty:
        return None, None

    # Top range
    top_range = subset.loc[subset["range_km"].idxmax()]
    range_car = {
        "Brand": top_range["brand"],
        "Model": top_range["model"],
        "Range (km)": round(top_range["range_km"], 1),
        "Battery (kWh)": round(top_range["battery_capacity_kWh"], 1),
        "Efficiency (Wh/km)": int(top_range["efficiency_wh_per_km"]),
        "Top Speed (km/h)": int(top_range["top_speed_kmh"]),
        "Torque (Nm)": int(top_range["torque_nm"]),
        "0‑100 km/h (s)": round(top_range["acceleration_0_100_s"], 1),
        "Fast Charge (kW)": int(top_range["fast_charging_power_kw_dc"])
        if pd.notna(top_range["fast_charging_power_kw_dc"])
        else "N/A",
        "Segment": top_range["segment"],
    }

    # Top performance (fastest 0‑100)
    top_perf = subset.loc[subset["acceleration_0_100_s"].idxmin()]
    perf_car = {
        "Brand": top_perf["brand"],
        "Model": top_perf["model"],
        "Range (km)": round(top_perf["range_km"], 1),
        "Battery (kWh)": round(top_perf["battery_capacity_kWh"], 1),
        "Efficiency (Wh/km)": int(top_perf["efficiency_wh_per_km"]),
        "Top Speed (km/h)": int(top_perf["top_speed_kmh"]),
        "Torque (Nm)": int(top_perf["torque_nm"]),
        "0‑100 km/h (s)": round(top_perf["acceleration_0_100_s"], 1),
        "Fast Charge (kW)": int(top_perf["fast_charging_power_kw_dc"])
        if pd.notna(top_perf["fast_charging_power_kw_dc"])
        else "N/A",
        "Segment": top_perf["segment"],
    }

    return range_car, perf_car

# ---- Get all brands from data (cached) ----
@st.cache_data
def get_all_brands():
    return df["brand"].str.lower().unique().tolist()

all_brands = get_all_brands()

# ---- UI ----
user_input = st.text_area("Ask your EV question (e.g., Lucid vs Tesla?)…", height=100, key="ai_chat_input_v8")

if st.button("Ask AI", key="ai_chat_send_v8"):
    if not user_input.strip():
        st.info("Please type a question.")
    else:
        with st.spinner("Gemini + Brand Analysis…"):
            answer_text = ""
            brand_cars = {}  # {brand_lower: (range_car, perf_car)}
            detected_brands = []

            # ---- No API key → mock ----
            if not st.secrets.get("GEMINI_API_KEY", None):
                answer_text = (
                    "Gemini API key missing.\n"
                    "```bash\nexport GEMINI_API_KEY=\"your-key\"\n```\n"
                    "Example: Lucid Air (range) vs Tesla Model S Plaid (speed)."
                )
            else:
                try:
                    import google.generativeai as genai
                    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
                    model = genai.GenerativeModel("gemini-2.5-flash")

                    # Detect brands using data-driven list
                    user_lower = user_input.lower()
                    for brand in all_brands:
                        if brand in user_lower:
                            detected_brands.append(brand.title())
                            range_car, perf_car = get_brand_extremes(brand)
                            if range_car and perf_car:
                                brand_cars[brand] = (range_car, perf_car)

                    brand_hints = ""
                    for b, (r, p) in brand_cars.items():
                        brand_hints += (
                            f"\n• {b.title()} Top range: {r['Model']} ({r['Range (km)']} km)\n"
                            f"• {b.title()} Fastest: {p['Model']} ({p['0‑100 km/h (s)']} s)"
                        )

                    prompt = f"""
                    You are an EV expert. Use real data:
                    • Avg range: {avg_range:.1f} km
                    • Predicted: {prediction:.1f} km
                    • Inputs: {battery_capacity:.0f} kWh, {efficiency:.0f} Wh/km
                    {brand_hints}

                    Question: {user_input}
                    Compare the brands head-to-head. Answer in 2–3 short sentences.
                    """

                    response = model.generate_content(prompt)
                    answer_text = response.text.strip()

                except Exception as e:
                    answer_text = f"Gemini Error: {e}"

            # -----------------------------
            # 1. AI ANSWER
            # -----------------------------
            st.markdown("### AI Answer")
            st.markdown(
                f"<div style='background:rgba(0,245,255,0.1);padding:15px;border-radius:12px;"
                f"border-left:4px solid #00f5ff;'><p style='margin:0;'>{answer_text}</p></div>",
                unsafe_allow_html=True,
            )

            # -----------------------------
            # 2. TABLES FOR EACH BRAND
            # -----------------------------
            if brand_cars:
                for brand, (range_car, perf_car) in brand_cars.items():
                    brand_title = brand.title()
                    st.markdown(f"### {brand_title}: Top Range Model")
                    df_range = pd.DataFrame([range_car])
                    st.dataframe(
                        df_range,
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            "Brand": None,
                            "Model": st.column_config.TextColumn("Model"),
                            "Range (km)": st.column_config.NumberColumn("Range", format="%.1f"),
                            "Battery (kWh)": st.column_config.NumberColumn("Battery", format="%.1f"),
                            "Efficiency (Wh/km)": st.column_config.NumberColumn("Efficiency", format="%d"),
                            "Top Speed (km/h)": st.column_config.NumberColumn("Top Speed", format="%d"),
                            "Torque (Nm)": st.column_config.NumberColumn("Torque", format="%d"),
                            "0‑100 km/h (s)": st.column_config.NumberColumn("0‑100", format="%.1f"),
                            "Fast Charge (kW)": st.column_config.TextColumn("Fast Charge"),
                            "Segment": st.column_config.TextColumn("Segment"),
                        },
                    )

                    st.markdown(f"### {brand_title}: Top Performance Model (Fastest 0‑100)")
                    df_perf = pd.DataFrame([perf_car])
                    st.dataframe(
                        df_perf,
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            "Brand": None,
                            "Model": st.column_config.TextColumn("Model"),
                            "Range (km)": st.column_config.NumberColumn("Range", format="%.1f"),
                            "Battery (kWh)": st.column_config.NumberColumn("Battery", format="%.1f"),
                            "Efficiency (Wh/km)": st.column_config.NumberColumn("Efficiency", format="%d"),
                            "Top Speed (km/h)": st.column_config.NumberColumn("Top Speed", format="%d"),
                            "Torque (Nm)": st.column_config.NumberColumn("Torque", format="%d"),
                            "0‑100 km/h (s)": st.column_config.NumberColumn("0‑100", format="%.1f"),
                            "Fast Charge (kW)": st.column_config.TextColumn("Fast Charge"),
                            "Segment": st.column_config.TextColumn("Segment"),
                        },
                    )
            else:
                st.info("No brands detected. Try *'Lucid vs Tesla?'*")

            # -----------------------------
            # 3. CROSS-BRAND RADAR: If 2+ brands
            # -----------------------------
            if len(brand_cars) >= 2:
                brand_list = list(brand_cars.keys())
                brand1, brand2 = brand_list[0], brand_list[1]
                range_car1, perf_car1 = brand_cars[brand1]
                range_car2, perf_car2 = brand_cars[brand2]

                # RADAR 1: Top Range
                st.markdown("### Cross-Brand Radar: Top Range Models")
                metrics = [
                    "Range (km)",
                    "Battery (kWh)",
                    "Efficiency (Wh/km)",
                    "Top Speed (km/h)",
                    "Torque (Nm)",
                    "0‑100 km/h (s)",
                ]

                def normalize_for_radar(vals1, vals2):
                    # Invert lower‑is‑better
                    max_eff = max(vals1[2], vals2[2])
                    max_acc = max(vals1[5], vals2[5])
                    v1, v2 = vals1.copy(), vals2.copy()
                    v1[2] = max_eff - v1[2] + 100
                    v2[2] = max_eff - v2[2] + 100
                    v1[5] = max_acc - v1[5] + 2
                    v2[5] = max_acc - v2[5] + 2
                    all_v = v1 + v2
                    min_v, max_v = min(all_v), max(all_v)
                    return [(x - min_v)/(max_v - min_v) for x in v1], [(x - min_v)/(max_v - min_v) for x in v2]

                # Range radar
                r1, r2 = normalize_for_radar(
                    [range_car1["Range (km)"], range_car1["Battery (kWh)"], range_car1["Efficiency (Wh/km)"],
                     range_car1["Top Speed (km/h)"], range_car1["Torque (Nm)"], range_car1["0‑100 km/h (s)"]],
                    [range_car2["Range (km)"], range_car2["Battery (kWh)"], range_car2["Efficiency (Wh/km)"],
                     range_car2["Top Speed (km/h)"], range_car2["Torque (Nm)"], range_car2["0‑100 km/h (s)"]]
                )
                fig1 = go.Figure()
                fig1.add_trace(go.Scatterpolar(r=r1, theta=metrics, fill='toself',
                    name=f"{range_car1['Brand']} {range_car1['Model']}", line_color="#00f5ff"))
                fig1.add_trace(go.Scatterpolar(r=r2, theta=metrics, fill='toself',
                    name=f"{range_car2['Brand']} {range_car2['Model']}", line_color="#ff00c8"))
                fig1.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                                   showlegend=True, template="plotly_dark", height=500,
                                   title=f"Range Kings: {brand1.title()} vs {brand2.title()}")
                st.plotly_chart(fig1, use_container_width=True)

                # Performance radar
                p1, p2 = normalize_for_radar(
                    [perf_car1["Range (km)"], perf_car1["Battery (kWh)"], perf_car1["Efficiency (Wh/km)"],
                     perf_car1["Top Speed (km/h)"], perf_car1["Torque (Nm)"], perf_car1["0‑100 km/h (s)"]],
                    [perf_car2["Range (km)"], perf_car2["Battery (kWh)"], perf_car2["Efficiency (Wh/km)"],
                     perf_car2["Top Speed (km/h)"], perf_car2["Torque (Nm)"], perf_car2["0‑100 km/h (s)"]]
                )
                fig2 = go.Figure()
                fig2.add_trace(go.Scatterpolar(r=p1, theta=metrics, fill='toself',
                    name=f"{perf_car1['Brand']} {perf_car1['Model']}", line_color="#00f5ff"))
                fig2.add_trace(go.Scatterpolar(r=p2, theta=metrics, fill='toself',
                    name=f"{perf_car2['Brand']} {perf_car2['Model']}", line_color="#ff00c8"))
                fig2.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                                   showlegend=True, template="plotly_dark", height=500,
                                   title=f"Speed Demons: {brand1.title()} vs {brand2.title()}")
                st.plotly_chart(fig2, use_container_width=True)

            elif len(brand_cars) == 1:
                # Single brand: intra-brand radar
                brand = next(iter(brand_cars))
                range_car, perf_car = brand_cars[brand]
                st.markdown(f"### Radar: {brand.title()} Range vs Performance")
                # ... (same as before, reuse normalize_for_radar)
                r_vals = [range_car["Range (km)"], range_car["Battery (kWh)"], range_car["Efficiency (Wh/km)"],
                          range_car["Top Speed (km/h)"], range_car["Torque (Nm)"], range_car["0‑100 km/h (s)"]]
                p_vals = [perf_car["Range (km)"], perf_car["Battery (kWh)"], perf_car["Efficiency (Wh/km)"],
                          perf_car["Top Speed (km/h)"], perf_car["Torque (Nm)"], perf_car["0‑100 km/h (s)"]]
                r_norm, p_norm = normalize_for_radar(r_vals, p_vals)
                fig = go.Figure()
                fig.add_trace(go.Scatterpolar(r=r_norm, theta=metrics, fill='toself',
                    name=f"{range_car['Model']} (Range)", line_color="#00f5ff"))
                fig.add_trace(go.Scatterpolar(r=p_norm, theta=metrics, fill='toself',
                    name=f"{perf_car['Model']} (Speed)", line_color="#ff00c8"))
                fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                                  showlegend=True, template="plotly_dark", height=500)
                st.plotly_chart(fig, use_container_width=True)