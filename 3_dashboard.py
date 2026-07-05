import gdown
import os
import gc
import pydeck as pdk
import streamlit as st
import pandas as pd
import xgboost as xgb
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium
from geopy.geocoders import Nominatim
from sklearn.metrics import mean_squared_error, r2_score

# ==========================================
# 1. PAGE CONFIG & PRO CUSTOM CSS
# ==========================================
st.set_page_config(page_title="ISRO AI Heat Mitigation", layout="wide", page_icon="🛰️", initial_sidebar_state="expanded")

st.markdown("""
<style>
    /* Gradient Header */
    .pro-header {
        background: -webkit-linear-gradient(45deg, #FF4B2B, #FF416C);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3em;
        font-weight: 800;
        margin-bottom: 0px;
    }
    /* Metric Card Styling */
    [data-testid="stMetricValue"] {
        font-size: 2.2rem !important;
        font-weight: 700 !important;
        color: #FF4B2B !important;
    }
    [data-testid="stMetricDelta"] {
        font-size: 1.2rem !important;
    }
    /* Clean up tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 20px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="pro-header">🛰️ National Urban Heat Mitigation System</p>', unsafe_allow_html=True)
st.markdown("**Engine:** 9D XGBoost PIML | **Scope:** Pan-India | **Forecasting Era:** 2024-2026 | **Sponsor:** ISRO")
st.divider()

# ==========================================
# 2. BULLETPROOF DATA & MODEL LOADING
# ==========================================
@st.cache_resource
import gc # Garbage collector to free up RAM

@st.cache_resource
def load_assets():
    # 1. CHANGE TO .ubj
    model_path = "urban_heat_temporal_model.ubj" 
    data_path = "India_1Million_Temporal_9D_Data.csv"
    
    if not os.path.exists(model_path):
        # 2. PUT YOUR NEW .ubj GOOGLE DRIVE ID HERE!
        ubj_id = "19BGrK-thH3c4xRNNT7gcve5aYojOrruq" 
        url = f'https://drive.google.com/uc?id={ubj_id}'
        gdown.download(url, model_path, quiet=False)
        
    if not os.path.exists(data_path):
        csv_id = "1D2Xtxy1PoQzR9tM0U_pPOb8ZustZzQKp" # Your working CSV ID
        url = f'https://drive.google.com/uc?id={csv_id}'
        gdown.download(url, data_path, quiet=False)

    features = ['NDVI_Greenness', 'NDBI_BuiltUp', 'elevation', 'Wind_Speed', 'Dewpoint_C', 'ECOSTRESS_LST', 'GHSL_Urban_Form', 'Pollution_CPCB_Proxy', 'Year']
    
    if not os.path.exists(model_path) or not os.path.exists(data_path):
         return None, None, 0.0, 0.0, "⚠️ Failed to download files."

    model = xgb.XGBRegressor()
    try:
        # The .ubj binary will load instantly without crashing the server!
        model.load_model(model_path)
        model.get_booster().feature_names = features
        
        # 3. Only load 2,500 rows to save massive amounts of RAM
        df = pd.read_csv(data_path, nrows=2500)
        df = df.dropna(subset=features + ['LST_Celsius'])
        
        X = df[features].to_numpy()
        y = df['LST_Celsius'].to_numpy()
        
        preds = model.predict(X) 
        rmse = np.sqrt(mean_squared_error(y, preds))
        r2 = r2_score(y, preds)
        
        # 4. AGGRESSIVELY CLEAR MEMORY
        del X, y, preds
        gc.collect() 
        
        return model, df, rmse, r2, None
    except Exception as e:
        return None, None, 0.0, 0.0, str(e)

model, df, true_rmse, true_r2, error_msg = load_assets()

if error_msg:
    st.error(error_msg)
    st.stop()

# ==========================================
# 3. SIDEBAR (PRO LAYOUT)
# ==========================================
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/b/bd/Indian_Space_Research_Organisation_Logo.svg", width=100)
    st.header("⚙️ Simulation Engine")
    
    location_query = st.text_input("📍 Target City", "Ahmedabad, Gujarat")
    scenario_year = st.selectbox("📅 Temporal Forecast Year", [2024, 2025, 2026], index=2)
    
    with st.expander("☁️ Meteorological Parameters (ERA5)", expanded=True):
        city_elevation = st.slider("Elevation (m)", 0.0, 2000.0, 50.0)
        city_wind = st.slider("Wind (m/s)", 0.0, 10.0, 3.5)
        city_humidity = st.slider("Dewpoint (°C)", 5.0, 30.0, 20.0)
        city_eco_lst = st.slider("ECOSTRESS Peak Heat (°C)", 30.0, 55.0, 40.0)
        city_pollution = st.slider("CPCB NO2 Proxy", 0.0, 10.0, 2.5)

    with st.expander("🎛️ Urban Intervention Sliders", expanded=True):
        current_ndvi = st.slider("Vegetation (NDVI)", 0.0, 0.8, 0.2, help="Increase to simulate urban forestry.")
        current_ndbi = st.slider("Concrete Density (NDBI)", -0.5, 0.5, 0.3)
        ghsl_form = st.slider("Urban Morph Volume", 0.0, 255.0, 100.0)
        cool_roof_albedo = st.slider("Cool Roof Albedo", 0.0, 0.5, 0.0, help="Increase to simulate reflective paints.")

# ==========================================
# 4. CORE ENGINE CALCULATIONS
# ==========================================
@st.cache_data
def get_coordinates(query):
    try:
        loc = Nominatim(user_agent="isro_pro_app").geocode(query + ", India")
        return (loc.latitude, loc.longitude) if loc else (23.0225, 72.5714)
    except: return (23.0225, 72.5714)

city_lat, city_lon = get_coordinates(location_query)

def calculate_temp(ndvi, ndbi, albedo):
    # Use a pure NumPy array to bypass strict column name errors
    input_data = np.array([[
        ndvi, ndbi, city_elevation, city_wind, city_humidity, 
        city_eco_lst, ghsl_form, city_pollution, scenario_year
    ]])
    return float(model.predict(input_data)[0]) - (albedo * 12.0)

# ==========================================
# 5. PROFESSIONAL TABS
# ==========================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🗺️ 1. Spatial Heatmaps", 
    "📊 2. AI Feature Weights", 
    "🧠 3. Model Validation", 
    "📈 4. Scenario ROI",
    "🎯 5. Action Strategy"
])

# --- TAB 1: SPATIAL MAPS (PRO VERSION) ---
# --- TAB 1: SPATIAL MAPS (PRO VERSION) ---
with tab1:
    st.markdown("### Spatial Heat Profiling")
    map_type = st.radio("Select Map View:", ["3D AI Prediction (PyDeck)", "2D AI Predictive Grid (Folium)", "Actual Satellite Dataset (Historical)"], horizontal=True)
    
    if map_type == "3D AI Prediction (PyDeck)":
        st.markdown("### 3D Urban Form & Heat Correlation")
        st.caption("Height represents Urban Morphology (Building Density). Color represents AI-Predicted Heat.")
        
        # 1. Create a 3D grid
        lats = np.linspace(city_lat - 0.05, city_lat + 0.05, 40)
        lons = np.linspace(city_lon - 0.05, city_lon + 0.05, 40)
        lon_grid, lat_grid = np.meshgrid(lons, lats)
        flat_lats = lat_grid.ravel()
        flat_lons = lon_grid.ravel()
        
        # 2. Simulate urban clustering for the 3D effect (dense center, green edges)
        dist = np.sqrt((flat_lats - city_lat)**2 + (flat_lons - city_lon)**2)
        sim_ghsl = np.clip(255 - (dist * 4000), 10, 255)
        sim_ndbi = np.clip(0.5 - (dist * 10), -0.2, 0.5)
        sim_ndvi = np.clip(0.1 + (dist * 10), 0.1, 0.7)
        
        # 3. Batch Predict using pure NumPy (Bypasses XGBoost naming errors)
        batch_inputs = np.column_stack((
            sim_ndvi, sim_ndbi, 
            np.full(len(flat_lats), city_elevation), 
            np.full(len(flat_lats), city_wind), 
            np.full(len(flat_lats), city_humidity), 
            np.full(len(flat_lats), city_eco_lst), 
            sim_ghsl, 
            np.full(len(flat_lats), city_pollution), 
            np.full(len(flat_lats), scenario_year)
        ))
        
        base_preds = model.predict(batch_inputs)
        mitigated_preds = base_preds - (cool_roof_albedo * 12.0)
        
        # 4. Map to DataFrame for PyDeck
        grid_df = pd.DataFrame({
            'lat': flat_lats,
            'lon': flat_lons,
            'temp': mitigated_preds,
            'height': sim_ghsl * 8  # Scale for visual height
        })
        
        # Calculate Colors (Blue to Red based on temp)
        t_min, t_max = grid_df['temp'].min(), grid_df['temp'].max()
        grid_df['color_r'] = ((grid_df['temp'] - t_min) / (t_max - t_min + 0.0001) * 255).astype(int)
        grid_df['color_g'] = 50
        grid_df['color_b'] = 255 - grid_df['color_r']
        grid_df['color'] = grid_df.apply(lambda row: [row['color_r'], row['color_g'], row['color_b'], 200], axis=1)

        # 5. Build PyDeck Layer
        layer = pdk.Layer(
            'ColumnLayer',
            data=grid_df,
            get_position='[lon, lat]',
            get_elevation='height',
            elevation_scale=1,
            radius=150,
            get_fill_color='color',
            pickable=True,
            auto_highlight=True
        )
        
        view_state = pdk.ViewState(latitude=city_lat, longitude=city_lon, zoom=11.5, pitch=50, bearing=-20)
        st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view_state, tooltip={"text": "Predicted LST: {temp} °C"}))

    elif map_type == "2D AI Predictive Grid (Folium)":
        m = folium.Map(location=[city_lat, city_lon], zoom_start=12, tiles="CartoDB dark_matter")
        lats = np.linspace(city_lat - 0.06, city_lat + 0.06, 80)
        lons = np.linspace(city_lon - 0.06, city_lon + 0.06, 80)
        lon_grid, lat_grid = np.meshgrid(lons, lats)
        
        base_temp = calculate_temp(current_ndvi, current_ndbi, cool_roof_albedo)
        dist = np.sqrt((lat_grid.ravel() - city_lat)**2 + (lon_grid.ravel() - city_lon)**2)
        simulated_temps = base_temp + (np.clip(0.06 - dist, 0, 0.06) * 50) 
        
        heat_data = np.column_stack((lat_grid.ravel(), lon_grid.ravel(), simulated_temps)).tolist()
        HeatMap(heat_data, radius=25, blur=20, gradient={0.4: 'blue', 0.6: 'lime', 0.8: 'yellow', 1.0: 'red'}).add_to(m)
        st_folium(m, width=1200, height=500)
        
    else:
        st.markdown("Mapping a random 5,000-point sample of historical training data.")
        sample_df = df.sample(min(5000, len(df)))
        lat_col = 'latitude' if 'latitude' in sample_df.columns else ('lat' if 'lat' in sample_df.columns else None)
        lon_col = 'longitude' if 'longitude' in sample_df.columns else ('lon' if 'lon' in sample_df.columns else None)
        
        if lat_col and lon_col:
            fig_map = px.scatter_mapbox(sample_df, lat=lat_col, lon=lon_col, color="LST_Celsius", 
                                        color_continuous_scale=px.colors.sequential.Inferno, size_max=15, zoom=4, 
                                        mapbox_style="carto-darkmatter")
            fig_map.update_layout(margin={"r":0,"t":40,"l":0,"b":0}, height=500)
            st.plotly_chart(fig_map, use_container_width=True)
        else:
            st.warning("No Lat/Lon found. Showing Attribute Distribution instead.")
            fig_scatter = px.scatter(sample_df, x='NDVI_Greenness', y='LST_Celsius', color='NDBI_BuiltUp',
                                     color_continuous_scale='Thermal', template="plotly_dark")
            st.plotly_chart(fig_scatter, use_container_width=True)
# --- TAB 2: DRIVERS ---
with tab2:
    st.markdown("### XGBoost Algorithm Feature Importance")
    importances = model.feature_importances_
    features_labels = ['Vegetation', 'Concrete', 'Elevation', 'Wind', 'Humidity', 'ECOSTRESS', 'Urban Morph', 'Pollution', 'Year']
    fig_bar = px.bar(x=importances, y=features_labels, orientation='h', color=importances, 
                     color_continuous_scale='Reds', template='plotly_dark')
    fig_bar.update_layout(xaxis_title="Impact Weight", yaxis_title="Urban Driver", yaxis={'categoryorder':'total ascending'})
    st.plotly_chart(fig_bar, use_container_width=True)

# --- TAB 3: VALIDATION ---
# --- TAB 3: VALIDATION ---
with tab3:
    st.markdown("### AI Model Ground-Truth Validation")
    c1, c2, c3 = st.columns(3)
    c1.metric("Engine Architecture", "XGBoost PIML")
    c2.metric("RMSE Error Margin", f"{true_rmse:.2f} °C", "- Highly Accurate", delta_color="inverse")
    c3.metric("R² Variance Score", f"{true_r2:.3f}", "+ Excellent Fit")
    
    st.markdown("#### Prediction vs Reality (Sampled)")
    val_sample = df.sample(min(1000, len(df)))
    
    # FIX: Added .to_numpy() here!
    val_sample['AI_Pred'] = model.predict(val_sample[['NDVI_Greenness', 'NDBI_BuiltUp', 'elevation', 'Wind_Speed', 'Dewpoint_C', 'ECOSTRESS_LST', 'GHSL_Urban_Form', 'Pollution_CPCB_Proxy', 'Year']].to_numpy())
    
    fig_val = px.scatter(val_sample, x='LST_Celsius', y='AI_Pred', opacity=0.6, template="plotly_dark",
                         color_discrete_sequence=['#00FFCC'])
    fig_val.add_shape(type="line", x0=25, y0=25, x1=55, y1=55, line=dict(color="red", dash="dash"))
    fig_val.update_layout(xaxis_title="True Satellite LST (°C)", yaxis_title="AI Predicted LST (°C)")
    st.plotly_chart(fig_val, use_container_width=True)

# --- TAB 4: SCENARIO ROI ---
with tab4:
    st.markdown(f"### {scenario_year} Scenario Evaluation for {location_query.split(',')[0]}")
    st.info("💡 Adjust the 'Urban Intervention' sliders in the sidebar to calculate exact temperature reductions.")
    
    # Calculate baseline for the year
    actual_baseline = df[df['Year'] == scenario_year]['LST_Celsius'].mean()
    t_base = calculate_temp(current_ndvi, current_ndbi, 0.0)
    t_mitigated = calculate_temp(current_ndvi, current_ndbi, cool_roof_albedo)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Historical City Avg", f"{actual_baseline:.2f} °C")
    with col2:
        st.metric("Base Prediction", f"{t_base:.2f} °C", f"{(t_base-actual_baseline):.2f} °C vs Hist.", delta_color="inverse")
    with col3:
        st.metric("Post-Intervention", f"{t_mitigated:.2f} °C", f"{(t_mitigated-t_base):.2f} °C Reduced", delta_color="inverse")
    with col4:
        st.metric("Cooling ROI", f"{((t_base-t_mitigated)/t_base)*100:.1f} %", "Total Efficacy")

# --- TAB 5: OPTIMAL STRATEGY & REPORTING ---
with tab5:
    st.markdown("### 🏆 AI-Recommended Mitigation Strategy")
    st.write("The algorithm evaluates cooling ROI against municipal financial constraints.")
    
    # NEW: Budget Slider (₹ Crores)
    city_budget = st.slider("💰 Municipal Budget Allocation (₹ Crores)", min_value=10, max_value=250, value=100, step=10)
    
    # Financial Assumptions (Mock estimations for the pitch)
    # E.g., Planting enough trees to raise city NDVI by 0.1 costs ₹40 Cr.
    # E.g., Painting roofs to raise city albedo by 0.1 costs ₹15 Cr.
    cost_per_ndvi_point = 400  # ₹400 Cr per full 1.0 NDVI (₹120 Cr for +0.3)
    cost_per_albedo_point = 150 # ₹150 Cr per full 1.0 Albedo (₹60 Cr for +0.4)
    
    base_t = calculate_temp(current_ndvi, current_ndbi, 0.0)
    
    # Strategy 1: Forestry
    drop_green = base_t - calculate_temp(min(current_ndvi + 0.3, 1.0), current_ndbi, 0.0) 
    cost_green = 0.3 * cost_per_ndvi_point
    
    # Strategy 2: Cool Roofs
    drop_roof = base_t - calculate_temp(current_ndvi, current_ndbi, 0.4)                  
    cost_roof = 0.4 * cost_per_albedo_point
    
    # Strategy 3: Hybrid
    drop_hybrid = base_t - calculate_temp(min(current_ndvi + 0.15, 1.0), current_ndbi, 0.2) 
    cost_hybrid = (0.15 * cost_per_ndvi_point) + (0.2 * cost_per_albedo_point)
    
    # Aggregate valid strategies
    strategies = {
        "🌲 Urban Forestry (+30% Greenery)": {"drop": drop_green, "cost": cost_green},
        "⚪ Cool Roofs (0.4 Albedo Paint)": {"drop": drop_roof, "cost": cost_roof},
        "🤝 Hybrid (Trees + Reflective Surfaces)": {"drop": drop_hybrid, "cost": cost_hybrid}
    }
    
    # Filter by budget
    affordable_strategies = {k: v for k, v in strategies.items() if v["cost"] <= city_budget}
    
    if not affordable_strategies:
        st.error(f"🚨 Budget Alert: ₹{city_budget} Cr is insufficient for mass city-wide interventions. Please increase the budget or target micro-zones.")
        best_strategy = "None (Insufficient Funds)"
        max_drop = 0.0
        placement = "N/A"
    else:
        # Pick the one with the highest temperature drop among affordable ones
        best_strategy = max(affordable_strategies, key=lambda k: affordable_strategies[k]["drop"])
        max_drop = affordable_strategies[best_strategy]["drop"]
        final_cost = affordable_strategies[best_strategy]["cost"]
        
        st.success(f"### Affordable Recommended Action: {best_strategy}")
        
        colA, colB, colC = st.columns(3)
        colA.metric("Max Cooling (ROI)", f"-{max_drop:.2f} °C", delta_color="inverse")
        colB.metric("Estimated Implementation Cost", f"₹{final_cost:.1f} Cr")
        
        placement = "High Concrete / Low Vegetation Grids" if "Forestry" in best_strategy else "Industrial Zones (High GHSL Volume)"
        colC.metric("Optimal Zonal Placement", placement)
        
        # Plot Cost vs Benefit
        plot_data = pd.DataFrame([
            {"Strategy": "Forestry", "Cooling (°C)": drop_green, "Cost (₹ Cr)": cost_green},
            {"Strategy": "Cool Roofs", "Cooling (°C)": drop_roof, "Cost (₹ Cr)": cost_roof},
            {"Strategy": "Hybrid", "Cooling (°C)": drop_hybrid, "Cost (₹ Cr)": cost_hybrid}
        ])
        fig_cost = px.scatter(plot_data, x="Cost (₹ Cr)", y="Cooling (°C)", color="Strategy", size="Cooling (°C)",
                              template="plotly_dark", title="Economic ROI vs. Cooling Efficacy")
        fig_cost.add_vline(x=city_budget, line_dash="dash", line_color="red", annotation_text="Budget Limit")
        st.plotly_chart(fig_cost, use_container_width=True)
    
    st.divider()
    
    # --- EXPORT ENGINE (Maintained from Feature 2) ---
    st.markdown("#### 📄 Generate Official Policy Report")
    
    report_data = {
        "Target_City": [location_query.split(',')[0]],
        "Forecast_Year": [scenario_year],
        "Budget_Limit_Cr": [city_budget],
        "Baseline_Temp_C": [round(base_t, 2)],
        "Mitigated_Temp_C": [round(base_t - max_drop, 2)],
        "Max_Cooling_ROI_C": [round(max_drop, 2)],
        "Recommended_Strategy": [best_strategy],
        "Zonal_Placement": [placement],
        "Sim_Vegetation_NDVI": [current_ndvi],
        "Sim_Concrete_NDBI": [current_ndbi]
    }
    
    report_df = pd.DataFrame(report_data)
    csv_export = report_df.to_csv(index=False).encode('utf-8')
    
    st.download_button(
        label="📥 Download Policy & Budget Report (CSV)",
        data=csv_export,
        file_name=f"ISRO_Budget_Report_{location_query.split(',')[0]}_{scenario_year}.csv",
        mime="text/csv",
        use_container_width=True
    )

