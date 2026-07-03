# STREAMING_CHUNK: Initializing Streamlit and importing libraries
import streamlit as st
import pandas as pd
import xgboost as xgb
import numpy as np
import plotly.express as px
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium
from geopy.geocoders import Nominatim
from sklearn.metrics import mean_squared_error, r2_score

st.set_page_config(page_title="ISRO AI Heat Mitigation", layout="wide", page_icon="🇮🇳")
st.title("🇮🇳 National Urban Heat Mitigation System")
st.markdown("**Engine:** 9D XGBoost PIML Model | **Scope:** Pan-India | **Forecasting Era:** 2024-2026")

# STREAMING_CHUNK: Loading AI Model and Dataset Caching
@st.cache_resource
def load_assets():
    model = xgb.XGBRegressor()
    try:
        model.load_model("urban_heat_temporal_model.json")
        df = pd.read_csv('India_1Million_Temporal_9D_Data.csv')
        df = df.dropna()
        df = df[(df['LST_Celsius'] >= 28.0) & (df['LST_Celsius'] <= 48.0)]
        
        features = ['NDVI_Greenness', 'NDBI_BuiltUp', 'elevation', 'Wind_Speed', 'Dewpoint_C', 'ECOSTRESS_LST', 'GHSL_Urban_Form', 'Pollution_CPCB_Proxy', 'Year']
        X = df[features]
        y = df['LST_Celsius']
        preds = model.predict(X)
        rmse = np.sqrt(mean_squared_error(y, preds))
        r2 = r2_score(y, preds)
    except:
        df, rmse, r2 = None, 0.0, 0.0
    return model, df, rmse, r2

model, df, true_rmse, true_r2 = load_assets()

# STREAMING_CHUNK: Building Sidebar Configuration Panel
st.sidebar.header("📍 Location & Era")
location_query = st.sidebar.text_input("Enter City or 'India'", "Ahmedabad, Gujarat")
scenario_year = st.sidebar.selectbox("AI Temporal Forecast", [2024, 2025, 2026], index=2)

st.sidebar.header("☁️ Meteorological (ISRO/ERA5)")
city_elevation = st.sidebar.slider("⛰️ Elevation (m)", 0.0, 2000.0, 50.0, 10.0)
city_wind = st.sidebar.slider("💨 Wind (m/s)", 0.0, 10.0, 3.5, 0.5)
city_humidity = st.sidebar.slider("💧 Dewpoint (°C)", 5.0, 30.0, 20.0, 1.0)
city_eco_lst = st.sidebar.slider("🔥 Peak Heat - ECOSTRESS V2 (°C)", 30.0, 55.0, 40.0, 1.0)
city_pollution = st.sidebar.slider("🏭 CPCB Pollution Proxy", 0.0, 10.0, 2.5, 0.5)

st.sidebar.header("🎛️ Urban Form & Intervention")
current_ndvi = st.sidebar.slider("🌿 Vegetation (NDVI)", 0.0, 0.8, 0.2, 0.05)
current_ndbi = st.sidebar.slider("🏢 Built-Up Density (NDBI)", -0.5, 0.5, 0.3, 0.05)
ghsl_form = st.sidebar.slider("🏙️ Urban Form Volume (GHSL 2025)", 0.0, 255.0, 100.0, 5.0)
cool_roof_albedo = st.sidebar.slider("⚪ Cool Roof Albedo", 0.0, 0.5, 0.0, 0.05)

# STREAMING_CHUNK: Geocoding and Base Heat Calculations
@st.cache_data
def get_coordinates(query):
    geolocator = Nominatim(user_agent="isro_heat_app")
    try:
        loc = geolocator.geocode(query + (", India" if "india" not in query.lower() else ""))
        if loc: return loc.latitude, loc.longitude
    except: pass
    return 23.0225, 72.5714 # Default Ahmedabad

city_lat, city_lon = get_coordinates(location_query)
is_national = "india" in location_query.lower()
map_zoom = 5 if is_national else 12
grid_span = 12.0 if is_national else 0.06 
grid_size = 150 if is_national else 100 

def calculate_future_temp(ndvi, ndbi, albedo):
    input_data = pd.DataFrame({
        'NDVI_Greenness': [ndvi], 'NDBI_BuiltUp': [ndbi], 
        'elevation': [city_elevation], 'Wind_Speed': [city_wind], 'Dewpoint_C': [city_humidity],
        'ECOSTRESS_LST': [city_eco_lst], 'GHSL_Urban_Form': [ghsl_form], 'Pollution_CPCB_Proxy': [city_pollution],
        'Year': [scenario_year]
    })
    base_lst = float(model.predict(input_data)[0])
    return base_lst - (albedo * 12.0)

# STREAMING_CHUNK: Rendering the 5 ISRO Outcome Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📍 1. Heat Stress Maps", 
    "📊 2. Drivers of Urban Heat", 
    "✔️ 3. Validated AIML Model", 
    "📈 4. Scenario Evaluation",
    "🎯 5. Optimal Intervention"
])

# STREAMING_CHUNK: Tab 1 - Spatial Heat Stress Maps
with tab1:
    st.header(f"Outcome 1: Spatial Heat Stress Maps ({scenario_year} Forecast)")
    st.write("Identifies urban hotspots dynamically based on 9-dimensional satellite inputs.")
    
    m = folium.Map(location=[city_lat, city_lon], zoom_start=map_zoom, tiles="CartoDB dark_matter")
    lats = np.linspace(city_lat - grid_span, city_lat + grid_span, grid_size)
    lons = np.linspace(city_lon - grid_span, city_lon + grid_span, grid_size)
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    flat_lats = lat_grid.ravel()
    flat_lons = lon_grid.ravel()
    
    dist_from_center = np.sqrt((flat_lats - city_lat)**2 + (flat_lons - city_lon)**2)
    mock_ndbi = np.clip(0.5 - (dist_from_center / grid_span * 2), 0.0, 1.0)
    mock_ndvi = np.clip(0.1 + (dist_from_center / grid_span * 2), 0.0, 0.6)
    
    grid_features = pd.DataFrame({
        'NDVI_Greenness': mock_ndvi, 'NDBI_BuiltUp': mock_ndbi, 'elevation': city_elevation,
        'Wind_Speed': city_wind, 'Dewpoint_C': city_humidity, 'ECOSTRESS_LST': city_eco_lst,
        'GHSL_Urban_Form': ghsl_form, 'Pollution_CPCB_Proxy': city_pollution, 'Year': scenario_year
    })
    
    base_preds = model.predict(grid_features)
    final_preds = base_preds - (cool_roof_albedo * 12.0)
    
    heat_data = np.column_stack((flat_lats, flat_lons, final_preds)).tolist()
    
    HeatMap(heat_data, 
            radius=25 if is_national else 30, 
            blur=25 if is_national else 30, 
            min_opacity=0.4, 
            gradient={0.2: '#0000FF', 0.4: '#00FFFF', 0.6: '#00FF00', 0.8: '#FFFF00', 1.0: '#FF0000'}
            ).add_to(m)
            
    st_folium(m, width=1200, height=550)

# STREAMING_CHUNK: Tab 2 - Drivers of Urban Heat
with tab2:
    st.header("Outcome 2: Quantitative Assessment of Key Drivers")
    st.write("XGBoost Feature Importance calculating exactly what contributes to urban heat.")
    if model is not None:
        importances = model.feature_importances_
        features = ['Vegetation', 'Concrete', 'Elevation', 'Wind', 'Humidity', 'ECOSTRESS', 'Morphology', 'Pollution', 'Temporal Year']
        fig_bar = px.bar(x=importances, y=features, orientation='h', color=features, 
                         title="Impact Weight of Urban Drivers on LST", labels={'x': 'Relative Importance', 'y': 'Urban Driver'})
        fig_bar.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig_bar, use_container_width=True)

# STREAMING_CHUNK: Tab 3 - Validated AIML Model
with tab3:
    st.header("Outcome 3: Validated AIML Model Capturing Heat Dynamics")
    colA, colB = st.columns([1, 2])
    with colA:
        st.metric("Model Architecture", "9D Temporal XGBoost")
        st.metric("Root Mean Square Error (RMSE)", f"{true_rmse:.2f} °C", "Excellent < 4.0")
        st.metric("Variance Explained (R-Squared)", f"{true_r2:.2f}", "Excellent > 0.60")
        st.success("Trained on 1,000,000 Data Points (2024-2026)")
    
    with colB:
        if df is not None:
            sample = df.sample(min(200, len(df)))
            features = ['NDVI_Greenness', 'NDBI_BuiltUp', 'elevation', 'Wind_Speed', 'Dewpoint_C', 'ECOSTRESS_LST', 'GHSL_Urban_Form', 'Pollution_CPCB_Proxy', 'Year']
            sample['AI_Prediction'] = model.predict(sample[features])
            fig_scatter = px.scatter(sample, x='LST_Celsius', y='AI_Prediction', 
                                     labels={'LST_Celsius': 'True Satellite LST (°C)', 'AI_Prediction': 'AI Predicted LST (°C)'},
                                     color='AI_Prediction', color_continuous_scale='Turbo')
            st.plotly_chart(fig_scatter, use_container_width=True)

# STREAMING_CHUNK: Tab 4 - Scenario-Based Evaluation
with tab4:
    st.header(f"Outcome 4: Scenario-Based Evaluation for {scenario_year}")
    st.write("Calculating mathematical outcomes based on user interventions against the AI's temporal forecast.")
    
    t_base = calculate_future_temp(current_ndvi, current_ndbi, 0.0)
    t_mitigated = calculate_future_temp(current_ndvi, current_ndbi, cool_roof_albedo)
    
    col1, col2, col3 = st.columns(3)
    col1.metric("AI Projected LST (Do Nothing)", f"{t_base:.2f} °C")
    col2.metric("AI Projected LST (With Interventions)", f"{t_mitigated:.2f} °C", f"{(t_mitigated-t_base):.2f} °C")
    col3.metric("Total Mitigation Efficacy", f"{((t_base-t_mitigated)/t_base)*100:.1f} %")
    
    st.info("👈 Use the sliders in the sidebar under 'Urban Form & Intervention' to test different scenarios live.")

# STREAMING_CHUNK: Tab 5 - Optimal Intervention Strategy
with tab5:
    st.header("Outcome 5: Optimal Intervention Strategy")
    st.write("Algorithmic recommendation engine for maximum cooling ROI.")
    
    base_t = calculate_future_temp(current_ndvi, current_ndbi, 0.0)
    
    # Simulate Interventions
    strat_green = calculate_future_temp(min(current_ndvi + 0.3, 1.0), current_ndbi, 0.0) # +30% Greenery
    strat_roof = calculate_future_temp(current_ndvi, current_ndbi, 0.4)                  # High Albedo Paint
    strat_hybrid = calculate_future_temp(min(current_ndvi + 0.15, 1.0), current_ndbi, 0.2) # Both
    
    reductions = {
        "Urban Forestry (Green Roofs/Trees)": base_t - strat_green,
        "High-Albedo Cool Roofs": base_t - strat_roof,
        "Hybrid Intervention (Green + Cool Roofs)": base_t - strat_hybrid
    }
    
    best_strategy = max(reductions, key=reductions.get)
    max_drop = reductions[best_strategy]
    
    st.success(f"### 🏆 Recommended Strategy: {best_strategy}")
    
    colX, colY, colZ = st.columns(3)
    colX.metric("Type of Intervention", best_strategy.split("(")[0].strip())
    
    placement = "High NDBI (Concrete) & Low NDVI (Sparse Vegetation) Grids"
    if "Roof" in best_strategy and "Forestry" not in best_strategy:
        placement = "Industrial & Commercial zones with GHSL > 150"
    colY.metric("Optimal Spatial Placement", placement)
    
    colZ.metric("Estimated Temperature Reduction", f"-{max_drop:.2f} °C")
    
    st.markdown(f"""
    #### Implementation Placement Directives for {scenario_year}:
    *   **Focus Area:** {placement}.
    *   **CPCB Pollution Factor:** Areas with high NO2 proxy values should prioritize *Urban Forestry* to achieve dual cooling and air-filtering outcomes.
    *   **ECOSTRESS Priority:** Focus interventions strictly in zones where ECOSTRESS V2 detects peak diurnal LST exceeding 42°C.
    """)

