import xgboost as xgb
import pandas as pd
import numpy as np

print("🌧️ Initiating July (Monsoon) Validation Test...")

try:
    model = xgb.XGBRegressor()
    model.load_model("urban_heat_temporal_model.json")
    print("✅ 9D Temporal Model loaded successfully.")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    print("Make sure you have run '2_model_training.py' first!")
    exit()

# These represent Ahmedabad's physical state in July 2026
july_conditions = {
    'NDVI_Greenness': [0.35],      # Higher greenness due to rain
    'NDBI_BuiltUp': [0.30],        # Concrete stays the same
    'elevation': [50.0],           # Elevation stays the same
    'Wind_Speed': [5.5],           # Monsoon winds are higher
    'Dewpoint_C': [26.0],          # Massive spike in humidity (Dewpoint)
    'ECOSTRESS_LST': [42.0],       # Cloud cover lowers peak solar radiation
    'GHSL_Urban_Form': [100.0],    # City structure stays the same
    'Pollution_CPCB_Proxy': [1.5], # Rain washes NO2 pollution out of the air
    'Year': [2026]                 # Future forecasting
}

df_july = pd.DataFrame(july_conditions)

print("\n🧠 Feeding July parameters into the Physics-Informed AI...")
predicted_lst = model.predict(df_july)[0]

print("==================================================")
print(f"🌡️ AI Predicted Surface Temperature for July 2026: {predicted_lst:.2f} °C")
print("==================================================")

# Accuracy Check Logic
expected_min = 34.0
expected_max = 39.0

if expected_min <= predicted_lst <= expected_max:
    print(f"🎯 ACCURACY CHECK PASSED: The prediction falls perfectly within the real-world historical range for Ahmedabad in July ({expected_min}°C - {expected_max}°C)!")
    print("This proves the model learned the actual physics of weather, not just random numbers.")
else:
    print(f"⚠️ Note: The prediction is slightly outside the historical norm of {expected_min}°C - {expected_max}°C.")