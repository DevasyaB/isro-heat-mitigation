import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import time

print("🧠 Loading massive 9-Dimensional Temporal Dataset...")
df = pd.read_csv('India_1Million_Temporal_9D_Data.csv')

df = df.dropna()
# Restrict to realistic Indian terrestrial temperature ranges (28°C to 48°C)
df = df[(df['LST_Celsius'] >= 28.0) & (df['LST_Celsius'] <= 48.0)]

print(f"Dataset successfully loaded with {len(df)} rows of clean data!")

# The AI explicitly learns the 'Year' to naturally forecast future macro-climate trends
features = [
    'NDVI_Greenness', 
    'NDBI_BuiltUp', 
    'elevation', 
    'Wind_Speed', 
    'Dewpoint_C', 
    'ECOSTRESS_LST', 
    'GHSL_Urban_Form', 
    'Pollution_CPCB_Proxy', 
    'Year'
]
X = df[features]
y = df['LST_Celsius']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\n🚀 Initializing High-Accuracy XGBoost on NVIDIA RTX GPU...")

model = xgb.XGBRegressor(
    n_estimators=4000,       
    learning_rate=0.01,      
    max_depth=10,             
    subsample=0.85,           
    colsample_bytree=0.85,    
    reg_alpha=0.5,           # L1 Regularization: Drops useless features, reducing future error
    reg_lambda=1.5,          # L2 Regularization: Prevents overfitting, allowing it to predict future years
    tree_method='hist',
    device='cuda',           # Harnesses the NVIDIA RTX GPU for ultra-fast training
    random_state=42
)

start_time = time.time()
model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=500)
end_time = time.time()

print(f"\n✅ Training Complete in {end_time - start_time:.2f} seconds!")

y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\n📊 9-Dimensional Temporal Model Performance Metrics:")
print(f"RMSE: {rmse:.2f} °C")
print(f"R-Squared: {r2:.2f}")

model.save_model("urban_heat_temporal_model.json")
print("\n💾 Temporal model saved successfully as 'urban_heat_temporal_model.json'.")
print("You are now ready to launch the dashboard!")
