import xgboost as xgb

print("Loading massive JSON model...")
model = xgb.XGBRegressor()
model.load_model("urban_heat_temporal_model.json")

print("Saving as lightweight Binary model...")
# Saving it as .ubj (Universal Binary JSON) shrinks it and stops RAM spikes!
model.save_model("urban_heat_temporal_model.ubj") 

print("Done! You can now use the .ubj file.")
