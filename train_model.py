import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
import joblib
import os

# Check for dataset
dataset_file = "preprocessed_dataset.xlsx"
if not os.path.exists(dataset_file):
    raise FileNotFoundError(f"No dataset found! Place '{dataset_file}' in the folder.")

# Load dataset
df = pd.read_excel(dataset_file)

target_column = "Total_Laval_count"

X = df.drop(columns=[target_column, "Date"], errors="ignore")
y = df[target_column]

# Scale features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Train model
model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
model.fit(X_scaled, y)

# Save model, scaler, and feature columns
joblib.dump(model, "population_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(X.columns.tolist(), "feature_columns.pkl")

print("✅ Model, scaler, and feature columns saved!")