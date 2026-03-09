# # app.py
# import streamlit as st
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import joblib
# from sklearn.metrics import r2_score, mean_absolute_error

# # -----------------------------
# # Load Pre-trained Model
# # -----------------------------
# @st.cache_resource
# def load_model():
#     model = joblib.load("population_model.pkl")
#     scaler = joblib.load("scaler.pkl")
#     feature_columns = joblib.load("feature_columns.pkl")
#     return model, scaler, feature_columns

# model, scaler, feature_columns = load_model()

# st.set_page_config(page_title="Population Monitoring System", layout="wide")
# st.title("🦋 Larval Population Monitoring Dashboard")

# # -----------------------------
# # Upload Dataset
# # -----------------------------
# uploaded_file = st.file_uploader("Upload population dataset (CSV or Excel)", type=["csv", "xlsx"])

# if uploaded_file:

#     # Load dataset
#     if uploaded_file.name.endswith(".xlsx"):
#         df = pd.read_excel(uploaded_file)
#     else:
#         df = pd.read_csv(uploaded_file)

#     st.subheader("Dataset Preview")
#     st.dataframe(df.tail())

#     # -----------------------------
#     # Preprocessing
#     # -----------------------------
#     target_column = "Total_Laval_count"

#     # Ensure correct feature order
#     X = df[feature_columns]
#     y = df[target_column]
#     X_scaled = scaler.transform(X)

#     # -----------------------------
#     # Predictions
#     # -----------------------------
#     y_pred_full = model.predict(X_scaled)

#     # -----------------------------
#     # Accuracy Calculation
#     # -----------------------------
#     r2 = r2_score(y, y_pred_full)
#     mae = mean_absolute_error(y, y_pred_full)

#     col1, col2 = st.columns(2)
#     col1.metric("R² Score", round(r2, 3))
#     col2.metric("MAE", round(mae, 2))

#     # -----------------------------
#     # 7-Day Forecast
#     # -----------------------------
#     last_rows = df.iloc[-7:].copy()
#     future_preds = []

#     for i in range(7):
#         X_last = last_rows[feature_columns]
#         X_last_scaled = scaler.transform(X_last)
#         pred = model.predict([X_last_scaled[-1]])[0]
#         future_preds.append(pred)

#         # Append predicted row for next iteration
#         new_row = last_rows.iloc[-1].copy()
#         new_row[target_column] = pred
#         last_rows = pd.concat([last_rows, pd.DataFrame([new_row])], ignore_index=True)

#     st.subheader("📅 7-Day Population Forecast")
#     forecast_df = pd.DataFrame({
#         "Day Ahead": [f"Day +{i}" for i in range(1,8)],
#         "Predicted Population": np.round(future_preds, 2)
#     })
#     st.dataframe(forecast_df)

#     # -----------------------------
#     # Plot Historical + Forecast
#     # -----------------------------
#     st.subheader("📈 Population Trend")
#     fig, ax = plt.subplots(figsize=(10,5))
#     ax.plot(df.index, df[target_column], label="Historical")
#     ax.plot(range(len(df), len(df)+7), future_preds, label="Forecast")
#     ax.set_xlabel("Time")
#     ax.set_ylabel("Total Larval Count")
#     ax.legend()
#     st.pyplot(fig)

#     # -----------------------------
#     # Outbreak Alert
#     # -----------------------------
#     if future_preds[0] > df[target_column].mean() * 1.5:
#         st.error("⚠️ High Outbreak Risk Detected!")
#     else:
#         st.success("Population within normal range.")

# else:
#     st.info("Please upload your dataset to start monitoring.")

from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
import os
from sklearn.metrics import r2_score, mean_absolute_error

app = Flask(__name__)

# -----------------------------
# Load Pre-trained Model
# -----------------------------
model = joblib.load("population_model.pkl")
scaler = joblib.load("scaler.pkl")
feature_columns = joblib.load("feature_columns.pkl")

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs("static", exist_ok=True)

@app.route("/", methods=["GET", "POST"])
def index():
    preview_data = None
    forecast_data = None
    metrics = None
    alert_message = None
    alert_type = None
    plot_path = None
    error_message = None

    if request.method == "POST":
        file = request.files.get("dataset")

        if not file or file.filename == "":
            error_message = "Please upload a CSV or Excel file."
            return render_template(
                "index.html",
                error_message=error_message
            )

        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)

        try:
            # -----------------------------
            # Load dataset
            # -----------------------------
            if file.filename.endswith(".xlsx"):
                df = pd.read_excel(filepath)
            elif file.filename.endswith(".csv"):
                df = pd.read_csv(filepath)
            else:
                error_message = "Unsupported file type. Upload CSV or XLSX."
                return render_template(
                    "index.html",
                    error_message=error_message
                )

            # Clean column names
            df.columns = df.columns.str.strip()

            target_column = "Total_Laval_count"

            if target_column not in df.columns:
                error_message = f"Target column '{target_column}' not found in dataset."
                return render_template(
                    "index.html",
                    error_message=error_message
                )

            missing_cols = [col for col in feature_columns if col not in df.columns]
            if missing_cols:
                error_message = f"Missing columns in dataset: {missing_cols}"
                return render_template(
                    "index.html",
                    error_message=error_message
                )

            # -----------------------------
            # Dataset preview
            # -----------------------------
            preview_data = df.tail().to_dict(orient="records")
            preview_columns = df.columns.tolist()

            # -----------------------------
            # Preprocessing
            # -----------------------------
            X = df[feature_columns]
            y = df[target_column]
            X_scaled = scaler.transform(X)

            # -----------------------------
            # Predictions
            # -----------------------------
            y_pred_full = model.predict(X_scaled)

            # -----------------------------
            # Accuracy metrics
            # -----------------------------
            r2 = r2_score(y, y_pred_full)
            mae = mean_absolute_error(y, y_pred_full)

            metrics = {
                "r2": round(r2, 3),
                "mae": round(mae, 2)
            }

            # -----------------------------
            # 7-Day Forecast
            # -----------------------------
            last_rows = df.iloc[-7:].copy()
            future_preds = []

            for i in range(7):
                X_last = last_rows[feature_columns]
                X_last_scaled = scaler.transform(X_last)
                pred = model.predict([X_last_scaled[-1]])[0]
                future_preds.append(float(pred))

                new_row = last_rows.iloc[-1].copy()
                new_row[target_column] = pred
                last_rows = pd.concat([last_rows, pd.DataFrame([new_row])], ignore_index=True)

            forecast_data = [
                {
                    "day": f"Day +{i+1}",
                    "population": round(future_preds[i], 2)
                }
                for i in range(7)
            ]

            # -----------------------------
            # Plot Historical + Forecast
            # -----------------------------
            plt.figure(figsize=(10, 5))
            plt.plot(df.index, df[target_column], label="Historical")
            plt.plot(range(len(df), len(df) + 7), future_preds, label="Forecast")
            plt.xlabel("Time")
            plt.ylabel("Total Larval Count")
            plt.title("Population Trend")
            plt.legend()
            plt.tight_layout()

            plot_path = "static/plot.png"
            plt.savefig(plot_path)
            plt.close()

            # -----------------------------
            # Outbreak Alert
            # -----------------------------
            if future_preds[0] > df[target_column].mean() * 1.5:
                alert_message = "High Outbreak Risk Detected!"
                alert_type = "danger"
            else:
                alert_message = "Population within normal range."
                alert_type = "success"

            return render_template(
                "index.html",
                preview_data=preview_data,
                preview_columns=preview_columns,
                forecast_data=forecast_data,
                metrics=metrics,
                alert_message=alert_message,
                alert_type=alert_type,
                plot_path=plot_path
            )

        except Exception as e:
            error_message = f"Error: {str(e)}"

    return render_template(
        "index.html",
        preview_data=preview_data,
        forecast_data=forecast_data,
        metrics=metrics,
        alert_message=alert_message,
        alert_type=alert_type,
        plot_path=plot_path,
        error_message=error_message
    )

if __name__ == "__main__":
    app.run(debug=True)