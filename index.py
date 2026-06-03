import os
import pandas as pd
import joblib
from flask import Flask, render_template, request

app = Flask(__name__)

# Dynamically find the absolute path of the directory this script lives in
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'xgboost_fire_model.pkl')

# Safely load the model
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
    print("SUCCESS: Model loaded perfectly.")
else:
    model = None
    print(f"ERROR: Model not found at {MODEL_PATH}")

METRICS = {
    "accuracy": "0.93",
    "report": """
               precision    recall  f1-score   support

         Low       0.98      0.93      0.96       613
      Medium       0.69      0.92      0.79       107
        High       0.00      0.00      0.00         3

    accuracy                           0.93       723
   macro avg       0.56      0.62      0.58       723
weighted avg       0.94      0.93      0.93       723
"""
}

RISK_MAPPING = {0: "Low", 1: "Medium", 2: "High"}

@app.route("/", methods=["GET", "POST"])
def home():
    prediction_label = None
    inputs = {
        "NDVI": 0.35,
        "neighborhood_avg_ndvi": 0.38,
        "LST": 34.5,
        "rainfall_deficit": 45.0
    }

    if request.method == "POST":
        inputs["NDVI"] = float(request.form.get("NDVI", 0.35))
        inputs["neighborhood_avg_ndvi"] = float(request.form.get("neighborhood_avg_ndvi", 0.38))
        inputs["LST"] = float(request.form.get("LST", 34.5))
        inputs["rainfall_deficit"] = float(request.form.get("rainfall_deficit", 45.0))

        if model:
            features_df = pd.DataFrame([{
                'NDVI': inputs["NDVI"],
                'neighborhood_avg_ndvi': inputs["neighborhood_avg_ndvi"],
                'LST': inputs["LST"],
                'rainfall_deficit': inputs["rainfall_deficit"]
            }])
            pred_idx = int(model.predict(features_df)[0])
            prediction_label = RISK_MAPPING.get(pred_idx, "Unknown")
        else:
            prediction_label = "Model File Missing"

    return render_template("index.html", prediction=prediction_label, metrics=METRICS, inputs=inputs)

# Essential for local terminal execution
if __name__ == "__main__":
    app.run(debug=True)