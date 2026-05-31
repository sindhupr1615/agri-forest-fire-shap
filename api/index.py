from flask import Flask, render_template, request
import pandas as pd
import xgboost as xgb
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.join(current_dir, '..')

app = Flask(
    __name__,
    template_folder=os.path.join(root_dir, "templates"),
    static_folder=os.path.join(root_dir, "static")
)

# Static validation figures from your notebook evaluation
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

def load_and_train_live_model():
    csv_path = os.path.join(root_dir, 'agri_forest_fire_data_spatial.csv')
    df = pd.read_csv(csv_path)
    
    # Core modeling features matching your pipeline layout
    features = ['NDVI', 'neighborhood_avg_ndvi', 'LST', 'rainfall_deficit']
    X = df[features]
    y = df['fire_risk_category']
    
    # Convert string categories to numerical index labels for XGBoost execution
    if y.dtype == 'object':
        y = y.map({'Low': 0, 'Medium': 1, 'High': 2})
        
    model = xgb.XGBClassifier(n_estimators=50, max_depth=4, learning_rate=0.1, random_state=42)
    model.fit(X, y)
    return model

# Global initialization wrapper to safely track the model state
try:
    ML_MODEL = load_and_train_live_model()
except Exception as e:
    ML_MODEL = None
    print("Inference model baseline load failure:", e)

@app.route("/", methods=["GET"])
def home():
    # Initial page loads without any active user predictions
    return render_template("index.html", metrics=METRICS, prediction=None)

@app.route("/predict", methods=["POST"])
def predict():
    if ML_MODEL is None:
        return render_template("index.html", metrics=METRICS, prediction="Error: Model Uninitialized")
        
    try:
        # Fetch individual pixel entries straight from the browser form inputs
        user_features = pd.DataFrame([{
            'NDVI': float(request.form.get('NDVI', 0.35)),
            'neighborhood_avg_ndvi': float(request.form.get('neighborhood_avg_ndvi', 0.38)),
            'LST': float(request.form.get('LST', 34.5)),
            'rainfall_deficit': float(request.form.get('rainfall_deficit', 45.0))
        }])
        
        # Calculate real-time array prediction indices
        raw_pred = int(ML_MODEL.predict(user_features)[0])
        idx_to_label = {0: "Low", 1: "Medium", 2: "High"}
        predicted_class = idx_to_label.get(raw_pred, "Unknown")
        
    except Exception as err:
        predicted_class = f"Pipeline Error: {str(err)}"
        
    return render_template("index.html", metrics=METRICS, prediction=predicted_class)

app.index = app