import os
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score
import xgboost as xgb
import joblib

# 1. Load the dataset
df = pd.read_csv('agri_forest_fire_data_spatial.csv')

# 2. Spatial Block Split
train_mask = df['spatial_block'] != 'NE'
test_mask = df['spatial_block'] == 'NE'

features = ['NDVI', 'neighborhood_avg_ndvi', 'LST', 'rainfall_deficit']

X_train = df.loc[train_mask, features]
y_train = df.loc[train_mask, 'fire_risk_category']

X_test = df.loc[test_mask, features]
y_test = df.loc[test_mask, 'fire_risk_category']

# 3. Explicitly compute class focal multipliers to fix the "Low Risk" bias
classes = np.unique(y_train)
total_samples = len(y_train)
class_weights_dict = {}

for c in classes:
    count = np.sum(y_train == c)
    # Inverse frequency balancing
    class_weights_dict[c] = total_samples / (len(classes) * count)

# Assign a custom weight to each individual row sample
sample_weights = y_train.map(class_weights_dict)

# 4. Train XGBoost Classifier
model = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.05,
    objective='multi:softprob',
    random_state=42
)

# Train using our computed weights
model.fit(X_train, y_train, sample_weight=sample_weights)

# Save the trained model file out to the root directory
joblib.dump(model, 'xgboost_fire_model.pkl')
print("SUCCESS: re-weighted model saved as xgboost_fire_model.pkl")

# 5. Local validation check
y_pred = model.predict(X_test)
print("\n--- Updated Model Performance Evaluation ---")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, zero_division=0))