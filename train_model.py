import pandas as pd
from sklearn.metrics import classification_report, accuracy_score
from sklearn.utils.class_weight import compute_sample_weight
import xgboost as xgb
import joblib

# 1. Load the dataset
df = pd.read_csv('agri_forest_fire_data_spatial.csv')

# 2. Spatial Block Split
# Train on NW, SW, SE regions; Test on completely unseen NE region
train_mask = df['spatial_block'] != 'NE'
test_mask = df['spatial_block'] == 'NE'

features = ['NDVI', 'neighborhood_avg_ndvi', 'LST', 'rainfall_deficit']

X_train = df.loc[train_mask, features]
y_train = df.loc[train_mask, 'fire_risk_category']

X_test = df.loc[test_mask, features]
y_test = df.loc[test_mask, 'fire_risk_category']

# 3. Handle Class Imbalance
# Compute weights so the model pays extra attention to the rare 'High Risk' pixels
sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)

# 4. Train XGBoost
model = xgb.XGBClassifier(
    n_estimators=150,
    max_depth=4,
    learning_rate=0.08,
    objective='multi:softprob',
    random_state=42
)

model.fit(X_train, y_train, sample_weight=sample_weights)

# SAVE THE TRAINED MODEL
joblib.dump(model, 'xgboost_fire_model.pkl')
print("Model saved successfully as xgboost_fire_model.pkl")

# 5. Evaluate on Unseen Geographic Region
y_pred = model.predict(X_test)
print(f"\n--- Evaluation on Unseen Spatial Block (NE Region) ---")
print("Spatial-Validation Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=['Low', 'Medium', 'High']))