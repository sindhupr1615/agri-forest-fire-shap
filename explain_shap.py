import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
import shap

df = pd.read_csv('agri_forest_fire_data_spatial.csv')
features = ['NDVI', 'neighborhood_avg_ndvi', 'LST', 'rainfall_deficit']
X = df[features]

# Quick refit on full data for visualization mapping
model = xgb.XGBClassifier(n_estimators=150, max_depth=4, learning_rate=0.08, random_state=42)
model.fit(X, df['fire_risk_category'])

# Compute SHAP values
explainer = shap.TreeExplainer(model)
shap_values = explainer(X)

# 1. Save Summary Plots for both Medium (1) and High (2) Risk
for class_idx, class_name in [(1, 'Medium_Risk'), (2, 'High_Risk')]:
    plt.figure(figsize=(10, 5))
    shap.summary_plot(shap_values[:, :, class_idx], X, show=False)
    plt.title(f"Feature Contributions driving {class_name.replace('_', ' ')}")
    plt.tight_layout()
    plt.savefig(f'shap_summary_{class_name}.png')
    plt.close()

# 2. Advanced Spatial Mapping: Comparing Drivers Across the Map
df['SHAP_LST_HighRisk'] = shap_values[:, 'LST', 2].values
df['SHAP_NeighNDVI_HighRisk'] = shap_values[:, 'neighborhood_avg_ndvi', 2].values

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7), sharey=True)

# Map 1: LST contribution to High Risk
sc1 = ax1.scatter(df['longitude'], df['latitude'], c=df['SHAP_LST_HighRisk'], cmap='coolwarm', s=15, alpha=0.6)
fig.colorbar(sc1, ax=ax1, label='SHAP Value')
ax1.set_title('Where Land Surface Temp (LST) Increases High Fire Risk')
ax1.set_xlabel('Longitude')
ax1.set_ylabel('Latitude')

# Map 2: Neighborhood NDVI contribution to High Risk
sc2 = ax2.scatter(df['longitude'], df['latitude'], c=df['SHAP_NeighNDVI_HighRisk'], cmap='PiYG', s=15, alpha=0.6)
fig.colorbar(sc2, ax=ax2, label='SHAP Value')
ax2.set_title('Where Neighborhood Dryness (Low NDVI) Increases High Fire Risk')
ax2.set_xlabel('Longitude')

plt.suptitle("Spatial SHAP Patterns: What is Driving Risk at the Agri-Forest Boundary?", fontsize=16, y=0.98)
plt.tight_layout()
plt.savefig('advanced_spatial_shap_map.png')
plt.close()

print("Advanced Spatial SHAP complete! Check 'advanced_spatial_shap_map.png'.")