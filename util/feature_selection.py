import xgboost as xgb
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestRegressor

# =======================
# 1️⃣  Load Data
# =======================
df = pd.read_csv("data/cleaned_data.csv")  # Replace with your data path
# df = pd.read_csv("data/pca_data.csv")  # Replace with your data path
df = df.drop(columns=["Date", "index"], errors="ignore")  # Drop non-numeric columns

# Target variable & Features
X = df.drop(columns=["Close"])  # Assuming 'Close' is the target value
y = df["Close"]

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =======================
# 2️⃣  Calculate Feature Importance using XGBoost
# =======================
xgb_model = xgb.XGBRegressor(objective="reg:squarederror", random_state=42)
xgb_model.fit(X_train, y_train)

xgb_importance = xgb_model.feature_importances_
xgb_importance_df = pd.DataFrame(
    {"Feature": X.columns, "XGB_Importance": xgb_importance}
)
xgb_importance_df = xgb_importance_df.sort_values(by="XGB_Importance", ascending=False)

print("\n🎯 XGBoost Feature Importance:")
print(xgb_importance_df)

# =======================
# 3️⃣  Calculate Feature Contributions using SHAP
# =======================
explainer = shap.Explainer(xgb_model, X_test)
shap_values = explainer(X_test)

shap_importance = np.abs(shap_values.values).mean(
    axis=0
)  # Take the mean of absolute values
shap_importance_df = pd.DataFrame(
    {"Feature": X.columns, "SHAP_Importance": shap_importance}
)
shap_importance_df = shap_importance_df.sort_values(
    by="SHAP_Importance", ascending=False
)

print("\n🔥 SHAP Feature Contributions:")
print(shap_importance_df)

# Plot SHAP Importance
shap.summary_plot(shap_values, X_test)

# =======================
# 4️⃣  Recursive Feature Elimination (RFE)
# =======================
rfe_model = RandomForestRegressor(random_state=42)
rfe_selector = RFE(
    rfe_model, n_features_to_select=10
)  # Select 10 most important features
rfe_selector.fit(X_train, y_train)

rfe_selected_features = X.columns[rfe_selector.support_]
rfe_importance_df = pd.DataFrame(
    {"Feature": rfe_selected_features, "RFE_Selected": True}
)

print("\n✅ RFE Selected Features:")
print(rfe_selected_features)

# =======================
# 5️⃣  Final Feature Selection
# =======================
final_features_df = xgb_importance_df.merge(shap_importance_df, on="Feature").merge(
    rfe_importance_df, on="Feature", how="left"
)
final_features_df["RFE_Selected"] = final_features_df["RFE_Selected"].fillna(False)

# Calculate final score: XGB + SHAP + (RFE selected features weighted)
final_features_df["Final_Score"] = (
    final_features_df["XGB_Importance"] * 0.4
    + final_features_df["SHAP_Importance"] * 0.4
    + final_features_df["RFE_Selected"].astype(int) * 0.2
)

# Sort by final score
final_features_df = final_features_df.sort_values(by="Final_Score", ascending=False)
print("\n📌 Final Selected Features:")
print(final_features_df)

# =======================
# 6️⃣  Plot: Final Feature Importance Comparison
# =======================
plt.figure(figsize=(12, 6))
plt.barh(
    final_features_df["Feature"], final_features_df["Final_Score"], color="skyblue"
)
plt.xlabel("Final Feature Score")
plt.ylabel("Features")
plt.title("Final Feature Selection Based on XGB + SHAP + RFE")
plt.gca().invert_yaxis()
plt.show()
