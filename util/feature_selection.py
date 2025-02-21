import xgboost as xgb
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestRegressor


# Function to load data
def load_data(data_path, target):
    df = pd.read_csv(data_path)  # Replace with your data path
    df = df.drop(columns=["Date", "index"], errors="ignore")  # Drop non-numeric columns
    X = df.drop(columns=[target])
    y = df[target]
    return X, y


# Function to calculate XGBoost feature importance
def calculate_xgb_importance(X_train, y_train, X):
    xgb_model = xgb.XGBRegressor(
        objective="reg:squarederror",
    )
    xgb_model.fit(X_train, y_train)

    xgb_importance = xgb_model.feature_importances_
    xgb_importance_df = pd.DataFrame(
        {"Feature": X.columns, "XGB_Importance": xgb_importance}
    )
    xgb_importance_df = xgb_importance_df.sort_values(
        by="XGB_Importance", ascending=False
    )

    print("\n🎯 XGBoost Feature Importance:")
    print(xgb_importance_df)

    return xgb_importance_df, xgb_model


# Function to calculate SHAP feature contributions
def calculate_shap_importance(xgb_model, X_test, show_plot=True):
    explainer = shap.Explainer(xgb_model, X_test)
    shap_values = explainer(X_test)

    shap_importance = np.abs(shap_values.values).mean(axis=0)
    shap_importance_df = pd.DataFrame(
        {"Feature": X_test.columns, "SHAP_Importance": shap_importance}
    )
    shap_importance_df = shap_importance_df.sort_values(
        by="SHAP_Importance", ascending=False
    )

    print("\n🔥 SHAP Feature Contributions:")
    print(shap_importance_df)

    # Only plot if show_plot is True
    if show_plot:
        shap.summary_plot(shap_values, X_test)

    return shap_importance_df


# Function to calculate RFE (Recursive Feature Elimination) importance
def calculate_rfe_importance(X_train, y_train, n_features_to_select=10):
    rfe_model = RandomForestRegressor()
    rfe_selector = RFE(rfe_model, n_features_to_select=n_features_to_select)
    rfe_selector.fit(X_train, y_train)

    rfe_selected_features = X_train.columns[rfe_selector.support_]
    rfe_importance_df = pd.DataFrame(
        {"Feature": rfe_selected_features, "RFE_Selected": True}
    )

    print("\n✅ RFE Selected Features:")
    print(rfe_selected_features)

    return rfe_importance_df


# Function to calculate final feature score
def calculate_final_feature_score(
    xgb_importance_df, shap_importance_df, rfe_importance_df
):
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

    return final_features_df


# Function to plot feature importance comparison
def plot_feature_importance(final_features_df):
    plt.figure(figsize=(12, 6))
    plt.barh(
        final_features_df["Feature"], final_features_df["Final_Score"], color="skyblue"
    )
    plt.xlabel("Final Feature Score")
    plt.ylabel("Features")
    plt.title("Final Feature Selection Based on XGB + SHAP + RFE")
    plt.gca().invert_yaxis()
    plt.show()


# Function to select top N features
def select_top_n_features(final_features_df, n=10):
    return final_features_df.head(n)


# Function to perform the entire feature selection process
def perform_feature_selection(data_path, target, n_top_features=10, show_plot=True):
    # 1️⃣ Load Data
    X, y = load_data(data_path, target)

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
    )

    # 2️⃣ Calculate Feature Importance using XGBoost
    xgb_importance_df, xgb_model = calculate_xgb_importance(X_train, y_train, X)

    # 3️⃣ Calculate Feature Contributions using SHAP
    shap_importance_df = calculate_shap_importance(xgb_model, X_test, show_plot)

    # 4️⃣ Recursive Feature Elimination (RFE)
    rfe_importance_df = calculate_rfe_importance(X_train, y_train)

    # 5️⃣ Final Feature Selection
    final_features_df = calculate_final_feature_score(
        xgb_importance_df, shap_importance_df, rfe_importance_df
    )

    # 6️⃣ Optionally Plot the feature importance comparison
    if show_plot:
        plot_feature_importance(final_features_df)

    # 7️⃣ Get the best n features based on the final score
    best_features = select_top_n_features(final_features_df, n=n_top_features)

    return best_features


target = "Return"
data_path = "data/cleaned_data.csv"
n_top_features = 10

best_features_without_plot = perform_feature_selection(
    data_path, target, n_top_features, show_plot=True
)
print(f"Best {n_top_features} features:")
print(best_features_without_plot["Feature"].values)

# Close ==> ['Low' 'High' 'EMA_50' 'Open' 'EMA_10' 'SMA_10' 'SMA_50' 'BB_Lower' 'BB_Upper' 'BB_Mid']
# Return ==> ['Close','RSI_14','Volume','Sentiment_Negative','MACD','Open','Signal_Line','Sentiment_Positive','Weekday','EMA_50','Sentiment_Neutral']
# Merged:
#     [
#         "Low",
#         "High",
#         "EMA_50",
#         "Open",
#         "EMA_10",
#         "SMA_10",
#         "SMA_50",
#         "BB_Mid",
#         "Sentiment_Positive",
#         "Close",
#         "RSI_14",
#         "MACD",
#         "Volume",
#         "Signal_Line",
#         "Sentiment_Negative",
#         "BB_Lower",
#         "BB_Upper",
#         "Weekday",
#     ]
