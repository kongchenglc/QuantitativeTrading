import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Should be used only on train dataset instead of full dataset
# And pca can cause some non-linear relationships loss in LSTM
def pca(df, target="Return"):
    print("PCAing...")
    if target not in df.columns:
        raise ValueError(f"The input DataFrame must contain {target} columns.")

    target_column = df[target]

    df_cleaned = df.drop(
        columns=["index", "Date"], errors="ignore"
    )  # Ignore if columns don't exist

    print(f"Cleaned columns: {df_cleaned.columns.tolist()}")

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df_cleaned)

    pca = PCA(n_components=10)
    principal_components = pca.fit_transform(scaled_features)

    df_pca = pd.DataFrame(
        principal_components, columns=[f"PC{i}" for i in range(1, 11)], index=df.index
    )

    df_pca[target] = target_column

    print(f"Columns of df_pca: {df_pca.columns.tolist()}")

    if df_pca[target].isna().any():
        print(f"Warning: {target} column contains NaN values.")

    df_pca.to_csv("data/pca_data.csv", index=True)

    print("PCA Done")
    return df_pca
