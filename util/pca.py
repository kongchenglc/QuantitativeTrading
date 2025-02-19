import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def pca(df):

    df_cleaned = df.drop(columns=["index", "Date"], errors="ignore")

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df_cleaned.drop(columns=["Close"]))

    pca = PCA(n_components=10)
    principal_components = pca.fit_transform(scaled_features)

    df_pca = pd.DataFrame(
        principal_components, columns=[f"PC{i}" for i in range(1, 11)]
    )
    df_pca["Close"] = df_cleaned["Close"]

    df_pca.to_csv("data/pca_data.csv", index=True)

    return df_pca
