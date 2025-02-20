import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def pca(df):
    # Check if 'Date' and 'Close' columns are in the dataframe
    if "Close" not in df.columns:
        raise ValueError("The input DataFrame must contain 'Close' columns.")

    # Save the 'Close' column separately before any transformations
    close_column = df["Close"]

    # Clean the dataframe by dropping unnecessary columns ('index' and 'Date')
    df_cleaned = df.drop(
        columns=["index", "Date"], errors="ignore"
    )  # Ignore if columns don't exist

    # Debug: Check the remaining columns after cleaning
    print(f"Cleaned columns: {df_cleaned.columns.tolist()}")

    # Standardize all features (excluding 'Close')
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df_cleaned)

    # Perform PCA with 10 components
    pca = PCA(n_components=10)
    principal_components = pca.fit_transform(scaled_features)

    # Create a new dataframe with the PCA results and maintain the original index
    df_pca = pd.DataFrame(
        principal_components, columns=[f"PC{i}" for i in range(1, 11)], index=df.index
    )

    # Add the 'Close' column back to the dataframe, ensuring proper index alignment
    df_pca["Close"] = close_column

    # Debug: Check the columns of the new PCA dataframe
    print(f"Columns of df_pca: {df_pca.columns.tolist()}")

    # Check if 'Close' column contains any NaN values
    if df_pca["Close"].isna().any():
        print("Warning: 'Close' column contains NaN values.")

    # Save the PCA results to a CSV file
    df_pca.to_csv("data/pca_data.csv", index=True)

    return df_pca
