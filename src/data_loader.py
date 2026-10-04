import pandas as pd
from ucimlrepo import fetch_ucirepo
import json
import os

def load_and_inspect_data():
    print("Fetching dataset from UCI ML Repo (ID: 45 - Heart Disease)...")
    # fetch dataset
    heart_disease = fetch_ucirepo(id=45)
    
    # data (as pandas dataframes)
    X = heart_disease.data.features
    y = heart_disease.data.targets
    
    # original raw dataset combines X and y
    df = pd.concat([X, y], axis=1)
    
    # save raw dataset
    raw_path = os.path.join("data", "raw", "heart_disease_uci.csv")
    df.to_csv(raw_path, index=False)
    print(f"Saved raw dataset to {raw_path}")
    
    print("\n--- Dataset Metadata ---")
    print(f"Name: {heart_disease.metadata['name']}")
    print(f"Data URL: {heart_disease.metadata['data_url']}")
    print(f"Number of Instances: {heart_disease.metadata['num_instances']}")
    print(f"Number of Features: {heart_disease.metadata['num_features']}")
    print(f"Missing Values: {heart_disease.metadata['has_missing_values']}")
    
    print("\n--- Dataset Inspection ---")
    print(f"Number of rows: {len(df)}")
    print(f"Number of columns: {len(df.columns)}")
    print(f"\nExact column names:\n{list(df.columns)}")
    
    print("\nData Types:")
    print(df.dtypes)
    
    print("\nMissing Values:")
    print(df.isnull().sum())
    
    print("\nHow missing values are represented:")
    print("Missing values in pandas are typically NaN. Let's see if there are any other placeholders like '?' or -9 (but ucimlrepo usually parses these).")
    
    print("\nDuplicate Rows:")
    duplicates = df[df.duplicated(keep=False)]
    print(f"Found {len(duplicates)} duplicate rows (this counts all occurrences of duplicated rows). Unique duplicated rows: {df.duplicated().sum()}")
    if len(duplicates) > 0:
        print(duplicates)
        
    print("\nTarget Column Info:")
    target_col = y.columns[0]
    print(f"Target column name: {target_col}")
    print(f"Unique values in target: {df[target_col].unique()}")
    print(f"Target class distribution:\n{df[target_col].value_counts(dropna=False)}")
    
    print("\nUnique values of every feature (to help identify categorical vs numerical):")
    for col in df.columns:
        unique_vals = df[col].unique()
        if len(unique_vals) < 15:
            print(f" - {col}: {sorted([v for v in unique_vals if pd.notnull(v)])} (Missing: {df[col].isnull().sum()})")
        else:
            print(f" - {col}: {len(unique_vals)} unique values (Numerical/Continuous) (Missing: {df[col].isnull().sum()})")

    # Let's check variables table from ucimlrepo
    print("\n--- Variable Definitions from UCI ---")
    print(heart_disease.variables[['name', 'role', 'type', 'description', 'missing_values']])

if __name__ == "__main__":
    load_and_inspect_data()
