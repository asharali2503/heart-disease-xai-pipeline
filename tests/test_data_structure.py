import pandas as pd
import os

def test_raw_data_structure():
    file_path = os.path.join("data", "raw", "heart_disease_uci.csv")
    assert os.path.exists(file_path), f"Dataset not found at {file_path}"
    
    df = pd.read_csv(file_path)
    
    # 1. Check dimensions
    assert len(df) == 303, f"Expected 303 rows, got {len(df)}"
    assert len(df.columns) == 14, f"Expected 14 columns, got {len(df.columns)}"
    
    # 2. Check exact columns
    expected_columns = [
        'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
        'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'num'
    ]
    assert list(df.columns) == expected_columns, f"Columns do not match expected list"
    
    # 3. Check target values
    unique_targets = set(df['num'].dropna().unique())
    assert unique_targets.issubset({0, 1, 2, 3, 4}), f"Unexpected values in target: {unique_targets}"
    
    # 4. Check missing values
    missing_counts = df.isnull().sum()
    assert missing_counts['ca'] == 4, f"Expected 4 missing in ca, got {missing_counts['ca']}"
    assert missing_counts['thal'] == 2, f"Expected 2 missing in thal, got {missing_counts['thal']}"
    assert missing_counts.sum() == 6, f"Expected 6 total missing values, got {missing_counts.sum()}"

if __name__ == "__main__":
    test_raw_data_structure()
    print("All tests passed! The dataset structure matches expectations.")
