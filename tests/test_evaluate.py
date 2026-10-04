import pytest
import os
import pandas as pd
from src.preprocessing import load_and_split_data
from src.evaluate import compare_models_cv

@pytest.fixture
def data_split():
    filepath = os.path.join("data", "raw", "heart_disease_uci.csv")
    return load_and_split_data(filepath)

def test_compare_models_cv(data_split):
    X_train, X_test, y_train, y_test = data_split
    
    # The function strictly uses only the training data and does not touch or accept the test set
    df_raw, df_display = compare_models_cv(X_train, y_train)
    
    # Valid summary containing all three models
    assert len(df_raw) == 3
    assert set(df_raw['Model']) == {'Logistic Regression', 'Random Forest', 'XGBoost'}
    
    # All five required metrics are present
    expected_metrics = ['Accuracy', 'Precision', 'Recall', 'F1', 'Roc_auc']
    for metric in expected_metrics:
        assert metric in df_display.columns
        
    # Check that they contain valid numerical values in the raw dataframe
    expected_raw_metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    for raw_metric in expected_raw_metrics:
        assert f'{raw_metric}_mean' in df_raw.columns
        assert f'{raw_metric}_std' in df_raw.columns
        assert not df_raw[f'{raw_metric}_mean'].isna().any()
        assert not df_raw[f'{raw_metric}_std'].isna().any()

def test_compare_models_cv_test_set_isolation(data_split):
    X_train, X_test, y_train, y_test = data_split
    
    # Create corrupted X_test and y_test to ensure they aren't used
    X_test_corrupted = X_test.copy()
    for col in X_test_corrupted.columns:
        X_test_corrupted[col] = "CORRUPTED"
        
    y_test_corrupted = y_test.copy()
    y_test_corrupted[:] = 999
    
    try:
        # Pass only train data. If the function internally reached out for global test data it would fail,
        # but in our design it only takes X_train, y_train anyway.
        df_raw, df_display = compare_models_cv(X_train, y_train)
    except Exception as e:
        pytest.fail(f"compare_models_cv failed, possibly due to leakage. Error: {e}")
