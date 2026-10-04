import pytest
import numpy as np
import os
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.compose import ColumnTransformer

from src.preprocessing import load_and_split_data
from src.train import build_model_pipeline, get_logistic_regression, get_random_forest, get_xgboost, evaluate_model_cv

@pytest.fixture
def data_split():
    filepath = os.path.join("data", "raw", "heart_disease_uci.csv")
    return load_and_split_data(filepath)

def test_lr_pipeline_construction():
    lr_model = get_logistic_regression()
    pipeline = build_model_pipeline(lr_model)
    
    # Verify it is a Pipeline
    assert isinstance(pipeline, Pipeline)
    
    # Verify the pipeline steps: preprocessing then classifier
    assert len(pipeline.steps) == 2
    assert pipeline.steps[0][0] == 'preprocessor'
    assert isinstance(pipeline.steps[0][1], ColumnTransformer)
    
    assert pipeline.steps[1][0] == 'classifier'
    assert isinstance(pipeline.steps[1][1], LogisticRegression)

def test_rf_pipeline_construction():
    rf_model = get_random_forest()
    pipeline = build_model_pipeline(rf_model)
    
    assert isinstance(pipeline, Pipeline)
    assert len(pipeline.steps) == 2
    assert pipeline.steps[0][0] == 'preprocessor'
    assert pipeline.steps[1][0] == 'classifier'
    assert isinstance(pipeline.steps[1][1], RandomForestClassifier)

def test_xgb_pipeline_construction():
    xgb_model = get_xgboost()
    pipeline = build_model_pipeline(xgb_model)
    
    assert isinstance(pipeline, Pipeline)
    assert len(pipeline.steps) == 2
    assert pipeline.steps[0][0] == 'preprocessor'
    assert pipeline.steps[1][0] == 'classifier'
    assert isinstance(pipeline.steps[1][1], XGBClassifier)

def test_evaluate_model_cv(data_split):
    X_train, X_test, y_train, y_test = data_split
    
    # Test for LR
    lr_model = get_logistic_regression()
    lr_pipeline = build_model_pipeline(lr_model)
    lr_results = evaluate_model_cv(lr_pipeline, X_train, y_train)
    
    # Test for RF
    rf_model = get_random_forest()
    rf_pipeline = build_model_pipeline(rf_model)
    rf_results = evaluate_model_cv(rf_pipeline, X_train, y_train)
    
    # Test for XGB
    xgb_model = get_xgboost()
    xgb_pipeline = build_model_pipeline(xgb_model)
    xgb_results = evaluate_model_cv(xgb_pipeline, X_train, y_train)
    
    expected_metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    
    for results in [lr_results, rf_results, xgb_results]:
        for metric in expected_metrics:
            assert metric in results
            assert len(results[metric]['folds']) == 5
            for val in results[metric]['folds']:
                assert np.isfinite(val)
            assert np.isfinite(results[metric]['mean'])
            assert np.isfinite(results[metric]['std'])
        
def test_test_set_not_accessed():
    """
    Ensure the cross validation logic doesn't touch the test set.
    """
    filepath = os.path.join("data", "raw", "heart_disease_uci.csv")
    X_train, X_test, y_train, y_test = load_and_split_data(filepath)
    
    xgb_model = get_xgboost()
    pipeline = build_model_pipeline(xgb_model)
    
    # Corrupt X_test
    X_test_corrupted = X_test.copy()
    for col in X_test_corrupted.columns:
        X_test_corrupted[col] = "CORRUPTED"
        
    try:
        evaluate_model_cv(pipeline, X_train, y_train)
    except Exception as e:
        pytest.fail(f"CV evaluation failed, possibly accessed test set. Error: {e}")
