import pytest
import os
import joblib
import numpy as np
from src.preprocessing import load_and_split_data
from src.explainability import (
    get_feature_names, 
    get_feature_coefficients, 
    setup_shap_explainer, 
    compute_global_shap, 
    explain_prediction
)

@pytest.fixture
def data_split():
    filepath = os.path.join("data", "raw", "heart_disease_uci.csv")
    return load_and_split_data(filepath)

@pytest.fixture
def final_pipeline():
    return joblib.load(os.path.join('models', 'saved_models', 'final_model_pipeline.joblib'))

def test_feature_extraction(final_pipeline):
    feature_names = get_feature_names(final_pipeline)
    
    # 25 features after one hot encoding
    assert len(feature_names) == 25
    
    # Assert no prefix
    assert not any(name.startswith('num__') or name.startswith('cat__') for name in feature_names)
    
def test_get_feature_coefficients(final_pipeline):
    coef_df = get_feature_coefficients(final_pipeline)
    
    assert len(coef_df) == 25
    assert 'Feature' in coef_df.columns
    assert 'Coefficient (β)' in coef_df.columns
    assert 'Odds Ratio' in coef_df.columns
    
    # Odds ratios are strictly positive
    assert (coef_df['Odds Ratio'] > 0).all()
    
    # Properly sorted by absolute coefficient
    abs_coefs = coef_df['Coefficient (β)'].abs().values
    assert all(abs_coefs[i] >= abs_coefs[i+1] for i in range(len(abs_coefs)-1))

def test_setup_shap_explainer(final_pipeline, data_split):
    X_train, _, _, _ = data_split
    
    # Should initialize without errors
    explainer, feature_names = setup_shap_explainer(final_pipeline, X_train)
    
    assert explainer is not None
    assert len(feature_names) == 25
    
def test_explain_prediction(final_pipeline, data_split):
    X_train, X_test, _, _ = data_split
    
    explainer, _ = setup_shap_explainer(final_pipeline, X_train)
    
    single_obs = X_test.iloc[[0]]
    explanation = explain_prediction(explainer, final_pipeline, single_obs)
    
    assert 'base_value' in explanation
    assert 'shap_summary_df' in explanation
    assert 'total_prediction_log_odds' in explanation
    
    df = explanation['shap_summary_df']
    assert len(df) == 25
    
    # Verify SHAP base value plus sum of SHAP values matches the model's decision function output
    classifier = final_pipeline.named_steps['classifier']
    preprocessor = final_pipeline.named_steps['preprocessor']
    X_transformed = preprocessor.transform(single_obs)
    
    decision_function_output = classifier.decision_function(X_transformed)[0]
    
    calculated_log_odds = explanation['total_prediction_log_odds']
    
    # within numerical tolerance
    np.testing.assert_almost_equal(decision_function_output, calculated_log_odds, decimal=5)
