import pytest
import pandas as pd
import numpy as np
from app.utils import convert_inputs_to_df, load_model

def test_convert_inputs_to_df():
    sample_input = {
        'age': 55, 'sex': 1, 'cp': 4, 'trestbps': 120, 'chol': 200, 
        'fbs': 0, 'restecg': 0, 'thalach': 150, 'exang': 0, 
        'oldpeak': 0.0, 'slope': 2, 'ca': 0, 'thal': 3
    }
    
    df = convert_inputs_to_df(sample_input)
    
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1
    assert set(df.columns) == set(sample_input.keys())
    assert df.iloc[0]['age'] == 55

def test_pipeline_on_example_profiles():
    # Load cached model
    pipeline = load_model()
    assert pipeline is not None
    
    # 1. Low Risk Example
    low_risk = {
        'age': 35, 'sex': 0, 'cp': 1, 'trestbps': 110, 'chol': 180, 
        'fbs': 0, 'restecg': 0, 'thalach': 180, 'exang': 0, 
        'oldpeak': 0.0, 'slope': 1, 'ca': 0, 'thal': 3
    }
    
    # 2. High Risk Example
    high_risk = {
        'age': 65, 'sex': 1, 'cp': 4, 'trestbps': 160, 'chol': 280, 
        'fbs': 1, 'restecg': 2, 'thalach': 110, 'exang': 1, 
        'oldpeak': 3.5, 'slope': 2, 'ca': 3, 'thal': 7
    }
    
    for profile in [low_risk, high_risk]:
        df = convert_inputs_to_df(profile)
        
        # Valid prediction
        pred = pipeline.predict(df)
        assert pred[0] in [0, 1]
        
        # Valid prob scores
        prob = pipeline.predict_proba(df)
        assert prob.shape == (1, 2)
        assert 0.0 <= prob[0][1] <= 1.0
