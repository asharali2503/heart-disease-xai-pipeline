import pytest
import os
import joblib
import numpy as np
from src.preprocessing import load_and_split_data
from src.evaluate import evaluate_final_model

@pytest.fixture
def data_split():
    filepath = os.path.join("data", "raw", "heart_disease_uci.csv")
    return load_and_split_data(filepath)

def test_final_evaluation_and_persistence(data_split):
    X_train, X_test, y_train, y_test = data_split
    
    # 1. Run final evaluation (this fits the model and saves it)
    metrics, save_path = evaluate_final_model(X_train, y_train, X_test, y_test)
    
    # 2. Verify all reported test metrics are valid finite numbers between 0.0 and 1.0
    metric_keys = ['Accuracy', 'Precision', 'Recall', 'F1', 'ROC-AUC']
    for k in metric_keys:
        assert k in metrics
        assert np.isfinite(metrics[k])
        assert 0.0 <= metrics[k] <= 1.0
        
    # 3. The confusion matrix components sum to exactly 61 (the test set size)
    assert metrics['TN'] + metrics['FP'] + metrics['FN'] + metrics['TP'] == len(y_test)
    assert len(y_test) == 61
    
    # 4. The saved model artifact file exists
    assert os.path.exists(save_path)
    
    # 5. It can be reloaded via joblib
    reloaded_pipeline = joblib.load(save_path)
    
    # 6. The reloaded pipeline can make valid binary predictions and probability estimates on new raw data
    y_pred_reloaded = reloaded_pipeline.predict(X_test)
    y_pred_proba_reloaded = reloaded_pipeline.predict_proba(X_test)
    
    assert set(np.unique(y_pred_reloaded)).issubset({0, 1})
    assert y_pred_proba_reloaded.shape == (len(y_test), 2)
    assert np.all((y_pred_proba_reloaded >= 0.0) & (y_pred_proba_reloaded <= 1.0))
