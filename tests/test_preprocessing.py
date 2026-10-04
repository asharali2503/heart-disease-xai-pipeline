import pytest
import pandas as pd
import numpy as np
import os
from sklearn.exceptions import NotFittedError
from src.preprocessing import load_and_split_data, build_preprocessor, get_cv_strategy, NUMERICAL_FEATURES, CATEGORICAL_FEATURES, RANDOM_SEED, TEST_SIZE

@pytest.fixture
def data_split():
    filepath = os.path.join("data", "raw", "heart_disease_uci.csv")
    return load_and_split_data(filepath)

def test_target_binarization_and_num_dropped(data_split):
    X_train, X_test, y_train, y_test = data_split
    
    # Verify num is not included
    assert 'num' not in X_train.columns
    assert 'num' not in X_test.columns
    
    # Verify target is not included in predictive features
    assert 'target' not in X_train.columns
    assert 'target' not in X_test.columns
    
    # Check target values are only 0 and 1
    assert set(y_train.unique()).issubset({0, 1})
    assert set(y_test.unique()).issubset({0, 1})

def test_split_reproducibility():
    filepath = os.path.join("data", "raw", "heart_disease_uci.csv")
    X_train1, X_test1, y_train1, y_test1 = load_and_split_data(filepath)
    X_train2, X_test2, y_train2, y_test2 = load_and_split_data(filepath)
    
    pd.testing.assert_frame_equal(X_train1, X_train2)
    pd.testing.assert_frame_equal(X_test1, X_test2)
    pd.testing.assert_series_equal(y_train1, y_train2)
    pd.testing.assert_series_equal(y_test1, y_test2)

def test_expected_feature_columns_present(data_split):
    X_train, X_test, _, _ = data_split
    expected_cols = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
    
    # Verify all 13 original predictive features are accounted for
    assert set(expected_cols) == set(X_train.columns)
    assert len(X_train.columns) == 13
    
    # Verify exact feature lists
    assert set(NUMERICAL_FEATURES) == {'age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'ca'}
    assert set(CATEGORICAL_FEATURES) == {'sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'thal'}

def test_train_test_split_proportions(data_split):
    X_train, X_test, y_train, y_test = data_split
    
    total_samples = len(X_train) + len(X_test)
    expected_test_size = int(total_samples * TEST_SIZE)
    assert abs(len(X_test) - expected_test_size) <= 1
    
    train_prop = y_train.mean()
    test_prop = y_test.mean()
    assert abs(train_prop - test_prop) < 0.05

def test_preprocessing_architecture():
    # 1. No preprocessing object is fitted merely by importing
    preprocessor = build_preprocessor()
    
    # We can check it's not fitted by attempting to transform and catching NotFittedError, 
    # or checking for the presence of the fitted attribute 'transformers_'
    assert not hasattr(preprocessor, 'transformers_') or not preprocessor.transformers_
    
    # Verify handle_unknown="ignore" is enabled
    # We can inspect the categorical pipeline
    cat_transformer = preprocessor.transformers[1][1] # ('cat', categorical_pipeline, CATEGORICAL_FEATURES)
    encoder = cat_transformer.named_steps['encoder']
    assert encoder.handle_unknown == "ignore"

def test_preprocessing_fit_transform(data_split):
    X_train, X_test, y_train, y_test = data_split
    
    preprocessor = build_preprocessor()
    
    # Create artificial missing values in test set to verify it can be handled
    # (Since ca and thal already have missing values, this just ensures robustness)
    X_test_missing = X_test.copy()
    X_test_missing.iloc[0, X_test_missing.columns.get_loc('age')] = np.nan
    X_test_missing.iloc[0, X_test_missing.columns.get_loc('sex')] = np.nan
    
    # Ensure preprocessor can be fitted on training data
    X_train_transformed = preprocessor.fit_transform(X_train)
    
    # Ensure preprocessor can then transform test data (handling missing values via imputation)
    X_test_transformed = preprocessor.transform(X_test_missing)
    
    # The transformed train and test matrices have the same number of columns
    assert X_train_transformed.shape[1] == X_test_transformed.shape[1]
    
    # Verify missing values are handled (no NaNs in transformed output)
    assert not np.isnan(X_train_transformed).any()
    assert not np.isnan(X_test_transformed).any()
    
    # Verify categorical encoding works (feature expansion)
    assert X_train_transformed.shape[1] > 13
    
def test_cv_strategy(data_split):
    X_train, _, y_train, _ = data_split
    cv = get_cv_strategy()
    
    # Verify basic config
    assert cv.n_splits == 5
    assert cv.random_state == RANDOM_SEED
    assert cv.shuffle == True
    
    # Use the CV object to generate splits on the training data
    splits = list(cv.split(X_train, y_train))
    
    # 1. Exactly 5 folds are created
    assert len(splits) == 5
    
    all_val_indices = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        # 2. No overlap between training and validation indices within any fold
        assert len(set(train_idx).intersection(set(val_idx))) == 0
        
        # 3. Every fold contains both classes
        y_val_fold = y_train.iloc[val_idx]
        assert set(y_val_fold.unique()) == {0, 1}
        
        all_val_indices.extend(val_idx)
        
    # 4. Every observation in the training set is used exactly once as a validation observation
    # by ensuring the sorted validation indices perfectly match the full index range
    all_val_indices_sorted = sorted(all_val_indices)
    assert all_val_indices_sorted == list(range(len(X_train)))
