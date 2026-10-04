import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_validate
from src.preprocessing import build_preprocessor, get_cv_strategy

def build_model_pipeline(model):
    """
    Combines the preprocessing pipeline with a given machine learning model.
    """
    preprocessor = build_preprocessor()
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])
    return pipeline

def get_logistic_regression():
    """
    Returns a configured Logistic Regression model.
    """
    return LogisticRegression(max_iter=1000, random_state=42)

def get_random_forest():
    """
    Returns a configured Random Forest model.
    """
    from sklearn.ensemble import RandomForestClassifier
    return RandomForestClassifier(n_estimators=100, random_state=42)

def get_xgboost():
    """
    Returns a configured XGBoost model.
    """
    from xgboost import XGBClassifier
    return XGBClassifier(random_state=42, eval_metric='logloss')

def evaluate_model_cv(pipeline, X, y):
    """
    Evaluates a model pipeline using StratifiedKFold cross-validation.
    Returns a dictionary of mean and std for various metrics.
    """
    cv_strategy = get_cv_strategy()
    
    scoring = {
        'accuracy': 'accuracy',
        'precision': 'precision',
        'recall': 'recall',
        'f1': 'f1',
        'roc_auc': 'roc_auc'
    }
    
    cv_results = cross_validate(
        pipeline, 
        X, 
        y, 
        cv=cv_strategy, 
        scoring=scoring,
        n_jobs=-1, # use all processors
        return_train_score=False
    )
    
    metrics_summary = {}
    for metric_name in scoring.keys():
        test_metric = cv_results[f'test_{metric_name}']
        metrics_summary[metric_name] = {
            'mean': np.mean(test_metric),
            'std': np.std(test_metric),
            'folds': test_metric.tolist()
        }
        
    return metrics_summary

if __name__ == "__main__":
    from src.preprocessing import load_and_split_data
    
    X_train, X_test, y_train, y_test = load_and_split_data()
    
    print("--- Logistic Regression ---")
    lr_model = get_logistic_regression()
    lr_pipeline = build_model_pipeline(lr_model)
    lr_results = evaluate_model_cv(lr_pipeline, X_train, y_train)
    for metric, stats in lr_results.items():
        print(f"{metric.capitalize()}: {stats['mean']:.4f} ± {stats['std']:.4f}")
        
    print("\n--- Random Forest ---")
    rf_model = get_random_forest()
    rf_pipeline = build_model_pipeline(rf_model)
    rf_results = evaluate_model_cv(rf_pipeline, X_train, y_train)
    for metric, stats in rf_results.items():
        print(f"{metric.capitalize()}: {stats['mean']:.4f} ± {stats['std']:.4f}")
        
    print("\n--- XGBoost ---")
    xgb_model = get_xgboost()
    xgb_pipeline = build_model_pipeline(xgb_model)
    xgb_results = evaluate_model_cv(xgb_pipeline, X_train, y_train)
    for metric, stats in xgb_results.items():
        print(f"{metric.capitalize()}: {stats['mean']:.4f} ± {stats['std']:.4f}")
