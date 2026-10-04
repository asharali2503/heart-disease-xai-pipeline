import pandas as pd
import numpy as np
import os
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from src.train import (
    build_model_pipeline, 
    get_logistic_regression, 
    get_random_forest, 
    get_xgboost, 
    evaluate_model_cv
)

def compare_models_cv(X_train, y_train):
    """
    Evaluates Logistic Regression, Random Forest, and XGBoost using CV on the training data.
    Returns a pandas DataFrame summarizing the metrics (Mean ± Std).
    """
    models = {
        'Logistic Regression': get_logistic_regression(),
        'Random Forest': get_random_forest(),
        'XGBoost': get_xgboost()
    }
    
    summary_data = []
    
    for model_name, model in models.items():
        pipeline = build_model_pipeline(model)
        results = evaluate_model_cv(pipeline, X_train, y_train)
        
        row = {'Model': model_name}
        
        # Store mean values for sorting
        for metric, stats in results.items():
            row[f'{metric}_mean'] = stats['mean']
            row[f'{metric}_std'] = stats['std']
            # Formatted string for display
            row[metric.capitalize()] = f"{stats['mean']:.4f} ± {stats['std']:.4f}"
            
        summary_data.append(row)
        
    df = pd.DataFrame(summary_data)
    
    # Sort by ROC-AUC primarily, then F1
    df = df.sort_values(by=['roc_auc_mean', 'f1_mean'], ascending=[False, False]).reset_index(drop=True)
    
    # Create a display dataframe with just the formatted columns
    display_cols = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1', 'Roc_auc']
    df_display = df[display_cols]
    
    return df, df_display

def evaluate_final_model(X_train, y_train, X_test, y_test):
    """
    Fits the final selected model (Logistic Regression) on the entire training set,
    evaluates it exactly once on the untouched test set, and saves the artifact.
    """
    print("\n--- Final Model Training and Evaluation ---")
    
    # 1. Build and fit pipeline
    model = get_logistic_regression()
    pipeline = build_model_pipeline(model)
    pipeline.fit(X_train, y_train)
    
    # 2. Predictions
    y_pred = pipeline.predict(X_test)
    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
    
    # 3. Metrics
    metrics = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1': f1_score(y_test, y_pred),
        'ROC-AUC': roc_auc_score(y_test, y_pred_proba)
    }
    
    # 4. Confusion Matrix
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    metrics['TN'] = tn
    metrics['FP'] = fp
    metrics['FN'] = fn
    metrics['TP'] = tp
    
    # 5. Save Artifact
    os.makedirs(os.path.join("models", "saved_models"), exist_ok=True)
    save_path = os.path.join("models", "saved_models", "final_model_pipeline.joblib")
    joblib.dump(pipeline, save_path)
    print(f"Model successfully saved to {save_path}")
    
    return metrics, save_path

if __name__ == "__main__":
    from src.preprocessing import load_and_split_data
    
    X_train, X_test, y_train, y_test = load_and_split_data()
    
    # CV comparison
    df_raw, df_display = compare_models_cv(X_train, y_train)
    print("\n--- Model Comparison CV Summary ---")
    print(df_display.to_string(index=False))
    
    # Final evaluation
    final_metrics, save_path = evaluate_final_model(X_train, y_train, X_test, y_test)
    print("\n--- Final Test Set Metrics ---")
    for k, v in final_metrics.items():
        if isinstance(v, float):
            print(f"{k}: {v:.4f}")
        else:
            print(f"{k}: {v}")
