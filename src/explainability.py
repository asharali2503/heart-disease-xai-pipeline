import pandas as pd
import numpy as np
import shap

def get_feature_names(pipeline):
    """
    Extracts feature names from the fitted ColumnTransformer.
    """
    preprocessor = pipeline.named_steps['preprocessor']
    # If the version of scikit-learn supports get_feature_names_out directly on ColumnTransformer
    feature_names = preprocessor.get_feature_names_out()
    
    # Optional: clean up the names to remove the prefix added by ColumnTransformer (e.g. 'num__', 'cat__')
    cleaned_names = [name.split('__')[-1] for name in feature_names]
    return cleaned_names

def get_feature_coefficients(pipeline):
    """
    Extracts the coefficients from the fitted LogisticRegression model.
    Calculates Odds Ratios and Returns a structured DataFrame sorted by absolute coefficient magnitude.
    """
    classifier = pipeline.named_steps['classifier']
    feature_names = get_feature_names(pipeline)
    
    coefs = classifier.coef_[0]
    odds_ratios = np.exp(coefs)
    
    df = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient (β)': coefs,
        'Odds Ratio': odds_ratios
    })
    
    df['Direction of Risk'] = df['Coefficient (β)'].apply(lambda x: 'Increases' if x > 0 else 'Decreases')
    
    # Sort by absolute coefficient magnitude
    df['abs_coef'] = df['Coefficient (β)'].abs()
    df = df.sort_values(by='abs_coef', ascending=False).drop(columns=['abs_coef']).reset_index(drop=True)
    
    return df

def setup_shap_explainer(pipeline, X_train):
    """
    Transforms X_train and initializes a shap.LinearExplainer using the fitted classifier.
    """
    preprocessor = pipeline.named_steps['preprocessor']
    classifier = pipeline.named_steps['classifier']
    
    X_train_transformed = preprocessor.transform(X_train)
    feature_names = get_feature_names(pipeline)
    
    # LinearExplainer works well with LogisticRegression
    # Note: masker can be simply the background dataset
    explainer = shap.LinearExplainer(classifier, X_train_transformed)
    
    return explainer, feature_names

def compute_global_shap(explainer, pipeline, X_sample):
    """
    Computes SHAP values for a sample and returns the mean absolute SHAP values per feature.
    """
    preprocessor = pipeline.named_steps['preprocessor']
    X_sample_transformed = preprocessor.transform(X_sample)
    
    feature_names = get_feature_names(pipeline)
    
    shap_values = explainer.shap_values(X_sample_transformed)
    
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    
    df = pd.DataFrame({
        'Feature': feature_names,
        'Mean |SHAP|': mean_abs_shap
    })
    
    df = df.sort_values(by='Mean |SHAP|', ascending=False).reset_index(drop=True)
    
    return df

def explain_prediction(explainer, pipeline, single_input_df):
    """
    Preprocesses a single row of raw input, computes SHAP values, and returns a clean mapping.
    """
    preprocessor = pipeline.named_steps['preprocessor']
    feature_names = get_feature_names(pipeline)
    
    X_transformed = preprocessor.transform(single_input_df)
    
    # Expected value (base value)
    base_value = explainer.expected_value
    
    # Compute SHAP values
    shap_values = explainer.shap_values(X_transformed)[0]
    
    # Also get the feature values to display alongside
    feature_values = X_transformed[0]
    
    df = pd.DataFrame({
        'Feature': feature_names,
        'Feature Value (Transformed)': feature_values,
        'SHAP Contribution': shap_values
    })
    
    df['Direction'] = df['SHAP Contribution'].apply(lambda x: 'Pushes towards disease' if x > 0 else 'Pushes away from disease')
    
    # Sort by absolute SHAP contribution
    df['abs_shap'] = df['SHAP Contribution'].abs()
    df = df.sort_values(by='abs_shap', ascending=False).drop(columns=['abs_shap']).reset_index(drop=True)
    
    return {
        'base_value': base_value,
        'shap_summary_df': df,
        'total_prediction_log_odds': base_value + shap_values.sum()
    }

if __name__ == "__main__":
    from src.preprocessing import load_and_split_data
    import joblib
    import os
    
    X_train, X_test, y_train, y_test = load_and_split_data()
    pipeline = joblib.load(os.path.join('models', 'saved_models', 'final_model_pipeline.joblib'))
    
    print("\n--- Top 5 Features Increasing Risk (Positive Coefficients) ---")
    coef_df = get_feature_coefficients(pipeline)
    print(coef_df[coef_df['Coefficient (β)'] > 0].head(5).to_string(index=False))
    
    print("\n--- Top 5 Features Decreasing Risk (Negative Coefficients) ---")
    print(coef_df[coef_df['Coefficient (β)'] < 0].head(5).to_string(index=False))
    
    explainer, f_names = setup_shap_explainer(pipeline, X_train)
    
    print("\n--- Top Global SHAP Features ---")
    global_shap_df = compute_global_shap(explainer, pipeline, X_train)
    print(global_shap_df.head(5).to_string(index=False))
    
    print("\n--- Single Instance Explanation ---")
    single_obs = X_test.iloc[[0]]
    explanation = explain_prediction(explainer, pipeline, single_obs)
    print(f"Base Value (Log-Odds): {explanation['base_value']:.4f}")
    print(f"Final Model Log-Odds: {explanation['total_prediction_log_odds']:.4f}")
    print("\nTop 5 contributing factors:")
    print(explanation['shap_summary_df'].head(5).to_string(index=False))
