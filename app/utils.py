import streamlit as st
import joblib
import pandas as pd
import os
import matplotlib.pyplot as plt
import shap

from src.explainability import setup_shap_explainer, get_feature_coefficients, explain_prediction
from src.preprocessing import load_and_split_data

@st.cache_resource
def load_model():
    """Load the final trained Logistic Regression pipeline."""
    model_path = os.path.join("models", "saved_models", "final_model_pipeline.joblib")
    return joblib.load(model_path)

@st.cache_resource
def get_shap_explainer(_pipeline):
    """Setup and cache the SHAP explainer."""
    X_train, _, _, _ = load_and_split_data(os.path.join("data", "raw", "heart_disease_uci.csv"))
    explainer, feature_names = setup_shap_explainer(_pipeline, X_train)
    return explainer, feature_names

@st.cache_data
def get_model_coefficients(_pipeline):
    """Cache and return global model coefficients."""
    return get_feature_coefficients(_pipeline)

def convert_inputs_to_df(inputs_dict):
    """
    Convert raw UI inputs into a pandas DataFrame matching X_train schema.
    """
    return pd.DataFrame([inputs_dict])

def plot_shap_waterfall(explanation_dict):
    """
    Generates a SHAP waterfall plot or bar plot from the explanation dictionary.
    Returns a matplotlib figure.
    """
    base_value = explanation_dict['base_value']
    shap_df = explanation_dict['shap_summary_df']
    
    # Sort by absolute contribution for better visualization
    shap_df = shap_df.sort_values(by='SHAP Contribution', key=abs, ascending=True)
    
    features = shap_df['Feature'].tolist()
    contributions = shap_df['SHAP Contribution'].tolist()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = ['red' if x > 0 else 'blue' for x in contributions]
    
    ax.barh(features, contributions, color=colors)
    
    ax.set_xlabel('SHAP Value (Impact on Log-Odds of Disease)')
    ax.set_title('Local Explanation: How features contributed to this specific prediction')
    ax.grid(axis='x', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    return fig
