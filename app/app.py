import streamlit as st
import pandas as pd
from app.utils import load_model, get_shap_explainer, get_model_coefficients, convert_inputs_to_df, plot_shap_waterfall
from src.explainability import explain_prediction

st.set_page_config(page_title="Heart Disease Prediction", layout="wide")

# A. Educational Disclaimer
st.warning(
    "**Educational Notice**: This system is an educational demonstration of machine learning methodology "
    "and explainable AI on historical data (UCI Cleveland). It is NOT a medical device, diagnostic tool, "
    "or clinical decision support system. It must never be used for medical diagnosis or clinical treatment."
)

st.title("Heart Disease Prediction — Explainable Machine Learning")

# Load Models & Explainers
with st.spinner("Loading models..."):
    pipeline = load_model()
    explainer, feature_names = get_shap_explainer(pipeline)
    coef_df = get_model_coefficients(pipeline)

tab1, tab2, tab3 = st.tabs(["Interactive Patient Playground", "Global Explainability", "Methodology & Performance"])

with tab1:
    st.header("Patient Profile Input")
    
    # Pre-loaded profiles
    profile_type = st.radio("Select Profile to Load:", ["Custom Input", "Low Risk Example", "High Risk Example"], horizontal=True)
    
    default_vals = {
        'age': 55, 'sex': 1, 'cp': 4, 'trestbps': 120, 'chol': 200, 
        'fbs': 0, 'restecg': 0, 'thalach': 150, 'exang': 0, 
        'oldpeak': 0.0, 'slope': 2, 'ca': 0, 'thal': 3
    }
    
    if profile_type == "Low Risk Example":
        default_vals = {
            'age': 35, 'sex': 0, 'cp': 1, 'trestbps': 110, 'chol': 180, 
            'fbs': 0, 'restecg': 0, 'thalach': 180, 'exang': 0, 
            'oldpeak': 0.0, 'slope': 1, 'ca': 0, 'thal': 3
        }
    elif profile_type == "High Risk Example":
        default_vals = {
            'age': 65, 'sex': 1, 'cp': 4, 'trestbps': 160, 'chol': 280, 
            'fbs': 1, 'restecg': 2, 'thalach': 110, 'exang': 1, 
            'oldpeak': 3.5, 'slope': 2, 'ca': 3, 'thal': 7
        }

    with st.form("patient_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            age = st.slider("Age", 20, 85, default_vals['age'])
            sex = st.selectbox("Sex", options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male", index=default_vals['sex'])
            cp = st.selectbox("Chest Pain Type (cp)", options=[1, 2, 3, 4], 
                              format_func=lambda x: {1: "Typical Angina", 2: "Atypical Angina", 3: "Non-anginal Pain", 4: "Asymptomatic"}[x],
                              index=[1, 2, 3, 4].index(default_vals['cp']))
            trestbps = st.number_input("Resting Blood Pressure (mm Hg)", 90, 200, default_vals['trestbps'])
            chol = st.number_input("Serum Cholesterol (mg/dl)", 100, 600, default_vals['chol'])
            
        with col2:
            fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (fbs)", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes", index=default_vals['fbs'])
            restecg = st.selectbox("Resting ECG (restecg)", options=[0, 1, 2], 
                                   format_func=lambda x: {0: "Normal", 1: "ST-T Wave Abnormality", 2: "Left Ventricular Hypertrophy"}[x],
                                   index=default_vals['restecg'])
            thalach = st.number_input("Max Heart Rate Achieved", 60, 220, default_vals['thalach'])
            exang = st.selectbox("Exercise Induced Angina (exang)", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes", index=default_vals['exang'])
            
        with col3:
            oldpeak = st.number_input("ST Depression Induced by Exercise (oldpeak)", 0.0, 6.5, float(default_vals['oldpeak']), step=0.1)
            slope = st.selectbox("Peak Exercise ST Segment Slope", options=[1, 2, 3], 
                                 format_func=lambda x: {1: "Upsloping", 2: "Flat", 3: "Downsloping"}[x],
                                 index=[1, 2, 3].index(default_vals['slope']))
            ca = st.slider("Number of Major Vessels Colored by Flourosopy (ca)", 0, 3, default_vals['ca'])
            thal = st.selectbox("Thalassemia (thal)", options=[3, 6, 7], 
                                format_func=lambda x: {3: "Normal", 6: "Fixed Defect", 7: "Reversible Defect"}[x],
                                index=[3, 6, 7].index(default_vals['thal']))

        submit = st.form_submit_button("Generate Prediction")

    if submit:
        input_dict = {
            'age': age, 'sex': sex, 'cp': cp, 'trestbps': trestbps, 'chol': chol,
            'fbs': fbs, 'restecg': restecg, 'thalach': thalach, 'exang': exang,
            'oldpeak': oldpeak, 'slope': slope, 'ca': ca, 'thal': thal
        }
        input_df = convert_inputs_to_df(input_dict)
        
        # C. Prediction Output
        prob = pipeline.predict_proba(input_df)[0][1]
        pred = pipeline.predict(input_df)[0]
        
        st.markdown("---")
        st.header("Prediction Results")
        
        col_res1, col_res2 = st.columns(2)
        with col_res1:
            if pred == 1:
                st.error("### High Risk of Heart Disease")
            else:
                st.success("### Low Risk of Heart Disease")
                
        with col_res2:
            st.metric(label="Predicted Probability of Disease", value=f"{prob * 100:.1f}%")
            st.progress(float(prob))

        # D. Local Explainability
        st.subheader("Local Explanation: What drove this prediction?")
        explanation = explain_prediction(explainer, pipeline, input_df)
        
        fig = plot_shap_waterfall(explanation)
        st.pyplot(fig)
        
        with st.expander("View SHAP Values Data Table"):
            st.dataframe(explanation['shap_summary_df'])

with tab2:
    st.header("Global Explainability")
    st.markdown("Understanding how the model makes decisions across the entire population.")
    
    st.subheader("Model Odds Ratios (Interpretable Log-Odds)")
    st.dataframe(coef_df)
    
with tab3:
    st.header("Methodology & Performance Overview")
    
    st.markdown("""
    ### Data Splitting Strategy
    - **Total Dataset:** 303 instances (UCI Cleveland Heart Disease)
    - **Training Set:** 80% (242 instances)
    - **Test Set:** 20% (61 instances, completely untouched during tuning)
    - **Cross-Validation:** 5-fold Stratified CV strictly on the training set
    
    ### Model Selection
    We evaluated three baseline models (untuned) on the training set using 5-Fold CV.
    *Logistic Regression was selected due to its highest performance, lowest variance, and native interpretability.*
    
    ### Final Test Set Evaluation
    The finalized Logistic Regression model achieved the following on the untouched test set:
    - **Accuracy:** 86.89%
    - **Recall (Sensitivity):** 92.86%
    - **ROC-AUC:** 0.9578
    
    **Confusion Matrix:**
    - True Negatives (Correctly predicted no disease): 27
    - False Positives (Predicted disease, actually no disease): 6
    - False Negatives (Predicted no disease, actually disease): 2
    - True Positives (Correctly predicted disease): 26
    """)
