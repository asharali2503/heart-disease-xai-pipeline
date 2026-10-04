# Heart Disease Prediction — Explainable Machine Learning

**🔴 Live Application:** [https://ashar-heart-xai.streamlit.app](https://ashar-heart-xai.streamlit.app)

![SHAP Local Explanation Chart showing risk factors](screenshot.png)

![Python](https://img.shields.io/badge/Python-3.14+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-FF4B4B.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.9+-F7931E.svg)
![SHAP](https://img.shields.io/badge/SHAP-Explainable_AI-black.svg)
![pytest](https://img.shields.io/badge/pytest-Passing-brightgreen.svg)

> **⚠️ EDUCATIONAL & PORTFOLIO NOTICE ⚠️**  
> This system is an educational demonstration of machine learning methodology and explainable AI (XAI) on historical data (UCI Cleveland). It is **NOT** a medical device, diagnostic tool, or clinical decision support system. It must **never** be used for medical diagnosis or clinical treatment. Always consult a qualified healthcare professional for medical advice.

---

## 📌 Project Overview
This project is a portfolio-grade machine learning application designed to predict the presence of heart disease using the UCI Cleveland Heart Disease dataset. 

Rather than chasing marginal accuracy gains with black-box ensembles, this project prioritizes **rigorous methodology** (strict leakage-prevention, stratified 5-fold cross-validation) and **model transparency** (Odds Ratios and SHAP). 

The final deliverable is an interactive Streamlit web application that acts as a "Patient Playground," breaking down complex probability scores into easily digestible visual explanations showing exactly *why* the model made a specific prediction.

## 🚀 Quick Start / Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/heart-disease-xai.git
   cd heart-disease-xai
   ```

2. **Create and activate a virtual environment (Optional but recommended):**
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Launch the interactive Streamlit Web App:**
   ```bash
   streamlit run app/app.py
   ```

## 🧠 Methodology & Model Selection

* **Dataset Preparation:** We binarized the original multi-class target (`0` = No Disease, `1-4` = Disease Present).
* **Data Splitting:** A strict 80/20 stratified split was executed. The 20% test set (61 rows) was completely isolated and never used for preprocessing fitting, model selection, or hyperparameter tuning.
* **Leakage-Free Preprocessing:** A scikit-learn `ColumnTransformer` handles Median Imputation & Standard Scaling for numerical features, and Most-Frequent Imputation & One-Hot Encoding for categorical features. These statistics are learned strictly on training folds.
* **Model Benchmarking:** We evaluated Logistic Regression, Random Forest, and XGBoost using Stratified 5-Fold Cross-Validation solely on the training data.
* **Selection:** **Logistic Regression** was selected. For a small tabular dataset (~240 training samples), linear models provide natural regularization that prevents the severe overfitting seen in complex tree ensembles (like XGBoost). Furthermore, linear models offer superior global interpretability via clinical Odds Ratios.

## 📊 Final Evaluation Results
Evaluated precisely **once** on the untouched 61-row test set, the final Logistic Regression pipeline achieved:

* **Accuracy:** 86.89%
* **Precision:** 81.25%
* **Recall (Sensitivity):** 92.86% (Highly desirable, missing only 2 actual disease cases)
* **F1 Score:** 86.67%
* **ROC-AUC:** 0.9578

*Confusion Matrix:* 27 True Negatives, 6 False Positives, 2 False Negatives, 26 True Positives.

## 🔍 Explainable AI (XAI) Focus
This project treats explainability as a first-class citizen:
* **Global Interpretability:** Calculates and displays explicit Odds Ratios ($e^{\beta}$) derived from the finalized pipeline, illustrating the population-wide impact of features like Asymptomatic Chest Pain or Number of Major Vessels (`ca`).
* **Local Interpretability (SHAP):** Utilizes `shap.LinearExplainer` on the transformed background training data to generate Waterfall and Bar plots. This empowers reviewers to see precisely how many log-odds points a specific patient's resting blood pressure or age contributed to their final risk score.

## 📂 Project Structure

```text
├── app/
│   ├── app.py                     # Main Streamlit application interface
│   └── utils.py                   # Caching, df formatting, and visualization helpers
├── data/
│   ├── raw/                       # Downloaded CSV dataset
│   └── README.md                  # Clinical data dictionary
├── models/
│   └── saved_models/              # Persisted joblib pipeline artifacts
├── notebooks/                       
│   ├── 01_eda_and_cleaning.ipynb  # Exploratory Data Analysis & Schema
│   ├── 02_model_training.ipynb    # 5-Fold CV, leakage prevention, and benchmarking
│   └── 03_explainability.ipynb    # SHAP explainer setup and Odds Ratio calculations
├── src/
│   ├── data_loader.py             # ucimlrepo integration
│   ├── evaluate.py                # CV comparison tables and final model evaluation
│   ├── explainability.py          # SHAP and Coefficient extraction functions
│   ├── preprocessing.py           # Feature classification, splits, and ColumnTransformer
│   └── train.py                   # Pipeline construction and model initializations
├── tests/                         # Comprehensive 22-test pytest suite
│   ├── test_app.py
│   ├── test_data_structure.py
│   ├── test_evaluate.py
│   ├── test_explainability.py
│   ├── test_final_evaluation.py
│   ├── test_preprocessing.py
│   └── test_train.py
├── requirements.txt               # Project dependencies
├── .gitignore                     # Standard Python/Jupyter/Git ignores
└── README.md                      # Project documentation (You are here)
```


## How to Run Locally

If you'd like to run this project on your own machine, follow these standard terminal commands:

```bash
# Clone the repository
git clone https://github.com/asharali2503/heart-disease-xai-pipeline.git
cd heart-disease-xai-pipeline

# Install the required dependencies
pip install -r requirements.txt

# Run the Streamlit web application
streamlit run app/app.py
```
