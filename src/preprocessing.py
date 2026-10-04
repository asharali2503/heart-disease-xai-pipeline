import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# --- Constants for Reproducibility ---
RANDOM_SEED = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# --- Feature Groups ---
NUMERICAL_FEATURES = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'ca']
CATEGORICAL_FEATURES = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'thal']

def load_and_split_data(filepath="data/raw/heart_disease_uci.csv"):
    """
    Loads raw data, creates binary target, drops original 'num', 
    and splits into train/test sets.
    """
    df = pd.read_csv(filepath)
    
    # 1. Target creation (binarize)
    df['target'] = df['num'].apply(lambda x: 1 if x > 0 else 0)
    
    # 2. Feature selection
    X = df.drop(columns=['num', 'target'])
    y = df['target']
    
    # 3. Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=TEST_SIZE, 
        stratify=y, 
        random_state=RANDOM_SEED
    )
    
    return X_train, X_test, y_train, y_test

def build_preprocessor():
    """
    Builds the scikit-learn ColumnTransformer for preprocessing.
    """
    # Numerical pipeline: Median imputation -> Standard scaling
    numeric_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Categorical pipeline: Most-frequent imputation -> One-hot encoding
    # drop='first' can be used, but since we are using tree models/L1 we might just use drop=None (default) 
    # Let's use handle_unknown='ignore' so it can handle any unexpected categories in production/test
    categorical_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    # Combine into a ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_pipeline, NUMERICAL_FEATURES),
            ('cat', categorical_pipeline, CATEGORICAL_FEATURES)
        ],
        remainder='drop' # Drop any unexpected columns not explicitly specified
    )
    
    return preprocessor

def get_cv_strategy():
    """
    Returns the StratifiedKFold cross-validation object.
    """
    return StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_and_split_data()
    print("--- Train/Test Sample Counts ---")
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    
    print("\n--- Target Distribution ---")
    print("y_train distribution:")
    print(y_train.value_counts(normalize=False))
    print(y_train.value_counts(normalize=True))
    print("y_test distribution:")
    print(y_test.value_counts(normalize=False))
    print(y_test.value_counts(normalize=True))
    
    preprocessor = build_preprocessor()
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)
    
    print("\n--- Transformed Feature Dimensions ---")
    print(f"Original X_train shape: {X_train.shape}")
    print(f"Transformed X_train shape: {X_train_transformed.shape}")
    print(f"Transformed X_test shape: {X_test_transformed.shape}")
