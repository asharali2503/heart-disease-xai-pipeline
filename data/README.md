# Heart Disease Dataset

## Source Information
- **Source**: UCI Machine Learning Repository
- **Dataset ID**: 45 (Heart Disease - Cleveland Database)
- **Data URL**: https://archive.ics.uci.edu/dataset/45/heart+disease
- **Citation/Attribution**: 
  Janosi, Andras, Steinbrunn, William, Pfisterer, Matthias, and Detrano, Robert. (1988). Heart Disease. UCI Machine Learning Repository. https://doi.org/10.24432/C52P4X.

## Dataset Statistics
- **Number of Observations (rows)**: 303
- **Number of Features**: 13 predictive features + 1 target (`num`)
- **Missing Values**: Yes, present in `ca` (4 missing) and `thal` (2 missing).

## Target Definition
- **Target Column Name**: `num`
- **Original Meaning**: Angiographic disease status (0 = < 50% diameter narrowing, 1-4 = > 50% diameter narrowing in varying severity).
- **Intended Binary Conversion**: 
  - `0` → no disease (164 observations)
  - `1, 2, 3, 4` → disease present (139 observations)
- **Class Balance**: The dataset has a moderate class imbalance (164 negative vs 139 positive examples). Stratification and appropriate evaluation metrics will be used.

## Feature Definitions

| Feature | Meaning | Data Type | Treatment (Num/Cat) | Missing Values |
| :--- | :--- | :--- | :--- | :--- |
| `age` | Age in years | `int64` | Numerical | 0 |
| `sex` | Sex (1 = male; 0 = female) | `int64` | Categorical | 0 |
| `cp` | Chest pain type (1 = typical angina, 2 = atypical angina, 3 = non-anginal pain, 4 = asymptomatic) | `int64` | Categorical | 0 |
| `trestbps` | Resting blood pressure (in mm Hg on admission to the hospital) | `int64` | Numerical | 0 |
| `chol` | Serum cholestoral in mg/dl | `int64` | Numerical | 0 |
| `fbs` | Fasting blood sugar > 120 mg/dl (1 = true; 0 = false) | `int64` | Categorical | 0 |
| `restecg` | Resting electrocardiographic results (0 = normal, 1 = having ST-T wave abnormality, 2 = showing probable or definite left ventricular hypertrophy) | `int64` | Categorical | 0 |
| `thalach` | Maximum heart rate achieved | `int64` | Numerical | 0 |
| `exang` | Exercise induced angina (1 = yes; 0 = no) | `int64` | Categorical | 0 |
| `oldpeak` | ST depression induced by exercise relative to rest | `float64` | Numerical | 0 |
| `slope` | The slope of the peak exercise ST segment (1 = upsloping, 2 = flat, 3 = downsloping) | `int64` | Categorical | 0 |
| `ca` | Number of major vessels (0-3) colored by flourosopy | `float64` | Numerical (Discrete) | 4 |
| `thal` | Thalassemia (3 = normal; 6 = fixed defect; 7 = reversable defect) | `float64` | Categorical | 2 |

## Missing Value Information
Missing values are represented as `NaN` (parsed from `?` in the original raw format by `ucimlrepo`).
- `ca`: 4 missing values
- `thal`: 2 missing values
These will need to be imputed (e.g., using median or mode) during preprocessing.

## Duplicate Rows
- **Duplicates**: 0 completely duplicated rows. (No removal needed).
