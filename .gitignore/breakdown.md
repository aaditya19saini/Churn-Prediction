# Churn Prediction Ensemble - Jupyter Notebook Breakdown

## Overview
This notebook implements an ensemble machine learning approach for customer churn prediction using XGBoost, CatBoost, and Gradient Boosting classifiers with weight optimization and isotonic regression calibration.

---

## Cell 1: Import Libraries and Environment Setup

**Code:**
```python
import numpy as np
import pandas as pd
import os
```

**Purpose:**
- Imports essential Python libraries for data manipulation and analysis
- `numpy`: Provides support for large multi-dimensional arrays and matrices, along with mathematical functions
- `pandas`: Offers data structures and operations for manipulating numerical tables and time series
- `os`: Enables operating system dependent functionality (used for file/directory operations)

**Note:** This appears to be running in a Kaggle environment, evidenced by comments about the kaggle/python Docker image and file paths like '/kaggle/input'.

---

## Cell 2: Load Data

**Code:**
```python
train=pd.read_csv(r"C:\Users\Aaditya Saini\Desktop\Churn\data\train.csv")
test=pd.read_csv(r"C:\Users\Aaditya Saini\Desktop\Churn\data\test.csv")
sub=pd.read_csv(r"C:\Users\Aaditya Saini\Desktop\Churn\data\sample_submission.csv")
```

**Purpose:**
- Loads three datasets using pandas' `read_csv()` function:
  1. **train.csv**: Training dataset containing features and target variable (Churn)
  2. **test.csv**: Test dataset for predictions (without target labels)
  3. **sample_submission.csv**: Template for submission format

**Function Breakdown:**
- `pd.read_csv()`: Reads comma-separated values file into DataFrame
- Raw string notation `r""` prevents escape character interpretation in Windows paths
- Each file is stored in separate DataFrames: `train`, `test`, `sub`

---

## Cell 3: Preview Training Data

**Code:**
```python
train.head()
```

**Purpose:**
- Displays the first 5 rows of the training dataset using pandas `head()` method
- Provides initial visual inspection of the data structure

**Output Analysis:**
- Dataset contains 21 columns including:
  - `id`: Unique identifier
  - `gender`: Customer gender (Male/Female)
  - `SeniorCitizen`: Binary indicator (0/1)
  - `Partner`: Whether customer has partner (Yes/No)
  - `Dependents`: Whether customer has dependents (Yes/No)
  - `tenure`: Number of months customer stayed
  - `PhoneService`: Phone service subscription (Yes/No)
  - `MultipleLines`: Multiple lines subscription (Yes/No/No phone service)
  - `InternetService`: Internet service type (DSL/Fiber optic/No)
  - `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`: Various service subscriptions
  - `StreamingTV`, `StreamingMovies`: Streaming services (Yes/No/No internet service)
  - `Contract`: Contract type (Month-to-month/One year/Two year)
  - `PaperlessBilling`: Paperless billing (Yes/No)
  - `PaymentMethod`: Payment method (Electronic check/Mailed check/Bank transfer/Credit card)
  - `MonthlyCharges`: Monthly billing amount (float)
  - `TotalCharges`: Total charges (float)
  - `Churn`: Target variable (Yes/No) - whether customer left

---

## Cell 4: Dataset Information

**Code:**
```python
train.info()
```

**Purpose:**
- Displays concise summary of the DataFrame including:
  - Index type and range
  - Column names and data types
  - Non-null count for each column
  - Memory usage

**Output Analysis:**
- **Dataset Size:** 594,194 entries (rows) × 21 columns
- **Data Types:**
  - `int64`: 3 columns (id, SeniorCitizen, tenure)
  - `float64`: 2 columns (MonthlyCharges, TotalCharges)
  - `str/object`: 16 columns (categorical features)
- **Missing Values:** No missing values (all columns show 594,194 non-null count)
- **Memory Usage:** 149.1 MB

**Key Insights:**
- Large dataset requiring efficient processing
- Mix of numerical and categorical features
- Binary classification problem (Churn: Yes/No)

---

## Cell 5: Separate Numerical and Categorical Columns

**Code:**
```python
num_cols=train.select_dtypes(include=np.number).columns.tolist()
cat_cols=train.select_dtypes(include="object").columns.tolist()
```

**Purpose:**
- Separates columns by data type for targeted preprocessing

**Function Breakdown:**
- `train.select_dtypes(include=np.number)`: Selects columns with numerical data types (int, float, etc.)
- `.columns.tolist()`: Extracts column names as a Python list
- `train.select_dtypes(include="object")`: Selects columns with object/string data types

**Output:**
- **num_cols**: ['id', 'SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges']
- **cat_cols**: ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod', 'Churn']

**Note:** A deprecation warning appears suggesting to use 'str' instead of 'object' for select_dtypes in future pandas versions.

---

## Cell 6: Value Counts for Categorical Columns

**Code:**
```python
for i in cat_cols:
    print(train[i].value_counts())
```

**Purpose:**
- Iterates through each categorical column and prints frequency count of unique values
- Helps understand distribution and cardinality of categorical features

**Output Analysis:**
Key findings for each categorical feature:

1. **gender**: Balanced distribution
   - Female: 298,738 (50.28%)
   - Male: 295,456 (49.72%)

2. **Partner**: Slightly more customers with partners
   - Yes: 309,554 (52.09%)
   - No: 284,640 (47.91%)

3. **Dependents**: Majority have no dependents
   - No: 414,362 (69.74%)
   - Yes: 179,832 (30.26%)

4. **PhoneService**: Most customers have phone service
   - Yes: 557,893 (93.89%)
   - No: 36,301 (6.11%)

5. **MultipleLines**: Three categories
   - No: 283,384 (47.69%)
   - Yes: 274,509 (46.20%)
   - No phone service: 36,301 (6.11%)

6. **InternetService**: Three types
   - Fiber optic: 272,386 (45.85%)
   - DSL: 181,081 (30.47%)
   - No: 140,727 (23.68%)

7. **OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport**: All have three categories (Yes/No/No internet service)

8. **StreamingTV, StreamingMovies**: Similar distribution patterns

9. **Contract**: Three types
   - Month-to-month: 298,918 (50.31%)
   - Two year: 186,943 (31.46%)
   - One year: 108,333 (18.23%)

10. **PaperlessBilling**: 
    - Yes: 365,579 (61.52%)
    - No: 228,615 (38.48%)

11. **PaymentMethod**: Four methods
    - Electronic check: 215,372 (36.23%)
    - Credit card (automatic): 133,705 (22.51%)
    - Mailed check: 123,757 (20.83%)
    - Bank transfer (automatic): 121,360 (20.43%)

12. **Churn** (Target Variable): Imbalanced
    - No: 460,377 (77.49%)
    - Yes: 133,817 (22.51%)

**Key Insight:** Significant class imbalance in the target variable - 77.49% non-churn vs 22.51% churn, which requires careful handling during model training.

---

## Cell 7: Boxplots for Numerical Columns

**Code:**
```python
import matplotlib.pyplot as plt 
import seaborn as sns

for i in num_cols:
    plt.figure()
    sns.boxplot(x=train[i])
    plt.title(f"Outlier of {i}")
    plt.show()
```

**Purpose:**
- Creates and displays boxplots for each numerical column to visualize:
  - Distribution of values
  - Median (central line)
  - Interquartile range (IQR) - box edges
  - Whiskers (1.5 * IQR from quartiles)
  - Outliers (points beyond whiskers)

**Function Breakdown:**
- `import matplotlib.pyplot as plt`: Standard plotting library
- `import seaborn as sns`: Statistical data visualization library built on matplotlib
- `plt.figure()`: Creates a new figure for each plot
- `sns.boxplot(x=train[i])`: Creates horizontal boxplot for column i
- `plt.title()`: Sets plot title
- `plt.show()`: Displays the plot

**Visualization Purpose:**
1. **Identify outliers**: Points outside whiskers indicate potential outliers
2. **Understand distribution**: Box shows median and quartiles
3. **Detect skew**: Asymmetric boxes indicate skewed data
4. **Compare scales**: Helps determine if scaling is needed

**Insights for Each Numerical Feature:**
- **id**: Uniform distribution (expected for identifier)
- **SeniorCitizen**: Binary values (0/1), no outliers
- **tenure**: Check for unusual values
- **MonthlyCharges**: Look for billing anomalies
- **TotalCharges**: Often has longer tail due to accumulation

---

## Cell 8: Drop ID Column

**Code:**
```python
train.drop("id",axis=1,inplace=True)
test.drop("id",axis=1,inplace=True)
```

**Purpose:**
- Removes the 'id' column from both train and test datasets as it's just an identifier
- ID columns don't provide predictive value for machine learning

**Function Breakdown:**
- `drop("id", axis=1)`: Drops column named 'id'
  - `axis=1`: Indicates column-wise operation (0 would be row-wise)
- `inplace=True`: Modifies the DataFrame directly without creating a new one

**Why Remove ID:**
- It's a unique identifier with no predictive power
- Can cause overfitting if included
- Reduces memory usage
- Simplifies feature set

---

## Cell 9: Remove ID from Numerical Columns List

**Code:**
```python
num_cols.remove("id")
```

**Purpose:**
- Updates the `num_cols` list to remove 'id' after dropping it from the dataset
- Ensures consistency between the actual DataFrame and the tracking lists

**Important:** This prevents errors in later preprocessing steps that would iterate through `num_cols` and try to access a non-existent column.

---

## Cell 10: Remove SeniorCitizen from Numerical Columns

**Code:**
```python
num_cols.remove("SeniorCitizen") 
'''removed because this is the feature we're trying to predict
error because theres churn is not a numerical column'''
```

**Purpose:**
- Removes 'SeniorCitizen' from the numerical columns list
- Comment suggests confusion (SeniorCitizen is a feature, not the target - Churn is the target)
- The actual reason: SeniorCitizen is binary (0/1) and better treated as categorical

**Note:** The comment contains an error - 'Churn' is the target, not 'SeniorCitizen'. SeniorCitizen is actually a feature indicating whether the customer is a senior citizen.

---

## Cell 11: Remove Churn from Categorical Columns

**Code:**
```python
cat_cols.remove("Churn")
```

**Purpose:**
- Removes 'Churn' from categorical columns list
- 'Churn' is the target variable and should be separated from features

**Why Remove:**
- Target variable should be treated separately
- Prevents accidental inclusion in feature preprocessing
- Avoids data leakage (using target to predict itself)

---

## Cell 12: Pairplot Visualization

**Code:**
```python
sns.pairplot(train[num_cols + ['Churn']], hue='Churn', corner=True)
plt.show()
```

**Purpose:**
- Creates a grid of scatter plots showing pairwise relationships between numerical features
- Color-coded by Churn status (hue='Churn')
- Helps identify patterns, correlations, and potential feature interactions

**Function Breakdown:**
- `train[num_cols + ['Churn']]`: Selects numerical columns plus Churn
- `sns.pairplot()`: Creates matrix of scatter plots
  - `hue='Churn'`: Colors points by Churn status (Yes/No)
  - `corner=True`: Shows only lower triangle (redu redundancy)

**Visualization Components:**
1. **Diagonal**: Kernel density estimation (KDE) showing distribution of each feature
2. **Off-diagonal**: Scatter plots showing relationship between pairs of features
3. **Colors**: Different colors for churned vs non-churned customers

**Key Insights:**
- Helps identify feature distributions
- Shows potential correlation between features
- Reveals if feature values differ significantly between churn classes
- Can detect clusters or separability patterns
- Identifies potential issues like high collinearity

**Computational Note:** This can be slow for large datasets (594K rows). The `corner=True` parameter helps reduce visual redundancy.

---

## Cell 13: Categorical Features Distribution by Churn

**Code:**
```python
for i in cat_cols:
    plt.figure(figsize=(12, 6))
    sns.countplot(data=train, x=i, hue='Churn')
    plt.title(f"{i} Distribution")
    plt.xticks(rotation=45)
    plt.show()
```

**Purpose:**
- Creates countplots for each categorical feature showing distribution across Churn classes
- Helps visualize how Churn rates vary across different categorical feature values

**Function Breakdown:**
- Loop iterates through each categorical column
- `plt.figure(figsize=(12, 6))`: Creates figure with specific dimensions
- `sns.countplot()`: Creates bar chart showing counts
  - `data=train`: Uses training DataFrame
  - `x=i`: Plots column i on x-axis
  - `hue='Churn'`: Separates bars by Churn status (Yes/No)
- `plt.xticks(rotation=45)`: Rotates x-axis labels 45 degrees for readability

**Insights from Each Feature:**

1. **gender**: Churn rates similar across genders (no strong predictive power)

2. **Partner**: Customers without partners show higher churn rates

3. **Dependents**: Customers without dependents more likely to churn

4. **PhoneService**: Similar churn patterns, not strong predictor

5. **MultipleLines**: Slight variation in churn across categories

6. **InternetService**: 
   - Fiber optic customers have highest churn
   - No internet service customers have lowest churn

7. **OnlineSecurity**: 
   - Customers without online security show high churn
   - Those with security features churn less

8. **OnlineBackup**: Similar pattern to OnlineSecurity

9. **DeviceProtection**: Protected customers churn less

10. **TechSupport**: 
    - No tech support = higher churn
    - Tech support availability reduces churn

11. **StreamingTV & StreamingMovies**: Moderate impact on churn

12. **Contract**: 
    - Month-to-month contracts have highest churn
    - Two-year contracts have lowest churn
    - Strong predictive feature

13. **PaperlessBilling**: 
    - Paperless billing customers churn more
    - Might correlate with tech-savvy customers

14. **PaymentMethod**: 
    - Electronic check shows highest churn
    - Automatic payments show lowest churn

**Feature Engineering Insights:**
- Contract type appears to be one of the strongest predictors
- Internet service and related features show clear patterns
- Security/support features are meaningful
- Gender has minimal impact

---

## Cell 14: Train-Test Split

**Code:**
```python
from sklearn.model_selection import train_test_split, cross_val_score

X=train.drop("Churn",axis=1)
y=train["Churn"]

y=y.replace({"No":0,"Yes":1})

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.05,random_state=42)
```

**Purpose:**
- Prepares data for model training by:
  1. Separating features (X) and target (y)
  2. Encoding target variable
  3. Creating train and validation sets

**Function Breakdown:**

**Step 1: Feature-Target Separation**
- `train.drop("Churn", axis=1)`: Creates DataFrame X with all columns except 'Churn'
- `train["Churn"]`: Creates Series y with only the target variable

**Step 2: Target Encoding**
- `y.replace({"No":0,"Yes":1})`: Converts binary categorical target to numeric
  - "No" → 0 (customer stayed)
  - "Yes" → 1 (customer churned)
- Required for machine learning algorithms that expect numeric targets

**Step 3: Train-Validation Split**
- `train_test_split(X, y, test_size=0.05, random_state=42)`: Splits data
  - `test_size=0.05`: 5% for validation (29,710 samples), 95% for training (564,484 samples)
  - `random_state=42`: Ensures reproducibility with fixed random seed
  - Returns four arrays: X_train, X_test, y_train, y_test

**Why 5% Validation:**
- With 594K+ samples, 5% provides statistically significant validation set
- Keeps more data for training
- Common practice in large datasets

**Note:** This split is for validation during training. The actual 'test' set loaded earlier is for final predictions and doesn't have labels.

---

## Cell 15: Install CatBoost Library

**Code:**
```bash
pip install catboost
```

**Purpose:**
- Installs the CatBoost library from PyPI (Python Package Index)
- Required for gradient boosting with categorical features support

**Output Analysis:**
- Successfully downloads and installs catboost-1.2.10
- Downloads 100.2 MB wheel file
- Installation takes ~36 seconds
- Checks and verifies dependencies:
  - graphviz: Already installed
  - matplotlib, numpy, pandas, scipy, plotly: Already installed
  - python-dateutil, tzdata: Already installed

**About CatBoost:**
- Open-source gradient boosting library developed by Yandex
- Specifically designed to handle categorical features natively
- Offers GPU acceleration for faster training
- Known for good performance with minimal hyperparameter tuning

**Why Install Here:**
- Not included in default Python environment
- Required for model training in subsequent cells
- One-time installation (persists in environment)

---

## Cell 16: Verify XGBoost Installation

**Code:**
```bash
pip install xgboost
```

**Purpose:**
- Verifies XGBoost is installed (already present in environment)
- Ensures compatibility before training

**Output Analysis:**
- xgboost-2.1.4 already installed
- Confirms dependencies are satisfied
- No installation needed

**About XGBoost:**
- eXtreme Gradient Boosting library
- Highly efficient implementation of gradient boosting
- Supports parallel processing
- Industry standard for tabular data competitions

---

## Cell 17: Model Training with Stratified K-Fold Ensemble

**Code:**
```python
from sklearn.model_selection import StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score

from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), cat_cols)])

n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

oof_xgb = np.zeros(len(X_train))
oof_cat = np.zeros(len(X_train))
oof_gbc = np.zeros(len(X_train))

test_xgb = np.zeros(len(test))
test_cat = np.zeros(len(test))
test_gbc = np.zeros(len(test))

for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
    print(f"--- Fold {fold+1} ---")
    
    X_tr, X_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
    y_tr, y_val = y_train.iloc[tr_idx], y_train.iloc[val_idx]
    
    xgb_pipe = Pipeline([
        ('prep', preprocessor),
        ('model', XGBClassifier(
            n_estimators=4000,
            max_depth=8,
            subsample=0.8,
            colsample_bytree=0.8,
            learning_rate=0.01,
            eval_metric='auc',
            random_state=42,
            verbosity=0,
            device='cuda'
        ))
    ])
    
    cat_pipe = Pipeline([
        ('prep', preprocessor),
        ('model', CatBoostClassifier(
            iterations=4000,
            depth=8,
            bootstrap_type='Bernoulli',
            subsample=0.8,
            learning_rate=0.01,
            loss_function='Logloss',
            random_state=42,
            verbose=0,
            task_type='GPU',
            devices='0',
            eval_metric='AUC'
        ))
    ])
    
    gbc_pipe = Pipeline([
        ('prep', preprocessor),
        ('model', GradientBoostingClassifier(
            n_estimators=2000,
            max_depth=7,
            subsample=0.8,
            learning_rate=0.02,
            random_state=42
        ))
    ])

    gbc_pipe.fit(X_tr, y_tr)
    xgb_pipe.fit(X_tr, y_tr)
    cat_pipe.fit(X_tr, y_tr)
    
    oof_xgb[val_idx] = xgb_pipe.predict_proba(X_val)[:, 1]
    oof_cat[val_idx] = cat_pipe.predict_proba(X_val)[:, 1]
    oof_gbc[val_idx] = gbc_pipe.predict_proba(X_val)[:, 1]

    test_xgb += xgb_pipe.predict_proba(test)[:, 1] / n_splits
    test_cat += cat_pipe.predict_proba(test)[:, 1] / n_splits
    test_gbc += gbc_pipe.predict_proba(test)[:, 1] / n_splits
```

**Purpose:**
- Implements stratified 5-fold cross-validation with three ensemble models
- Uses out-of-fold (OOF) predictions for robust ensemble building
- Averages test predictions across all folds for final submission

**Detailed Function Breakdown:**

**Imports:**
- `StratifiedKFold`: Ensures each fold has same proportion of target classes
- `ColumnTransformer`: Applies different preprocessing to different column subsets
- `StandardScaler`: Normalizes numerical features (mean=0, std=1)
- `OneHotEncoder`: Converts categorical features to binary columns
- `Pipeline`: Chains preprocessing and model steps
- `roc_auc_score`: Evaluation metric (Area Under ROC Curve)

**Preprocessor Setup:**
```python
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), num_cols),
    ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), cat_cols)
])
```
- Numerical columns: Scaled to standard normal distribution
- Categorical columns: One-hot encoded with first category dropped (prevents multicollinearity)
- `handle_unknown='ignore'`: Handles unseen categories in test set

**StratifiedKFold:**
- `n_splits=5`: Creates 5 different train/validation splits
- `shuffle=True`: Shuffles data before splitting
- `random_state=42`: Reproducible splits
- Maintains class distribution in each fold

**Out-of-Fold (OOF) Predictions:**
- `oof_xgb/cat/gbc`: Arrays to store validation predictions for each fold
- Each row gets predicted exactly once when it's in validation set
- Used for ensemble weight optimization

**Test Predictions:**
- Accumulated across all folds and averaged
- More robust than single model predictions

**Model Specifications:**

**1. XGBoost Classifier:**
- `n_estimators=4000`: Number of boosting rounds
- `max_depth=8`: Maximum tree depth
- `subsample=0.8`: Use 80% of samples per tree
- `colsample_bytree=0.8`: Use 80% of features per tree
- `learning_rate=0.01`: Step size shrinkage
- `device='cuda'`: GPU acceleration

**2. CatBoost Classifier:**
- `iterations=4000`: Number of trees
- `depth=8`: Tree depth
- `bootstrap_type='Bernoulli'`: Bootstrap sampling method
- `task_type='GPU'`: GPU acceleration
- `loss_function='Logloss'`: Binary cross-entropy
- Native categorical feature handling (though preprocessor handles this)

**3. Gradient Boosting Classifier:**
- `n_estimators=2000`: Fewer estimators ( sklearn implementation is slower)
- `max_depth=7`: Tree depth
- `subsample=0.8`: Stochastic gradient boosting
- Runs on CPU (no GPU support in sklearn)

**Training Loop:**
1. Iterates through 5 folds
2. For each fold:
   - Splits data into train/validation
   - Creates separate pipelines for each model
   - Fits models on training data
   - Predicts probabilities on validation set (stores in OOF arrays)
   - Predicts probabilities on test set (accumulates)

**Error Encountered:**
```
ValueError: Supported target types are: ('binary', 'multiclass'). Got 'unknown' instead.
```
This occurs because `y_train` may not be properly encoded or has NaN/unknown values. The code expects binary labels.

**How to Fix:**
Ensure y_train is properly encoded as 0/1 before splitting:
```python
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(train['Churn'])
```

---

## Cell 18: Weight Optimization for Ensemble

**Code:**
```python
best_auc = 0
best_weights = None

for w_xgb in np.arange(0, 1.01, 0.05):
    for w_cat in np.arange(0, 1.01 - w_xgb, 0.05):
        w_gbc = 1.0 - w_xgb - w_cat
        
        blend_oof = (w_xgb * oof_xgb) + (w_cat * oof_cat) + (w_gbc * oof_gbc)
        
        auc = roc_auc_score(y_train, blend_oof)
        
        if auc > best_auc:
            best_auc = auc
            best_weights = (w_xgb, w_cat, w_gbc)
```

**Purpose:**
- Finds optimal weights for combining three models' predictions
- Uses grid search over weight combinations
- Maximizes AUC score on validation data

**Detailed Algorithm:**

**Weight Constraints:**
- `w_xgb`: Weight for XGBoost predictions (0 to 1.0, step 0.05)
- `w_cat`: Weight for CatBoost predictions (0 to 1-w_xgb)
- `w_gbc`: Weight for GradientBoost predictions (1 - w_xgb - w_cat)
- All weights must sum to 1.0

**Grid Search Process:**
1. Iterates through possible XGBoost weights (0, 0.05, 0.10, ..., 1.0)
2. For each XGBoost weight, iterates through CatBoost weights
3. GradientBoost weight is determined by constraint
4. Tests all valid combinations (weights sum to 1)

**Weight Combination:**
```python
blend_oof = (w_xgb * oof_xgb) + (w_cat * oof_cat) + (w_gbc * oof_gbc)
```
- Creates weighted average of OOF predictions
- Each model contributes proportionally to its weight

**Evaluation:**
- `roc_auc_score(y_train, blend_oof)`: Calculates AUC for blend
- Compares with current best AUC
- Updates best weights if improvement found

**Optimization Strategy:**
- Grid search gives global optimum (not local)
- Fine granularity (0.05 steps) balances precision and speed
- Uses OOF predictions to avoid overfitting
- Tests ~400 weight combinations

**Example Weight Combinations Tested:**
- (0.0, 0.0, 1.0): Only GradientBoost
- (0.5, 0.5, 0.0): Equal XGBoost and CatBoost
- (0.4, 0.4, 0.2): Weighted combination
- Optimal combination stored in `best_weights`

**Time Complexity:**
- O(n*w_xgb*w_cat) where n = len(y_train)
- Approximately 400 iterations
- Fast because using pre-computed predictions

---

## Cell 19: Isotonic Regression Calibration

**Code:**
```python
from sklearn.isotonic import IsotonicRegression

w_xgb, w_cat, w_gbc = best_weights

ensemble_oof = (
    w_xgb * oof_xgb +
    w_cat * oof_cat +
    w_gbc * oof_gbc
)

iso = IsotonicRegression(out_of_bounds='clip')
iso.fit(ensemble_oof, y_train)
```

**Purpose:**
- Applies isotonic regression to calibrate ensemble probabilities
- Corrects any misalignment between predicted probabilities and actual outcomes
- Ensures predicted probabilities are well-calibrated

**Detailed Function Breakdown:**

**IsotonicRegression:**
- Non-parametric calibration method
- Fits a piecewise constant, monotonically increasing function
- Maps predicted probabilities to calibrated probabilities
- Better than Platt scaling (sigmoid) when calibration curve is non-linear

**Why Calibration Matters:**
- Raw ensemble scores may not reflect true probabilities
- Example: Predicted 0.7 may not correspond to 70% actual churn rate
- Calibration ensures probabilities are meaningful
- Important for business decisions (e.g., customer retention campaigns)

**Weight Extraction:**
```python
w_xgb, w_cat, w_gbc = best_weights
```
- Unpacks optimal weights from Cell 18

**Ensemble Creation:**
```python
ensemble_oof = w_xgb * oof_xgb + w_cat * oof_cat + w_gbc * oof_gbc
```
- Creates final ensemble predictions using optimal weights
- Weighted average of three models' OOF predictions

**Isotonic Regression Parameters:**
- `out_of_bounds='clip'`: 
  - Handles predictions outside training range
  - Clips to min/max values seen during training
  - Alternative: 'nan', 'raise'

**Fitting Process:**
1. Input: ensemble OOF predictions (x-values)
2. Target: actual labels (y-values)
3. Finds optimal mapping to minimize calibration error
4. Learns which predicted scores map to actual probabilities

**Mathematical Foundation:**
- Finds function f such that: y = f(x)
- Minimizes: Σ(y_i - f(x_i))²
- Subject to: x_i ≤ x_j → f(x_i) ≤ f(x_j)
- Ensures monotonicity (higher predicted score → higher probability)

**Benefits:**
1. Non-parametric: No assumptions about distribution
2. Monotonic: Preserves order of predictions
3. Flexible: Captures complex calibration curves
4. Proven: Works well for ensemble methods

**When to Use:**
- Binary classification with probability estimates
- When calibration curve is non-linear
- After model ensembling
- When true probabilities matter

---

## Cell 20: Create Ensemble Test Predictions

**Code:**
```python
ensemble_test = (
    w_xgb * test_xgb +
    w_cat * test_cat +
    w_gbc * test_gbc
)

probs = iso.transform(ensemble_test)
```

**Purpose:**
- Applies trained ensemble weights to test set predictions
- Calibrates final probabilities using isotonic regression
- Produces final churn probabilities for submission

**Detailed Breakdown:**

**Weighted Test Ensemble:**
```python
ensemble_test = w_xgb * test_xgb + w_cat * test_cat + w_gbc * test_gbc
```
- Combines test predictions from all three models
- Uses optimal weights determined in Cell 18
- `test_xgb/cat/gbc`: Averaged predictions from all folds (Cell 17)
- Creates a single ensemble score per test sample

**Isotonic Transformation:**
```python
probs = iso.transform(ensemble_test)
```
- Applies learned calibration function to test predictions
- Maps ensemble scores to calibrated probabilities
- Uses `transform()` method (not `fit_transform()`)
- `iso` was already fitted on training data (Cell 19)

**Probability Calibration Effect:**
- Before: Raw scores (may not be proper probabilities)
- After: Calibrated probabilities in [0, 1] range
- Probabilities better reflect actual churn likelihood

**Output:**
- `probs`: NumPy array of shape (n_test_samples,)
- Each value ∈ [0, 1] representing churn probability
- Calibrated for interpretability

---

## Cell 21: Assign Predictions to Submission

**Code:**
```python
sub["Churn"]=probs
```

**Purpose:**
- Assigns calibrated probabilities to submission DataFrame
- Prepares predictions in required format

**Function Breakdown:**
- `sub`: DataFrame loaded from sample_submission.csv (Cell 2)
- `sub["Churn"]`: Accesses/creates 'Churn' column
- `probs`: Calibrated probability array from Cell 20

**Expected Output Format:**
- DataFrame with columns: ['id', 'Churn']
- 'id': From original submission template
- 'Churn': Prediction probabilities (0-1)

**Why Probabilities Instead of Labels:**
- Competition requires probability predictions
- AUC evaluation metric needs probabilities
- Binary predictions (0/1) would lose information
- Allows threshold tuning for business applications

---

## Cell 22: Save Submission File

**Code:**
```python
sub.to_csv("submission.csv", index=False)
```

**Purpose:**
- Saves final predictions to CSV file for submission
- Creates 'submission.csv' in current directory

**Function Breakdown:**
- `sub.to_csv()`: Writes DataFrame to CSV file
- `"submission.csv"`: Output filename
- `index=False`: Excludes row indices from output

**Why Exclude Index:**
- Pandas adds row numbers by default
- Competition submissions don't need index column
- Cleaner file format
- Smaller file size

**Expected File Structure:**
```
id,Churn
0,0.234
1,0.567
2,0.123
...
```

**Output:**
- CSV file with predictions
- Ready for submission to competition or deployment
- Contains probability predictions for all test samples

**Best Practices:**
- Always check submission format matches requirements
- Verify file has correct number of rows
- Ensure no missing values
- Confirm probability range is [0, 1]

---

## Summary

### Pipeline Overview:
1. **Data Loading**: Load train/test data
2. **EDA**: Explore distributions (value counts, boxplots, pairplots)
3. **Preprocessing**: Separate feature types, drop ID
4. **Model Training**: 5-fold cross-validation with 3 models
5. **Ensemble Optimization**: Grid search for optimal weights
6. **Probability Calibration**: Isotonic regression
7. **Prediction**: Generate and save final predictions

### Models Used:
1. **XGBoost**: Gradient boosting with GPU acceleration
2. **CatBoost**: Gradient boosting with categorical handling
3. **GradientBoostingClassifier**: Scikit-learn implementation

### Key Techniques:
- Stratified K-Fold cross-validation
- Out-of-fold predictions
- Grid search weight optimization
- Probability calibration
- Ensemble averaging

### Feature Engineering Insights:
- Contract type is strongest predictor
- Internet service and features show clear patterns
- Gender has minimal impact
- Security and support services reduce churn

### Performance Considerations:
- Large dataset (594K samples) requires efficient processing
- GPU acceleration used for XGBoost and CatBoost
- Grid search explores ~400 weight combinations
- Calibrated probabilities improve interpretability