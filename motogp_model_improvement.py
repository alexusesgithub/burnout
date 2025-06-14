import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder, RobustScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.feature_selection import SelectFromModel
import warnings
warnings.filterwarnings('ignore')

# 1. Load Data
print("Loading data...")
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')
print('Train shape:', train_data.shape)
print('Test shape:', test_data.shape)

# 2. Advanced Outlier Handling
print("\nHandling outliers...")
Q1 = train_data['Lap_Time_Seconds'].quantile(0.25)
Q3 = train_data['Lap_Time_Seconds'].quantile(0.75)
IQR = Q3 - Q1
lower = Q1 - 2.5 * IQR  # More aggressive outlier removal
upper = Q3 + 2.5 * IQR
filtered_train = train_data[(train_data['Lap_Time_Seconds'] >= lower) & (train_data['Lap_Time_Seconds'] <= upper)]
print(f'Original train size: {len(train_data)}, After outlier removal: {len(filtered_train)}')

# 3. Feature Engineering
print("\nPerforming feature engineering...")
def engineer_features(df):
    df = df.copy()
    
    # Speed-based features
    if 'Avg_Speed_kmh' in df.columns:
        df['Speed_kmh_squared'] = df['Avg_Speed_kmh'] ** 2
        df['Speed_kmh_cubed'] = df['Avg_Speed_kmh'] ** 3
    
    # Temperature interaction features
    if all(col in df.columns for col in ['Ambient_Temperature_Celsius', 'Track_Temperature_Celsius']):
        df['Temp_Difference'] = df['Track_Temperature_Celsius'] - df['Ambient_Temperature_Celsius']
        df['Temp_Interaction'] = df['Ambient_Temperature_Celsius'] * df['Track_Temperature_Celsius']
    
    # Circuit complexity features
    if all(col in df.columns for col in ['Circuit_Length_km', 'Corners_per_Lap']):
        df['Circuit_Complexity'] = df['Corners_per_Lap'] / df['Circuit_Length_km']
    
    # Rider experience features
    if all(col in df.columns for col in ['starts', 'podiums', 'wins']):
        df['Podium_Rate'] = df['podiums'] / df['starts'].replace(0, 1)
        df['Win_Rate'] = df['wins'] / df['starts'].replace(0, 1)
    
    # Time-based features
    if 'year_x' in df.columns and 'max_year' in df.columns and 'min_year' in df.columns:
        df['Years_Active'] = df['max_year'] - df['min_year']
    
    return df

filtered_train = engineer_features(filtered_train)
test_data = engineer_features(test_data)

# 4. Preprocessing: Label encode high-cardinality categoricals, one-hot encode a few key ones
print("\nPreprocessing data...")
# Identify categorical columns
cat_cols = filtered_train.select_dtypes(include=['object']).columns.tolist()
# Choose a few key categorical columns for one-hot encoding (low cardinality)
onehot_cols = [col for col in cat_cols if filtered_train[col].nunique() <= 10]
# The rest will be label encoded
label_cols = [col for col in cat_cols if col not in onehot_cols]

# Label encode high-cardinality columns
for col in label_cols:
    le = LabelEncoder()
    filtered_train[col] = le.fit_transform(filtered_train[col].astype(str))
    test_data[col] = le.transform(test_data[col].astype(str))

# Now set up the preprocessor
numerical_cols = filtered_train.select_dtypes(include=[np.number]).columns.drop('Lap_Time_Seconds')
preprocessor = ColumnTransformer([
    ('num', RobustScaler(), numerical_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), onehot_cols)
])

X = filtered_train.drop('Lap_Time_Seconds', axis=1)
y = filtered_train['Lap_Time_Seconds']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocess features
X_train_prep = preprocessor.fit_transform(X_train)
X_val_prep = preprocessor.transform(X_val)
X_prep = preprocessor.fit_transform(X)
test_prep = preprocessor.transform(test_data)

# 5. Advanced Model Training
print("\nTraining models...")
# XGBoost
xgb = XGBRegressor(
    n_estimators=1000,
    learning_rate=0.01,
    max_depth=7,
    min_child_weight=1,
    subsample=0.8,
    colsample_bytree=0.8,
    gamma=0,
    reg_alpha=0.1,
    reg_lambda=1,
    random_state=42
)

# LightGBM
lgbm = LGBMRegressor(
    n_estimators=1000,
    learning_rate=0.01,
    max_depth=7,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1,
    random_state=42
)

# CatBoost
catboost = CatBoostRegressor(
    iterations=1000,
    learning_rate=0.01,
    depth=7,
    l2_leaf_reg=3,
    bootstrap_type='Bernoulli',
    subsample=0.8,
    random_seed=42,
    verbose=False
)

# Train models
xgb.fit(X_train_prep, y_train)
lgbm.fit(X_train_prep, y_train)
catboost.fit(X_train_prep, y_train)

# 6. Model Evaluation and Ensemble
print("\nEvaluating models...")
xgb_pred = xgb.predict(X_val_prep)
lgbm_pred = lgbm.predict(X_val_prep)
catboost_pred = catboost.predict(X_val_prep)

# Weighted ensemble
ensemble_pred = (0.4 * xgb_pred + 0.3 * lgbm_pred + 0.3 * catboost_pred)
ensemble_rmse = np.sqrt(mean_squared_error(y_val, ensemble_pred))
print(f'Ensemble RMSE: {ensemble_rmse:.4f}')

# 7. Final Model Training and Predictions
print("\nGenerating final predictions...")
# Train final models on full training data
xgb.fit(X_prep, y)
lgbm.fit(X_prep, y)
catboost.fit(X_prep, y)

# Generate predictions
xgb_test_pred = xgb.predict(test_prep)
lgbm_test_pred = lgbm.predict(test_prep)
catboost_test_pred = catboost.predict(test_prep)

# Weighted ensemble for test predictions
final_predictions = (0.4 * xgb_test_pred + 0.3 * lgbm_test_pred + 0.3 * catboost_test_pred)

# Create submission file
submission = pd.DataFrame({
    'Unique ID': test_data['Unique ID'],
    'Lap_Time_Seconds': final_predictions
})
submission.to_csv('submission.csv', index=False)
print('Final submission file created!')

# 8. Feature Importance Analysis
print("\nAnalyzing feature importance...")
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'xgb_importance': xgb.feature_importances_,
    'lgbm_importance': lgbm.feature_importances_,
    'catboost_importance': catboost.feature_importances_
})

feature_importance['avg_importance'] = (
    feature_importance['xgb_importance'] * 0.4 +
    feature_importance['lgbm_importance'] * 0.3 +
    feature_importance['catboost_importance'] * 0.3
)

print("\nTop 10 most important features:")
print(feature_importance.sort_values('avg_importance', ascending=False).head(10))