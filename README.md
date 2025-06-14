# MotoGP Lap Time Prediction

## Model Architecture
Ensemble model combining three algorithms with optimized weights:
- XGBoost (40%): n_estimators=1000, learning_rate=0.01, max_depth=7
- LightGBM (30%): n_estimators=1000, learning_rate=0.01, max_depth=7
- CatBoost (30%): iterations=1000, learning_rate=0.01, depth=7

## Technical Features
- Memory-efficient preprocessing using label encoding for high-cardinality features
- One-hot encoding for low-cardinality categorical features
- Robust scaling for numerical features
- Advanced outlier removal using 2.5 * IQR method
- Feature engineering for speed, temperature, and circuit metrics

## Implementation
```python
# Key preprocessing steps
preprocessor = ColumnTransformer([
    ('num', RobustScaler(), numerical_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), onehot_cols)
])

# Ensemble prediction
final_predictions = (0.4 * xgb_test_pred + 
                    0.3 * lgbm_test_pred + 
                    0.3 * catboost_test_pred)
```

## Requirements
- Python 3.7+
- pandas
- numpy
- scikit-learn
- xgboost
- lightgbm
- catboost
- matplotlib
- seaborn

## Installation
```bash
pip install -r requirements.txt
```

## Usage
1. Place data files:
   - `train.csv`: Training data
   - `test.csv`: Test data
2. Run model:
```bash
python motogp_model_improvement.py
```
3. Output: `submission.csv` with predictions

## Model Performance
- RMSE evaluation on validation set
- Feature importance analysis included
- Cross-validation for robust evaluation 