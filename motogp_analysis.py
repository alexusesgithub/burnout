import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import xgboost as xgb
import time
import gc  # For garbage collection
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV

# Limit CPU usage
os.environ['OMP_NUM_THREADS'] = '2'  # Limit OpenMP threads
os.environ['MKL_NUM_THREADS'] = '2'  # Limit MKL threads

# Set random seed for reproducibility
np.random.seed(42)

def load_data_in_chunks(file_path, chunk_size=50000):  # Reduced chunk size
    """Load data in chunks to manage memory usage"""
    chunks = []
    for chunk in pd.read_csv(file_path, chunksize=chunk_size):
        chunks.append(chunk)
        gc.collect()  # Collect garbage after each chunk
    return pd.concat(chunks, ignore_index=True)

def plot_feature_importance(model, feature_names, top_n=15, title="Feature Importance"):
    """Plot feature importance for the model"""
    plt.figure(figsize=(12, 6))
    
    # Get feature importance
    if isinstance(model, xgb.XGBRegressor):
        importance = model.feature_importances_
    else:  # Random Forest
        importance = model.feature_importances_
    
    # Create DataFrame for easier plotting
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance
    })
    
    # Sort and get top N features
    importance_df = importance_df.sort_values('Importance', ascending=False).head(top_n)
    
    # Plot
    sns.barplot(x='Importance', y='Feature', data=importance_df)
    plt.title(title)
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.close()
    
    return importance_df

def plot_residuals(y_true, y_pred, title="Residuals vs Predicted Values"):
    """Plot residuals against predicted values"""
    residuals = y_true - y_pred
    
    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title(title)
    plt.tight_layout()
    plt.savefig('residuals_plot.png')
    plt.close()

def plot_performance_comparison(metrics_dict):
    """Plot performance comparison across different models"""
    plt.figure(figsize=(12, 6))
    
    # Create DataFrame for plotting
    metrics_df = pd.DataFrame(metrics_dict).T
    
    # Plot RMSE and MAE
    metrics_df.plot(kind='bar', y=['RMSE', 'MAE'])
    plt.title('Model Performance Comparison')
    plt.xlabel('Model Stage')
    plt.ylabel('Error')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('performance_comparison.png')
    plt.close()

def select_features(X, y, model, threshold=0.01):
    """Select features based on importance threshold"""
    importance = model.feature_importances_
    selected_features = X.columns[importance > threshold].tolist()
    print(f"\nSelected {len(selected_features)} features with importance > {threshold}")
    return selected_features

def plot_predicted_vs_actual(y_true, y_pred, title="Predicted vs Actual Values"):
    """Plot predicted vs actual values with a perfect prediction line"""
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', label='Perfect Prediction')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig('predicted_vs_actual.png')
    plt.close()

# Load data in chunks
print("Loading data in chunks...")
start_time = time.time()
train_df = load_data_in_chunks('train.csv')
val_df = load_data_in_chunks('val.csv')
test_df = load_data_in_chunks('test.csv')
print(f"Data loading completed in {time.time() - start_time:.2f} seconds")

# Basic EDA
print("\nDataset shapes:")
print(f"Training set: {train_df.shape}")
print(f"Validation set: {val_df.shape}")
print(f"Test set: {test_df.shape}")

print("\nMissing values in training set:")
print(train_df.isnull().sum())

# Preprocessing
print("\nPreprocessing data...")
def preprocess(df, fit_encoders=False, encoders=None):
    df = df.copy()
    
    # Handle missing values
    if 'Penalty' in df.columns:
        df['Penalty'] = df['Penalty'].fillna('None')
    
    # Get categorical columns
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    for col in ['Lap_Time_Seconds', 'Unique ID']:
        if col in cat_cols:
            cat_cols.remove(col)
    
    # Encode categoricals
    if fit_encoders:
        encoders = {col: LabelEncoder().fit(df[col].astype(str)) for col in cat_cols}
    for col in cat_cols:
        df[col] = encoders[col].transform(df[col].astype(str))
    
    return df, encoders

# Prepare data
print("Preparing features...")
train_X = train_df.drop(['Lap_Time_Seconds', 'Unique ID'], axis=1)
train_y = train_df['Lap_Time_Seconds']
train_X, encoders = preprocess(train_X, fit_encoders=True)

# Store feature names for importance analysis
feature_names = train_X.columns.tolist()

# Clear memory
del train_df
gc.collect()

val_X = val_df.drop(['Lap_Time_Seconds', 'Unique ID'], axis=1)
val_y = val_df['Lap_Time_Seconds']
val_X, _ = preprocess(val_X, fit_encoders=False, encoders=encoders)

# Clear memory
del val_df
gc.collect()

test_ids = test_df['Unique ID']
test_X = test_df.drop(['Unique ID'], axis=1)
test_X, _ = preprocess(test_X, fit_encoders=False, encoders=encoders)

# Clear memory
del test_df
gc.collect()

# Train Random Forest (ultra-lightweight version)
print("\nTraining Random Forest...")
rf = RandomForestRegressor(
    n_estimators=25,      # Further reduced number of trees
    max_depth=8,          # Further limited depth
    min_samples_split=200,  # Further increased to reduce complexity
    n_jobs=2,             # Limit parallel jobs
    random_state=42
)

start_time = time.time()
rf.fit(train_X, train_y)
rf_time = time.time() - start_time
print(f"Random Forest training completed in {rf_time:.2f} seconds")

# Evaluate Random Forest
rf_val_preds = rf.predict(val_X)
rf_rmse = np.sqrt(mean_squared_error(val_y, rf_val_preds))
rf_mae = mean_absolute_error(val_y, rf_val_preds)
print(f"Random Forest RMSE: {rf_rmse:.4f}")
print(f"Random Forest MAE: {rf_mae:.4f}")

# Clear memory
del rf_val_preds
gc.collect()

# Train XGBoost (ultra-lightweight version)
print("\nTraining XGBoost...")
xgbr = xgb.XGBRegressor(
    n_estimators=25,      # Further reduced number of trees
    max_depth=8,          # Further limited depth
    learning_rate=0.1,
    tree_method='hist',   # More memory efficient
    n_jobs=2,             # Limit parallel jobs
    random_state=42
)

start_time = time.time()
xgbr.fit(train_X, train_y)
xgb_time = time.time() - start_time
print(f"XGBoost training completed in {xgb_time:.2f} seconds")

# Evaluate XGBoost
xgb_val_preds = xgbr.predict(val_X)
xgb_rmse = np.sqrt(mean_squared_error(val_y, xgb_val_preds))
xgb_mae = mean_absolute_error(val_y, xgb_val_preds)
print(f"XGBoost RMSE: {xgb_rmse:.4f}")
print(f"XGBoost MAE: {xgb_mae:.4f}")

# Select best model based on RMSE and training time
if xgb_rmse < rf_rmse and xgb_time < rf_time:
    best_model = xgbr
    print("\nXGBoost selected as best model (better RMSE and faster training)")
else:
    best_model = rf
    print("\nRandom Forest selected as best model")

# Analyze feature importance
print("\nAnalyzing feature importance...")
importance_df = plot_feature_importance(best_model, feature_names, title="Initial Model Feature Importance")
print("\nTop 15 most important features:")
print(importance_df)

# Select important features
selected_features = select_features(train_X, train_y, best_model, threshold=0.01)
train_X_selected = train_X[selected_features]
val_X_selected = val_X[selected_features]
test_X_selected = test_X[selected_features]

# Train model with selected features
print("\nTraining model with selected features...")
selected_model = xgb.XGBRegressor(
    n_estimators=25,
    max_depth=8,
    learning_rate=0.1,
    tree_method='hist',
    n_jobs=2,
    random_state=42
)

selected_model.fit(train_X_selected, train_y)
selected_val_preds = selected_model.predict(val_X_selected)
selected_rmse = np.sqrt(mean_squared_error(val_y, selected_val_preds))
selected_mae = mean_absolute_error(val_y, selected_val_preds)
print(f"Selected features model RMSE: {selected_rmse:.4f}")
print(f"Selected features model MAE: {selected_mae:.4f}")

# Hyperparameter tuning for XGBoost
print("\nPerforming hyperparameter tuning...")
param_grid = {
    'n_estimators': [20, 25, 30],
    'max_depth': [6, 8, 10],
    'learning_rate': [0.05, 0.1, 0.15]
}

grid_search = GridSearchCV(
    xgb.XGBRegressor(
        tree_method='hist',
        n_jobs=2,
        random_state=42
    ),
    param_grid,
    cv=3,
    scoring='neg_root_mean_squared_error',
    n_jobs=2
)

grid_search.fit(train_X_selected, train_y)
print(f"Best parameters: {grid_search.best_params_}")

# Train final model with best parameters
final_model = xgb.XGBRegressor(
    **grid_search.best_params_,
    tree_method='hist',
    n_jobs=2,
    random_state=42
)

final_model.fit(train_X_selected, train_y)
final_val_preds = final_model.predict(val_X_selected)
final_rmse = np.sqrt(mean_squared_error(val_y, final_val_preds))
final_mae = mean_absolute_error(val_y, final_val_preds)
print(f"Final model RMSE: {final_rmse:.4f}")
print(f"Final model MAE: {final_mae:.4f}")

# Plot final model feature importance
final_importance_df = plot_feature_importance(
    final_model, 
    selected_features,
    title="Final Model Feature Importance"
)

# Plot residuals for final model
plot_residuals(val_y, final_val_preds)

# Compare performance across different stages
performance_metrics = {
    'Baseline': {'RMSE': xgb_rmse, 'MAE': xgb_mae},
    'Feature Selection': {'RMSE': selected_rmse, 'MAE': selected_mae},
    'Hyperparameter Tuning': {'RMSE': final_rmse, 'MAE': final_mae}
}

plot_performance_comparison(performance_metrics)

# Print performance comparison table
print("\nPerformance Comparison:")
print("=" * 50)
print("Stage\t\tRMSE\t\tMAE")
print("-" * 50)
for stage, metrics in performance_metrics.items():
    print(f"{stage}\t\t{metrics['RMSE']:.4f}\t\t{metrics['MAE']:.4f}")
print("=" * 50)

# After final model evaluation, add:
plot_predicted_vs_actual(val_y, final_val_preds)

# Save validation predictions
val_predictions = pd.DataFrame({
    'Unique ID': val_df['Unique ID'],
    'Lap_Time_Seconds': final_val_preds
})
val_predictions.to_csv('val_predictions.csv', index=False)

# Clear memory
del train_X, train_y, val_X, val_y, xgb_val_preds
gc.collect()

# Generate predictions for test set
print("\nGenerating test predictions...")
test_preds = final_model.predict(test_X_selected)

# Create submission file
submission = pd.DataFrame({
    'Unique ID': test_ids,
    'Lap_Time_Seconds': test_preds
})

# Save submission
submission.to_csv('submission.csv', index=False)
print("Submission file created: submission.csv")

# Print summary of important features
print("\nFeature Importance Summary:")
print("=" * 50)
print("Top 5 most important features:")
for idx, row in final_importance_df.head().iterrows():
    print(f"{row['Feature']}: {row['Importance']:.4f}")
print("=" * 50) 