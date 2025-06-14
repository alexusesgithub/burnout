import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Load validation data
val_data = pd.read_csv('val.csv')
y_true = val_data['Lap_Time_Seconds']

# Load validation predictions
val_preds = pd.read_csv('val_predictions.csv')
y_pred = val_preds['Lap_Time_Seconds']

# Calculate metrics
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
mae = mean_absolute_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)
mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

print("\nModel Accuracy Metrics:")
print("=" * 50)
print(f"RMSE (Root Mean Square Error): {rmse:.4f} seconds")
print(f"MAE (Mean Absolute Error): {mae:.4f} seconds")
print(f"R² Score: {r2:.4f}")
print(f"Mean Absolute Percentage Error: {mape:.2f}%")
print("=" * 50)

# Print some example predictions
print("\nExample Predictions:")
print("=" * 50)
print("Actual\t\tPredicted\tDifference")
print("-" * 50)
for i in range(5):
    actual = y_true.iloc[i]
    predicted = y_pred.iloc[i]
    diff = predicted - actual
    print(f"{actual:.2f}\t\t{predicted:.2f}\t\t{diff:+.2f}")
print("=" * 50) 