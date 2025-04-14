from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

def calculate_regression_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    return {
        'mse': mse,
        'r2': r2,
        'rmse': np.sqrt(mse)
    }

def print_metrics(train_metrics, test_metrics):

    print("=== Regression Metrics ===")
    print(f"Train MSE: {train_metrics['mse']:.4f}")
    print(f"Test MSE: {test_metrics['mse']:.4f}")
    print(f"Train R²: {train_metrics['r2']:.4f}")
    print(f"Test R²: {test_metrics['r2']:.4f}")
    print(f"Train RMSE: {train_metrics['rmse']:.4f}")
    print(f"Test RMSE: {test_metrics['rmse']:.4f}")

def print_probabilistic_metrics(coverage):
    print("=== Probabilistic Metrics ===")
    print(f"95% Confidence Interval Coverage: {coverage:.2%}")