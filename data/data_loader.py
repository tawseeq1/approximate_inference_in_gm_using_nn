import pandas as pd
import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_wine_data():
    wine = load_wine()
    X = pd.DataFrame(wine.data, columns=wine.feature_names)
    y = wine.target
    
    return X, y

def prepare_inference_data(X, target_feature, selected_features, test_size=0.2, random_state=42):

    X_inference = X[selected_features]
    y_inference = X[target_feature]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_inference, y_inference, test_size=test_size, random_state=random_state)
    
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)
    
    y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).flatten()
    y_test_scaled = scaler_y.transform(y_test.values.reshape(-1, 1)).flatten()
    
    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'X_train_scaled': X_train_scaled,
        'X_test_scaled': X_test_scaled,
        'y_train_scaled': y_train_scaled,
        'y_test_scaled': y_test_scaled,
        'scaler_X': scaler_X,
        'scaler_y': scaler_y
    }

def generate_conditional_distribution_data(X_train, feature_to_vary, selected_features, scaler_X):

    feature_idx = selected_features.index(feature_to_vary)
    
    feature_min = X_train[feature_to_vary].min()
    feature_max = X_train[feature_to_vary].max()
    varied_values = np.linspace(feature_min, feature_max, 100)
    
    median_values = X_train.median().values
    test_points = np.tile(median_values, (100, 1))
    test_points[:, feature_idx] = varied_values
    
    test_points_scaled = scaler_X.transform(test_points)
    
    return test_points_scaled, varied_values