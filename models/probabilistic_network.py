import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input, Concatenate

def gaussian_nll_loss(y_true, y_pred):

    mean, log_var = y_pred[..., :1], y_pred[..., 1:]
    var = tf.exp(log_var)
    return 0.5 * tf.reduce_mean(
        tf.math.log(2 * np.pi * var) + tf.square(y_true - mean) / var
    )

def build_probabilistic_inference_network(input_dim, hidden_layers=[64, 32, 16], 
                                       dropout_rates=[0.3, 0.2, 0], learning_rate=0.001):

    inputs = Input(shape=(input_dim,))
    
    x = Dense(hidden_layers[0], activation='relu')(inputs)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rates[0])(x)
    
    for i in range(1, len(hidden_layers)):
        x = Dense(hidden_layers[i], activation='relu')(x)
        if i < len(dropout_rates) and dropout_rates[i] > 0:
            x = BatchNormalization()(x)
            x = Dropout(dropout_rates[i])(x)
    
    mean = Dense(1)(x)
    log_var = Dense(1)(x)
    
    outputs = Concatenate()([mean, log_var])
    model = keras.Model(inputs=inputs, outputs=outputs)
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=gaussian_nll_loss
    )
    
    return model

def get_probabilistic_predictions(model, X_test_scaled, scaler_y):

    predictions = model.predict(X_test_scaled)
    means = predictions[:, 0]
    log_vars = predictions[:, 1]
    stds = np.sqrt(np.exp(log_vars))
    
    means_original = scaler_y.inverse_transform(means.reshape(-1, 1)).flatten()
    stds_original = stds * scaler_y.scale_
    
    return means_original, stds_original

def calculate_interval_coverage(y_true, means, stds, confidence=1.96):

    within_interval = np.sum(
        (y_true >= means - confidence * stds) & 
        (y_true <= means + confidence * stds)
    )
    coverage = within_interval / len(y_true)
    return coverage