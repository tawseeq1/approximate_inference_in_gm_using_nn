from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
import numpy as np

def build_inference_network(input_dim, hidden_layers=[64, 32, 16], dropout_rates=[0.3, 0.2, 0], 
                          learning_rate=0.001):

    model = keras.Sequential()
    
    model.add(Dense(hidden_layers[0], activation='relu', input_shape=(input_dim,)))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rates[0]))
    
    for i in range(1, len(hidden_layers)):
        model.add(Dense(hidden_layers[i], activation='relu'))
        if i < len(dropout_rates) and dropout_rates[i] > 0:
            model.add(BatchNormalization())
            model.add(Dropout(dropout_rates[i]))
    
    model.add(Dense(1))  
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='mse'
    )
    
    return model

def create_early_stopping(patience=20):

    return keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=patience,
        restore_best_weights=True
    )

def train_model(model, X_train, y_train, batch_size=16, epochs=200, validation_split=0.2, callbacks=None):

    if callbacks is None:
        callbacks = [create_early_stopping()]
        
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        callbacks=callbacks,
        verbose=1
    )
    
    return history