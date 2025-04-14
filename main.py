import numpy as np
import matplotlib.pyplot as plt
import networkx as nx  # Make sure to install this dependency


from data.data_loader import load_wine_data, prepare_inference_data, generate_conditional_distribution_data
from models.inference_network import build_inference_network, train_model, create_early_stopping
from models.probabilistic_network import (build_probabilistic_inference_network, 
                                         get_probabilistic_predictions, 
                                         calculate_interval_coverage)
from utils.metrics import calculate_regression_metrics, print_metrics, print_probabilistic_metrics
from visualization.visualize import (plot_correlation_graph, plot_correlation_matrix, plot_target_distribution,
                                   plot_training_history, plot_predictions,
                                   plot_probabilistic_predictions, plot_graphical_model,
                                   plot_conditional_distribution)
import config

def main():
    print("Loading Wine dataset...")
    X, y = load_wine_data()
    print(f"Dataset features: {X.columns.tolist()}")
    print(f"Dataset shape: {X.shape}")
    
    plot_correlation_matrix(X, figsize=config.CORRELATION_FIG_SIZE)
    
    print(f"Preparing data for inference of {config.TARGET_FEATURE} from {config.SELECTED_FEATURES}...")
    data = prepare_inference_data(
        X, 
        config.TARGET_FEATURE, 
        config.SELECTED_FEATURES, 
        test_size=config.TEST_SIZE, 
        random_state=config.RANDOM_SEED
    )
    
    plot_target_distribution(
        data['y_train'], 
        config.TARGET_FEATURE,
        figsize=config.PREDICTION_FIG_SIZE
    )
    
    input_dim = data['X_train_scaled'].shape[1]
    # regression_model = build_inference_network(
    #     input_dim,
    #     hidden_layers=config.HIDDEN_LAYERS,
    #     dropout_rates=config.DROPOUT_RATES,
    #     learning_rate=config.LEARNING_RATE
    # )
    
    early_stopping = create_early_stopping()
    
    # history = train_model(
    #     regression_model,
    #     data['X_train_scaled'],
    #     data['y_train_scaled'],
    #     batch_size=config.BATCH_SIZE,
    #     epochs=config.EPOCHS,
    #     validation_split=config.VALIDATION_SPLIT,
    #     callbacks=[early_stopping]
    # )
    
    # Plot training history
    # plot_training_history(history, figsize=config.HISTORY_FIG_SIZE)
    
    # # Make predictions
    # y_train_pred_scaled = regression_model.predict(data['X_train_scaled'])
    # y_test_pred_scaled = regression_model.predict(data['X_test_scaled'])
    
    # Inverse transform predictions to original scale
    # y_train_pred = data['scaler_y'].inverse_transform(y_train_pred_scaled)
    # y_test_pred = data['scaler_y'].inverse_transform(y_test_pred_scaled)
    
    # Calculate and print metrics
    # train_metrics = calculate_regression_metrics(data['y_train'], y_train_pred)
    # test_metrics = calculate_regression_metrics(data['y_test'], y_test_pred)
    # print_metrics(train_metrics, test_metrics)
    
    # # Plot predictions
    # plot_predictions(
    #     data['y_train'],
    #     y_train_pred,
    #     data['y_test'],
    #     y_test_pred,
    #     config.TARGET_FEATURE,
    #     figsize=config.HISTORY_FIG_SIZE
    # )
    
    # Build and train probabilistic model
    print("Building and training probabilistic model...")
    prob_model = build_probabilistic_inference_network(
        input_dim,
        hidden_layers=config.HIDDEN_LAYERS,
        dropout_rates=config.DROPOUT_RATES,
        learning_rate=config.LEARNING_RATE
    )
    
    prob_history = train_model(
        prob_model,
        data['X_train_scaled'],
        data['y_train_scaled'],
        batch_size=config.BATCH_SIZE,
        epochs=config.EPOCHS,
        validation_split=config.VALIDATION_SPLIT,
        #callbacks=[early_stopping]     #it can be turned on if the model overfits
    )
    
    test_means, test_stds = get_probabilistic_predictions(
        prob_model, 
        data['X_test_scaled'], 
        data['scaler_y']
    )
    
    plot_probabilistic_predictions(
        data['y_test'],
        test_means,
        test_stds,
        config.TARGET_FEATURE,
        figsize=config.PREDICTION_FIG_SIZE
    )
    
    coverage = calculate_interval_coverage(data['y_test'].values, test_means, test_stds)
    print_probabilistic_metrics(coverage)
    
    
    correlation_matrix = X.corr()
    plot_correlation_graph(
        X,
        correlation_matrix,
        threshold=config.CORRELATION_THRESHOLD,
        figsize=config.GRAPHICAL_MODEL_FIG_SIZE
    )

    plot_graphical_model(
        config.SELECTED_FEATURES,
        config.TARGET_FEATURE,
        figsize=config.GRAPHICAL_MODEL_FIG_SIZE
    )
    
    feature_to_vary = 'proline'
    print(f"Generating conditional distribution P({config.TARGET_FEATURE} | {feature_to_vary})...")
    
    test_points_scaled, varied_values = generate_conditional_distribution_data(
        data['X_train'],
        feature_to_vary,
        config.SELECTED_FEATURES,
        data['scaler_X']
    )
    
    varied_means, varied_stds = get_probabilistic_predictions(
        prob_model, 
        test_points_scaled, 
        data['scaler_y']
    )
    
    plot_conditional_distribution(
        varied_values,
        varied_means,
        varied_stds,
        feature_to_vary,
        config.TARGET_FEATURE,
        figsize=config.PREDICTION_FIG_SIZE
    )
    
    print("Saving model...")
    prob_model.save(config.PROBABILISTIC_MODEL_PATH)
    #print(f"Models saved to {config.REGRESSION_MODEL_PATH} and {config.PROBABILISTIC_MODEL_PATH}")
    print(f"Models saved to {config.PROBABILISTIC_MODEL_PATH}")
    print("Approximate inference complete!")

if __name__ == "__main__":
    main()