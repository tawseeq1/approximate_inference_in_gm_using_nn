# visualization/visualize.py
"""
Visualization functions for the approximate inference project.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import networkx as nx

def plot_correlation_graph(X, correlation, threshold=0.4, figsize=(12, 10)):
    G = nx.Graph()

    for feature in X.columns:
        G.add_node(feature)

    for i, feature1 in enumerate(X.columns):
        for j, feature2 in enumerate(X.columns):
            if i < j:
                correlation_value = correlation.iloc[i, j]
                if abs(correlation_value) > threshold:
                    G.add_edge(feature1, feature2)

    plt.figure(figsize=figsize)
    pos = nx.kamada_kawai_layout(G)

    nx.draw(
        G, pos,
        with_labels=True,
        node_size=3000,
        node_color='lightblue',
        font_size=10,
        font_weight='bold',
        edge_color='gray'
    )

    plt.title(f'Undirected Graph of Features (|correlation| > {threshold})')
    plt.tight_layout()
    plt.show()


def plot_correlation_matrix(X, figsize=(12, 10)):

    plt.figure(figsize=figsize)
    correlation = X.corr()
    mask = np.triu(np.ones_like(correlation, dtype=bool))
    sns.heatmap(correlation, mask=mask, annot=True, fmt=".2f", cmap='coolwarm',
                square=True, linewidths=.5)
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    plt.show()

def plot_target_distribution(y, target_feature, figsize=(10, 6)):

    plt.figure(figsize=figsize)
    sns.histplot(y, kde=True)
    plt.title(f'Distribution of {target_feature}')
    plt.xlabel(target_feature)
    plt.ylabel('Frequency')
    plt.show()

def plot_training_history(history, figsize=(12, 4)):
    plt.figure(figsize=figsize)
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()

def plot_predictions(y_train, y_train_pred, y_test, y_test_pred, target_feature, figsize=(10, 6)):

    plt.subplot(1, 2, 2)
    plt.scatter(y_train, y_train_pred, alpha=0.5, label='Train')
    plt.scatter(y_test, y_test_pred, alpha=0.5, label='Test')
    
    all_vals = np.concatenate([y_train, y_test, y_train_pred.flatten(), y_test_pred.flatten()])
    min_val, max_val = all_vals.min(), all_vals.max()
    
    plt.plot([min_val, max_val], [min_val, max_val], 'k--')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title(f'Actual vs Predicted {target_feature}')
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_probabilistic_predictions(y_test, means, stds, target_feature, figsize=(10, 6)):

    plt.figure(figsize=figsize)
    plt.errorbar(range(len(y_test)), means, 
                 yerr=1.96*stds, fmt='o', alpha=0.5, 
                 label='Predicted with 95% CI')
    plt.scatter(range(len(y_test)), y_test.values, color='red', label='Actual')
    plt.xlabel('Test Sample')
    plt.ylabel(target_feature)
    plt.title(f'Probabilistic Inference for {target_feature}')
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_graphical_model(selected_features, target_feature, figsize=(10, 8)):

    plt.figure(figsize=figsize)
    G = nx.DiGraph()
    
    nodes = selected_features + [target_feature]
    for node in nodes:
        G.add_node(node)
    
    for feature in selected_features:
        G.add_edge(feature, target_feature)
    
    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, with_labels=True, node_size=3000, node_color='lightblue', 
            font_size=10, font_weight='bold', arrows=True, 
            arrowsize=20, connectionstyle='arc3,rad=0.1')
    
    plt.title('Graphical Model Structure for Wine Dataset')
    plt.tight_layout()
    plt.show()

def plot_conditional_distribution(varied_values, means, stds, feature_to_vary, target_feature, figsize=(10, 6)):

    plt.figure(figsize=figsize)
    plt.plot(varied_values, means, 'b-', label='Predicted Mean')
    plt.fill_between(varied_values, 
                     means - 1.96*stds,
                     means + 1.96*stds,
                     alpha=0.3, label='95% Confidence Interval')
    plt.xlabel(feature_to_vary)
    plt.ylabel(f'Predicted {target_feature}')
    plt.title(f'Conditional Distribution: P({target_feature} | {feature_to_vary})')
    plt.legend()
    plt.tight_layout()
    plt.show()