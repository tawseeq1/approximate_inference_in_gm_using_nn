# Approximate Inference in Graphical Models using Neural Networks

This project demonstrates approximate inference in graphical models using neural networks. Using the UCI Wine dataset, we model relationships between features as a graphical model and use neural networks to learn approximate inference.

## Project Overview

The goal is to perform approximate inference in a graphical model where we want to predict one variable (e.g., alcohol content) given other variables in the model. Instead of exact inference which can be computationally expensive, we use neural networks to learn the conditional distributions.

## Features

- Load and preprocess the UCI Wine dataset
- Build a graphical model structure from selected features
- Train regression and probabilistic neural networks for approximate inference
- Visualize the graphical model and conditional distributions
- Evaluate inference performance with various metrics

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/approximate-inference-graphical-models.git
cd approximate-inference-graphical-models

# Install dependencies
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow networkx
```

## Project Structure

```
approximate-inference-graphical-models/
│
├── main.py                   # Main script that orchestrates the workflow
├── data/
│   └── data_loader.py        # Functions for loading and preparing data
│
├── models/
│   ├── inference_network.py  # Neural network model definitions
│   └── probabilistic_network.py  # Probabilistic network implementation
│
├── visualization/
│   └── visualize.py          # All visualization functions
│
├── utils/
│   └── metrics.py            # Evaluation metrics and utility functions
│
├── config.py                 # Configuration parameters
│
└── README.md                 # Project documentation
```

## Usage

Simply run the main script:

```bash
python main.py
```

This will:
1. Load the Wine dataset
2. Prepare the data for inference
3. Train regression and probabilistic models
4. Perform inference and evaluate