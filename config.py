import numpy as np
import tensorflow as tf

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

LEARNING_RATE = 0.001
BATCH_SIZE = 16
EPOCHS = 200
VALIDATION_SPLIT = 0.2
TEST_SIZE = 0.2

HIDDEN_LAYERS = [64, 32, 16]
DROPOUT_RATES = [0.3, 0.2, 0]

TARGET_FEATURE = 'alcohol'
#SELECTED_FEATURES = ['color_intensity', 'flavanoids', 'proline', 'od280/od315_of_diluted_wines']
SELECTED_FEATURES = ['malic_acid', 'ash', 'alcalinity_of_ash', 'magnesium', 'total_phenols',
       'flavanoids', 'nonflavanoid_phenols', 'proanthocyanins',
       'color_intensity', 'hue', 'od280/od315_of_diluted_wines', 'proline']

REGRESSION_MODEL_PATH = 'wine_inference_model.h5'
PROBABILISTIC_MODEL_PATH = 'wine_probabilistic_inference_model.h5'


CORRELATION_FIG_SIZE = (12, 10)
HISTORY_FIG_SIZE = (12, 4)
PREDICTION_FIG_SIZE = (10, 6)
CORRELATION_THRESHOLD = 0.4
GRAPHICAL_MODEL_FIG_SIZE = (10, 8)