import numpy as np
import pandas as pd
import random
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from keras.models import load_model

# Parameters
n_features = 256  # Number of original features (equivalent to number of time steps per signal, for BioAmps EEG sensor)
n_samples = 500  # Total samples to generate
seizure_proportion = 0.5  # Proportion of seizure samples
n_channels = 8  # Number of EEG channels (BioAmps often has 8 or more channels)

# Frequency bands (Delta, Theta, Alpha, Beta, Gamma)
frequency_bands = {
    'Delta': (0.5, 4),   # Slow waves (sleep)
    'Theta': (4, 8),     # Light sleep and relaxation
    'Alpha': (8, 12),    # Calm alertness
    'Beta': (12, 30),    # Active thinking, alertness
    'Gamma': (30, 40)    # High-level processing, cognition
}


