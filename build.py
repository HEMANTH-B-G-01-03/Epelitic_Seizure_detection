import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, classification_report
from keras import models, layers
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv('data.csv')

# Prepare input features and binary labels
X = df.iloc[:, 1:-1].values  # Exclude ID and label columns
y = df['y'].values           # Labels

# Convert to binary classification: Seizure (1) vs. No Seizure (0)
y = np.where(y == 1, 1, 0)

# Dimensionality reduction (reduce features to 45)
pca = PCA(n_components=45)
X = pca.fit_transform(X)

# Split into training, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=1)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=1)

# Save test set for future use
np.save('X_test.npy', X_test)
np.save('y_test.npy', y_test)

# Normalize input data
X_train = (X_train - X_train.mean(axis=0)) / X_train.std(axis=0)
X_val = (X_val - X_val.mean(axis=0)) / X_val.std(axis=0)
X_test = (X_test - X_test.mean(axis=0)) / X_test.std(axis=0)

# Handle potential NaN or infinity values in normalized data
X_train = np.nan_to_num(X_train)
X_val = np.nan_to_num(X_val)
X_test = np.nan_to_num(X_test)

# Define the model
model = models.Sequential([
    layers.Input(shape=(45,)),
    layers.Dense(256, activation='relu'),
    layers.Dense(128, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(16, activation='relu'),
    layers.Dense(1, activation='sigmoid')  # Binary classification output
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
