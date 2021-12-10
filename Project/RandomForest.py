import numpy as np

# ------ Load data
# Training set
X_train = np.load("data/train_set.npy", allow_pickle=True)
y_train = np.load("data/train_labels.npy", allow_pickle=True)

# Testing set
X_test = np.load("data/test_set.npy", allow_pickle=True)
y_test = np.load("data/test_labels.npy", allow_pickle=True)

# Columns names
columns = np.load("data/columns.npy", allow_pickle=True)



# -- Train Random Forst


# -- Tune Random Forest


