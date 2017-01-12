"""
=======================================================
Drug activity prediction
=======================================================

Please, dowload the dataset from:

  https://drive.google.com/file/d/0B6VyRTVgbDFeWFFUUVRidUR3MFE/view?usp=sharing (25MB)

and change the path in the following script.

This script provides two precomputed kernel matrices (K_train, K_test) along with the corresponding labels for each line K(x_i, ...) (namely y_train and y_test).

Author: Maxime Sangnier (Telecom ParisTech)
Date: 18-Nov-2015

"""

import numpy as np
path = "/home/maxime/matlab/toolboxes/drug_activity/data"

# Load the data
K = np.loadtxt(path + "/ncicancer_kernel_hf_ex0.txt") # Load the kernel
y = np.loadtxt(path + "/ncicancer_targets_ex0.txt")[:, 0] # Load the targets
y = (y-np.min(y)) / (np.max(y)-np.min(y)) # Scale the targets

# Split train/test sets
indices = np.random.permutation(K.shape[0])
train_idx, test_idx = indices[:K.shape[0]/2], indices[K.shape[0]/2:]
K_train = K[train_idx][:, train_idx]
y_train = y[train_idx]
K_test = K[test_idx][:, train_idx]
y_test = y[test_idx]
