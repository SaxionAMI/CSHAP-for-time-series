# Import packages
import sys
import torch 
import os
import random

import numpy as np

from .cshap import CSHAPInformed, CustomDecomposition, EMDDecomposition, DWTDecomposition
from .utils import load_model, load_array, pickle_array

# Script used to obtain concept-informed C-SHAP values


# Set seeds
np.random.seed(10)
torch.manual_seed(10)
random.seed(10)

# Settings
dataset = 'opportunity_uni'         # Dataset to calculate SHAP values for
exact = True                # If True calculate exact SHAP values, else use KernelSHAP
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Device to load model and data to

# Select decomposition to use
decomposition_def = 'trend+bias+scale*(low_freq+var*high_freq)'
max_imfs = 4            # Number of IMFs to include if EMD is used
noise_sigma = 1         # Used for the low_freq component if custom decomposition is used

# Input paths
data_path = './data/prepared/' + dataset
model_path = './models/' + dataset + '/' + decomposition_def
model_path = model_path.replace('*', 'x')

# Path to store calculated SHAP values to
output_path = './data/shap/' + dataset + '/multi_channel/' + decomposition_def
output_path = output_path.replace('*', 'x')

# Select how many models you want to calculate SHAP values for.
n_runs = 10

# Update settings based on dataset
if dataset == 'fordA':
    class_dict = {0: 'Normal', 1: 'Anomalous'}      # Mapping from target values to class names
    bidirectional = True                            # Whether to use bidirectional GRU layers
    task = 'classification'                         # Regression or classification
    softmax = True                                  # Whether to apply softmax to model output
    var_window_size = 50                            # Used for the variance component in custom decomposition
elif dataset == 'opportunity_uni':
    class_dict = {0:'Stand', 1:'Walk', 2:'Sit', 3:'Lie'}
    bidirectional = True
    task = 'classification'
    softmax = True
    var_window_size = 27
elif dataset == 'turbofan':
    class_dict = None
    bidirectional = False
    task = 'regression'
    softmax = False
    var_window_size = 30

# If output path does not yet exist, create it
if not os.path.exists(output_path):
    os.makedirs(output_path)
print('output path:', output_path)

# Load data
X_train = load_array(data_path + '/X_train.pkl')
X_test = load_array(data_path +  '/X_test.pkl')

y_train = load_array(data_path +  '/y_train.pkl')
y_test = load_array(data_path +  '/y_test.pkl')

# Placeholder time steps
t = np.linspace(0, X_train.shape[1] + 1, X_train.shape[1])

# Load background samples
background_samples = load_array(data_path + '/background_samples.pkl')

# Reformat input data if necessary
if len(X_train.shape) == 2:
    X_train = X_train[:, :, None]
    X_test = X_test[:, :, None]
    background_samples = background_samples[:, :, None]

# Instantiate the selected decomposition
if decomposition_def == 'EMD':
    decomposition = EMDDecomposition(max_imfs=max_imfs)
elif decomposition_def == 'DWT':
    decomposition = DWTDecomposition(level=3)
else:
    decomposition = CustomDecomposition(decomposition_def, noise_sigma=noise_sigma, 
                                        var_window_size=var_window_size)

# Decompose data
X_train = decomposition.decompose(t, X_train)
X_test = decomposition.decompose(t, X_test)
background_samples = decomposition.decompose(t, background_samples)

for i in range(n_runs):
    # Load model
    model = load_model(model_path, task, bidirectional, X_train.shape[-1], class_dict, device, model_name='model_' + str(i) + '.pt')

    # Calculate C-SHAP values
    cshap = CSHAPInformed(t, background_samples, softmax=softmax)
    shap_values, base_values = cshap.shap_values(t, X_test, model, device, exact)

    # Store results
    pickle_array(shap_values, output_path + '/shap_values' + str(i) + '_temp.pkl')
    pickle_array(base_values, output_path + '/base_values' + str(i) + '_temp.pkl')