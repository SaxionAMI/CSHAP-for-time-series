# Import packages
import sys
import torch 
import os
import random
import shap

import numpy as np

from .utils import load_model, load_array, pickle_array, WindowshapWrapper
from .windowshap import SlidingWindowSHAP, StationaryWindowSHAP

# Script used to obtain concept-agnostic C-SHAP values

# Set seeds
np.random.seed(10)
torch.manual_seed(10)
random.seed(10)

# Settings
dataset = 'opportunity_uni'         # Data set to calculate SHAP values for
exact = True                # If True calculate exact SHAP values, else use KernelSHAP
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu") # Device to load model and data to

# Select decomposition to use
max_imfs = 4            # Number of IMFs to include if EMD is used
noise_sigma = 1         # Used for the low_freq component if custom decomposition is used

# Input paths
data_path = './data/prepared/' + dataset
model_path = './models/' + dataset + '/uni_channel'

# Path to store calculated SHAP values to
output_path = './data/windowshap_stationary/' + dataset + '/uni_channel/'

# Select how many models you want to calculate SHAP values for.
n_runs = 1

# Update settings based on dataset
if dataset == 'fordA':
    class_dict = {0: 'Normal', 1: 'Anomalous'}      # Mapping from target values to class names
    bidirectional = True                            # Whether to use bidirectional GRU layers
    task = 'classification'                         # Regression or classification
    softmax = True                                  # Whether to apply softmax to model output
    n_classes = len(class_dict)
elif dataset == 'opportunity_uni':
    class_dict = {0:'Stand', 1:'Walk', 2:'Sit', 3:'Lie'}
    bidirectional = True
    task = 'classification'
    softmax = True
    n_classes = len(class_dict)
elif dataset == 'turbofan':
    class_dict = None
    bidirectional = False
    task = 'regression'
    softmax = False
    n_classes = 1

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

# Iterate through models, calculate C-SHAP values for each one and store results
for i in range(n_runs):
    # Load model
    model = load_model(model_path, task, bidirectional, X_train.shape[-1], class_dict, device, model_name='model_' + str(i) + '.pt')
    wrapper = WindowshapWrapper(model, device)

    # Calculate C-SHAP values
    # explainer = SlidingWindowSHAP(wrapper, stride=8, window_len=15, 
    #                              B_ts=background_samples, test_ts=X_test)
    explainer = StationaryWindowSHAP(wrapper, window_len=15, 
                                B_ts=background_samples, test_ts=X_test)
    shap_values = explainer.shap_values(num_output=n_classes)

    # Store results
    pickle_array(shap_values, output_path + '/shap_values' + str(i) + '_temp.pkl')
    # pickle_array(base_values, output_path + '/base_values' + str(i) + '_temp.pkl')