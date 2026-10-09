# Import packages
import sys
import torch 
import os
import random

import numpy as np

from ..cshap import CSHAP, CustomDecomposition
from ..utils import load_model, load_array, pickle_array

# Script used for hyperparameter sensitivity analysis of C-SHAP with custom decomposition

# Settings
datasets = ['opportunity_uni', 'fordA', 'turbofan']         # Dataset to calculate SHAP values for
exact = True                # If True calculate exact SHAP values, else use KernelSHAP
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu") # Device to load model and data to

noise_sigma_start = 1           # DEFAULT sigma value

for dataset in datasets:        # Iterate through datasets

    # Input paths
    data_path = './data/prepared/' + dataset
    model_path = './models/' + dataset + '/uni_channel'

    # Select how many models you want to calculate SHAP values for.
    n_runs = 5

    # Update settings based on dataset
    if dataset == 'fordA':
        class_dict = {0: 'Normal', 1: 'Anomalous'}      # Mapping from target values to class names
        bidirectional = True                            # Whether to use bidirectional GRU layers
        task = 'classification'                         # Regression or classification
        softmax = True                                  # Whether to apply softmax to model output
        var_window_size_start = 50                      # DEFAULT variance window size
        decomposition_def = 'scale*(low_freq+var*high_freq)'

    elif dataset == 'opportunity_uni':
        class_dict = {0:'Stand', 1:'Walk', 2:'Sit', 3:'Lie'}
        bidirectional = True
        task = 'classification'
        softmax = True
        var_window_size_start = 27
        decomposition_def = 'trend+bias+scale*(low_freq+var*high_freq)'

    elif dataset == 'turbofan':
        class_dict = None
        bidirectional = False
        task = 'regression'
        softmax = False
        var_window_size_start = 30
        decomposition_def = 'trend+bias+scale*(low_freq+var*high_freq)'

    # Path to store calculated SHAP values to
    output_path = './data/results/sensitivity_analysis/' + dataset + '/uni_channel/' + decomposition_def
    output_path = output_path.replace('*', 'x')

    # If output path does not yet exist, create it
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    print('output path:', output_path)

    # Values to analyze
    var_window_sizes = [var_window_size_start - 10, var_window_size_start - 5, var_window_size_start - 2, var_window_size_start + 2, var_window_size_start + 5, var_window_size_start + 10]
    noise_sigmas = [0.25, 0.5, 0.75, 0.9, 1.1, 1.25, 1.5, 1.75]

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


    # Iterate through sigma values
    for noise_sigma in noise_sigmas:
        # Set seeds
        np.random.seed(10)
        torch.manual_seed(10)
        random.seed(10)

        # Set default window size
        var_window_size = var_window_size_start

        # Initialize decomposition using current parameters
        decomposition = CustomDecomposition(decomposition_def, noise_sigma=noise_sigma, 
                                            var_window_size=var_window_size)

        # Iterate through models, calculate C-SHAP values for each one and store results
        for i in range(n_runs):
            # Load model
            model = load_model(model_path, task, bidirectional, X_train.shape[-1], class_dict, device, model_name='model_' + str(i) + '.pt')

            # Calculate C-SHAP values
            cshap = CSHAP(t, background_samples, decomposition, softmax=softmax)
            shap_values, base_values = cshap.shap_values(t, X_test, model, device, exact)

            # Store results
            pickle_array(shap_values, output_path + '/v_w_size_' + str(var_window_size) + '_noise_' + str(noise_sigma) + '_shap_values_' + str(i) + '.pkl')
            pickle_array(base_values, output_path + '/v_w_size_' + str(var_window_size) + '_noise_' + str(noise_sigma) + '_base_values_' + str(i) + '.pkl')


    # Iterate through window sizes
    for var_window_size in var_window_sizes:
        # Set seeds
        np.random.seed(10)
        torch.manual_seed(10)
        random.seed(10)

        # Set default sigma
        noise_sigma = 1

        # Initialize decomposition using current parameters
        decomposition = CustomDecomposition(decomposition_def, noise_sigma=noise_sigma, 
                                            var_window_size=var_window_size)

        # Iterate through models, calculate C-SHAP values for each one and store results
        for i in range(n_runs):
            # Load model
            model = load_model(model_path, task, bidirectional, X_train.shape[-1], class_dict, device, model_name='model_' + str(i) + '.pt')

            # Calculate C-SHAP values
            cshap = CSHAP(t, background_samples, decomposition, softmax=softmax)
            shap_values, base_values = cshap.shap_values(t, X_test, model, device, exact)

            # Store results
            pickle_array(shap_values, output_path + '/v_w_size_' + str(var_window_size) + '_noise_' + str(noise_sigma) + '_shap_values_' + str(i) + '.pkl')
            pickle_array(base_values, output_path + '/v_w_size_' + str(var_window_size) + '_noise_' + str(noise_sigma) + '_base_values_' + str(i) + '.pkl')
