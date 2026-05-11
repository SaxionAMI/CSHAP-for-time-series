# Import packages
import sys
sys.path.append("./../") 

import numpy as np
import torch
from torch.utils.data import DataLoader

from cshap import CustomDecomposition

from utils import load_array, pickle_array, load_param_dict
from models.utils import train
from data_utils import normalize_data, numpy_to_dataset, get_weighted_sampler
from evaluation_utils import delete_concept

# Script for the single deletion evaluation experiments where we repeatedly
# train models on the data where a single concept (component) has been removed
# we only did this evaluation for the concept-agnostic models

# Initialisation settings
dataset = 'opportunity_uni'             # Dataset to test on
n_reps = 1                              # Number of times to repeat each model training configuration for variance
decomposition_def = 'trend+bias+scale*(low_freq+var*high_freq)' # Decomposition to use for component selection
noise_sigma = 1                           # Noise parameters to use for low_freq component in custom decomposition
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu") # Device to train on
batch_size = 32                                                         # Batch size for training

data_path = './../../data/prepared/' + dataset  # Path to prepared data
model_path = './../../models/' + dataset + '/uni_channel' # Path to model, only used to extract parameters from params.txt

# Update settings based on dataset
if dataset == 'fordA':
    class_dict = {0: 'Normal', 1: 'Anomalous'}      # Mapping from target values to class names
    bidirectional = True                            # Whether to use bidirectional GRU layers
    task = 'classification'                         # Regression or classification
    var_window_size = 50                            # Used for the variance component in custom decomposition
elif dataset == 'opportunity_uni':
    class_dict = {0:'Stand', 1:'Walk', 2:'Sit', 3:'Lie'}
    bidirectional = True
    task = 'classification'
    var_window_size = 27
elif dataset == 'turbofan':
    class_dict = None
    bidirectional = False
    task = 'regression'
    var_window_size = 30

# Load hyperparameters from given model path
params = load_param_dict(model_path)

# Concept-agnostic -> univariate data, so only one feature as model input
input_dim = 1

# Determine output dimension, which is the number of output neurons
if task == 'regression':
    output_dim = 1
else:
    output_dim = len(class_dict)

# Load prepared data
X_train = load_array(data_path + '/X_train.pkl')
X_test = load_array(data_path +  '/X_test.pkl')
y_train = load_array(data_path +  '/y_train.pkl')
y_test = load_array(data_path +  '/y_test.pkl')

# Placeholder time dimension
t = np.linspace(0, X_train.shape[1] + 1, X_train.shape[1])

# If necessary put data into desired shape (last dimension has size 1)
if len(X_train.shape) == 2:
    X_train = X_train[:, :, None]
    X_test = X_test[:, :, None]

# Instantiate decomposition object
decomposition = CustomDecomposition(decomposition_def, noise_sigma=noise_sigma, var_window_size=var_window_size)

# Determine number of concepts
n_concepts = len(decomposition.component_names)

# Instantiate dictionary to store results   
performance_dict = {i: [None for _ in range(n_reps)] for i in range(n_concepts + 1)}

# Run experiments
for r in range(n_reps):                 # Experiments are repeated n_reps times
    for i in range(n_concepts + 1):     # Select one concept to delete (or none for the last iteration)
        # Remove selected concept from the data
        X_train_filtered = delete_concept(t, X_train, decomposition, i)
        X_test_filtered = delete_concept(t, X_test, decomposition, i)

        # Normalize resulting data
        X_train_filtered, mean_train, std_train = normalize_data(X_train_filtered)
        X_test_filtered, _, _ = normalize_data(X_test_filtered, mean_train, std_train)

        # Put data into dataloaders to train with
        train_dataset = numpy_to_dataset(X_train_filtered, y_train)
        if task == 'classification':
            sampler = get_weighted_sampler(y_train)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # Put data into dataloaders to test with
        test_dataset = numpy_to_dataset(X_test_filtered, y_test)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

        # Train model and get resulting test performance (RMSE for regression, weighted accuracy in percentage for classification)
        test_performance, model = train(task, train_loader, test_loader, params, bidirectional, input_dim, output_dim, class_dict, device)
        performance_dict[i][r] = test_performance   # Keep track of results

# Store dictionary containing all obtained results
pickle_array(performance_dict, './../../data/_results/' + dataset + '/single_deletion_temp.pkl')