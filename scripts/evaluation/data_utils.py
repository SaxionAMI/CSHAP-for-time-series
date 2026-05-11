# Import packages
import numpy as np

import torch
from torch.utils.data import Dataset, WeightedRandomSampler

from collections import Counter

# Scripts used to deal with data aspects in various scripts and notebooks

class MyDataset(Dataset):
    """
    Basic Pytorch.Dataset object for supervised datasets
    """
    def __init__(self, x, y):
        """
        x: measurements
        y: targets
        """
        self.x = x
        self.y = y

    def __len__(self):
        """
        return: length of dataset
        """
        return len(self.y)
    
    def __getitem__(self, idx):
        """
        idx: index of element to return
        returns: measurement at index and matching target at index
        """
        return self.x[idx], self.y[idx]

def normalize_data(X, mean=None, std=None):
    """
    Apply normalization across samples in given data, by substracting the mean and dividing by
    the standard deviation. The mean and standard deviation should either be specified or
    are calculated from the given data.
    X: data to normalize
    mean: mean to apply for normalization
    std: standard deviation to apply for normalization
    returns: normalized data
    """
    if mean is None:
        mean = np.mean(X, axis=(0, 1))
        std = np.std(X, axis=(0, 1))
    X = (X - mean) / std
    return X, mean, std

def numpy_to_dataset(X, y):
    """
    Create a MyDataset object using the given measurements and targets.
    X: measurements of shape (# samples, # sample length, 1) or (# samples, # sample length)
    y: targets of shape(# samples)
    dataset: MyDataset object in which X (as shape (# samples, # sample length, 1)) and y are stored
    """
    tensor = torch.from_numpy(X)    # Cast to tensor
    if len(X.shape) == 2:           # If necessary put into shape (# samples, # sample length, 1)
        tensor = tensor.unsqueeze(-1).to(torch.float32)
    else:
        tensor = tensor.to(torch.float32)
    dataset = MyDataset(tensor, y)  # Instantiate dataset
    return dataset

def get_weighted_sampler(y):
    """
    Create a torch.utils.data.WeightedRandomSampler based on the occurence of different classes in y
    using oversampling.
    y: targets in datasets
    returns: torch.utils.data.WeightedRandomSampler using oversampling to balance unbalanced data
    """
    class_counts = Counter(y)           # Count the number of occurrences of each class in y
    n_samples = len(y)                  # Determine the total number of samples

    # Determine weights to apply (inversly proportional to the number of occurrences)
    class_weights = [n_samples/class_counts[i] for i in range(len(class_counts))]       
    weights = [class_weights[y[i]] for i in range(n_samples)]

    # Instantiate sampler
    sampler = WeightedRandomSampler(torch.DoubleTensor(weights), n_samples, replacement=True)
    return sampler