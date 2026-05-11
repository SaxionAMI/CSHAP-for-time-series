# Import packages
import numpy as np

from torch.utils.data import Dataset

# Helper functions used for C-SHAP

class TempDataset(Dataset):
    """
    Dataset object to process perturbed instances for SHAP in batches.
    This speeds up the inference process.
    """
    def __init__(self, x):
        self.x = x              # Store data

    def __len__(self):
        return len(self.x)      # Access data length
    
    def __getitem__(self, idx):
        return self.x[idx]      # Return sample at index idx    


def encode_data(x, start_idx=0):
    """
    Encodes data for wrapping purposes, inspired by WindowSHAP.
    We map each component to an index.
    These indices can be used for masking purposes when they get shuffled by SHAP.
    x: The input data in shape 
    start_idx: Index where to start the indexation from. In the end the background data and
    the input data should receive distinct indices, so they can be selected.
    returns: Encoded data in an array of (# samples, # components)
    """
    n_components = x.shape[2] # Extract number of components

    # For each sample create one element in the array that refers to each component.
    # In the unshuffled encoded data (prior to SHAP)
    # each element should refer to the current index of the sample. When they indices get 
    # shuffled, they can be used to retrieve features from other samples.
    x_ = [[i]*n_components for i in range(start_idx, start_idx + x.shape[0])]
    return np.array(x_)