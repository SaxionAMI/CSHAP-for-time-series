# Import packages
import os
import pickle

import numpy as np

import torch
from torch.utils.data import DataLoader, Dataset

from .models import MyGRU

# Helper functions used in various scripts and notebooks

def load_param_dict(model_path):
    """
    Load parameters of the MyGRU model (see model.py) stored at model_path/params.txt.
    params.txt should contain the parameters: lr (learning rate), n_epochs, hidden_dim, num_layers and dropout (rate).
    Parameter settings should be formated as 'param_name param_value\n'.
    model_path: path to folder containing params.txt file (not including params.txt)
    returns: parameter dictionary as {param_name: param_value}
    """
    param_dict = {}     # For storing parameter dictionary
    with open(os.path.join(model_path, 'params.txt'), 'r') as f:    # Open file
        lines = f.readlines()                                       # Read all lines
        for line in lines:                                          # Parse line by line
            param, value = line.split(' ')                          # Split parameter name and value
            value = value.replace('\n', '')                         # Remove new line symbol
            if '.' in value:                                        # Cast parameter value to float or integer
                value = float(value)
            else:
                value = int(value)
            param_dict[param] = value                               # Store parameter value
    return param_dict

def load_model(model_path, task, bidirectional, input_dim, class_dict, device, model_name=None):
    """
    Load MyGRU model (see model.py) from model path for model weights stored using: torch.save(model.state_dict(), PATH).
    model_path: folder file containing model weights is located at
    task: classification or regression
    bidirectional: whether GRU layers should be bidirectional
    input_dim: number of features model should accept (1 for univariate)
    class_dict: if task is classification, the class dictionary, length is assumed to be the number of output classes
    device: device to cast model to (cpu, cuda, ...)
    model_name: file name of model on model_path, if None file name is assumed to be model.pt
    """
    # Load parameters for MyGRU model
    param_dict = load_param_dict(model_path)

    # Extract parameters
    hidden_dim = param_dict['hidden_dim']
    num_layers = param_dict['num_layers']
    dropout = param_dict['dropout']

    # Determine number of output neurons of model
    if task == 'classification':
        output_dim = len(class_dict)
    else:
        output_dim = 1

    # Instantiate model
    model = MyGRU(input_dim, bidirectional, hidden_dim, num_layers, output_dim, dropout=dropout)

    # Load model weights from specified file
    if model_name is None:
        model.load_state_dict(torch.load(os.path.join(model_path, 'model.pt'), weights_only=True))
    else:
        model.load_state_dict(torch.load(os.path.join(model_path, model_name), weights_only=True))

    # Cast model to specified device
    model.to(device)
    return model

def pickle_array(array, file):
    """
    Helper function to pickle a numpy.array.
    array: any numpy.array
    file: file path to store pickled file
    """
    with open(file, 'wb') as file: 
        pickle.dump(array, file) 

def load_array(file_name):
    """
    Helper function to unpickle a numpy.array.
    file_name: path to pickled file
    returns: unpickled numpy.array
    """
    with open(file_name, 'rb') as f:
        x = pickle.load(f)
    return x


# Helper functions KernelSHAP
class UnlabelledDataset(Dataset):
    def __init__(self, x):
        self.x = x

    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return self.x[idx]


def shap_wrapper(np_array, model, device, batch_size=1024, softmax=True):
    if len(np_array.shape) == 2:
        np_array = np_array[:, :, None]
    tensor = torch.tensor(np_array, dtype=torch.float32, device=device)

    dataset = UnlabelledDataset(tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model.eval()
    output_list = []
    for i, data in enumerate(dataloader):
        with torch.no_grad():
            data = data.to(device)
            out = model(data)
            if softmax:
                out = torch.nn.Softmax(dim=-1)(out)
            out = out.cpu().numpy()
            output_list.append(out)
    output = np.concatenate(output_list, axis=0)
    return output

# Helper functions windowshap
class WindowshapWrapper:
    def __init__(self, model, device):
        self.model = model
        self.device = device

    def predict(self, x):
        output = shap_wrapper(x, self.model, self.device, batch_size=1024, softmax=True)
        if len(output.shape) == 1:
            output = output[:, None]
        return output        