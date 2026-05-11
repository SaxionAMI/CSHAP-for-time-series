from .cshap_utils import encode_data, TempDataset

import numpy as np

import torch
from torch.utils.data import DataLoader

import shap

class CSHAPInformed():
    """
    The CSHAP for concept-informed models using decomposition for concept construction. 
    """
    def __init__(self, t, background_data, softmax=True):
        """
        Calculates the SHAP values for the given decomposition.
        t: time mapping, used for decomposing
        background_data: the data used for concept masking in SHAP
        softmax: whether to apply softmax to model output before applying SHAP (generally True for classification)
        """
        self.n_components = background_data.shape[-1]  
        self.len_ts = background_data.shape[1]
        self.num_background = background_data.shape[0]      
        self.softmax = softmax
        self.background_data = background_data

        # Encode background data into indices array of shape (# samples, # components) for SHAP purposes
        self.bg_coded = encode_data(self.background_data)

    def shap_wrapper(self, x_in, model, device, batch_size=2048):
        x_coded = x_in.copy()       # For safety reasons copy data
        x_coded = x_coded.astype(int)   # Convert to integers

        masked = np.zeros((x_coded.shape[0], self.len_ts, x_coded.shape[1]))        # For storing masked data
        for i_x in range(x_coded.shape[0]):                                         # Iterate through samples
            for i_c in range(x_coded.shape[-1]):
                # Select replacement feature from data based on index in coded data
                masked[i_x, :, i_c] = self.all_data[x_coded[i_x, i_c], :, i_c] 

        # Put the masked vector through the model
        with torch.no_grad():
            input_tensor = torch.tensor(masked, dtype=torch.float32).to(device)

        # Construct dataset and dataloader to supply samples to model
        dataset = TempDataset(input_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        # Put samples through model and store the outputs as list of numpy arrays
        model.eval()
        output_list = []
        for _, data in enumerate(dataloader):
            with torch.no_grad():
                data = data.to(device)
                if self.softmax:
                    out = model(data, softmax=True).cpu().numpy()
                else:
                    out = model(data).cpu().numpy()
                output_list.append(out)

        # Concatenate list of numpy arrays and we obtain the masked data in the format needed for SHAP
        output = np.concatenate(output_list, axis=0) 
        return output
    
    def shap_values(self, t, x_samples, model, device, exact=True):
        """
        Function for calculating the SHAP values using C-SHAP.
        t: time mapping, used for decomposing
        x_samples: samples for which to calculate SHAP values in shape (# samples, # time series length, 1)
        model: PyTorch model for which to determine SHAP values
        device: device on which the model is located
        exact: whether to use exact SHAP or KernelSHAP
        returns: SHAP values for C-SHAP
        data in shape # (# samples, # time series length, # components)
        """

        # Create an array containing both the background data and the input data for which to calculate SHAP
        self.all_data = np.concatenate([self.background_data, x_samples], axis=0)

        # Encode the input data, start encoding from index after background data, so they can be distinguished
        x_coded = encode_data(x_samples, start_idx=self.num_background)

        # Use Exact SHAP or KernelEXPLAINER
        if exact:
            explainer = shap.explainers.Exact(lambda x: self.shap_wrapper(x, model, device), self.bg_coded)
            explanation = explainer(x_coded)
            complete_shap_values = explanation.values
            base_values = explanation.base_values
        else:
            explainer = shap.KernelExplainer(lambda x: self.shap_wrapper(x, model, device), self.bg_coded)
            complete_shap_values = explainer.shap_values(x_coded)
            base_values = explainer.expected_value

        # Extract output dimension of predictive model
        if len(complete_shap_values.shape) > 2:
            output_dim = complete_shap_values.shape[-1]
        else:
            output_dim = 1

        # Reshape the SHAP values into the proper output shape
        shap_values = complete_shap_values.reshape((x_coded.shape[0], self.n_components, output_dim))
        return shap_values, base_values


