import numpy as np

import shap

import torch
from torch.utils.data import DataLoader

from .cshap_utils import encode_data, TempDataset


class CSHAP():
    """
    The CSHAP for concept-agnostic models using decomposition for concept construction. 
    """
    def __init__(self, t, background_data, decomposition, softmax=True):
        """
        Calculates the SHAP values for the given decomposition.
        t: time mapping, used for decomposing
        background_data: the data used for concept masking in SHAP
        softmax: whether to apply softmax to model output before applying SHAP (generally True for classification)
        """
        self.decomposition = decomposition     
        self.n_components = len(decomposition.component_names)   
        self.num_background = background_data.shape[0]      
        self.softmax = softmax

        # Decompose background data and put it in the correct shape
        self.bg_components = decomposition.decompose(t, background_data)

        # Encode background data into indices array of shape (# samples, # components) for SHAP purposes
        self.bg_coded = encode_data(self.bg_components, start_idx=0)

    def shap_wrapper(self, x_in, model, device, batch_size=2048):
        """
        Wrapper for the model predictions to give to SHAP. Takes encoded input data, to which SHAP has
        applied masking. The encoded input data contains per sample indices for each component. After SHAP's
        masking the index will either refer to the sample itself (features is not masked) or
        to the background sample from which to extract the feature (feature is masked).
        x_in: encoded data of shape (# masked samples to process, # components)
        model: PyTorch model to which to apply C-SHAP
        device: device the data should be located to (same as model)
        batch_size: to speed up parallel computing, the masked data is provided to the model in batches
        returns: the masked model input required by SHAP
        """

        x_coded = x_in.copy()       # For safety reasons copy data
        x_coded = x_coded.astype(int)   # Convert to integers

        # Determine number of components
        n_components = self.all_data.shape[2]

        # Used for storing the data 'reconstructed' from the masked encoded data
        x_c_reconstructed = np.zeros((x_coded.shape[0], self.all_data.shape[1], n_components))

        # Reshaping the input samples such that components can be index easily
        x_coded = x_coded.reshape((x_coded.shape[0], n_components))

        # Extract data using the indices in the encoded data. Each index refers either to the original
        # sample or the background sample
        for c in range(n_components):
            background_indices = x_coded[:, c]
            x_c_reconstructed[:, :, c] = self.all_data[background_indices, :, c]

        # Compose the reconstructed components into input samples for the model
        x_input = self.decomposition.compose(x_c_reconstructed)
        x_input = x_input[:, :, None]

        # Create PyTorch tensor out of input samples
        input_tensor = torch.tensor(x_input, dtype=torch.float32, device=device)

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
        returns: SHAP values for C-SHAP in shape (# samples, # time series length, # components)
        """

        # Initialize array where to store the components extracted from the input data
        x_components = np.zeros((x_samples.shape[0], x_samples.shape[1], self.n_components))

        # Iterate through the samples and decompose them one by one
        x_components = self.decomposition.decompose(t, x_samples)

        # Create an array containing both the background data and the input data for which to calculate SHAP
        self.all_data = np.concatenate([self.bg_components, x_components], axis=0)

        # Encode the input data, start encoding from index after background data, so they can be distinguished
        x_coded = encode_data(x_components, start_idx=self.num_background)

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
