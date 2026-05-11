# Import packages
import torch.nn as nn

# Class for GRU model used in the various experiments

class MyGRU(nn.Module):
    """
    Class used as predictive model for the C-SHAP experiments.
    """
    def __init__(self, input_dim, bidirectional, hidden_dim, num_layers, output_dim, dropout=0):
        """
        Build predictive model including a number of GRU layers and a single Linear layer.
        input_dim: number of features in input data
        bidirectional: whether GRU layers should be bidirectional as boolean
        hidden_dim: number of features in the GRU hidden state
        num_layers: number of layers in the GRU block (note this excludes the last single Linear layer)
        output_dim: number of output neurons
        dropout: dropout-rate to use for GRU layers, default is 0.
        """
        super(MyGRU, self).__init__()
        # Initialize GRU block using the specified arguments
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=bidirectional, 
                          dropout=dropout)
        # Initialize a single Linear layer mapping GRU to output
        self.fc = nn.Linear(hidden_dim, output_dim)
        # Store the model's output dimensionality
        self.output_dim = output_dim
        
    def forward(self, x, softmax=False):
        """
        Pass data through model.
        x: data to pass through model as torch.tensor in shape (# samples, # time steps, # input features)
        softmax: whether to apply softmax at the end
        returns: model output as torch.tensor in shape (# samples) if output_dim is 1 or (# samples, output_dim)
        """
        _, hn = self.gru(x)         # Pass through GRU block
        hn = hn[-1]                 # Extract hidden state
        x_out = self.fc(hn)         # Pass through linear layer

        if softmax:                 # Apply softmax if indicated
            return nn.Softmax(dim=1)(x_out)
        elif self.output_dim == 1:  # Put into proper shape
            return x_out[:, 0]
        else:
            return x_out