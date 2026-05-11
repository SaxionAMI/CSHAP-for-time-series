This module contains all scripts necessary for C-SHAP and used in the experiments in our article.
```
scripts
│
└───  cshap - Folder containing C-SHAP scripts
    │    
    └───    decomposition: contains classes for time series decomposition
            |
            └───    decomposition.py:  contains abstract class Decomposition for the decomposition classes
            └───    custom_decomposition.py: contains class CustomDecomposition for the custom decomposition method
            └───    DWT_decomposition.py: contains class DWTDecomposition for the Discrete Wavelet Transform decomposition method
            └───    EMD_decomposition.py: contains class EMDDecomposition for the Empirical Mode Decomposition method
    │
    └───    cshap.py: contains CSHAP class to apply C-SHAP to concept-agnostic models
    └───    cshap_informed.py: contains CSHAPInformed class to apply C-SHAP to concept-informed models
    └───    cshap_utils.py: contains helper methods for the previously described classes
    └───    plotting.py: contains plotting functions for the C-SHAP results
│
└───  evaluation: contains scripts for feasibility evaluation
    |
    └───    single_deletion.py: script to execute for reproducing single deletion experiments
    └───    single_insertion.py: script to execute for reproducing single insertion experiments
    └───    evaluation_utils.py: contains helper functions for the evaluation
    └───    data_utils.py: contains helper functions for loading and storing data
│
└───  models: contains scripts for dealing with predictive models
    |
    └───    model.py: contains MyGRU class used for the predictive model in our experiments
    └───    utils.py: contains helper functions for loading, training and evaluating models
│
└───  model_training: contains scripts for data preparation and model training (the results of these are available in the data/prepared and models folders)
    |
    └───    data_processing.py: functions used for data preparation
    └───    train_model_nruns.py: script used for training the different predictive models
    └───    utils.py: contains helper functions for data preparation and model training
│
└───  run_cshap_agnostic.py: script used to calculate C-SHAP values for concept-agnostic models
└───  run_cshap_informed.py: script used to calculate C-SHAP values for concept-informed models
└───  run_kernelshap.py: script used to calculate SHAP values using KernelSHAP
└───  run_windowshap.py: script used to calculate C-SHAP values for WindowSHAP
└───  windowshap.py: script from: https://github.com/vsubbian/WindowSHAP for running WindowSHAP, adjusted to match formatting
└───  utils.py: contains helper functions for the C-SHAP calculations
```
