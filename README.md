# C-SHAP for time series: An approach to high-level temporal explanations

C-SHAP offers concept-based explanations for black box models trained on time series data. This code accompanies our article "C-SHAP for time series: An approach to high-level temporal explanations" [1]. C-SHAP offers both a fully post-hoc and a model-informed approach. The current code only supports PyTorch.

## C-SHAP explained

<img width="800" alt="Overview of concept-informed and concept-agnostic C-SHAP approaches" src="https://github.com/user-attachments/assets/b82d1205-8e8d-4621-824a-e6073f7f637d" />

C-SHAP uses SHAP [2] to determine the influence of concepts on a model outcome. We provide two implementations of C-SHAP, one fully post-hoc approach (concept-agnostic C-SHAP) and one partially ante-hoc approach (concept-informed C-SHAP). For the concept-agnostic approach, a predictive model has been trained using the raw model input. For the concept-informed approach, the model is trained on the concepts, passed to the model as channels. 

To calculate the SHAP values, concepts are masked by replacing them by background data sampled from the training data. For the concept-agnostic approach, the masked signals are aggregated to obtain a masked model input. For the concept-informed approach, the masked concepts can directly be passed ot the predictive model. The SHAP values are calculated using ExactSHAP from the [SHAP](https://https://github.com/shap/shap) package.

For the construction of concepts, time series decomposition is used. Currently three decomposition types are supported: Emperical Mode Decomposition, Discrete Wavelet Transform decomposition and a custom decomposition. The latter decomposition allows custom selection of the following components: Trend, Bias, Scale, Variance, High frequency, Low frequency. The decomposition can be defined using multiplicative and additive relations between the components.

## Examples
C-SHAP can generate both local and global explanations. See the examples below.

<img width="300" alt="Example of a local explanation for a sample from the Turbofan dataset" src="https://github.com/user-attachments/assets/f7e17fff-b1c3-4494-ad99-2b42c39c980d" />
<p><b>Figure 1:</b> Local explanation for a sample from the Turbofan dataset where the number of remaining useful life cycles is predicted. The custom decomposition: Trend+Bias+Scale*(Low_freq+Var*High_freq) is used. The explanation shows how the Trend informs the model of a low number of remaining useful cycles.</p>

<img width="300" alt="Example of a global explanation for the OPPORTUNITY dataset" src="https://github.com/user-attachments/assets/a9928639-7368-471b-91f8-a5fe7b0d3b57" />
<p><b>Figure 2:</b> Global explanation for the OPPORTUNITY dataset. The custom decomposition: `trend+bias+scale*(low_freq+var*high_freq)` was used. The explanation shows a high contribution of the concept Bias (in this case the Bias represents the angle of the accelerometer).</p>

## Running C-SHAP
The experiments have been tested in Python 3.8.11, the required Python libraries can be found in `requirements.txt`.

We defined two classes for the two C-SHAP approaches.
* `scripts.cshap.cshap.CSHAP()`: defines the concept-agnostic C-SHAP approach.
* `scripts.cshap.cshap.CSHAPInformed()`: defines the concept-informed C-SHAP approach.

Below an example of how to calculate C-SHAP values. CSHAPInformed works the same way.
```
    from .cshap import CSHAP
    cshap = CSHAP(t, background_samples, decomposition, softmax=softmax)
    shap_values, base_values = cshap.shap_values(t, X, model, device, exact)
```
C-SHAP currently supports three types of decompositions: Discrete Wavelet Transform (DWT), Empirical Mode Decomposition (EMD) and our custom decomposition (see our article). These can be imported from the cshap module. Below examples of how to decompose data using our defined decompositions:
```
  from .cshap import CustomDecomposition
  decomposition_def = 'trend+bias+scale*(low_freq+var*high_freq)'
  decomposition = CustomDecomposition(decomposition_def, noise_sigma=noise_sigma, var_window_size=var_window_size)
  components = self.decomposition.decompose(t, X)
```
```
  from .cshap import EMDDecomposition
  decomposition = EMDDecomposition(max_imfs=5)
  components = self.decomposition.decompose(t, X)
```
```
  from .cshap import DWTDecomposition
  decomposition = DWTDecomposition(level=3)
  components = self.decomposition.decompose(t, X)
```

Further example code can be found in `./scripts/run_cshap_informed.py` and `./scripts/run_cshap_agnostic.py`.

## Reproducing experiments
The C-SHAP values on which the results in our article are based can be calculated as follows, for the concept-agnostic model:
```
  python -m scripts.run_cshap_agnostic
```
For the concept-informed model:
```
  python -m scripts.run_cshap_agnostic
```
Switch the dataset parameter and decomposition definition to obtain the results for different datasets. The scripts and notebooks described below require the calculation of the C-SHAP values beforehand.

The anecdotal evidence in terms of the calculated C-SHAP can be visualized using `./notebooks/anecdotal_evidence.ipynb`. The extended analysis for the OPPORTUNITY dataset can be reproduced using `./notebooks/opportunity_extended_analysis.ipynb`.

The results for the feasibility analysis can be reproduced using `./scripts/evaluation/single_deletion.py` and `./scripts/evaluation/single_insertion.py`. The visualization of the results can be made using `./notebooks/feasbility_analysis.ipynb`.

The stability analysis can be reproduced using `./notebooks/stability_analysis.ipynb`.

The comparison between concept-agnostic and concept-specific models from our supplementary materials can be reproduced using `./notebooks/model_comparison.ipynb`.

The hyperparameter sensitivity analysis from our supplementary materials can be reproduced using `./scripts/evaluation/sensitivity_analysis.py` and visualized using `./notebooks/sensitivity_analysis.ipynb`.


## Data used
The scripts `./scripts/run_cshap_agnostic.py` and `./scripts/run_cshap_informed.py` require the prepared data given in ./data/prepared. The data was prepared from the datasets listed below (for the details regarding data preparation we refer to our article):
* [OPPORTUNITY](https://doi.org/10.24432/C5M027) [3]: a dataset for Human Activity Recognition from sensor data. We used the accelerometer data from a single sensor for locomotion classification.
* [FordA](https://www.timeseriesclassification.com/description.php?Dataset=FordA) [4]: a simulated dataset for the diagnosis of anomalies in an automotive subsystem from engine noise.
* [Turbofan](https://data.phmsociety.org/nasa/) [5]: a dataset of engine degradation of a simulated aero-propulsion system.

The C-SHAP values we calculated for the article can be found in `./data/shap`. The results of the feasibility experiments can be found in `./data/results`.

The models trained for the experiments in the article can be found in `./models`. The models are objects of the class `MyGRU` (`./scripts/models/model.py`) The folder contains four types of models for each dataset. One model trained directly on the (prepared) data and three models trained on different decompositions. Models were trained on the Discrete Wavelet Transform (DWT), Empirical Mode Decomposition (EMD) and the custom decomposition. For the custom decomposition, the decomposition used is denoted as the folder name.

## References
[1] Jutte, A., Ahmed, F., Linssen, J., & van Keulen, M. (2025). C-SHAP for time series: An approach to high-level temporal explanations. arXiv preprint arXiv:2504.11159. doi: https://doi.org/10.48550/arXiv.2504.11159

[2] Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model predictions. Advances in neural information processing systems, 30.

[3]  Roggen, D., Calatroni, A., Rossi, M., Holleczek, T., Förster, K., Tröster, G., ... & Millàn, J. D. R. (2010, June). Collecting complex activity datasets in highly rich networked sensor environments. In 2010 Seventh international conference on networked sensing systems (INSS) (pp. 233-240). IEEE. doi: https://doi.org/10.1109/INSS.2010.5573462.

[4]  Dau, H. A., Bagnall, A., Kamgar, K., Yeh, C. C. M., Zhu, Y., Gharghabi, S., ... & Keogh, E. (2019). The UCR time series archive. IEEE/CAA Journal of Automatica Sinica, 6(6), 1293-1305. doi: https://doi.org/10.1109/JAS.2019.1911747.

[5] Saxena, A., Goebel, K., Simon, D., & Eklund, N. (2008, October). Damage propagation modeling for aircraft engine run-to-failure simulation. In 2008 international conference on prognostics and health management (pp. 1-9). IEEE. doi: https://doi.org/10.1109/PHM.2008.4711414.

[5] Nayebi, A., Tipirneni, S., Reddy, C. K., Foreman, B., & Subbian, V. (2023). WindowSHAP: An efficient framework for explaining time-series classifiers based on Shapley values. Journal of biomedical informatics, 144, 104438. doi: https://doi.org/10.1016/j.jbi.2023.104438.
