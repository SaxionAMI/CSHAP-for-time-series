import numpy as np
import pywt
import re

from .decomposition import Decomposition

class CustomDecomposition(Decomposition):
    """
    Custom decomposition used for article, specialized for sensor data
    """

    def __init__(self, input_string, noise_sigma=1, var_window_size=30):
        """
        Initialise CustomDecomposition
        input_string: string determining which components to include, should be formatted as comp1(.)comp2(.)...(.)compn,
        brackets may additionally be included to deal with ordering. compi is one of: bias, trend, var (=variance), 
        low_freq (=low frequency), scale, high_freq (=high frequency), and (.) is * or +. 
        high_freq is always assumed to be the last component in the definition by the code. Do not supply duplicate
        components.
        An example input_string is trend+bias+scale*(low_freq+var*high_freq).
        noise_sigma: used when extracting a low frequency component (low_freq), a higher value leads to more strict
        filtering with less components being included.
        var_window_size: used when extracting a variance component (var), determines the size of the windowing used
        to calculate the standard deviation.
        """
        # Parse input string into a list of components to extract and the associated operators
        component_names, operators = parse_decomposition(input_string)  

        # Store extracted information and input arguments
        self.component_names = component_names
        self.operators = operators
        self.operators_filtered = [opp for opp in operators if (opp != '(' and opp != ')')]
        self.noise_sigma = noise_sigma
        self.var_window_size = var_window_size

    def decompose(self, t, x_samples):
        """
        Decompose a sample, currently only a single sample at a time is supported.
        t: time mapping, used for decomposing as numpy.array in shape (# time steps)
        x: sample to decompose as numpy.array in shape (# time steps)
        returns: components from decomposition as numpy.array in shape (# components, # time steps)
        """
        # x_intermediate = x_samples[:]       # Make a copy of the input signal
        components = np.zeros((x_samples.shape[0], x_samples.shape[1], len(self.component_names))) # List to store extracted components
        # x_steps = []     
                   
        for i_x, _ in enumerate(x_samples):
            x_intermediate = x_samples[i_x, :, 0]
            for i_c, comp_name in enumerate(self.component_names):  # Iterate through components to extract in set order
                if comp_name == 'bias':
                    x_component, x_intermediate = debias(t, x_intermediate)
                elif comp_name == 'trend':
                    x_component, x_intermediate = detrend(t, x_intermediate)
                elif comp_name == 'scale':
                    x_component, x_intermediate = normalize_scale(t, x_intermediate)
                elif comp_name == 'var':
                    x_component, x_intermediate = stabilize_variance(t, x_intermediate, 
                                                                    window_size=self.var_window_size, smooth=True)
                elif comp_name == 'low_freq':
                    x_component, x_intermediate = filter_high_freq(t, x_intermediate, sigma=self.noise_sigma, wavelet='db6')
                elif comp_name == 'high_freq':
                    components[i_x, :, i_c] = x_intermediate   # Store remaining signal, assumed to be high frequency component
                    # x_steps.append(x_intermediate)
                    break
                components[i_x, :, i_c] = x_component          # Store extracted component
                # x_steps.append(x_intermediate)
            # self.components = np.array(components)
            # self.steps = x_steps
        components = np.array(components)
        return components
    
    def compose(self, components):
        '''
        Compose sample (or multiple samples) from components
        components: components to compose as numpy.array in shape (# samples, # time steps, # components)
        returns: composed samples in shape (# samples, # time steps)
        '''
        components = np.array(components)   # Copy array
        i_component_lst = list(reversed(range(components.shape[-1])))   # Traverse in reverse through components for operator applciations

        for i, i_component in enumerate(i_component_lst):           # Component aggregation
            if i == 0:
                rec_construct = components[:, :, i_component]    # Start with first (=last) component
            else:
                if self.operators_filtered[i_component] == '+':     # Check which operator to apply for aggregation
                    rec_construct += components[:, :, i_component]   # Aggregate
                if self.operators_filtered[i_component] == '*':
                    rec_construct *= components[:, :, i_component]
        return rec_construct
    

def parse_decomposition(input_string):
    '''
    Parse string defining custom decomposition, determines which components to include using which operators
    input_string: string determining which components to include, should be formatted as comp1(.)comp2(.)...(.)compn,
    brackets may additionally be included to deal with ordering. compi is one of: bias, trend, var (=variance), 
    low_freq (=low frequency), scale, high_freq (=high frequency), and (.) is * or +. 
    high_freq is always assumed to be the last component in the definition by the code. Do not supply duplicate
    components.
    returns: list containing component_names each name is one of bias, trend, var, low_freq, scale and high_freq and
    list containing operators: (, ), + or *.
    '''
    component_pattern = r'[a-zA-Z0-9\_]+'
    operator_pattern = r'[+\*\(\)]'
    
    component_names = re.findall(component_pattern, input_string)
    operators = re.findall(operator_pattern, input_string)
    return component_names, operators


def debias(t, x):
    '''
    Extract bias of the given signal as its mean
    '''
    bias = np.mean(x)                       # Bias defined as mean of signal
    x_bias = np.array([bias for _ in t])    # Bias is a repeated constant value
    x = x - x_bias                          # Extract additive component
    return x_bias, x
    
def detrend(t, x):
    '''
    Extract trend of the given signal as linear fit
    '''
    t = np.linspace(0, len(x), len(x))
    trend, bias = np.polyfit(t, x, 1)
    x_trend = trend * t
    x = x - x_trend
    return x_trend, x

def _moving_average(x, window_size):
    '''
    Calculating moving average using the windowsize
    '''
    half_window = int(window_size / 2)
    x_MA = []
    # First windows will not fully fit
    for i in range(half_window):
        x_MA.append(np.mean(x[:i + half_window]))

    for i in range(half_window, len(x) - half_window):
        x_MA.append(np.mean(x[i - half_window : i + half_window]))

    # Last windows will not fully fit
    for i in range(half_window):
        x_MA.append(np.mean(x[len(x) - window_size + i:]))

    x_MA = np.array(x_MA)
    return x_MA

def filter_high_freq(t, x, sigma = 1, wavelet='db6'):
    """
    Capture low frequency behaviour using wavelet denoising
    """
    wavelet = 'db6'
    coeffs = pywt.wavedec(x, wavelet)
    # Using fixed threshold from Donoho: https://web.stanford.edu/dept/statistics/cgi-bin/donoho/wp-content/uploads/2018/08/denoiserelease3.pdf
    # More info: https://pyyawt.readthedocs.io/_modules/pyyawt/denoising.html#thselect, https://scikit-image.org/docs/stable/api/skimage.restoration.html#skimage.restoration.denoise_wavelet
    threshold = sigma * np.sqrt(2 * np.log(len(t)))

    denoised_coeffs = coeffs[:]
    denoised_coeffs[1:] = (pywt.threshold(i, value=threshold, mode='soft') for i in denoised_coeffs[1:])
    denoised_signal = pywt.waverec(denoised_coeffs, wavelet)
    denoised_signal = denoised_signal[:x.shape[0]] # Fix for odd numbered series
    x = x - denoised_signal
    return denoised_signal, x

def stabilize_variance(t, x, window_size, smooth=False):
    """
    Determine variance of signal using windowing with the given window size.
    If smooth, the result is smoothened by vinding the moving average.
    """
    half_window = int(window_size / 2)
    x_variance = []
    for i in range(half_window):
        x_variance.append(np.std(x[:i + half_window]))

    for i in range(half_window, len(x) - half_window):
        x_variance.append(np.std(x[i - half_window : i + half_window]))

    for i in range(half_window):
        x_variance.append(np.std(x[len(x) - window_size + i:]))

    x_variance = np.array(x_variance)

    if smooth:
        x_variance = _moving_average(x_variance, window_size)

    x_variance /= np.max(x_variance)

    x = (x / x_variance)
    return x_variance, x


def normalize_scale(t, x):
    """
    Determine the scale of the data as the maximum amplitude.
    """
    scale = np.max(np.abs(x))
    x_scale = np.array([scale for _ in t])
    x_scaled = x / x_scale
    return x_scale, x_scaled