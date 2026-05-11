import numpy as np
import pywt

from .decomposition import Decomposition

class DWTDecomposition(Decomposition):
    """
    Discrete Wavelet Transform-based decomposition
    """

    def __init__(self, level=3):
        """
        Initialise DWTDecomposition
        level: number of components to extract: Approximation + level - 1 Detail components
        """
        self.components = None
        self.level = level
        self.component_names = ['Approximation'] + [('Detail-' + str(i)) for i in range(1, level + 1)]

    def decompose(self, t, x, wavelet='db6'):
        '''
        Decompose a sample, currently only a single sample at a time is supported. 
        t: time mapping, used for decomposing as numpy.array in shape (# time steps)
        x: sample to decompose as numpy.array in shape (# time steps)
        wavelet: wavelet to use for the DWT
        returns: components from decomposition as numpy.array in shape (# components, # time steps)
        '''
        self.components = np.array(pywt.mra(x, wavelet, transform='dwt', level=self.level))
        return self.components
    
    def compose(self, components):
        '''
        Compose sample (or multiple samples) from components
        components: components to compose as numpy.array in shape (# samples, # time steps, # components)
        returns: composed samples in shape (# samples, # time steps)
        '''
        return np.sum(components, axis=-1)

