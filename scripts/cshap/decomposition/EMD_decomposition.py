import numpy as np
import emd

from .decomposition import Decomposition

class EMDDecomposition(Decomposition):
    """
    Empirical Mode Decomposition
    """
    def __init__(self, max_imfs=10):
        """
        Initialise EMDDecomposition
        max_imfs: number of IMFs to extract, total number of components return will be max_imfs + 1
        implementation sets IMFs of higher levels not returned by the algorithm to zero vectors
        """
        self.components = None
        self.n_imfs = max_imfs
        self.component_names = ['IMF-' + str(i) for i in range(1, max_imfs + 1)] + ['Trend']

    def decompose(self, t, x):
        '''
        Decompose a sample, currently only a single sample at a time is supported. 
        t: time mapping, used for decomposing as numpy.array in shape (# time steps)
        x: sample to decompose as numpy.array in shape (# time steps)
        wavelet: wavelet to use for the DWT
        returns: components from decomposition as numpy.array in shape (# components, # time steps)
        '''
        emd_components = emd.sift.sift(x, max_imfs=self.n_imfs).T
        n_imfs_found = emd_components.shape[0] - 1
        self.components = np.zeros((self.n_imfs + 1, len(x)))
        self.components[:n_imfs_found] = emd_components[:-1]
        self.components[-1] = emd_components[-1]
        return self.components
    
    def compose(self, components):
        '''
        Compose sample (or multiple samples) from components
        components: components to compose as numpy.array in shape (# samples, # time steps, # components) 
        returns: composed samples in shape (# samples, # time steps)
        '''
        return np.sum(components, axis=-1)