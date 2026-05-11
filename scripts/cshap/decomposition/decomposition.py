from abc import ABC, abstractmethod


class Decomposition(ABC):

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def decompose(self, t, x):
        '''
        Decompose a sample, currently only a single sample at a time is supported.
        t: time mapping, used for decomposing as numpy.array in shape (# time steps)
        x: sample to decompose as numpy.array in shape (# time steps)
        wavelet: wavelet to use for the DWT
        returns: components from decomposition as numpy.array in shape (# components, # time steps)
        '''
        pass

    @abstractmethod
    def compose(self, components):
        '''
        Compose sample (or multiple samples) from components
        components: components to compose as numpy.array in shape (# samples, # time steps, # components)
        returns: composed samples in shape (# samples, # time steps)
        '''
        pass