import torch
import torch.nn as nn
import numpy as np
import math
import time

from kernels import large_matrix_matrix_product

"""
These are the numpy versions of the random projection modules.
They are currently not used in the framework but are left to be used in other work.
"""

class OPUModuleNumpy(object):
    """
    Simulated Optical Random Features for the OPU kernel.

    scale is the sqrt(var) of the kernel, default: 1.
    bias is similar as the one for the polynomial kernel, default: 0.
    However, it ends up being squared in the resulting kernel: x' @ y' = bias**2 * (x @ y)
    degree is the degree after the modulous operation, default: 2 (m in the paper).
    """    

    def __init__(self, device_config, input_features, output_features, scale=1., bias=0, degree=2.):
        super(OPUModuleNumpy, self).__init__()

        self.input_features = input_features + 1 # to account for the bias
        
        self.real_matrix = np.random.normal(loc=0.0, scale=np.sqrt(0.5),
                                size=(self.input_features, output_features)).astype(dtype)
        self.img_matrix = np.random.normal(loc=0.0, scale=np.sqrt(0.5),
                                size=(self.input_features, output_features)).astype(dtype)
        
        self.scale = scale / np.sqrt(output_features)
        self.bias = bias
        self.degree = degree

        self.device_config = device_config
        
    def forward(self, data):
        # append bias to the data
        bias_vector = (np.ones((len(data), 1)) * self.bias)
        data = np.hstack([data, bias_vector])
        data = data.astype('float32')

        out_real = data.dot(self.real_matrix) ** 2
        out_img = data.dot(self.img_matrix) ** 2
        
        output = (out_real + out_img) ** (self.degree // 2)

        return self.scale * output

class RBFModuleNumpy(object):
    """
    Random Fourier Features for the RBF kernel.
    
    lengthscale is the initialization value for the lengthscale.
    if 'auto', it is initialized to sqrt(n_features / 2), which makes sense for std-normalized features.

    scale is the sqrt(var) of the kernel, default: 1.
    """

    def __init__(self, input_features, output_features, lengthscale='auto', scale=1., dtype='float32'):
        super(RBFModuleNumpy, self).__init__()
        
        if lengthscale == 'auto':
            self.lengthscale = np.sqrt(input_features / 2)
        else:
            self.lengthscale = lengthscale

        self.scale = scale / np.sqrt(output_features)

        self.projection_matrix = np.random.normal(size=(input_features, output_features)).astype(dtype)
        self.bias = np.random.uniform(low=0.0, high=2*np.pi, size=(output_features)).astype(dtype)

        self.dtype = dtype
        
    def forward(self, data):
        data = data.astype(self.dtype)

        data = data / self.lengthscale

        output = data @ self.projection_matrix
        output += self.bias
        output = np.cos(output)

        output *= np.sqrt(2.) * self.scale

        return output
