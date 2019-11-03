import torch
import torch.nn as nn
import numpy as np
import math
import time

class RandomProjectionModule(nn.Module):
    """
    This module is simply used to compute X @ W with W ~ N(mean, std).
    """

    def __init__(self, input_features, output_features, mean=0., std=1., dtype=torch.FloatTensor):
        super(RandomProjectionModule, self).__init__()
        self.input_features = input_features
        self.output_features = output_features

        self.weight = nn.Parameter(torch.zeros(output_features, input_features).type(dtype), requires_grad=False)
        torch.nn.init.normal_(self.weight, mean, std)

    def forward(self, input):
        return input.mm(self.weight.t())

class OPUModulePyTorch(nn.Module):
    """
    Simulated Optical Random Features for the OPU kernel.

    scale is the sqrt(var) of the kernel, default: 1.
    bias is the same as for the polynomial kernel, default: 0.
    exponent is the exponent after the modulous operation, default: 2 (m in the paper).

    if tunable_kernel is set to True, the scale is tuned.
    """

    def __init__(self, input_features, output_features, scale=1., bias=0, exponent=2., tunable_kernel=False, dtype=torch.FloatTensor):
        super(OPUModulePyTorch, self).__init__()

        self.input_features = input_features + 1 # to account for the bias
        self.output_features = output_features
        
        # initialize the scale to 1/sqrt(output_features) * scale
        self.log_scale = -0.5 * np.log(output_features) + np.log(scale)
        
        self.proj_real = RandomProjectionModule(self.input_features, self.output_features,
                            mean=0., std=np.sqrt(0.5), dtype=dtype)
        self.proj_im = RandomProjectionModule(self.input_features, self.output_features,
                            mean=0., std=np.sqrt(0.5), dtype=dtype)
        
        # log makes sure that scale stays positive during optimization
        self.log_scale = nn.Parameter(
            torch.ones(1).type(dtype) * log_scale,
            requires_grad=tunable_kernel
        )

        self.log_bias = nn.Parameter(
            torch.ones(1).type(dtype) * np.log(bias),
            requires_grad=tunable_kernel
        )
        
        self.log_exponent = nn.Parameter(
            torch.ones(1).type(dtype) * np.log(exponent),
            requires_grad=tunable_kernel
        )

        self.dtype = dtype
        
    def forward(self, data):
        data = data.type(self.dtype)
        bias_vector = torch.ones(len(data), 1).type(self.dtype) * torch.exp(self.log_bias)
        data = torch.cat([data, bias_vector], dim=1)

        out_real = self.proj_real(data) ** 2
        out_img = self.proj_im(data) ** 2
        
        output = (out_real + out_img) ** (torch.exp(self.log_exponent) // 2)
        
        # we scale with the original scale factor (leads to kernel variance)
        return torch.exp(self.log_scale) * output
    
    
class OPUModuleNumpy(object):
    """
    Simulated Optical Random Features for the OPU kernel.

    scale is the sqrt(var) of the kernel, default: 1.
    bias is the same as for the polynomial kernel, default: 0.
    exponent is the exponent after the modulous operation, default: 2 (m in the paper).
    """    

    def __init__(self, input_features, output_features, scale=1., bias=0, exponent=2., dtype='float32'):
        super(OPUModuleNumpy, self).__init__()

        self.input_features = input_features + 1 # to account for the bias
        
        self.real_matrix = np.random.normal(loc=0.0, scale=np.sqrt(0.5),
                                size=(self.input_features, output_features)).astype(dtype)
        self.img_matrix = np.random.normal(loc=0.0, scale=np.sqrt(0.5),
                                size=(self.input_features, output_features)).astype(dtype)
        
        self.scale = scale / np.sqrt(output_features)
        self.bias = bias
        self.exponent = exponent

        self.dtype = dtype
        
    def forward(self, data):
        # append bias to the data
        data = data.astype(self.dtype)
        bias_vector = np.ones((len(data), 1)).astype(self.dtype) * self.bias
        data = np.hstack([data, bias_vector])

        out_real = data.dot(self.real_matrix) ** 2
        out_img = data.dot(self.img_matrix) ** 2
        
        output = (out_real + out_img) ** (self.exponent // 2)

        return self.scale * output
    
    
class OPUModuleReal(object):
    """
    Optical Random Features for the OPU kernel.

    scale is the sqrt(var) of the kernel, default: 1.
    bias is the same as for the polynomial kernel, default: 0.
    However, it needs to be integer-valued for the OPU!
    exponent is the exponent after the modulous operation, default: 2 (m in the paper).
    exposure_us is the exposure time of the camera in micro-seconds.

    The optical random features are VERY tricky!
    The features are normalized to account for the default variance given by the physical setup.
    This variance also comes from the exposure time.

    Although this normalization is not necessary for plain linear regression,
    it becomes very important as soon as L2-regularization is added!

    In this case, we prefer to tune over the same hyperparameter range as for the simulated device!

    The input for the OPU needs to be in binary format!
    """

    def __init__(self, input_features, output_features, scale=1., bias=0, exponent=2., exposure_us=400):
        
        

        # One way to seed would be to move the camera ROI
        # self.random_mapping.opu.device.cam_ROI = ([x_offset, y_offset], [width, height])
        # However, it is easier to oversample from the output space and subsample afterwards
        self.eposure_us = exposure_us
        self.output_features = output_features

        self.scale = scale / np.sqrt(output_features)
        self.bias = bias
        self.exponent = exponent

    def estimate_opu_variance(self, data, projection):
        """
        projection = (X @ W)**2 + (X @ V)**2 with V, W ~ N(0, OPU_var)
        
        For every input x_i:
        E[projection_i] = E[(x_i @ w)**2 + (x_i @ v)**2]
                      = 2 * OPU_var * norm(x_i)**2
        E[projection_i] ~ mean(projection_i)
        => OPU_var ~ mean(projection_i) / (2*norm(x_i)**2)

        Since OPU_var is the same for all x_i, we can improve the estimate:
        OPU_var ~ mean_i(mean(projection_i) / (2*norm(x_i)**2))
        OPU_var ~ mean(projection / (2*norm(X)**2))
        """

        data_norm = np.linalg.norm(data, axis=1, keepdims=True)
        return np.mean(projection / (2*data_norm**2))

    def forward(self, data):
        # append bias to the data
        bias_vector = np.ones((len(data), self.bias)).astype('uint8')
        data = np.hstack([data, bias_vector])

        # this part communicates with the physical device
        from lightonml.projections.sklearn import OPUMap
        from lightonopu.opu import OPU
        
        with OPU(n_components=self.output_features) as opu_dev:
            random_mapping = OPUMap(opu=opu_dev, n_components=self.output_features, ndims=1)
            random_mapping.opu.device.exposure_us = self.eposure_us
            random_mapping.opu.device.frametime_us = self.eposure_us+100
            output = random_mapping.transform(data.astype('uint8')).astype('float32')

        # Now we have to be careful:
        # The OPU has an unknown variance defined by physical settings
        # However, we can estimate it and normalize the projection to var=0.5
        opu_var = self.estimate_opu_variance(data, output)
        # We have OPU(x)_i = (w_i @ x)**2 + (v_i @ x)**2
        # = opu_var * (N(0,I) @ x)**2 + (N(0,I) @ x)**2
        # => We need to divide the projection by opu_var and multiply by 0.5
        output = output / opu_var * 0.5
        # => We have sampled all real entries from N(0, 0.5) like for the simulated device!

        output = output**(self.exponent // 2)

        return self.scale * output

    
class RBFModulePyTorch(nn.Module):
    """
    Random Fourier Features for the RBF kernel.
    
    lengthscale is the initialization value for the lengthscale.
    if 'auto', it is initialized to sqrt(n_features / 2) (sklearn convention)

    scale is the sqrt(var) of the kernel, default: 1.

    if tunable_kernel is set to True, ARD for the lengthscales is activated and the scale is tuned.
    """

    def __init__(self, input_features, output_features, lengthscale='auto', scale=1., tunable_kernel=False, dtype=torch.FloatTensor):
        super(RBFModulePyTorch, self).__init__()
        
        self.input_features = input_features
        self.output_features = output_features
        
        self.proj = RandomProjectionModule(input_features, output_features, mean=0., std=1., dtype=dtype)
        self.bias = nn.Parameter(
            torch.zeros(output_features).uniform_(0, 2 * np.pi).type(dtype),
            requires_grad=False
        )
        
        # initialize gamma to 1. / input_features
        if lengthscale == 'auto':
            # log_gamma = -np.log(input_features)
            # since gamma = 1/(2l^2)
            # we have log l = 0.5 * log(input_features / 2)
            # log_lengthscales = 0.5 * torch.log(T([input_features]).type(dtype) / 2.)
            log_lengthscale = 0.5 * np.log(input_features / 2.)
        else:
            log_lengthscale = np.log(lengthscale)
        
        self.log_lengthscales = nn.Parameter(
            # if the kernel is tunable, we have ARD
            torch.ones(input_features).type(dtype) * log_lengthscale,
            requires_grad=tunable_kernel
        )

        # initialize the scale to 1/sqrt(output_features) * scale
        self.log_scale = -0.5 * np.log(output_features) + np.log(scale)
        
        self.log_scale = nn.Parameter(
            # log makes sure that scale stays positive during optimization
            torch.ones(1).type(dtype) * log_scale,
            requires_grad=tunable_kernel
        )
        
    def forward(self, data):
        # scale features using lengthscales
        data = data / torch.exp(self.log_lengthscales)
        output = self.proj(data)

        output = torch.cos(self.proj(data) + self.bias)

        return  np.sqrt(2.) * torch.exp(self.log_scale) * output
        
class RBFModuleNumpy(object):
    """
    Random Fourier Features for the RBF kernel.
    
    lengthscale is the initialization value for the lengthscale.
    if 'auto', it is initialized to sqrt(n_features / 2) (sklearn convention)

    scale is the sqrt(var) of the kernel, default: 1.
    """

    def __init__(self, input_features, output_features, lengthscale='auto', scale=1., dtype='float32'):
        super(RBFModuleNumpy, self).__init__()
        
        if lengthscale == 'auto':
            self.lengthscale = np.sqrt(input_features / 2)
        else:
            self.lengthscale = lengthscale

        self.scale = scale / np.sqrt(output_features)

        self.projection_matrix = np.random.normal(size=(input_features, output_features))
        self.bias = np.random.uniform(low=0.0, high=2*np.pi, size=(output_features))
        
    def forward(self, data):
        data = data / self.lengthscale

        output = data @ self.projection_matrix
        output += self.bias
        output = np.cos(output)

        output *= np.sqrt(2.) * self.scale


projections = {
    'opu_pytorch': OPUModulePyTorch,
    'opu_numpy': OPUModuleNumpy,
    'opu_physical': OPUModuleReal,
    'rbf_pytorch': RBFModulePyTorch,
    'rbf_numpy': RBFModuleNumpy
}

## TODO: project_big_np_matrix + Tests in main!
    
def project_big_np_matrix(data, out_dim=int(1e4), chunk_size=int(1e4), projection='opu',
                          framework='pytorch', dtype=torch.FloatTensor, cuda=True,
                          log_lengthscale_init='auto', exponent=2):
    since = time.time()
    
    projection_module = projections['_'.join([projection, framework])]
    
    if projection == 'rbf':
        proj_mod = projection_module(data.shape[1], out_dim, dtype=dtype, log_lengthscale_init=log_lengthscale_init)
    elif projection == 'opu':
        proj_mod = projection_module(data.shape[1], out_dim, dtype=dtype, exponent=exponent)
    else:
        proj_mod = projection_module(data.shape[1], out_dim, dtype=dtype)
    
    if cuda and framework=='pytorch':
        proj_mod = proj_mod.cuda()
    
    N = len(data)
    n_chunks = math.ceil(N / chunk_size)
    
    output_chunks = []
    
    for i in range(n_chunks):
        data_chunk = data[i*chunk_size:(i+1)*chunk_size]
        
        if framework=='pytorch':
            data_chunk = torch.from_numpy(data_chunk).type(dtype)
            if cuda:
                data_chunk = data_chunk.cuda()
            
        print('Processing chunk of size:', data_chunk.shape)
        
        if framework == 'pytorch':
            with torch.no_grad():
                if cuda:
                    output = proj_mod.forward(data_chunk).cpu().numpy()
                else:
                    output = proj_mod.forward(data_chunk).numpy()
        else:
            output = proj_mod.forward(data_chunk)

        output_chunks.append(output)
        
    output_chunks = np.vstack(output_chunks)
        
    elapsed = time.time() - since
    print('Total time elapsed (seconds):', elapsed)
    print('Time per chunk (seconds):', elapsed / n_chunks)
    
    return output_chunks, elapsed
