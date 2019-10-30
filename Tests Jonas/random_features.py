import torch
import torch.nn as nn
import torch.tensor as T
import numpy as np
import math
import time

from sklearn.kernel_approximation import RBFSampler

class RandomProjectionModule(nn.Module):
    def __init__(self, input_features, output_features, mean=0., std=1., dtype=torch.FloatTensor):
        super(RandomProjectionModule, self).__init__()
        self.input_features = input_features
        self.output_features = output_features

        self.weight = nn.Parameter(torch.zeros(output_features, input_features).type(dtype), requires_grad=False)
        torch.nn.init.normal_(self.weight, mean, std)

    def forward(self, input):
        return input.mm(self.weight.t())

class OPUModulePyTorch(nn.Module):
    def __init__(self, input_features, output_features, initial_log_scale='auto', tunable_kernel=False, exponent=1, dtype=torch.FloatTensor):
        super(OPUModulePyTorch, self).__init__()

        self.input_features = input_features
        self.output_features = output_features
        
        if initial_log_scale == 'auto':
            # 1. / np.sqrt(input_features)
            initial_log_scale = -0.5 * np.log(input_features)
        
        self.proj_real = RandomProjectionModule(input_features, output_features, mean=0., std=np.sqrt(0.5), dtype=dtype)
        self.proj_im = RandomProjectionModule(input_features, output_features, mean=0., std=np.sqrt(0.5), dtype=dtype)
        
        self.log_scale = nn.Parameter(
            # log makes sure that scale stays positive during optimization
            torch.ones(1).type(dtype) * initial_log_scale,
            requires_grad=tunable_kernel
        )
        
        self.exponent = exponent
        
    def forward(self, input):
        out_real = self.proj_real(input) ** 2
        out_img = self.proj_im(input) ** 2
        
        output = (out_real + out_img) ** self.exponent
        
        # we scale with the original scale factor (leads to kernel variance)
        return torch.exp(self.log_scale) * output
    
    def get_projection_and_rawscale(self, data):
        # norm of the inputs taken to the power of self.exponent
        data_norm = data.norm(dim=1, keepdims=True)**(self.exponent*2)
        projection = self.forward(data)
        
        raw_scale = (projection / data_norm).mean()
        
        return projection, raw_scale
    
    
class OPUModuleNumpy(object):
    def __init__(self, input_features, output_features, initial_log_scale='auto', exponent=1, dtype='float32'):
        super(OPUModuleNumpy, self).__init__()
        
        self.real_matrix = np.random.normal(loc=0.0, scale=np.sqrt(0.5), size=(input_features, output_features)).astype(dtype)
        self.img_matrix = np.random.normal(loc=0.0, scale=np.sqrt(0.5), size=(input_features, output_features)).astype(dtype)
        
        self.exponent = exponent
        
        if initial_log_scale == 'auto':
            self.log_scale = -0.5 * np.log(input_features)
        else:
            self.log_scale = initial_log_scale
        
    def project(self, data, matrix):
        return np.dot(data, matrix)
        
    def forward(self, data):
        out_real = self.project(data, self.real_matrix) ** 2
        out_img = self.project(data, self.img_matrix) ** 2
        
        output = (out_real + out_img) ** self.exponent

        return np.exp(self.log_scale) * output
    
    def get_projection_and_rawscale(self, data):
        # norm of the inputs taken to the power of self.exponent
        data_norm = np.linalg.norm(data, axis=1, keepdims=True)**(self.exponent*2)
        projection = self.forward(data)
        
        raw_scale = (projection / data_norm).mean()
        
        return projection, raw_scale
    
    
class OPUModuleReal(object):
    def __init__(self, input_features, output_features, activation=None, bias=False, initial_log_scale=0, exponent=1, exposure_us=400):
        
        # One way to seed would be to move the camera ROI
        # self.random_mapping.opu.device.cam_ROI = ([x_offset, y_offset], [width, height])
        # However, it is easier to oversample from the output space and subsample afterwards
        self.eposure_us = exposure_us
        self.output_features = output_features
        
        if initial_log_scale == 'auto':
            self.log_scale = -0.5 * np.log(input_features)
        else:
            self.log_scale = initial_log_scale
            
        self.exponent = exponent
        
    def forward(self, data):
        from lightonml.projections.sklearn import OPUMap
        from lightonopu.opu import OPU
        
        with OPU(n_components=self.output_features) as opu_dev:
            random_mapping = OPUMap(opu=opu_dev, n_components=self.output_features, ndims=1)
            random_mapping.opu.device.exposure_us = self.eposure_us
            random_mapping.opu.device.frametime_us = self.eposure_us+100
            output = random_mapping.transform(data.astype('uint8'))**self.exponent
        # random_mapping.opu.close()
        
        if self.log_scale != 0:
            output = output * np.exp(self.log_scale)

        return output
    
    def get_projection_and_rawscale(self, data):
        # norm of the inputs taken to the power of self.exponent
        data_norm = np.linalg.norm(data, axis=1, keepdims=True)**(self.exponent*2)
        projection = self.forward(data)
        
        raw_scale = (projection / data_norm).mean()
        
        return projection, raw_scale

    
class RBFModulePyTorch(nn.Module):
    def __init__(self, input_features, output_features, log_lengthscale_init='auto', tunable_kernel=False, dtype=torch.FloatTensor):
        super(RBFModulePyTorch, self).__init__()
        
        self.input_features = input_features
        self.output_features = output_features
        
        self.proj = RandomProjectionModule(input_features, output_features, mean=0., std=1., dtype=dtype)
        self.bias = nn.Parameter(
            torch.zeros(output_features).uniform_(0, 2 * np.pi).type(dtype),
            requires_grad=False
        )
        
        # initialize gamma to 1. / input_features since gamma = 1/(2l^2)
        if log_lengthscale_init == 'auto':
            log_lengthscale_init = 0.5 * torch.log(T([input_features]).type(dtype) / 2.)
        
        self.log_lengthscales = nn.Parameter(
            torch.ones(input_features).type(dtype) * log_lengthscale_init,
            # torch.zeros(input_features).type(torch.FloatTensor),
            requires_grad=tunable_kernel
        )
        
        self.scale = nn.Parameter(
            torch.sqrt(T([2.]).type(dtype)),
            # we do not set the scale to trainable because it overparameterizes the weights in the next layer
            requires_grad=False
        )
        
    def forward(self, data):
        # scale features using lengthscales
        data = data / torch.exp(self.log_lengthscales)

        output = torch.cos(self.proj(data) + self.bias)


        # the rest is done by sigma^2 (the same in DGP code)
        # scale = torch.sqrt(2 * torch.exp(self.log_var))

        # sqrt(output_features) is already done when followed by OPU layer or trainable linear layer
        return self.scale * output
        
class RBFModuleNumpy(object):
    def __init__(self, input_features, output_features, log_lengthscale_init='auto', dtype='float32'):
        super(RBFModuleNumpy, self).__init__()
        
        if log_lengthscale_init=='auto':
            log_lengthscale_init = 0.5 * np.log(input_features / 2.)
        
        gamma = 1. / (2.*np.exp(log_lengthscale_init)**2)
        self.sampler = RBFSampler(gamma=gamma, n_components=output_features, random_state=1)
        
    def forward(self, data):
        if not hasattr(self.sampler, 'random_weights_'):
            return self.sampler.fit_transform(data)
        else:
            return self.sampler.transform(data)

projections = {
    'opu_pytorch': OPUModulePyTorch,
    'opu_numpy': OPUModuleNumpy,
    'opu_physical': OPUModuleReal,
    'rbf_pytorch': RBFModulePyTorch,
    'rbf_numpy': RBFModuleNumpy
}
    
def project_big_np_matrix(data, out_dim=int(1e4), chunk_size=int(1e4), projection='opu',
                          framework='pytorch', dtype=torch.FloatTensor, cuda=True,
                          log_lengthscale_init='auto', exponent=1):
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
