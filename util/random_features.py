import torch
import torch.nn as nn
import numpy as np
import math
import time

from .kernels import large_matrix_matrix_product

class RandomProjectionModule(nn.Module):
    """
    This module is simply used to compute X @ W with W ~ N(mean, std).
    """

    def __init__(self, input_features, output_features, mean=0., std=1.):
        super(RandomProjectionModule, self).__init__()
        self.input_features = input_features
        self.output_features = output_features

        self.weight = nn.Parameter(torch.zeros(output_features, input_features).type(torch.FloatTensor), requires_grad=False)
        torch.nn.init.normal_(self.weight, mean, std)

    def forward(self, input):
        return input.mm(self.weight.t())

class OPUModulePyTorch(nn.Module):
    """
    Simulated Optical Random Features for the OPU kernel.

    scale is the sqrt(var) of the kernel, default: 1.
    bias is similar as the one for the polynomial kernel, default: 0.
    However, it ends up being squared in the resulting kernel: x' @ y' = bias**2 * (x @ y)
    degree is the degree after the modulous operation, default: 2 (m in the paper).

    if tunable_kernel is set to True, the scale is tuned.

    device_config defines whether to use a partitioned matrix-matrix product.
    """

    def __init__(self, device_config, input_features, output_features, scale=1., bias=0, degree=2., tunable_kernel=False):
        super(OPUModulePyTorch, self).__init__()

        self.input_features = input_features + 1 # to account for the bias
        self.output_features = output_features

        self.proj_real = RandomProjectionModule(self.input_features, self.output_features,
                            mean=0., std=np.sqrt(0.5))
        self.proj_im = RandomProjectionModule(self.input_features, self.output_features,
                            mean=0., std=np.sqrt(0.5))
        
        # initialize the scale to 1/sqrt(output_features) * scale
        self.log_scale = -0.5 * np.log(output_features) + np.log(scale)
        # log makes sure that scale stays positive during optimization
        self.log_scale = nn.Parameter(
            torch.ones(1).type(torch.FloatTensor) * self.log_scale,
            requires_grad=tunable_kernel
        )

        self.log_bias = nn.Parameter(
            # add noise to avoid log(0)
            torch.ones(1).type(torch.FloatTensor) * np.log(bias + 1e-9),
            requires_grad=tunable_kernel
        )
        
        self.log_degree = nn.Parameter(
            torch.ones(1).type(torch.FloatTensor) * np.log(degree),
            requires_grad=tunable_kernel
        )

        self.device_config = device_config
        
    def forward(self, data):
        # bias_vector = torch.ones(len(data), 1) * torch.exp(self.log_bias)
        bias_vector = torch.exp(self.log_bias).repeat(len(data)).view(-1, 1)
        data = torch.cat([data, bias_vector], dim=1)

        if self.device_config['use_cpu_memory']:
            # memory-saving version
            output = large_matrix_matrix_product(self.device_config, data, self.proj_real.weight.t(), bias=0, p=2.)
            large_matrix_matrix_product(self.device_config, data, self.proj_im.weight.t(), bias=0, p=2., add_overwrite=output)
        else:
            output = self.proj_real(data) ** 2
            output += self.proj_im(data) ** 2
        
        # in-place operations to decrease memory-usage.
        output = output.pow_(torch.exp(self.log_degree) // 2)
        # output = output ** (torch.exp(self.log_degree) // 2)
        output = output.mul_(torch.exp(self.log_scale))
        # output = torch.exp(self.log_scale) * output
        
        # we scale with the original scale factor (leads to kernel variance)
        return output
    
    
class OPUModuleReal(object):
    """
    Optical Random Features for the OPU kernel.

    scale is the sqrt(var) of the kernel, default: 1.
    bias is similar as the one for the polynomial kernel, default: 0.
    However, it ends up being squared in the resulting kernel: x' @ y' = bias**2 * (x @ y)
    However, it needs to be integer-valued for the OPU!
    degree is the degree after the modulous operation, default: 2 (m in the paper).
    exposure_us is the exposure time of the camera in micro-seconds.

    The optical random features are VERY tricky!
    The features are normalized to account for the default variance given by the physical setup.
    This variance also comes from the exposure time.

    Although this normalization is not necessary for plain linear regression,
    it becomes very important as soon as L2-regularization is added!

    In this case, we prefer to tune over the same hyperparameter range as for the simulated device!

    The input for the OPU needs to be in binary format!
    """

    def __init__(self, input_features, output_features, scale=1., bias=0, degree=2., exposure_us=400):
        # One way to seed would be to move the camera ROI
        # self.random_mapping.opu.device.cam_ROI = ([x_offset, y_offset], [width, height])
        # However, it is easier to oversample from the output space and subsample afterwards
        self.eposure_us = exposure_us
        self.output_features = output_features

        self.scale = scale / np.sqrt(output_features)
        self.bias = bias
        self.degree = degree

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
        # The opu needs data to be in numpy uint8 binary format!
        data = data.numpy().astype('uint8')

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
            output = random_mapping.transform(data).astype('float32')

        # Now we have to be careful:
        # The OPU has an unknown variance defined by physical settings
        # However, we can estimate it and normalize the projection to var=0.5
        opu_var = self.estimate_opu_variance(data, output)
        # We have OPU(x)_i = (w_i @ x)**2 + (v_i @ x)**2
        # = opu_var * (N(0,I) @ x)**2 + (N(0,I) @ x)**2
        # => We need to divide the projection by opu_var and multiply by 0.5
        output = output / opu_var * 0.5
        # => We have sampled all real entries from N(0, 0.5) like for the simulated device!

        output = output**(self.degree // 2)
        output = self.scale * output

        # We convert the data back to PyTorch format
        return torch.from_numpy(output).type(torch.FloatTensor)

    
class RBFModulePyTorch(nn.Module):
    """
    Random Fourier Features for the RBF kernel.
    
    lengthscale is the initialization value for the lengthscale.
    if 'auto', it is initialized to sqrt(n_features / 2), which makes sense for std-normalized features.

    scale is the sqrt(var) of the kernel, default: 1.

    if tunable_kernel is set to True, ARD for the lengthscales is activated and the scale is tuned.

    device_config defines whether to use a partitioned matrix-matrix product.
    """

    def __init__(self, device_config, input_features, output_features, lengthscale='auto', scale=1., tunable_kernel=False):
        super(RBFModulePyTorch, self).__init__()
        
        self.input_features = input_features
        self.output_features = output_features
        
        self.proj = RandomProjectionModule(input_features, output_features, mean=0., std=1.)
        self.bias = nn.Parameter(
            torch.zeros(output_features).uniform_(0, 2 * np.pi).type(torch.FloatTensor),
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
            torch.ones(input_features).type(torch.FloatTensor) * log_lengthscale,
            requires_grad=tunable_kernel
        )

        # initialize the scale to 1/sqrt(output_features) * scale
        self.log_scale = -0.5 * np.log(output_features) + np.log(scale)
        
        self.log_scale = nn.Parameter(
            # log makes sure that scale stays positive during optimization
            torch.ones(1).type(torch.FloatTensor) * self.log_scale,
            requires_grad=tunable_kernel
        )

        self.device_config = device_config
        
    def forward(self, data):
        # scale features using lengthscales
        data = data / torch.exp(self.log_lengthscales)

        if self.device_config['use_cpu_memory']:
            output = large_matrix_matrix_product(self.device_config, data, self.proj.weight.t(), bias=0, p=1.)
        else:
            output = self.proj(data)

        output = torch.cos(self.proj(data) + self.bias)

        return  np.sqrt(2.) * torch.exp(self.log_scale) * output


projection_modules = {
    'opu': OPUModulePyTorch,
    'opu_physical': OPUModuleReal,
    'rbf': RBFModulePyTorch
}

    
def project_data(data, device_config, num_features=int(1e4), projection='opu',
                    gamma='auto', scale=1., degree=2., bias=0):
    """
    This function produces the desired random features for the input data (pytorch tensors).

    device_config controls the use of GPUs and their memory.

    num_features is the projection dimension.

    gamma, scale, degree and bias are kernel parameters.
    Only a subset needs to be adapted for the desired kernel.
    """
    
    print('Computing random projection...')

    # convert gamma to lengthscale
    if gamma == 'auto':
        lengthscale = 'auto'
    else:
        lengthscale = np.sqrt(1./(2*gamma))
    
    since = time.time()

    try:
        projection_module = projection_modules[projection]
    except KeyError:
        raise RuntimeError("No {} module available!".format(projection))
    
    if projection == 'rbf':
        proj_mod = projection_module(device_config, data.shape[1], num_features, lengthscale=lengthscale, scale=scale)
    elif projection == 'opu':
        proj_mod = projection_module(device_config, data.shape[1], num_features, scale=scale, bias=bias, degree=degree)
    else:
        proj_mod = projection_module(data.shape[1], num_features)

    if not device_config['use_cpu_memory']:
        # we keep data in GPU memory
        if len(device_config['active_gpus']) == 0:
            raise RuntimeError("You have to activate the flag use_cpu_memory in the config if no GPUs are used!")

        main_gpu = torch.device('cuda:' + str(device_config['active_gpus'][0]))
        proj_mod = nn.DataParallel(proj_mod, device_ids=device_config['active_gpus'])
        proj_mod = proj_mod.to(main_gpu)

    with torch.no_grad():
        output = proj_mod.forward(data)

    elapsed = time.time() - since
    return output, elapsed


if __name__ == '__main__':
    ## Some tests to check that the modules work correctly!
    device_config = {
        "active_gpus": [1,2,3],
        "use_cpu_memory": True,
        "memory_limit_gb": 0.3,
        "preferred_chunk_size_gb": 0.1,
        "matrix_prod_batch_size": 1000
    }
    data = torch.randn(10000,784).type(torch.FloatTensor)

    # OPU Tests:
    # opu_pytorch, _ = project_data(data, device_config, projection='opu', num_features=10000, scale=np.sqrt(0.5), degree=4, bias=0)

    # from kernels import opu_kernel
    # true_kernel = opu_kernel(device_config, data, var=0.5, bias=0, degree=4)

    # kernel_pytorch = torch.matmul(opu_pytorch, opu_pytorch.t())

    # print(torch.mean(torch.abs(kernel_pytorch.to('cpu') - true_kernel) / true_kernel))
    
    # RBF Tests:
    rbf_pytorch, _ = project_data(data, device_config, projection='rbf', num_features=10000, scale=np.sqrt(0.5), lengthscale='auto')

    from kernels import rbf_kernel
    true_kernel = rbf_kernel(device_config, data, var=0.5, lengthscale='auto')

    kernel_pytorch = torch.matmul(rbf_pytorch, rbf_pytorch.t())

    print(torch.mean(torch.abs(kernel_pytorch.to('cpu') - true_kernel) / true_kernel))
    
    print('Done!')