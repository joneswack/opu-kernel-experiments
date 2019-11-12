import math
import time

import numpy as np
import scipy

import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import load_device_config, get_dataloader


class EcoLinear(nn.Module):
    """
    This is a linear module that uses already existing weights to save memory.
    """

    def __init__(self, weights):
        super(EcoLinear, self).__init__()
        self.input_features = weights.shape[0]
        self.output_features = weights.shape[1]

        self.weight = nn.Parameter(weights, requires_grad=False)

    def forward(self, input):
        return torch.matmul(input, self.weight)


class PairwiseDistances(nn.Module):
    """
    This class is a helper module used to compute pairwise squared euclidean distances with Y.
    p=2 corresponds to the euclidean distance.
    """
    def __init__(self, Y, p=2., squared=True):
        super(PairwiseDistances, self).__init__()
        self.Y = nn.Parameter(Y, requires_grad=False)
        self.p = p
        self.squared = squared

    def forward(self, x):
        dists = torch.cdist(x, self.Y, p = self.p)
        if self.squared:
            dists = dists.pow_(2)
        return dists



def iterate_over_column_chunks(device_config, Y):
    """
    This function is a generator that divides the matrix Y into column-wise chunks.
    It yields (start index, offset) pairs.

    device_config defines the size of the chunks in GB as well as when to start chunking.
    """

    # on a GPU, we have to partition the matrix-matrix-product
    # and store intermediate results in memory
    num_weights = Y.shape[0] * Y.shape[1]
    # memory size in GB
    weight_memory_size = num_weights * 32. / 8. / 1024. / 1024. / 1024.

    if weight_memory_size > device_config['memory_limit_gb']:
        # divide Y into chunks of size Y_CHUNK_SIZE in case it exceeds Y_MEMORY_LIMIT
        num_chunks = math.ceil(weight_memory_size / device_config['preferred_chunk_size_gb'])
    else:
        num_chunks = 1

    Y_column_chunk_size = int(Y.shape[1] / num_chunks)

    for i in range(num_chunks):
        if i < (num_chunks - 1):
            output_dim = Y_column_chunk_size
        elif Y.shape[1] % Y_column_chunk_size == 0:
            output_dim = Y_column_chunk_size
        else:
            output_dim = Y.shape[1] % Y_column_chunk_size

        print('Processing chunk {}'.format(i))
        print('Y shape: {}'.format([Y.shape[0], output_dim]))

        yield (i*Y_column_chunk_size, output_dim)


def large_matrix_matrix_product(device_config, X, Y, bias=0, p=1., add_overwrite=None):
    """
    This function computes X @ Y in a memory-efficient way while making use of multiple GPUs.
    Y is split into chunks separated between columns. Only one chunk is kept in GPU memory at a time.
    A chunk is copied to each GPU.
    X is processed in batches and in parallel using nn.DataParallel with multiple GPUs.

    X @ Y is therefore assembled from intermediate results across chunks and batches.

    If p is changed, the computation changes to (X @ Y)^p.
    Therefore, this function can also be used to compute linear/polynomial kernels.

    Takes around 12 min for Y = X.T (100 000 x 60 000), batchsize=3000, 6GB chunks, 3 GPUs.

    For CPU computation, please define active_gpus=[] in the device_config.
    Otherwise, the GPU ids to be used should be passed in the list.

    add_overwrite (default: None) allows to pass a matrix to which the results are added.
    This allows to save memory in case the return value should be added to a matrix.
    """

    if len(device_config['active_gpus']) > 0:
        main_gpu = torch.device('cuda:' + str(device_config['active_gpus'][0]))
    cpu = torch.device('cpu')

    dataloader = get_dataloader(X, labels=None, batchsize=device_config['matrix_prod_batch_size'], shuffle=False)

    # If add_overwrite is given, we do not need to allocate new memory for the output!
    if add_overwrite is not None:
        output_tensor = add_overwrite
    # Otherwise, we pre-allocate the memory in order to avoid unnecessary buffers.
    else:
        output_tensor = torch.zeros(X.size(0), Y.size(1)).type(torch.FloatTensor)

    print('Computing matrix-matrix product: {} x {}'.format(X.shape, Y.shape))
    since = time.time()

    for start_index, offset in iterate_over_column_chunks(device_config, Y):

        mat_mult = EcoLinear(Y[:, start_index : start_index + offset])

        if len(device_config['active_gpus']) > 0:
            mat_mult.to(main_gpu)
            mat_mult = nn.DataParallel(mat_mult, device_ids=device_config['active_gpus'])

        for idx, batch in enumerate(dataloader):
            print('Progress: {0:.2f}%'.format(idx / len(dataloader) * 100))
            # There are no labels!
            batch = batch[0]

            xTy = mat_mult(batch)

            if bias != 0:
                xTy = xTy.add_(bias)
            if p != 1:
                xTy = xTy.pow_(p)

            if len(device_config['active_gpus']) > 0:
                # partial results are stored in cpu memory after being processed
                xTy = xTy.to(cpu)

            output_tensor[
                idx*len(batch) : (idx+1)*len(batch),
                start_index : start_index + offset
            ] += xTy

    print('Elapsed: {0:.2f} seconds'.format(time.time() - since))

    if add_overwrite is None:
        return output_tensor


def large_pairwise_distances(device_config, X, Y, p=2., squared=True):
    """
    This function computes pairwise distances between X and Y in a memory-efficient way.
    It makes use of multiple GPUs.
    Y is split into chunks separated between columns. Only one chunk is kept in GPU memory at a time.
    A chunk is copied to each GPU.
    X is processed in batches and in parallel using nn.DataParallel with multiple GPUs.

    dist(X, Y) is therefore assembled from intermediate results across chunks and batches.

    For CPU computation, please define active_gpus=[] in the device_config.
    Otherwise, the GPU ids to be used should be passed in the list.
    """


    if len(device_config['active_gpus']) > 0:
        main_gpu = torch.device('cuda:' + str(device_config['active_gpus'][0]))
    cpu = torch.device('cpu')

    dataloader = get_dataloader(X, labels=None, batchsize=device_config['matrix_prod_batch_size'], shuffle=False)

    output_tensor = torch.zeros(X.size(0), Y.size(0)).type(torch.FloatTensor)
    print('Computing pairwise distances of dimension: {} x {}'.format(X.shape, Y.shape))
    since = time.time()

    # here we need to split Y by rows (column chunks of Y.t())
    for start_index, offset in iterate_over_column_chunks(device_config, Y.t()):

        # The weights need to be transposed (PyTorch convention)
        pd_module = PairwiseDistances(Y[start_index : start_index + offset], p=p, squared=squared)

        if len(device_config['active_gpus']) > 0:
            pd_module.to(main_gpu)
            pd_module = nn.DataParallel(pd_module, device_ids=device_config['active_gpus'])

        for idx, batch in enumerate(dataloader):
            print('Progress: {0:.2f}%'.format(idx / len(dataloader) * 100))
            # There are no labels!
            batch = batch[0]

            d_x_y = pd_module(batch)

            if len(device_config['active_gpus']) > 0:
                # partial results are stored in cpu memory after being processed
                d_x_y = d_x_y.to(cpu)

            output_tensor[
                idx*len(batch) : (idx+1)*len(batch),
                start_index : start_index + offset
            ] = d_x_y

    print('Elapsed: {0:.2f} seconds'.format(time.time() - since))
    return output_tensor


def opu_kernel(device_config, X, Y=None, var=1., bias=0, degree=2.):
    """
    This function computes the OPU kernel for even degrees.
    It also supports large-scale GPU computations. This should be used when Y is large.

    X and Y are input matrices of dimension (n_samples x feature_dimension).
    var is the scaling factor of the OPU kernel normally defined by the scaling of the optical RFs.

    For CPU computation, please use device_config['active_gpus']=[].
    Otherwise, the GPU ids to be used should be passed in the list.
    """

    if degree % 2 != 0:
        raise RuntimeError("This implementation only supports the OPU kernel for even degrees!")

    if Y is None:
        Y = X

    kernel = torch.zeros(X.size(0), Y.size(0)).type(torch.FloatTensor)
    s = int(degree // 2)
    s_fac_sq = math.factorial(s)**2

    if device_config['use_cpu_memory']:
        xyT = large_matrix_matrix_product(device_config, X, Y.t(), bias=bias, p=1.)
    else:
        xyT = torch.matmul(X, Y.t())

    # needs to become more memory-efficient!
    norm_x = X.norm(p='fro', dim=1, keepdim=True)
    norm_y = Y.norm(p='fro', dim=1, keepdim=True)
    norm_x_norm_y_T = torch.matmul(norm_x, norm_y.t())

    for i in range(s+1):
        # compute the sum shown in the paper
        coef = s_fac_sq * scipy.special.binom(s, i)**2
        if i > 0:
            # please note: we only take xTy**i because xTy**2 is computed beforehand
            xyT = xyT.pow_(2) # computes xyT**(2*i)
            kernel = kernel.add_(coef * xyT * norm_x_norm_y_T ** (2*(s-i)))
        else:
            kernel = kernel.add_(coef * norm_x_norm_y_T ** (2*s))

    kernel = kernel.mul_(var)
    
    return kernel


def polynomial_kernel(device_config, X, Y=None, var=1., bias=0, degree=2.):
    """
    This function computes the polynomial kernel.
    It also supports large-scale GPU computations. This should be used when Y is large.

    X and Y are input matrices of dimension (n_samples x feature_dimension).
    var is the scaling factor of the OPU kernel normally defined by the scaling of the optical RFs.

    For CPU computation, please use device_config['active_gpus']=[].
    Otherwise, the GPU ids to be used should be passed in the list.
    """

    if Y is None:
        Y = X

    if device_config['use_cpu_memory']:
        kernel = large_matrix_matrix_product(device_config, X, Y.t(), bias=bias, p=degree)
    else:
        kernel = torch.matmul(X, y.t())
        kernel.add_(bias)
        kernel.pow_(degree)
    kernel.mul_(var)
    
    return kernel


def rbf_kernel(device_config, X, Y=None, var=1., lengthscale='auto'):
    """
    This function computes the RBF kernel.
    It also supports large-scale GPU computations. This should be used when Y is large.

    X and Y are input matrices of dimension (n_samples x feature_dimension).
    var is the scaling factor of the OPU kernel normally defined by the scaling of the optical RFs.
    if lengthscale is set to 'auto', it is set to sqrt(feature_dim / 2), which makes sense for std-normalized features

    For CPU computation, please use device_config['active_gpus']=[].
    Otherwise, the GPU ids to be used should be passed in the list.
    """

    if Y is None:
        Y = X

    if lengthscale == 'auto':
        lengthscale = np.sqrt(X.shape[1] / 2)

    if device_config['use_cpu_memory']:
        kernel = large_pairwise_distances(device_config, X, Y, p=2., squared=True)
    else:
        kernel = PairwiseDistances(Y, p=2., squared=True).forward(X)

    kernel = kernel.div_(2 * lengthscale**2)
    kernel = torch.exp(-kernel, out=kernel)
    kernel = kernel.mul_(var)

    return kernel


if __name__ == '__main__':
    ## Test large matrix product divided across chunks
    # device_config = load_device_config('Tests Jonas/config/devices/mult_gpu_low_mem.json')
    device_config = {
        "active_gpus": [1,2,3],
        "use_cpu_memory": True,
        "memory_limit_gb": 0.3,
        "preferred_chunk_size_gb": 0.1,
        "matrix_prod_batch_size": 1000
    }

    # X = torch.randn(10000,10000).type(torch.FloatTensor)
    # Y = torch.randn(10000,10000).type(torch.FloatTensor)
    
    # result = large_matrix_matrix_product(device_config, X, Y)
    # result2 = torch.matmul(X, Y)

    # print(result.shape)
    # print(result2.shape)

    # print(result[:2,:2])
    # print(result2[:2,:2])

    # print(torch.mean(torch.abs(result - result2)))


    ## Test OPU kernel computation
    X = torch.randn(20000, 784).type(torch.FloatTensor)
    Y = X

    # since = time.time()
    # result = opu_kernel(device_config, X, var=1., degree=2).numpy()
    # print('Elapsed:', time.time() - since)

    # norm_x_sq = np.linalg.norm(X.numpy(), ord=2, axis=1, keepdims=True) ** 2
    # norm_y_sq = np.linalg.norm(Y.numpy(), ord=2, axis=1, keepdims=True) ** 2
    # result2 = np.dot(norm_x_sq, norm_y_sq.T) + (X.numpy() @ Y.t().numpy())**2

    # print('Error', np.mean(np.abs(result - result2) / result2))

    # result = large_pairwise_distances(device_config, X, Y).numpy()
    # from scipy.spatial.distance import cdist
    # result2 = cdist(X.numpy(), Y.numpy(), metric='sqeuclidean')

    # print('Error', np.mean(np.abs(result - result2)))

    from sklearn.metrics.pairwise import rbf_kernel as skl_rbf_kernel

    result = rbf_kernel(device_config, X, Y, lengthscale=np.sqrt(0.5)).numpy()
    result2 = skl_rbf_kernel(X, Y=Y, gamma=1.)

    print('Error', np.mean(np.abs(result - result2)))
