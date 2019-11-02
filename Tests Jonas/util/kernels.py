import math
import time

import numpy as np
import scipy

import torch
import torch.nn as nn

from dataset import get_dataloader


def large_matrix_matrix_product(X, Y, bias=0, p=1., dtype=torch.FloatTensor,
                                gpu_ids=[1,2,3], batchsize=3000, Y_MEMORY_LIMIT = 12, Y_CHUNK_SIZE = 6):
    """
    This function computes X @ Y in a memory-efficient way while making use of multiple GPUs.
    Y is split into chunks separated between columns. Only one chunk is kept in GPU memory at a time.
    A chunk is copied to each GPU.
    X is processed in batches and in parallel using nn.DataParallel with multiple GPUs.

    X @ Y is therefore assembled from intermediate results across chunks and batches.

    If p is changed, the computation changes to (X @ Y)^p.
    Therefore, this function can also be used to compute linear/polynomial kernels.

    Takes around 12 min for Y = X.T (100 000 x 60 000), batchsize=3000, 6GB chunks, 3 GPUs.

    For CPU computation, please use gpu_ids=[].
    Otherwise, the GPU ids to be used should be passed in the list.
    """


    if len(gpu_ids) > 0:
        main_gpu = torch.device('cuda:' + str(gpu_ids[0]))
    cpu = torch.device('cpu')

    dataloader = get_dataloader(X, labels=None, batchsize=batchsize, shuffle=False, dtype=dtype)

    # on a GPU, we have to partition the matrix-matrix-product
    # and store intermediate results in memory
    num_weights = Y.shape[0] * Y.shape[1]
    # memory size in GB
    weight_memory_size = num_weights * 32. / 8. / 1024. / 1024. / 1024.

    if weight_memory_size > Y_MEMORY_LIMIT:
        # divide Y into chunks of size Y_CHUNK_SIZE in case it exceeds Y_MEMORY_LIMIT
        num_chunks = math.ceil(weight_memory_size / Y_CHUNK_SIZE)
    else:
        num_chunks = 1

    Y_column_chunk_size = int(Y.shape[1] / num_chunks)

    chunk_results = []

    for i in range(num_chunks):
        if i < (num_chunks - 1):
            output_dim = Y_column_chunk_size
        elif Y.shape[1] % Y_column_chunk_size == 0:
            output_dim = Y_column_chunk_size
        else:
            output_dim = Y.shape[1] % Y_column_chunk_size

        print('Processing chunk {}'.format(i))
        print('Y shape: {}'.format([Y.shape[0], output_dim]))

        mat_mult = nn.Linear(in_features=Y.shape[0], out_features=output_dim, bias=False)
        # The weights need to be transposed (PyTorch convention)
        mat_mult.weight.data = torch.from_numpy(
                                    Y[:, i*Y_column_chunk_size : i*Y_column_chunk_size + output_dim].T
                                ).type(dtype)

        if len(gpu_ids) > 1:
            mat_mult.to(main_gpu)
            mat_mult = nn.DataParallel(mat_mult, device_ids=gpu_ids)
        elif len(gpu_ids) == 1:
            mat_mult = mat_mult.to(main_gpu)
        else:
            mat_mult = mat_mult.to(cpu)

        with torch.no_grad():
            results = []
            for idx, batch in enumerate(dataloader):
                print('Progress: {}'.format(idx / len(dataloader)))
                # There are no labels!
                batch = batch[0]

                if len(gpu_ids) == 1:
                    batch = batch.to(main_gpu)

                xTy = mat_mult(batch)

                if bias != 0:
                    xTy += bias
                if p != 1:
                    xTy = xTy ** p

                results.append(xTy)

        results = torch.cat([result.to(cpu) if len(gpu_ids) > 0
                                else result for result in results], dim=0, out=None)
        chunk_results.append(results.numpy())

    return np.hstack(chunk_results)


def opu_kernel(X, Y=None, gamma=1., bias=0, degree=2., dtype=torch.FloatTensor, gpu_ids=[1,2,3]):
    """
    This function computes the OPU kernel for even degrees.
    It also supports large-scale GPU computations. This should be used when Y is large.

    X and Y are input matrices of dimension (n_samples x feature_dimension).
    Gamma is the scaling factor of the OPU kernel normally defined by the scaling of the optical RFs.

    For CPU computation, please use gpu_ids=[].
    Otherwise, the GPU ids to be used should be passed in the list.
    """

    if degree % 2 != 0:
        raise RuntimeError("This implementation only supports the OPU kernel for even degrees!")

    if Y is None:
        Y = X

    kernel = 0
    s = degree // 2
    s_fac_sq = math.factorial(s)**2

    xTy = large_matrix_matrix_product(X, Y.T, bias=bias, p=2., dtype=torch.FloatTensor,
                                        gpu_ids=gpu_ids, batchsize=3000, Y_MEMORY_LIMIT = 12, Y_CHUNK_SIZE = 6)

    norm_x = np.linalg.norm(X, ord=2, axis=1, keepdims=True)
    norm_y = np.linalg.norm(Y, ord=2, axis=1, keepdims=True)
    norm_x_norm_y_T = np.dot(norm_x, norm_y.T)

    for i in range(s+1):
        # compute the sum shown in the paper
        coef = s_fac_sq * scipy.special.binom(s, i)**2
        if i > 0:
            # please note: we only take xTy**i because xTy**2 is computed beforehand
            kernel += coef * (xTy ** i) * (norm_x_norm_y_T ** (2*(s-i)))
        else:
            kernel += coef * norm_x_norm_y_T**(2*s)

    kernel *= gamma
    
    return kernel


if __name__ == '__main__':
    ## Test large matrix product divided across chunks
    # X = np.random.normal(size=(10000,10000)).astype('float32')
    # Y = np.random.normal(size=(10000,10000)).astype('float32')
    
    # result = large_matrix_matrix_product(X, Y, gpu_ids=[1,2,3], Y_MEMORY_LIMIT = 0.3, Y_CHUNK_SIZE = 0.1)
    # result2 = X @ Y

    # print(result.shape)
    # print(result2.shape)

    # print(result[:2,:2])
    # print(result2[:2,:2])

    # print(np.mean(np.abs(result - result2)))


    ## Test OPU kernel computation
    X = np.random.normal(size=(20000, 784)).astype('float32')
    Y = X

    since = time.time()
    # result = large_matrix_matrix_product(X, Y, gpu_ids=[1,2,3])
    # result = opu_kernel_gpu(X, gamma=1., gpu_ids=[1,2,3])
    result = opu_kernel(X, gamma=1., degree=2, gpu_ids=[])
    print('Elapsed:', time.time() - since)

    norm_x_sq = np.linalg.norm(X, ord=2, axis=1, keepdims=True) ** 2
    norm_y_sq = np.linalg.norm(Y, ord=2, axis=1, keepdims=True) ** 2
    result2 = np.dot(norm_x_sq, norm_y_sq.T) + (X @ Y.T)**2

    print('Error', np.mean(np.abs(result - result2) / result2))
