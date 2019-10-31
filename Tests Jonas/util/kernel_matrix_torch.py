import math
import time

import numpy as np
import scipy

import torch
import torch.nn as nn

from dataset import get_dataloader


def large_matrix_matrix_product(X, Y, p=1., dtype=torch.FloatTensor,
                                gpu_ids=[1,2,3], batchsize=3000, Y_MEMORY_LIMIT = 12, Y_CHUNK_SIZE = 6):
    """
    This function computes X @ Y in a memory-efficient way while making use of multiple GPUs.
    Y is split into chunks separated between columns. Only one chunk is kept in GPU memory at a time.
    A chunk is copied to each GPU.
    X is processed in batches and in parallel using nn.DataParallel with multiple GPUs.

    X @ Y is therefore assembled from intermediate results across chunks and batches.

    If p is changed, the computation changes to (X @ Y)^p

    Takes around 12 min for Y = X.T (100 000 x 60 000), batchsize=3000, 6GB chunks, 3 GPUs.
    """


    if len(gpu_ids) > 0:
        main_gpu = torch.device('cuda:' + str(gpu_ids[0]))
    cpu = torch.device('cpu')

    dataloader = get_dataloader(X, labels=None, batchsize=batchsize, shuffle=False)

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
        mat_mult.weight.data = torch.from_numpy(Y[:, i*Y_column_chunk_size : i*Y_column_chunk_size + output_dim].T)

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

                if p != 1:
                    results.append(mat_mult(batch) ** p)
                else:
                    results.append(mat_mult(batch))

        results = torch.cat([result.to(cpu) for result in results], dim=0, out=None)
        chunk_results.append(results.numpy())

    return np.hstack(chunk_results)



def compute_opu_kernel_gpu(X, Y=None, gamma=1., degree=2., dtype=torch.FloatTensor, gpu_ids=[1,2,3]):
    if Y is None:
        Y = X

    kernel = 0
    s = degree // 2
    s_fac_sq = math.factorial(s)**2

    xTy = large_matrix_matrix_product(X, Y.T, p=2., dtype=torch.FloatTensor,
                                        gpu_ids=[1,2,3], batchsize=3000, Y_MEMORY_LIMIT = 12, Y_CHUNK_SIZE = 6)

    norm_x = np.linalg.norm(X, ord=2, axis=1, keepdims=True)
    norm_y = np.linalg.norm(Y, ord=2, axis=1, keepdims=True)
    norm_x_norm_y_T = np.dot(norm_x, norm_y.T)

    for i in range(s):
        # compute the sum shown in the paper
        coef = s_fac_sq * scipy.special.binom(s, i)**2
        if i > 0:
            kernel += coef * (xTy ** (2*i)) / (norm_x_norm_y_T ** (2*s-2*i))
        else:
            kernel += coef * norm_x_norm_y_T**(2*s)

    # if degree == 2:
    #     kernel = large_matrix_matrix_product(X, Y.T, p=2., dtype=torch.FloatTensor,
    #                                     gpu_ids=[1,2,3], batchsize=3000, Y_MEMORY_LIMIT = 12, Y_CHUNK_SIZE = 6)

    #     norm_x_sq = np.linalg.norm(X, ord=2, axis=1, keepdims=True) ** 2
    #     norm_y_sq = np.linalg.norm(Y, ord=2, axis=1, keepdims=True) ** 2

    #     # corresponds to element-wise addition of norm_x^2 * norm_y^2
    #     kernel += np.dot(norm_x_sq, norm_y_sq.T)

    kernel *= gamma
    
    return kernel


if __name__ == '__main__':
    # X = np.random.normal(size=(10000,10000)).astype('float32')
    # Y = np.random.normal(size=(10000,10000)).astype('float32')
    
    # result = large_matrix_matrix_product(X, Y, gpu_ids=[1,2,3], Y_MEMORY_LIMIT = 0.3, Y_CHUNK_SIZE = 0.1)
    # result2 = X @ Y

    # print(result.shape)
    # print(result2.shape)

    # print(result[:2,:2])
    # print(result2[:2,:2])

    # print(np.mean(np.abs(result - result2)))

    
    # X = np.random.normal(size=(60000,100000)).astype('float32')
    X = np.random.normal(size=(60000, 784)).astype('float32')
    Y = X.T

    since = time.time()
    # result = large_matrix_matrix_product(X, Y, gpu_ids=[1,2,3])
    result = compute_opu_kernel_gpu(X, gamma=1., gpu_ids=[1,2,3])
    print('Elapsed:', time.time() - since)
