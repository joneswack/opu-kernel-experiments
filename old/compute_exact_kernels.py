import numpy as np
import time

from sklearn.preprocessing import LabelBinarizer
from kernel_ridge import KernelRidge
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.metrics.pairwise import polynomial_kernel

import logging

logger = logging.getLogger()
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s [%(levelname)s] - %(message)s',
    filename='exact_kernels_new.log')

import os
os.environ["CUDA_VISIBLE_DEVICES"]="1,2,3"


### Parameters:

# number of training points (N=60000 for all data)
N = 60000 # 60000
# fashion mnist has values between 0 and 255
threshold = 10

logger.info('N: {}'.format(N))
logger.info('Bin. Threshold: {}'.format(threshold))

### Loading the data:

train_data = np.load('../../datasets/export/fashion_mnist/numpy/train_data_fashion_mnist.npy').astype('uint8')
test_data = np.load('../../datasets/export/fashion_mnist/numpy/test_data_fashion_mnist.npy').astype('uint8')
train_labels = np.load('../../datasets/export/fashion_mnist/numpy/train_targets_fashion_mnist.npy').astype('uint8')
test_labels = np.load('../../datasets/export/fashion_mnist/numpy/test_targets_fashion_mnist.npy').astype('uint8')

# Convert one-hot to integers
train_labels = np.argmax(train_labels, axis=1)
test_labels = np.argmax(test_labels, axis=1)

D = train_data[0].reshape(-1).shape[0]

# Flatten the images
train_data = train_data.reshape(-1, D)
test_data = test_data.reshape(-1, D)


### Preprocessing:

def threshold_binarize(data, threshold):
    data_bin = np.where(data>threshold, 1, 0)
    return data_bin

train_data_bin = threshold_binarize(train_data, threshold).astype('float32')
test_data_bin = threshold_binarize(test_data, threshold).astype('float32')

# like one-hot encoding with 0 corresponding to -1
label_binarizer = LabelBinarizer(pos_label=1, neg_label=-1)
train_labels_bin = label_binarizer.fit_transform(train_labels).astype('float32')
test_labels_bin = label_binarizer.fit_transform(test_labels).astype('float32')

### Kernels to run:

def opu_kernel(x, y):
    kernel = polynomial_kernel(x, Y=y, degree=2, gamma=1, coef0=0)
    norm_x_sq = np.linalg.norm(x, ord=2, axis=1, keepdims=True) ** 2
    norm_y_sq = np.linalg.norm(y, ord=2, axis=1, keepdims=True) ** 2

    # corresponds to element-wise addition of norm_x^2 * norm_y^2
    kernel += np.dot(norm_x_sq, norm_y_sq.T)
    
    return kernel

configurations = [
    {
        'kernel_name': 'rbf',
        'kernel_params': {'gamma': 0.005},
        'kernel_function': rbf_kernel,
        'kernel_scale': 0.0001,
        'kernel_noise': 10
    },
    {
        'kernel_name': 'opu',
        'kernel_params': {},
        'kernel_function': opu_kernel,
        'kernel_scale': 0.001,
        'kernel_noise': 10
    },
#     {
#         'kernel_name': 'hom_poly2',
#         'kernel_params': {'degree': 2, 'gamma': 1, 'coef0': 0},
#         'kernel_function': polynomial_kernel,
#         'kernel_scale': 0.001,
#         'kernel_noise': 10
#     },
#     {
#         'kernel_name': 'poly2',
#         'kernel_params': {'degree': 2, 'gamma': 1, 'coef0': 1},
#         'kernel_function': polynomial_kernel,
#         'kernel_scale': 0.001,
#         'kernel_noise': 10
#     },
#     {
#         'kernel_name': 'hom_poly3',
#         'kernel_params': {'degree': 3, 'gamma': 1, 'coef0': 0},
#         'kernel_function': polynomial_kernel,
#         'kernel_scale': 0.001,
#         'kernel_noise': 10
#     },
#     {
#         'kernel_name': 'poly3',
#         'kernel_params': {'degree': 3, 'gamma': 1, 'coef0': 0},
#         'kernel_function': polynomial_kernel,
#         'kernel_scale': 0.001,
#         'kernel_noise': 10
#     }
]

### Process the kernels one by one

def train(kernel_matrix, target, alpha):
    clf = KernelRidge(alpha=alpha, kernel="precomputed")
    since = time.time()
    loss = clf.fit(kernel_matrix, target, cg=True, tol=1e-5, atol=0, max_iterations=15000, num_gpus=3)
    elapsed = time.time() - since
    logger.info('Training Time: {}'.format(elapsed))
    logger.info('Training Loss: {}'.format(loss))
    
    return clf

def test(clf, test_kernel, target):
    predictions = clf.predict(test_kernel)
    score = np.sum(np.equal(np.argmax(predictions, 1), np.argmax(test_labels_bin, 1))) / len(test_data_bin) * 100
    return score

for config in configurations:
    logger.info('-----------')
    logger.info('Computing {}-kernel'.format(config['kernel_name']))
    logger.info('-----------')
 
    kernel_matrix = config['kernel_scale'] * config['kernel_function'](train_data_bin[:N], train_data_bin[:N], **config['kernel_params'])
    clf = train(kernel_matrix, train_labels_bin[:N], config['kernel_noise'])

    test_kernel = config['kernel_scale'] * config['kernel_function'](test_data_bin, train_data_bin[:N], **config['kernel_params'])
    score = test(clf, test_kernel, test_labels_bin)

    logger.info('Score: {}'.format(score))
    logger.info('-----------\n')
            
print('Done!')
        