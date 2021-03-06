import numpy as np
import time
import itertools
import pandas as pd
import os
import gzip

import torch

from sklearn.linear_model import RidgeClassifier

import logging

save_name = 'opu_random_features_100'
feature_dir = 'fashion_mnist_features_opu'

logger = logging.getLogger()
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s [%(levelname)s] - %(message)s',
    filename=os.path.join('logs', save_name + '.log'))

# import os
# os.environ["CUDA_VISIBLE_DEVICES"]="2"


### Parameters:

# number of training points (N=60000 for all data)
N = 60000 # 60000
# fashion mnist has values between 0 and 255
threshold = 10

logger.info('-------------')
logger.info('New feature processing...')
logger.info('N: {}'.format(N))
logger.info('Bin. Threshold: {}'.format(threshold))

### Loading the data:

from lightonml.datasets import FashionMNIST

(train_data, train_labels), (test_data, test_labels) = FashionMNIST()
D = train_data[0].reshape(-1).shape[0]

# train_data = np.load('../datasets/export/fashion_mnist/numpy/train_data_fashion_mnist.npy').astype('uint8')
# test_data = np.load('../datasets/export/fashion_mnist/numpy/test_data_fashion_mnist.npy').astype('uint8')
# train_labels = np.load('../datasets/export/fashion_mnist/numpy/train_targets_fashion_mnist.npy').astype('uint8')
# test_labels = np.load('../datasets/export/fashion_mnist/numpy/test_targets_fashion_mnist.npy').astype('uint8')

# # Convert one-hot to integers
# train_labels = np.argmax(train_labels, axis=1)
# test_labels = np.argmax(test_labels, axis=1)

# D = train_data[0].reshape(-1).shape[0]

# Flatten the images
train_data = train_data.reshape(-1, D)
test_data = test_data.reshape(-1, D)


### Preprocessing:

def threshold_binarize(data, threshold):
    data_bin = np.where(data>threshold, 1, 0)
    return data_bin

train_data_bin = threshold_binarize(train_data, threshold).astype('uint8')
test_data_bin = threshold_binarize(test_data, threshold).astype('uint8')

### Features to compute:

from random_features import projections

# output_dims = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 2000, 3000, 4000, 5000, 10000]
output_dim = 100000 # allows us to use 5 seeds for up to 20K dimensions
# seeds = 5

configuration = {
    'kernel': 'opu',
    'framework': 'physical',
    'dummy_input': [False],
    'exposure_us': [100] # [300, 600, 700] # [300, 400, 500, 600]
}

### Process the kernels one by one

# for seed in seeds

total_number_macro = len(configuration['exposure_us']) * len(configuration['dummy_input'])
i = 0

for exposure_us in configuration['exposure_us']:
    for dummy in configuration['dummy_input']:
        logger.info('-----------')

        logger.info('Dummy: {}'.format(dummy))
        logger.info('Exposure mu_s: {}'.format(exposure_us))

        module = projections['_'.join([configuration['kernel'], configuration['framework']])]
        input_dim = len(train_data_bin[0])

        if dummy:
            input_dim += 1

        proj = module(input_dim, output_dim, exposure_us=exposure_us)
        data = np.vstack([train_data_bin[:N], test_data_bin])

        if dummy:
            data = np.hstack([np.ones((len(data), 1)).astype('uint8'), data])

        since = time.time()
        proj_data = proj.forward(data)
        train_time = time.time() - since

        logger.info('Projection Time: {}'.format(train_time))

        # save features
        dummy_dir = 'dummy' if dummy else 'no_dummy'
        train_filename = 'train_{}K.npy'.format(output_dim//1000)
        test_filename = 'test_{}K.npy'.format(output_dim//1000)

        out_dir = os.path.join(
            feature_dir,
            'exposure_{}'.format(exposure_us),
            dummy_dir
        )

        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        # train_file = gzip.GzipFile(os.path.join(out_dir, train_filename), "w")
        np.save(os.path.join(out_dir, train_filename), proj_data[:N])
        # train_file.close()

        # test_file = gzip.GzipFile(os.path.join(out_dir, test_filename), "w")
        np.save(os.path.join(out_dir, test_filename), proj_data[N:])
        # test_file.close()

        # for loading later on:
        # f = gzip.GzipFile('file.npy.gz', "r")
        # np.load(f)

        logger.info('-----------\n')
        i += 1
        print('Finished {} / {} kernels'.format(i, total_number_macro))
print('Done!')
        