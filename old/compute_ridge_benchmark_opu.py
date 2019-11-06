import numpy as np
import time
import itertools
import pandas as pd
import os
import gzip

import torch

from sklearn.linear_model import RidgeClassifier

import logging
import warnings

save_name = 'ridge_benchmark_opu_optimized_1000'
feature_dir = 'fashion_mnist_features_opu'

logger = logging.getLogger()
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s [%(levelname)s] - %(message)s',
    filename=os.path.join('logs', save_name + '.log'))

# import os
# os.environ["CUDA_VISIBLE_DEVICES"]="2"


# compute number of active pixels
from lightonml.datasets import FashionMNIST

(train_data, train_labels), (test_data, test_labels) = FashionMNIST()

def threshold_binarize(data, threshold):
    data_bin = np.where(data>threshold, 1, 0).astype('uint8')
    return data_bin

# fashion mnist has values between 0 and 255
threshold = 10

train_data_bin = threshold_binarize(train_data, threshold).reshape(len(train_data), -1)
test_data_bin = threshold_binarize(test_data, threshold).reshape(len(test_data), -1)

# compute number of active pixels
active_pixels = np.vstack([train_data_bin, test_data_bin]).sum(axis=1, keepdims=True)


### Parameters:

# number of training points (N=60000 for all data)
N = 60000 # 60000
# fashion mnist has values between 0 and 255
threshold = 10

logger.info('-------------')
logger.info('New benchmark')
logger.info('N: {}'.format(N))
logger.info('Bin. Threshold: {}'.format(threshold))

### Features to compute:

from random_features import projections

alphas = [10.0] # [0.1, 1, 10, 100]
kernel_scales = [0.001] # [1, 0.1, 0.01, 0.001, 0.0001, 0.00001]
output_dims = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 2000, 3000, 4000, 5000, 10000]
seeds = [0, 1, 2, 3, 4]

config = {
        'kernel': 'opu',
        'framework': 'physical',
        'dummy_input': [False], # [True, False],
        'activation': [None],
        'exposure_us': [1000] # [300, 400, 500, 600, 700, 1000]
}

### Process the kernels one by one

def train(data, target, alpha):
    clf = RidgeClassifier(alpha=alpha)
    
    warned = False
    
    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        clf.fit(data, target)

        for warning in caught_warnings:
            # if warning.category == UnsupportedWarning:
            print(str(warning.message))
            warned = True
    
    return clf, warned

def test(clf, data, target):
    score = clf.score(data, target)
    return score

# for seed in seeds

df = pd.DataFrame()

total_number_macro = len(config['exposure_us']) * len(config['dummy_input']) * \
                        len(alphas) * len(output_dims) * len(config['activation']) * len(kernel_scales)
i = 0

for exposure_time in config['exposure_us']:
    for dummy in config['dummy_input']:
        data_dir = os.path.join(
            feature_dir,
            'exposure_{}'.format(exposure_time),
            'dummy' if dummy else 'no_dummy'
        )
        
        print('Loading input file. This may take a while...')
        # train_file = gzip.GzipFile(os.path.join(data_dir, 'train_100K.npy.gz'), "r")
        train_data = np.load(os.path.join(data_dir, 'train_100K.npy')).astype('float32')
        
        # test_file = gzip.GzipFile(os.path.join(data_dir, 'test_100K.npy.gz'), "r")
        test_data = np.load(os.path.join(data_dir, 'test_100K.npy')).astype('float32')
        
        # compute the raw scale of the projections (2*sigma^2)
        raw_scale = (np.vstack([train_data, test_data]) / active_pixels).mean()
        
        for scale in kernel_scales:
            factor = scale / raw_scale
            train_data *= factor
            test_data *= factor
        
            # print('Data type:', type(train_data[0,0]), type(test_data[0,0]))
        
            for alpha in alphas:
                for output_dim in output_dims:
                    for activation in config['activation']:
                        for seed in seeds:
                            logger.info('Alpha: {}'.format(alpha))
                            logger.info('Output dim.: {}'.format(output_dim))
                            logger.info('Seed: {}'.format(seed))
                            logger.info('Dummy: {}'.format(dummy))
                            logger.info('Exposure Time: {}'.format(exposure_time))

                            since = time.time()

                            # artificial seeding through oversampling opu features
                            start_index = seed * output_dim
                            end_index = (seed+1) * output_dim

                            train_data_subsampled = train_data[:N, start_index:end_index]
                            test_data_subsampled = test_data[:, start_index:end_index]

                            if activation == 'sqrt':
                                train_data_subsampled = np.sqrt(train_data_subsampled)
                                test_data_subsampled = np.sqrt(test_data_subsampled)

                            clf, warned = train(train_data_subsampled, train_labels[:N], alpha)

                            train_time = time.time() - since

                            score = test(clf, test_data_subsampled, test_labels)
                            logger.info('Score: {}'.format(score))
                            logger.info('Training Time: {}'.format(train_time))
                            logger.info('Warned: {}'.format(warned))

                            param_dict = {
                                'test_score': score,
                                'training_time': train_time,
                                'alpha': alpha,
                                'scale': scale,
                                'output_dim': output_dim,
                                'dummy_input': True if dummy else False,
                                'exposure_us': exposure_time,
                                'seed': seed,
                                'activation': activation,
                                'inversion_warning': warned
                            }

                            df = df.append(param_dict, ignore_index=True)
                        i = i + 1
                        print('Finished {} / {} kernels (incl. seeds)'.format(i, total_number_macro))
                        # we update the dataframe after processing one set of seeds
                        df.to_csv(save_name + '.csv', index=False)
print('Done!')
        