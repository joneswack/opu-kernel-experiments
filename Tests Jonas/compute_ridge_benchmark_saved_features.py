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

save_name = 'ridge_benchmark_opu'
feature_dir = 'opu_fashion_mnist_features'

logger = logging.getLogger()
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s [%(levelname)s] - %(message)s',
    filename=save_name + '.log')

# import os
# os.environ["CUDA_VISIBLE_DEVICES"]="2"


from lightonml.datasets import FashionMNIST

(_, train_labels), (_, test_labels) = FashionMNIST()


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

alphas = [0, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]
output_dims = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 2000, 3000, 4000, 5000, 10000]
seeds = [0, 1, 2, 3, 4]

config = {
        'kernel': 'opu',
        'framework': 'physical',
        'dummy_input': [False], # [True, False],
        'activation': [None, 'sqrt'],
        # 'exposure_us': [300, 400, 500]
        'exposure_us': [400]
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
                        len(alphas) * len(output_dims) * len(config['activation'])
i = 0

for exposure_time in config['exposure_us']:
    for dummy in config['dummy_input']:
        data_dir = os.path.join(
            feature_dir,
            'exposure_{}'.format(exposure_time),
            'dummy' if dummy else 'no_dummy'
        )
        
        print('Loading input file. This may take a while...')
        train_file = gzip.GzipFile(os.path.join(data_dir, 'train_100K.npy.gz'), "r")
        train_data = np.load(train_file).astype('float32')
        
        test_file = gzip.GzipFile(os.path.join(data_dir, 'test_100K.npy.gz'), "r")
        test_data = np.load(test_file).astype('float32')
        
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
        