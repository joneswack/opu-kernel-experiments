import numpy as np
import time
import itertools
import pandas as pd

import torch

from sklearn.linear_model import RidgeClassifier

import logging

logger = logging.getLogger()
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s [%(levelname)s] - %(message)s',
    filename='ridge_benchmark.log')

import os
os.environ["CUDA_VISIBLE_DEVICES"]="2"


### Parameters:

# number of training points (N=60000 for all data)
N = 60000 # 60000
# fashion mnist has values between 0 and 255
threshold = 10

logger.info('-------------')
logger.info('New benchmark')
logger.info('N: {}'.format(N))
logger.info('Bin. Threshold: {}'.format(threshold))

### Loading the data:

# from lightonml.datasets import FashionMNIST

# (train_data, train_labels), (test_data, test_labels) = FashionMNIST()
# D = train_data[0].reshape(-1).shape[0]

train_data = np.load('../datasets/export/fashion_mnist/numpy/train_data_fashion_mnist.npy').astype('uint8')
test_data = np.load('../datasets/export/fashion_mnist/numpy/test_data_fashion_mnist.npy').astype('uint8')
train_labels = np.load('../datasets/export/fashion_mnist/numpy/train_targets_fashion_mnist.npy').astype('uint8')
test_labels = np.load('../datasets/export/fashion_mnist/numpy/test_targets_fashion_mnist.npy').astype('uint8')

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

### Features to compute:

from random_features import projections

alphas = [0, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]
output_dims = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 2000, 3000, 4000, 5000, 10000]
seeds = [0, 1, 2, 3, 4]

configurations = [
    {
        'kernel': 'opu',
        'framework': 'pytorch',
        'cuda': False,
        'dummy_input': False,
        'kernel_parameters': {
            'activation': [None, 'sqrt', 'cos'],
            'bias': [True, False]
        }
    }
#     {
#         'kernel': 'rbf',
#         'framework': 'pytorch',
#         'cuda': False,
#         'kernel_parameters': {
#             'log_lengthscale_init': ['auto']
#         }
#     }
]

### Process the kernels one by one

def train(data, target, alpha):
    clf = RidgeClassifier(alpha=alpha)
    clf.fit(data, target)
    
    return clf

def test(clf, data, target):
    score = clf.score(data, target)
    return score

# for seed in seeds

df = pd.DataFrame()

total_number_macro = len(alphas) * len(output_dims) * len(configurations)
i = 0

for alpha in alphas:
    for output_dim in output_dims:
        for config in configurations:
            logger.info('-----------')
            logger.info('Kernel: {}'.format(config['kernel']))
            logger.info('-----------')

            logger.info('Alpha: {}'.format(alpha))
            logger.info('Output dim.: {}'.format(output_dim))
            logger.info('Framework: {}'.format(config['framework']))
            
            kernel_parameters = config['kernel_parameters']
            # list of lists
            all_parameters = [kernel_parameters[key] for key in sorted(kernel_parameters.keys())]
            all_combinations = list(itertools.product(*all_parameters))
            all_combinations_dicts = []
            for combo in all_combinations:
                combo_dict = dict(list(zip(sorted(kernel_parameters.keys()), combo)))
                all_combinations_dicts.append(combo_dict)
            
            for combo_dict in all_combinations_dicts:
                for key, item in combo_dict.items():
                    logger.info('{}: {}'.format(key, item))
                    
                since = time.time()
                
                module = projections['_'.join([config['kernel'], config['framework']])]
                input_dim = len(train_data_bin[0])
                if 'dummy_input' in config and config['dummy_input']:
                    input_dim += 1

                proj = module(input_dim, output_dim, **combo_dict)
                data = np.vstack([train_data_bin[:N], test_data_bin])

                if 'dummy_input' in config and config['dummy_input']:
                    data = np.hstack([np.ones((len(data), 1)).astype('float32'), data])

                if config['framework'] == 'pytorch':
                    data = torch.from_numpy(data)

                if 'cuda' in config and config['cuda']:
                    proj = proj.cuda()
                    data = data.cuda()

                proj_data = proj.forward(data)

                if 'cuda' in config and config['cuda']:
                    proj_data = proj_data.cpu()

                if config['framework'] == 'pytorch':
                    proj_data = proj_data.numpy()

                clf = train(proj_data[:N], train_labels[:N], alpha)
                
                train_time = time.time() - since
                
                score = test(clf, proj_data[N:], test_labels)
                logger.info('Score: {}'.format(score))
                logger.info('Training Time: {}'.format(train_time))
                
                param_dict = {
                    'test_score': score,
                    'training_time': train_time,
                    'alpha': alpha,
                    'output_dim': output_dim,
                    'kernel': config['kernel'],
                    'framework': config['framework'],
                    'dummy_input': True if ('dummy_input' in config and config['dummy_input']) else False,
                    'cuda': True if ('cuda' in config and config['cuda']) else False
                }
                
                param_dict = {**param_dict, **combo_dict}
                df = df.append(param_dict, ignore_index=True)
            
            logger.info('-----------\n')
            i += 1
            print('Finished {} / {} kernels (incl. parameter tests)'.format(i, total_number_macro))

            # We update the CSV after every kernel
            df.to_csv('ridge_benchmark.csv', index=False)
print('Done!')
        