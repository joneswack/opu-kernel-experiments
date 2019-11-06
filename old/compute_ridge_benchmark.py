import numpy as np
import time
import itertools
import os
import pandas as pd

import torch

from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import GridSearchCV

from random_features import project_big_np_matrix

import logging
import warnings

save_name = 'opu_degree_4_optimized'

logger = logging.getLogger()
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s [%(levelname)s] - %(message)s',
    filename='logs/{}.log'.format(save_name))

# import os
# os.environ["CUDA_VISIBLE_DEVICES"]="1"


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

### Features to compute:

from random_features import projections

output_dims = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 2000, 3000, 4000, 5000, 10000]
# output_dims = [5000]
seeds = [0] # , 1, 2, 3, 4
output_dim = 50000

configurations = [
    {
        'kernel': 'opu',
        'framework': 'pytorch',
        'cuda': False,
        'alphas': [10],
        'scales': [0.001**2],
        'dummies': [0],
        'degrees': [2]
        # 'degrees': [0.5],
        # 'dummies': [20]
#         'kernel_parameters': {
#             'activation': [None], # [None, 'sqrt', 'cos'],
#             'bias': [False] # [True, False]
#         }
    }
#     {
#         'kernel': 'opu',
#         'framework': 'pytorch',
#         'cuda': True,
#         'dummy_input': [False], # [False, True],
#         'activation': 'sqrt',
#         'kernel_parameters': {
#             'activation': ['sqrt'], # [None, 'sqrt', 'cos'],
#             'bias': [False] # [True, False]
#         }
#     }
#     {
#         'kernel': 'rbf',
#         'framework': 'pytorch',
#         'cuda': True,
#         'dummy_input': [False],
#         'kernel_parameters': {
#             'log_lengthscale_init': [np.log(np.sqrt(1./(2*0.006)))] # ['auto']
#         }
#     }
]

### Process the kernels one by one

def train(train_data, train_target, test_data, test_target, alphas):

    warned = False
    
    start_time = time.time()
    
    test_scores = []
    
    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        
#         model = RidgeClassifier()
#         parameters = {'alpha':alphas}
#         if train_data.shape[1] <= 5000:
#             clf = GridSearchCV(model, parameters, cv=2, n_jobs=len(alphas))
#         else:
#             # decrease memory usage for 5K+ dimensions
#             clf = GridSearchCV(model, parameters, cv=2, n_jobs=1)
        
#         clf.fit(train_data, train_target)
#         val_scores = clf.cv_results_['mean_test_score']
        
        val_scores = []
        
        for alpha in alphas:
            # validation score
            end_index = int(len(train_data) * 0.8)
            
            clf = RidgeClassifier(alpha=alpha)
            clf.fit(train_data[:end_index], train_target[:end_index])
            val_scores.append(clf.score(train_data[end_index:], train_target[end_index:]))
        
        for alpha in alphas:
            clf = RidgeClassifier(alpha=alpha)
            clf.fit(train_data, train_target)
            test_scores.append(clf.score(test_data, test_target))

#         for warning in caught_warnings:
#             # if warning.category == UnsupportedWarning:
#             print(str(warning.message))
#             warned = True
    train_time = time.time() - start_time
    
    return val_scores, test_scores, caught_warnings, train_time

def evaluate_kernel(df, train_data, train_targets, test_data, test_targets, scales, alphas, kwargs):
    for scale in scales:
        validation_scores, test_scores, warnings, train_time = train(scale * train_data, train_targets, scale * test_data, test_targets, alphas)

        logger.info('Scale: {}'.format(scale))
        logger.info('Training Time: {}'.format(train_time))
        logger.info('Warned: {}'.format(len(warnings)))

        entries = zip(validation_scores, test_scores, config['alphas'])

        for val_score, test_score, alpha in entries:
            logger.info('Alpha: {}'.format(alpha))
            logger.info('Val Score: {}'.format(val_score))
            logger.info('Test Score: {}'.format(test_score))
            
            param_dict = {
                'validation_score': val_score,
                'test_score': test_score,
                'training_time': train_time,
                'alpha': alpha,
                'scale': scale,
                'warnings': len(warnings)
            }
            
            param_dict = {**param_dict, **kwargs}

            df = df.append(param_dict, ignore_index=True)
        
    return df

# for seed in seeds

df = pd.DataFrame()

total_number_opu_kernels = len(configurations[0]['degrees']) * len(configurations[0]['dummies'])
total_number_kernels = total_number_opu_kernels * len(output_dims) * len(seeds)
i = 0

for config in configurations:
    for degree in config['degrees']:
        for dummy in config['dummies']:
            
            # generate 100K dimensions once
            logger.info('-----------')
            logger.info('Kernel: {}'.format(config['kernel']))
            logger.info('Dummy: {}'.format(dummy))
            logger.info('Degree: {}'.format(degree))

            data = np.vstack([train_data_bin[:N], test_data_bin])

            if dummy > 0:
                data = np.hstack([np.ones((len(data), 1)).astype('float32') * dummy, data])


            print('Computing kernel features...')

            if config['kernel'] == 'rbf':
                print('Initializing RBF lengthscale')
                log_lengthscale_init = config['kernel_parameters']['log_lengthscale_init'][0]
            else:
                log_lengthscale_init = 'auto'

            proj_data, proj_time = project_big_np_matrix(
                                        data, out_dim=output_dim, chunk_size=5000, projection=config['kernel'],
                                        framework=config['framework'], dtype=torch.FloatTensor,
                                        cuda=config['cuda'], log_lengthscale_init=log_lengthscale_init,
                                        exponent=degree)
            print('Done!')

            logger.info('Projection Time: {}'.format(proj_time))
        
            # compute the raw scale of the projections (2*sigma^2)
            if config['kernel'] == 'opu':
                data_norm = np.linalg.norm(data, axis=1, keepdims=True)**(degree*2)
                raw_scale = (proj_data / data_norm).mean()
#                 elif config['activation'] == 'cos':
#                     bias = np.random.uniform(low=0.0, high=2 * np.pi, size=(1, output_dim))
#                     proj_data = np.cos(proj_data + bias)
#                     raw_scale = 1.
            else:
                raw_scale = 1.
        
            proj_data = proj_data / raw_scale
        
            for output_dim in output_dims:
                for seed in seeds:
                    logger.info('Output dim.: {}'.format(output_dim))
                    logger.info('Seed: {}'.format(seed))

                    # artificial seeding through oversampling opu features
                    start_index = seed * output_dim
                    end_index = (seed+1) * output_dim

                    train_data_subsampled = proj_data[:N, start_index:end_index]
                    test_data_subsampled = proj_data[N:, start_index:end_index]
                    
                    kwargs = {
                        'kernel': config['kernel'],
                        'output_dim': output_dim,
                        'dummy': dummy,
                        'degree': degree,
                        'seed': seed
                    }
                    
                    logger.info('Degree: {}'.format(degree))
                    logger.info('Dummy: {}'.format(dummy))
                    logger.info('Seed: {}'.format(seed))
                    
                    df = evaluate_kernel(
                            df,
                            train_data_subsampled, train_labels,
                            test_data_subsampled, test_labels,
                            config['scales'], config['alphas'], kwargs
                    )
                    
                    i = i + 1
                    print('Finished {} / {} kernels'.format(i, total_number_kernels))
                    # we update the dataframe after processing one set of seeds
                    df.to_csv(os.path.join('csv', save_name + '.csv'), index=False)
print('Done!')
