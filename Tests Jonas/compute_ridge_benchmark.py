import numpy as np
import time
import itertools
import pandas as pd

import torch

from sklearn.linear_model import RidgeClassifier

from random_features import project_big_np_matrix

import logging
import warnings

save_name = 'ridge_benchmark_syn_opu_optimized'

logger = logging.getLogger()
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s [%(levelname)s] - %(message)s',
    filename='logs/{}.log'.format(save_name))

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

# compute number of active pixels
active_pixels = np.vstack([train_data_bin, test_data_bin]).sum(axis=1, keepdims=True)

### Features to compute:

from random_features import projections

alphas = [10.0] # [0.1, 1, 10, 100]
kernel_scales = [0.001] # [1, 0.1, 0.01, 0.001, 0.0001, 0.00001]
output_dims = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 2000, 3000, 4000, 5000, 10000]
seeds = [0, 1, 2, 3, 4]
output_dim = 100000

configurations = [
    {
        'kernel': 'opu',
        'framework': 'pytorch',
        'cuda': True,
        'dummy_input': [False, True], # [False, True],
        'kernel_parameters': {
            'activation': [None, 'sqrt', 'cos'],
            'bias': [True, False]
        }
    }
#     {
#         'kernel': 'rbf',
#         'framework': 'pytorch',
#         'cuda': False,
#         'dummy_input': [False],
#         'kernel_parameters': {
#             'log_lengthscale_init': ['auto']
#         }
#     }
]

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

num_dummies = sum([len(config['dummy_input']) for config in configurations])
total_number_macro = num_dummies * len(alphas) * len(output_dims) * len(kernel_scales)
i = 0

for config in configurations:
    for dummy in config['dummy_input']:
        # generate 100K dimensions once
        logger.info('-----------')
        logger.info('Kernel: {}'.format(config['kernel']))
        logger.info('Dummy: {}'.format(dummy))

        input_dim = len(train_data_bin[0])

        if dummy:
            input_dim += 1

        data = np.vstack([train_data_bin[:N], test_data_bin])

        if dummy:
            data = np.hstack([np.ones((len(data), 1)).astype('float32'), data])
            
            
        print('Computing kernel features...')

        proj_data, proj_time = project_big_np_matrix(
                                    data, out_dim=output_dim,
                                    chunk_size=5000, projection=config['kernel'],
                                    framework=config['framework'], dtype=torch.FloatTensor,
                                    cuda=config['cuda'])
        print('Done!')

        logger.info('Projection Time: {}'.format(proj_time))
        
        # compute the raw scale of the projections (2*sigma^2)
        raw_scale = (np.vstack([train_data, test_data]) / active_pixels).mean()
        
        for scale in kernel_scales:
            factor = scale / raw_scale
            proj_data *= factor
        
            for alpha in alphas:
                for output_dim in output_dims:
                    for seed in seeds:
                        logger.info('Alpha: {}'.format(alpha))
                        logger.info('Output dim.: {}'.format(output_dim))
                        logger.info('Seed: {}'.format(seed))

                        # artificial seeding through oversampling opu features
                        start_index = seed * output_dim
                        end_index = (seed+1) * output_dim

                        train_data_subsampled = proj_data[:N, start_index:end_index]
                        test_data_subsampled = proj_data[N:, start_index:end_index]

                        # Going through all combinations of kernel parameters
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

                            if 'bias' in combo_dict:
                                if combo_dict['bias']:
                                    bias = np.random.uniform(low=0.0, high=2 * np.pi, size=(1, output_dim))
                                    train_data_subsampled += bias
                                    test_data_subsampled += bias

                            if 'activation' in combo_dict:
                                if combo_dict['activation'] == 'sqrt':
                                    train_data_subsampled = np.sqrt(train_data_subsampled)
                                    test_data_subsampled = np.sqrt(test_data_subsampled)
                                elif combo_dict['activation'] == 'cos':
                                    train_data_subsampled = np.cos(train_data_subsampled)
                                    test_data_subsampled = np.cos(test_data_subsampled)

                            clf, warned = train(train_data_subsampled, train_labels[:N], alpha)

                            train_time = time.time() - since

                            score = test(clf, test_data_subsampled, test_labels)
                            logger.info('Score: {}'.format(score))
                            logger.info('Training Time: {}'.format(train_time))
                            logger.info('Warned: {}'.format(warned))

                            param_dict = {
                                'kernel': config['kernel'],
                                'framework': config['framework'],
                                'test_score': score,
                                'training_time': train_time,
                                'alpha': alpha,
                                'scale': scale,
                                'output_dim': output_dim,
                                'dummy_input': True if dummy else False,
                                'seed': seed,
                                'inversion_warning': warned
                            }

                            param_dict = {**param_dict, **combo_dict}

                            df = df.append(param_dict, ignore_index=True)
                    i = i + 1
                    print('Finished {} / {} kernels (incl. seeds)'.format(i, total_number_macro))
                    # we update the dataframe after processing one set of seeds
                    df.to_csv(os.path.join('csv', save_name + '.csv'), index=False)
print('Done!')
