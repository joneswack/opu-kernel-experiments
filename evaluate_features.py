import os
import argparse
import json
import time
import itertools
import collections

import numpy as np
import torch
from sklearn.model_selection import train_test_split

import util.data
from util.ridge_regression import RidgeRegression
from util.random_features import project_data
from util.kernels import polynomial_kernel


def run_experiment(data, proj_params, alpha, cg_config, device_config):
    """
    Runs a regression for a single hyperparameter combination.
    Returns validation/test scores and projection/regression timings.
    """

    train_data, test_data, train_labels, test_labels = data

    # depending on device_config, we receive either a GPU tensor or a np matrix
    projection, projection_time = project_data(torch.cat([train_data, test_data], dim=0),
                                    device_config, **proj_params)

    if not device_config['use_cpu_memory']:
        # we need to move the labels to GPU
        train_labels = train_labels.to('cuda:' + str(device_config['active_gpus'][0]))
        test_labels = test_labels.to('cuda:' + str(device_config['active_gpus'][0]))

    since = time.time()

    try:
        if proj_params['num_features'] > 10000:
            if proj_params['num_features'] > len(train_labels):
                # in this case we prefer to invert the n x n matrix instead of the D x D one.
                # linear kernel <=> polynomial with degree 1.
                kernel = lambda device_config, X, Y: polynomial_kernel(device_config, X, Y, var=1., bias=0, degree=1.)
            else:
                kernel = None
            clf = RidgeRegression(device_config, solver='cg_torch', kernel=kernel, **cg_config)
            clf.fit(projection[:len(train_data)], train_labels, alpha)
        else:
            clf = RidgeRegression(device_config, solver='cholesky_torch', kernel=None)
            clf.fit(projection[:len(train_data)], train_labels, alpha)
    except RuntimeError as e:
        print(e)
        return 0, 0, 0

    regression_time = time.time() - since

    test_score = clf.score(projection[len(train_data):], test_labels)
    
    return test_score, projection_time, regression_time
    

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_config', type=str, required=True,
                        help='Path to dataset configuration file')
    parser.add_argument('--hyperparameter_config', type=str, required=True,
                        help='Path to hyperparameter configuration file')
    parser.add_argument('--device_config', type=str, required=True,
                        help='Path to device configuration file')

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = parse_args()

    print('Loading dataset: {}'.format(args.dataset_config))
    data = util.data.load_dataset(args.dataset_config, binarize_data=True)

    print('Loading hyperparameters: {}'.format(args.hyperparameter_config))
    hyperparameter_config = util.data.load_hyperparameters(args.hyperparameter_config)

    print('Loading device config: {}'.format(args.device_config))
    device_config = util.data.load_device_config(args.device_config)

    # iterate over all the kernel configs
    for kernel, params in hyperparameter_config.items():
        log_name = '_'.join([data[0], kernel])
        log_folder = 'feature_evaluation'
        csv_handler = util.data.DF_Handler(log_folder, log_name)
        log_handler = util.data.Log_Handler(log_folder, log_name)

        log_handler.append('Running experiments for kernel {}'.format(kernel))

        if 'precomputed_features' in params:
            # we can use precomputed OPU features
            precomputed = np.load(params['precomputed_features'])

        # iterate over all cv hyperparameters
        num_experiments = len(params['num_features'])
        for idx, feature_dim in enumerate(params['num_features']):
            print('Progress: {} / {} ({:.2f}%)'.format(idx, num_experiments, 100*float(idx) / num_experiments))

            log_handler.append('Feature Dimension: {}'.format(feature_dim))

            alpha = params['alpha']
            kernel_params = params['kernel_hyperparameters']
            cg_config = params['cg_config']
            proj_params = {**kernel_params, **{'projection': params['projection'], 'num_features': feature_dim}}

            if 'precomputed_features' in params:
                # we can use precomputed OPU features
                proj_params['precomputed'] = precomputed[:, :feature_dim]
                proj_params['raw_features'] = False

            # the lengthscale is sqrt(1 / 2*gamma)
            if 'gamma' in proj_params.keys():
                if proj_params['gamma'] == 'auto':
                    proj_params['lengthscale'] = proj_params.pop('gamma')
                else:
                    proj_params['lengthscale'] = np.sqrt(1. / (2*proj_params.pop('gamma')))

            test_score, proj_time, regr_time = run_experiment(
                data[1:], proj_params, alpha, cg_config, device_config)

            log_dictionary = {**proj_params, **{
                'alpha': alpha, 'test_score': test_score,
                'proj_time': proj_time, 'regr_time': regr_time
            }}

            if 'precomputed' in log_dictionary and log_dictionary['precomputed'] is not None:
                log_dictionary['precomputed'] = True

            csv_handler.append(log_dictionary)
            csv_handler.save()
            log_handler.append('Result: {}'.format(log_dictionary))

    log_handler.append('Experiments completed!')
