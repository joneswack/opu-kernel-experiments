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
from util.kernels import opu_kernel, rbf_kernel, polynomial_kernel


def run_experiment(data, projection, kernel_params, alpha, cg_config, device_config):
    """
    Runs a regression for a single hyperparameter combination.
    Returns validation/test scores and projection/regression timings.
    """

    train_data, test_data, train_labels, test_labels = data

    kernels = {
        'opu': opu_kernel,
        'rbf': rbf_kernel,
        'poly': polynomial_kernel
    }

    if projection == 'linear':
        kernel = kernels['poly']
        kernel_params['degree'] = 1
    else:
        try:
            kernel = kernels[projection]
        except KeyError:
            raise RuntimeError("Kernel {} not available.".format(projection))

    if not device_config['use_cpu_memory']:
        # we need to move the labels to GPU
        train_labels = train_labels.to('cuda:' + str(device_config['active_gpus'][0]))
        test_labels = test_labels.to('cuda:' + str(device_config['active_gpus'][0]))

    since = time.time()

    try:
        # in this case we prefer to invert the n x n matrix instead of the D x D one.
        # linear kernel <=> polynomial with degree 1.
        kernel_function = lambda device_config, X, Y: kernel(device_config, X, Y, **kernel_params)

        clf = RidgeRegression(device_config, solver='cg_torch', kernel=kernel_function, **cg_config)
        clf.fit(train_data, train_labels, alpha)
    except RuntimeError as e:
        print(e)
        return 0, 0

    regression_time = time.time() - since

    test_score = clf.score(test_data, test_labels)
    
    return test_score, regression_time
    

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
        log_folder = 'kernel_evaluation'
        csv_handler = util.data.DF_Handler(log_folder, log_name)
        log_handler = util.data.Log_Handler(log_folder, log_name)

        log_handler.append('Running experiments for kernel {}'.format(kernel))

        projection = params['projection']
        alpha = params['alpha']
        kernel_params = params['kernel_hyperparameters']
        cg_config = params['cg_config']

        # we need to convert the scale parameter to variance
        kernel_params['var'] = kernel_params.pop('scale') ** 2
        # also the bias is squared since it appears in xTy
        if 'bias' in kernel_params.keys():
            kernel_params['bias'] = kernel_params['bias'] ** 2
        # the lengthscale is sqrt(1 / 2*gamma)
        if 'gamma' in kernel_params.keys():
            if kernel_params['gamma'] == 'auto':
                kernel_params['lengthscale'] = kernel_params.pop('gamma')
            else:
                kernel_params['lengthscale'] = np.sqrt(1. / (2*kernel_params.pop('gamma')))

        test_score, regr_time = run_experiment(
            data[1:], projection, kernel_params, alpha, cg_config, device_config)

        log_dictionary = {**kernel_params, **{
            'projection': projection, 'alpha': alpha,
            'test_score': test_score, 'regr_time': regr_time
        }}
        csv_handler.append(log_dictionary)
        csv_handler.save()
        log_handler.append('Result: {}'.format(log_dictionary))

    log_handler.append('Experiments completed!')
