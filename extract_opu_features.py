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


def compute_opu_features(data, proj_params, device_config):
    """
    Returns OPU features for a certain configuration.
    """

    train_data, test_data, train_labels, test_labels = data

    # depending on device_config, we receive either a GPU tensor or a np matrix
    projection, _ = project_data(torch.cat([train_data, test_data], dim=0),
                                    device_config, **proj_params)
    
    return projection
    

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_config', type=str, required=True,
                        help='Path to dataset configuration file')
    parser.add_argument('--hyperparameter_config', type=str, required=True,
                        help='Path to hyperparameter configuration file')

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = parse_args()

    print('Loading dataset: {}'.format(args.dataset_config))
    data = util.data.load_dataset(args.dataset_config, binarize_data=True)

    print('Loading hyperparameters: {}'.format(args.hyperparameter_config))
    hyperparameter_config = util.data.load_hyperparameters(args.hyperparameter_config)

    device_config = {
        'use_cpu_memory': True
    }

    # iterate over all the kernel configs
    for kernel, params in hyperparameter_config.items():
        log_name = '_'.join([data[0], kernel])

        print('Computing features for kernel {}'.format(kernel))

        kernel_params = params['kernel_hyperparameters']
        proj_params = {**kernel_params, **{'projection': 'opu_physical', 'num_features': 100000, 'raw_features': True}}

        projection = compute_opu_features(data[1:], proj_params, device_config)

        util.data.save_numpy(projection.astype('uint8'), 'opu_features', log_name)


    print('Done!')
