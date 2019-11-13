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


def check_thresholds(data, alpha, threshold, device_config, cg_config):
    """
    Runs a regression for a single hyperparameter combination.
    Returns validation/test scores and projection/regression timings.
    """

    train_data, test_data, train_labels, test_labels = data

    if not device_config['use_cpu_memory']:
        # we need to move the labels to GPU
        train_data = train_data.to('cuda:' + str(device_config['active_gpus'][0]))
        test_data = test_data.to('cuda:' + str(device_config['active_gpus'][0]))
        train_labels = train_labels.to('cuda:' + str(device_config['active_gpus'][0]))
        test_labels = test_labels.to('cuda:' + str(device_config['active_gpus'][0]))

    since = time.time()

    try:
        if test_data.shape[1] > 10000:
            clf = RidgeRegression(device_config, solver='cg_torch', kernel=None, **cg_config)
            clf.fit(train_data, train_labels, alpha)
        else:
            clf = RidgeRegression(device_config, solver='cholesky_torch', kernel=None)
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
    parser.add_argument('--device_config', type=str, required=True,
                        help='Path to device configuration file')
    parser.add_argument('--alpha', type=float, required=True,
                        help='Regularization strength for regression')
    parser.add_argument('--threshold_min', type=float, required=True,
                        help='Minimum threshold value')
    parser.add_argument('--threshold_max', type=float, required=True,
                        help='Maximum threshold value')
    parser.add_argument('--threshold_step', type=float, required=True,
                        help='Threshold step size')    

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    cg_config = {
        "tol": 1e-6,
        "atol": 1e-9,
        "max_iterations": 180000
    }

    args = parse_args()

    print('Loading dataset: {}'.format(args.dataset_config))
    data = util.data.load_dataset(args.dataset_config, binarize_data=False)

    print('Loading device config: {}'.format(args.device_config))
    device_config = util.data.load_device_config(args.device_config)

    log_name = data[0]
    log_folder = 'thresholds'
    csv_handler = util.data.DF_Handler(log_folder, log_name)
    log_handler = util.data.Log_Handler(log_folder, log_name)

    for threshold in np.arange(args.threshold_min, args.threshold_max, args.threshold_step):
        train_data = (data[0].clone() > threshold).float()
        test_data = (data[1].clone() > threshold).float()

        test_score, regr_time = check_thresholds(
            (train_data, test_data, data[2], data[3]), args.alpha, threshold, device_config, cg_config)

        log_dictionary = {
            'test_score': test_score, 'alpha': args.alpha,
            'threshold': threshold, 'regr_time': regr_time
        }

        csv_handler.append(log_dictionary)
        csv_handler.save()
        log_handler.append('Result: {}'.format(log_dictionary))

    log_handler.append('Experiments completed!')
