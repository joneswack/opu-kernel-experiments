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


def generate_log_values(values, base=2):
    """
    Converts a log-range dictionary into a value-list.
    """

    if isinstance(values, dict):
        # convert the given range into a list containing the values
        min_value = values['min']
        max_value = values['max']
        step = values['step']

        return [base**i for i in range(min_value, max_value+1, step)]
    else:
        # simply convert the existing list of values
        return [base**i for i in values]

def convert_cv_hyperparameters_dict(hp_dict):
    """
    Converts log-range dictionaries inside the hyperparameter dictionary.
    """

    new_dict = {}

    for key, val in hp_dict.items():
        if key.endswith('_log2'):
            log_values = generate_log_values(val, base=2)
            new_key = key.replace("_log2", "")
            new_dict[new_key] = log_values
        elif key.endswith('_log10'):
            log_values = generate_log_values(val, base=10)
            new_key = key.replace("_log10", "")
            new_dict[new_key] = log_values
        else:
            new_dict[key] = val

    return new_dict

def hyperparameter_iterator(cv_hyperparameters):
    """
    This iterator iterates over all combinations of the given hyperparameter sets,
    i.e. set_1 x set_2 x ... x set_n
    """

    # take the alphas out because they are regression parameters
    alphas = cv_hyperparameters['alpha']

    kernel_params = {k:v for k,v in cv_hyperparameters.items() if k != 'alpha'}
    # we sort the dictionary by its keys
    kernel_params = collections.OrderedDict(sorted(kernel_params.items()))

    # we iterate over every combination of (alpha, kernel_params)
    for alpha in alphas:
        for combo in itertools.product(*list(kernel_params.values())):
            yield {'alpha': alpha, 'kernel_params': dict(zip(kernel_params.keys(), combo))}

def create_train_val_split(train_data, train_labels, train_size=0.8):
    """
    Splits the training data into training and validation data.
    train_size is the ratio of the training set.

    TODO: Seeding
    """

    train_size = int(train_size * len(train_labels))
    perm = torch.randperm(len(train_data))
    train_idxs = perm[:train_size]
    val_idxs = perm[train_size:]
    
    X_train = train_data[train_idxs]
    X_val = train_data[val_idxs]
    y_train = train_labels[train_idxs]
    y_val = train_labels[val_idxs]

    return X_train, X_val, y_train, y_val

def run_experiment(data, proj_params, alpha, device_config):
    """
    Runs a regression for a single hyperparameter combination.
    Returns validation/test scores and projection/regression timings.
    """

    train_data, test_data, train_labels, test_labels = data

    # depending on device_config, we receive either a GPU tensor or a np matrix
    projection, projection_time = project_data(torch.cat([train_data, test_data], dim=0),
                                    device_config, **proj_params)

    # compute train_test split on training data to create validation set
    X_train, X_val, y_train, y_val = create_train_val_split(projection[:len(train_data)], train_labels)

    if not device_config['use_cpu_memory']:
        # we need to move the labels to GPU
        y_train = y_train.to('cuda:' + str(device_config['active_gpus'][0]))
        y_val = y_val.to('cuda:' + str(device_config['active_gpus'][0]))
        test_labels = test_labels.to('cuda:' + str(device_config['active_gpus'][0]))

    since = time.time()
    
    try:
        clf = RidgeRegression(device_config, solver='cholesky_torch', kernel=None)
        clf.fit(X_train, y_train, alpha)
    except RuntimeError:
        return 0, 0, 0, 0

    regression_time = time.time() - since

    val_score = clf.score(X_val, y_val)
    test_score = clf.score(projection[len(train_data):], test_labels)
    
    return val_score, test_score, projection_time, regression_time
    

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_config', type=str, required=True,
                        help='Path to dataset configuration file')
    parser.add_argument('--hyperparameter_search_config', type=str, required=True,
                        help='Path to hyperparameter search configuration file')
    parser.add_argument('--device_config', type=str, required=True,
                        help='Path to device configuration file')

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = parse_args()

    print('Loading dataset: {}'.format(args.dataset_config))
    data = util.data.load_dataset(args.dataset_config, binarize_data=True)

    print('Loading hyperparameters: {}'.format(args.hyperparameter_search_config))
    hyperparameter_config = util.data.load_hyperparameters(args.hyperparameter_search_config)

    print('Loading device config: {}'.format(args.device_config))
    device_config = util.data.load_device_config(args.device_config)

    # iterate over all the kernel configs
    for kernel, params in hyperparameter_config.items():
        log_name = '_'.join([data[0], kernel])
        log_folder = 'hyperparameter_optimization'
        csv_handler = util.data.DF_Handler(log_folder, log_name)
        log_handler = util.data.Log_Handler(log_folder, log_name)

        log_handler.append('Running experiments for kernel {}'.format(kernel))

        converted_cv_hyperparameters = convert_cv_hyperparameters_dict(params['cv_hyperparameters'])
        other_hyperparams = {k:v for k,v in params.items() if k != 'cv_hyperparameters'}

        # iterate over all cv hyperparameters
        num_experiments = len(list(hyperparameter_iterator(converted_cv_hyperparameters)))
        for idx, cv_hyperparams in enumerate(hyperparameter_iterator(converted_cv_hyperparameters)):
            print('Progress: {} / {} ({:.2f}%)'.format(idx, num_experiments, 100*float(idx) / num_experiments))

            log_handler.append('Current configuration: {}'.format(cv_hyperparams))

            proj_params = {**cv_hyperparams['kernel_params'], **other_hyperparams}
            alpha = cv_hyperparams['alpha']

            val_score, test_score, proj_time, regr_time = run_experiment(
                data[1:], proj_params, alpha, device_config)

            log_dictionary = {**proj_params, **{
                'alpha': alpha, 'val_score': val_score,
                'test_score': test_score, 'proj_time': proj_time,
                'regr_time': regr_time
            }}
            csv_handler.append(log_dictionary)
            csv_handler.save()
            log_handler.append('Result: {}'.format(log_dictionary))

    log_handler.append('Experiments completed!')
