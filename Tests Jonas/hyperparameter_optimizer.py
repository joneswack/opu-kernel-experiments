import os
import argparse
import json
import time
import itertools
import collections

import numpy as np
import torch

import util.data
import util.cg_torch

# We avoid using CUDA_VISIBLE_DEVICES here and assume that this variable is set from the command line!
# os.environ["CUDA_VISIBLE_DEVICES"]="2,3"

def generate_log_values(values, base=2):
    if isinstance(values, dict):
        # convert the given range into a list containing the values
        min_value = values['min']
        max_value = values['max']
        step = values['step']

        return [base**i for i in range(min_value, max_value, step)]
    else:
        # simply convert the existing list of values
        return [base**i for i in values]

def convert_cv_hyperparameters_dict(hp_dict):
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
    # take the alphas out because they are regression parameters
    alphas = cv_hyperparameters['alpha']

    kernel_params = {k:v for k,v in cv_hyperparameters.items() if k != 'alpha'}
    # we sort the dictionary by its keys
    kernel_params = collections.OrderedDict(sorted(kernel_params.items()))

    # we iterate over every combination of (alpha, kernel_params)
    for alpha in alphas:
        for combo in itertools.product(*list(kernel_params.values())):
            yield {'alpha': alpha, 'kernel_params': dict(zip(kernel_params.keys(), combo))}

def run_experiment(train_data, test_data, train_labels, test_labels, hyperparams):
    return True

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_config', type=str, required=True,
                        help='Dataset configuration file')
    parser.add_argument('--hyperparameter_config', type=str, default=None,
                        help='Path to configuration file for hyperparameters to test')
    # parser.add_argument('--num_gpus', type=int, default=0,
    #                     help='Number of GPUs to use')

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = parse_args()

    print('Loading dataset: {}'.format(args.dataset_config))
    train_data, test_data, train_labels, test_labels = util.data.load_dataset(args.dataset_config, binarize_data=True)

    print('Loading hyperparameters: {}'.format(args.hyperparameter_config))
    hyperparameter_config = util.data.load_hyperparameters(args.hyperparameter_config)

    # iterate over all the kernel configs
    for kernel, params in hyperparameter_config.items():
        print('Running experiments for kernel {}'.format(kernel))
        converted_hyperparameters = convert_cv_hyperparameters_dict(params['cv_hyperparameters'])
        other_parameters = {k:v for k,v in params.items() if k != 'cv_hyperparameters'}

        # iterate over all cv hyperparameters
        for hyperparams in hyperparameter_iterator(converted_hyperparameters):
            print('Current configuration: {}'.format(hyperparams))
            run_experiment(train_data, test_data, train_labels, test_labels, hyperparams)

    # since = time.time()
    # dual_coef, iterations = util.cg_torch.MultiCGGPU(K, train_labels, tol=1e-11, atol=1e-9, max_iterations=10*N, cuda=True, num_gpus=2)

    # print('Done. Iterations:', iterations)
    # print('Time:', time.time() - since)