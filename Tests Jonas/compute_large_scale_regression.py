import os
import argparse
import json
import time

import numpy as np
import torch

import util.dataset
import util.cg_torch

# We avoid using CUDA_VISIBLE_DEVICES here and assume that this variable is set from the command line!
# os.environ["CUDA_VISIBLE_DEVICES"]="2,3"

def parse_args():
    available_kernels = ['polynomial', 'opu', 'rbf']


    parser = argparse.ArgumentParser()
    parser.add_argument('--kernel', choices=available_kernels, type=str, required=True,
                        help='Kernel')
    parser.add_argument('--degree', type=int, default=1,
                        help='Degree')

    parser.add_argument('--dataset_config', type=str, required=True,
                        help='Dataset configuration file')
    parser.add_argument('--projection_config', type=str, default=None,
                        help='Path to configuration file for projection')
    parser.add_argument('--num_gpus', type=int, default=0,
                        help='Number of GPUs to use')

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = parse_args()

    print('Loading {}'.format(args.dataset_config))
    train_data, test_data, train_labels, test_labels = util.dataset.load_dataset(args.dataset_config)

    since = time.time()

    dual_coef, iterations = util.cg_torch.MultiCGGPU(K, train_labels, tol=1e-11, atol=1e-9, max_iterations=10*N, cuda=True, num_gpus=2)

    print('Done. Iterations:', iterations)
    print('Time:', time.time() - since)



