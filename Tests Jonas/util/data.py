import numpy as np
import os
import json
import logging
import pandas as pd

from sklearn.preprocessing import LabelBinarizer

import torch
from torch.utils.data import DataLoader, TensorDataset

GPU_CONFIG_PATH = 'Tests Jonas/config/gpu.json'

class DF_Handler(object):
    def __init__(self, filename):
        super(DF_Handler, self).__init__()

        if not os.path.exists('csv'):
            os.makedirs('csv')

        self.filename = filename
        self.df = pd.DataFrame()
    
    def append(self, entry_dict):
        self.df = self.df.append(entry_dict, ignore_index=True)

    def save(self):
        self.df.to_csv(os.path.join('csv', self.filename + '.csv'), index=False)

class Log_Handler(object):
    def __init__(self, filename):
        super(Log_Handler, self).__init__()

        if not os.path.exists('logs'):
            os.makedirs('logs')

        self.logger = logging.getLogger()
        logging.basicConfig(
            level=logging.INFO, 
            format='%(asctime)s [%(levelname)s] - %(message)s',
            filename=os.path.join('logs', filename + '.log'))

    def append(self, line):
        print(line)
        self.logger.info(line)

def check_file(path):
    if os.path.isfile(path):
        return True
    else:
        raise RuntimeError('File not found: {}'.format(path))

def load_hyperparameters(config_path):
    check_file(config_path)

    with open(config_path) as json_file:
        config = json.load(json_file)
    
    return config

def load_gpu_config():
    check_file(GPU_CONFIG_PATH)

    with open(GPU_CONFIG_PATH) as json_file:
        gpu_params = json.load(json_file)

    return gpu_params

def load_dataset(config_path, binarize_data=True, dtype='float32'):
    check_file(config_path)

    with open(config_path) as json_file:
        config = json.load(json_file)

    check_file(config['train_data'])
    check_file(config['test_data'])
    check_file(config['train_labels'])
    check_file(config['test_labels'])

    train_data = np.load(config['train_data'])
    test_data = np.load(config['test_data'])
    train_labels = np.load(config['train_labels'])
    test_labels = np.load(config['test_labels'])

    # Flatten the images
    train_data = train_data.reshape(len(train_data), -1)
    test_data = test_data.reshape(len(test_data), -1)

    # Binarize the images
    if binarize_data:
        threshold = config['binary_threshold']
        train_data = np.where(train_data > threshold, 1, 0).astype(dtype)
        test_data = np.where(test_data > threshold, 1, 0).astype(dtype)

    # Convert labels to one-hot vectors
    if len(train_labels.shape) > 1:
        train_labels = np.argmax(train_labels, axis=1)
        test_labels = np.argmax(test_labels, axis=1)

    label_binarizer = LabelBinarizer(pos_label=1, neg_label=-1)
    train_labels = label_binarizer.fit_transform(train_labels).astype(dtype)
    test_labels = label_binarizer.fit_transform(test_labels).astype(dtype)

    return config['name'], train_data, test_data, train_labels, test_labels

def get_torch_dataset(data, labels=None, dtype=torch.FloatTensor):
    if labels is not None:
        return TensorDataset(
            torch.from_numpy(data).type(dtype),
            torch.from_numpy(labels).type(dtype)
        )
    else:
        return TensorDataset(torch.from_numpy(data).type(dtype))

def get_dataloader(data, labels=None, batchsize=3000, shuffle=True, dtype=torch.FloatTensor):
    return DataLoader(
        get_torch_dataset(data, labels, dtype=dtype),
        batch_size=batchsize,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=True
    )
