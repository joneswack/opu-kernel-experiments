import numpy as np
import os
import json
import logging
import pandas as pd

from sklearn.preprocessing import LabelBinarizer

import torch
from torch.utils.data import DataLoader, TensorDataset


class DF_Handler(object):
    def __init__(self, folder, filename):
        super(DF_Handler, self).__init__()

        self.folder = os.path.join('csv', folder)

        if not os.path.exists(self.folder):
            os.makedirs(self.folder)

        self.filename = filename
        self.df = pd.DataFrame()
    
    def append(self, entry_dict):
        self.df = self.df.append(entry_dict, ignore_index=True)

    def save(self):
        self.df.to_csv(os.path.join(self.folder, self.filename + '.csv'), index=False)

class Log_Handler(object):
    def __init__(self, folder, filename):
        super(Log_Handler, self).__init__()

        self.folder = os.path.join('logs', folder)

        if not os.path.exists(self.folder):
            os.makedirs(self.folder)

        self.logger = logging.getLogger()
        logging.basicConfig(
            level=logging.INFO, 
            format='%(asctime)s [%(levelname)s] - %(message)s',
            filename=os.path.join(self.folder, filename + '.log'))

    def append(self, line):
        print(line)
        self.logger.info(line)

def check_file(path):
    if os.path.isfile(path):
        return True
    else:
        raise RuntimeError('File not found: {}'.format(path))

def save_numpy(data, folder, filename):
    if not os.path.exists(folder):
        os.makedirs(folder)

    np.save(os.path.join(folder, filename + '.npy'), data)

def load_hyperparameters(config_path):
    check_file(config_path)

    with open(config_path) as json_file:
        config = json.load(json_file)
    
    return config

def load_device_config(config_path):
    check_file(config_path)

    with open(config_path) as json_file:
        device_params = json.load(json_file)

    return device_params

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

    # Convert everything to PyTorch FloatTensors
    train_data = torch.from_numpy(train_data).type(torch.FloatTensor)
    test_data = torch.from_numpy(test_data).type(torch.FloatTensor)
    train_labels = torch.from_numpy(train_labels).type(torch.FloatTensor)
    test_labels = torch.from_numpy(test_labels).type(torch.FloatTensor)

    return config['name'], train_data, test_data, train_labels, test_labels

def get_torch_dataset(data, labels=None):
    if labels is not None:
        return TensorDataset(data, labels)
    else:
        return TensorDataset(data)

def get_dataloader(data, labels=None, batchsize=3000, shuffle=True):
    return DataLoader(
        get_torch_dataset(data, labels),
        batch_size=batchsize,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=True
    )
