import numpy as np
import time

from sklearn.preprocessing import LabelBinarizer
from kernel_ridge import KernelRidge
from random_features import project_big_np_matrix

import torch

import logging

logger = logging.getLogger()
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s [%(levelname)s] - %(message)s',
    filename='large_projections.log')


### Parameters:

# number of training points (N=60000 for all data)
N = 1000
# projection dimension
D_OUT = 1000
# alpha regularization terms for kernel ridge
alphas = [0, 0.05, 0.1, 0.5]
# scale values for the opu kernel
gammas = [1.0]
# fashion mnist has values between 0 and 255
threshold = 10

logger.info('N: {}'.format(N))
logger.info('Bin. Threshold: {}'.format(threshold))

### Loading the data:

train_data = np.load('../datasets/export/fashion_mnist/numpy/train_data_fashion_mnist.npy').astype('uint8')
test_data = np.load('../datasets/export/fashion_mnist/numpy/test_data_fashion_mnist.npy').astype('uint8')
train_labels = np.load('../datasets/export/fashion_mnist/numpy/train_targets_fashion_mnist.npy').astype('uint8')
test_labels = np.load('../datasets/export/fashion_mnist/numpy/test_targets_fashion_mnist.npy').astype('uint8')

# Convert one-hot to integers
train_labels = np.argmax(train_labels, axis=1)
test_labels = np.argmax(test_labels, axis=1)

D = train_data[0].reshape(-1).shape[0]

# Flatten the images
train_data = train_data.reshape(-1, D)
test_data = test_data.reshape(-1, D)


### Preprocessing:

def threshold_binarize(data, threshold):
    data_bin = np.where(data>threshold, 1, 0).astype('uint8')
    return data_bin

train_data_bin = threshold_binarize(train_data, threshold)
test_data_bin = threshold_binarize(test_data, threshold)

# like one-hot encoding with 0 corresponding to -1
label_binarizer = LabelBinarizer(pos_label=1, neg_label=-1)
train_labels_bin = label_binarizer.fit_transform(train_labels)
test_labels_bin = label_binarizer.fit_transform(test_labels)

### Kernels to run:
features = {
    # rbf kernels with automatic lengthscale determination
    'rbf': lambda x: project_big_np_matrix(x, out_dim=D_OUT, chunk_size=1000, projection='rbf',
                          framework='pytorch', dtype=torch.FloatTensor, cuda=False),
    # opu kernel
    'opu': lambda x: project_big_np_matrix(x, out_dim=D_OUT, chunk_size=1000, projection='opu',
                          framework='pytorch', dtype=torch.FloatTensor, cuda=False)
}

### Process the kernels one by one

def train(data, target, alpha):
    clf = KernelRidge(alpha=alpha, kernel="linear")
    since = time.time()
    clf.fit(data, target, cg=True, tol=1e-5)
    elapsed = time.time() - since
    logger.info('Training Time: {}'.format(elapsed))
    
    return clf

def test(clf, data, target):
    predictions = clf.predict(data)
    score = np.sum(np.equal(np.argmax(predictions, 1), np.argmax(test_labels_bin, 1))) / len(test_data_bin) * 100
    return score

for alpha in alphas:
    for key, kernel_fun in features.items():
        logger.info('-----------')
        logger.info('Computing {}-kernel'.format(key))
        logger.info('-----------')
        
        logger.info('Alpha: {}'.format(alpha))
        
        projection = kernel_fun(np.vstack([train_data_bin[:N], test_data_bin]))
        clf = train(projection[:N], train_labels_bin[:N], alpha)
        score = test(clf, projection[N:], test_labels_bin)

        logger.info('Score: {}'.format(score))
        logger.info('-----------\n')
            
print('Done!')
        
