import numpy as np
import time

from sklearn.preprocessing import LabelBinarizer
from kernel_ridge import KernelRidge
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.metrics.pairwise import polynomial_kernel

import logging

logger = logging.getLogger()
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s [%(levelname)s] - %(message)s',
    filename='exact_kernels.log')

import os
os.environ["CUDA_VISIBLE_DEVICES"]="2"


### Parameters:

# number of training points (N=60000 for all data)
N = 60000 # 60000
# alpha regularization terms for kernel ridge
alphas = [0.05, 0.5, 1.0] # 0.05, 
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
    data_bin = np.where(data>threshold, 1, 0)
    return data_bin

train_data_bin = threshold_binarize(train_data, threshold).astype('float32')
test_data_bin = threshold_binarize(test_data, threshold).astype('float32')

# like one-hot encoding with 0 corresponding to -1
label_binarizer = LabelBinarizer(pos_label=1, neg_label=-1)
train_labels_bin = label_binarizer.fit_transform(train_labels).astype('float32')
test_labels_bin = label_binarizer.fit_transform(test_labels).astype('float32')

### Kernels to run:

def opu_kernel(x, y, gamma=1):
    kernel = polynomial_kernel(x, Y=y, degree=2, gamma=1, coef0=0)
    norm_x_sq = np.linalg.norm(x, ord=2, axis=1, keepdims=True) ** 2
    norm_y_sq = np.linalg.norm(y, ord=2, axis=1, keepdims=True) ** 2

    # corresponds to element-wise addition of norm_x^2 * norm_y^2
    kernel += np.dot(norm_x_sq, norm_y_sq.T)
    
    kernel *= gamma
    
    return kernel

kernels = {
    # rbf kernels with automatic lengthscale determination
    'rbf': lambda x, y: rbf_kernel(x, Y=y, gamma=None),
    # simplest polynomial kernel of degree 2
    'hom_poly2': lambda x, y: polynomial_kernel(x, Y=y, degree=2, gamma=1, coef0=0),
    # automatic choice of gamma
    'hom_poly2_auto': lambda x, y: polynomial_kernel(x, Y=y, degree=2, gamma=None, coef0=0),
    # inhomogeneous polynomial kernel
    'poly2': lambda x, y: polynomial_kernel(x, Y=y, degree=2, gamma=1, coef0=1),
    # automatic choice of gamma
    'poly2_auto': lambda x, y: polynomial_kernel(x, Y=y, degree=2, gamma=None, coef0=1),
    # opu kernel
    'opu': lambda x, y, gamma: opu_kernel(x, y, gamma=gamma)
}

### Process the kernels one by one

def train(kernel_matrix, target, alpha):
    clf = KernelRidge(alpha=alpha, kernel="precomputed")
    since = time.time()
    loss = clf.fit(kernel_matrix, target, cg=False, tol=1e-5, lr=1, bs=60000)
    elapsed = time.time() - since
    logger.info('Training Time: {}'.format(elapsed))
    logger.info('Training Loss: {}'.format(loss))
    
    return clf

def test(clf, test_kernel, target):
    predictions = clf.predict(test_kernel)
    score = np.sum(np.equal(np.argmax(predictions, 1), np.argmax(test_labels_bin, 1))) / len(test_data_bin) * 100
    return score

for alpha in alphas:
    for key, kernel_fun in kernels.items():
        logger.info('-----------')
        logger.info('Computing {}-kernel'.format(key))
        logger.info('-----------')
        
        logger.info('Alpha: {}'.format(alpha))
        
        if key == 'opu':
            for gamma in gammas:
                logger.info('Gamma: {}'.format(gamma))
                
                kernel_matrix = kernel_fun(train_data_bin[:N], train_data_bin[:N], gamma=gamma)
                clf = train(kernel_matrix, train_labels_bin[:N], alpha)

                test_kernel = kernel_fun(test_data_bin, train_data_bin[:N], gamma=gamma)
                score = test(clf, test_kernel, test_labels_bin)
                
                logger.info('Score: {}'.format(score))
        else:   
            kernel_matrix = kernel_fun(train_data_bin[:N], train_data_bin[:N])
            clf = train(kernel_matrix, train_labels_bin[:N], alpha)

            test_kernel = kernel_fun(test_data_bin, train_data_bin[:N])
            score = test(clf, test_kernel, test_labels_bin)

            logger.info('Score: {}'.format(score))
        logger.info('-----------\n')
            
print('Done!')
        