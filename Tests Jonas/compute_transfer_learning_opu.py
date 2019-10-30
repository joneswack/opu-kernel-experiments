import os
import numpy as np

from random_features import OPUModuleNumpy, RBFModuleNumpy

from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelBinarizer
import pandas as pd

import logging
import warnings
import time

save_name = 'ridge_benchmark_transfer_learning_opu_cifar10_full_newscale'
d_out = 10000
l1_ratio = 1.0

logger = logging.getLogger()
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s [%(levelname)s] - %(message)s',
    filename='logs/{}.log'.format(save_name))

logger.info('-------------')
logger.info('New benchmark')

feature_dir = 'conv_features'

rf_configs = [
    {
        'name': 'opu',
        'alphas': [100, 10, 1, 0.1, 0.01, 0],
        'scales': [1, 0.1, 0.01, 0.001, 0.0001, 0.00001],
        # 'dummies': [0, 1, 5, 10, 20, 100],
        'dummies': [0, 1, 10, 100],
        'degrees': [1] # 0.5, 2
    }
#     {
#         'name': 'rbf',
#         'alphas': [100, 10, 1, 0.1, 0.01, 0],
#         'scales': [1, 0.1, 0.01, 0.001, 0.0001, 0.00001]
#     },
#     {
#         'name': 'linear',
#         'alphas': [100, 10, 1, 0.1, 0.01, 0],
#         'scales': [1, 0.1, 0.01, 0.001, 0.0001, 0.00001]
#     }
]

def threshold_binarize(data, threshold):
    data_bin = np.where(data>threshold, 1, 0).astype('uint8')
    return data_bin

def clf_score(clf, data, target):
    predictions = clf.predict(data)
    
    accuracy = np.sum(np.equal(np.argmax(predictions, 1), np.argmax(target, 1))) / len(data)
    
    return accuracy

### Process the kernels one by one

def train(train_data, train_target, test_data, test_target, alphas):

    # model = RidgeClassifier()
    # parameters = {'alpha':alphas}
    # clf = GridSearchCV(model, parameters, cv=2, n_jobs=len(alphas))
    warned = False
    
    start_time = time.time()
    
    test_scores = []
    val_scores = []
    
    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        # clf.fit(train_data, train_target)
        # val_scores = clf.cv_results_['mean_test_score']
        
        for alpha in alphas:
            # validation score
            end_index = int(len(train_data) * 0.8)
            
            clf = RidgeClassifier(alpha=alpha, fit_intercept=False)
            # clf = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, fit_intercept=False, selection='random')
            clf.fit(train_data[:end_index], train_target[:end_index])
            
            val_scores.append(clf.score(train_data[end_index:], train_target[end_index:]))
            # val_scores.append(clf_score(clf, train_data[end_index:],train_target[end_index:]))
        
        for alpha in alphas:
            clf = RidgeClassifier(alpha=alpha, fit_intercept=False)
            # clf = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, fit_intercept=False, selection='random')
            clf.fit(train_data, train_target)
            
            test_scores.append(clf.score(test_data, test_target))
            # test_scores.append(clf_score(clf, test_data, test_target))

#         for warning in caught_warnings:
#             # if warning.category == UnsupportedWarning:
#             print(str(warning.message))
#             warned = True
    train_time = time.time() - start_time
    
    return val_scores, test_scores, caught_warnings, train_time

def evaluate_kernel(df, train_data, train_targets, test_data, test_targets, scales, alphas, kwargs):
    for scale in scales:
        validation_scores, test_scores, warnings, train_time = train(scale * train_data, train_targets, scale * test_data, test_targets, alphas)

        logger.info('Scale: {}'.format(scale))
        logger.info('Training Time: {}'.format(train_time))
        logger.info('Warned: {}'.format(len(warnings)))

        entries = zip(validation_scores, test_scores, config['alphas'])

        for val_score, test_score, alpha in entries:
            logger.info('Alpha: {}'.format(alpha))
            logger.info('Val Score: {}'.format(val_score))
            logger.info('Test Score: {}'.format(test_score))
            
            param_dict = {
                'validation_score': val_score,
                'test_score': test_score,
                'training_time': train_time,
                'alpha': alpha,
                'scale': scale,
                'warnings': len(warnings)
            }
            
            param_dict = {**param_dict, **kwargs}

            df = df.append(param_dict, ignore_index=True)
        
    return df

df = pd.DataFrame()

# datasets = [dataset for dataset in os.listdir(feature_dir) if not dataset.startswith('.')]
datasets = ['cifar10']
# models = ['vgg16_bn_avgpool.npz'] # , 
models = [
    'alexnet_2.npz',
    'alexnet_5.npz',
    'alexnet_avgpool.npz',
    'resnet34_1.npz',
    'resnet34_2.npz',
    'resnet34_3.npz',
    'resnet34.npz',
    'vgg16_bn_13.npz',
    'vgg16_bn_23.npz',
    'vgg16_bn_33.npz',
    'vgg16_bn_43.npz',
    'vgg16_bn_avgpool.npz'
]

num_opu = len(rf_configs[0]['degrees']) * len(rf_configs[0]['dummies'])
# num_opu = 0
# num_rbf = 10 # we always have 10 gammas to test
# num_linear = 1

num_rbf = 0
num_linear = 0
total_number_kernels = (num_opu + num_rbf + num_linear) * len(datasets) * len(models)

i = 0

for dataset in datasets:
    
    print('Looking at the dataset: {}'.format(dataset))
    
    model_dir = os.path.join(feature_dir, dataset, 'models')
    labels_file = os.path.join(feature_dir, dataset, 'labels.npz')
    labels = np.load(labels_file)
    labels_train = labels['train']
    labels_test = labels['test']
    
    # turn labels into binary format in order to use lasso etc.
    # label_binarizer = LabelBinarizer(pos_label=1, neg_label=-1)
    # labels_train = label_binarizer.fit_transform(labels_train).astype('float32')
    # labels_test = label_binarizer.fit_transform(labels_test).astype('float32')
    
    for model in models:
        if model.startswith('.'):
            # skip hidden files
            continue
            
        print('Computing scores for the model: {}'.format(model))
        logger.info('Dataset: {}'.format(dataset))
        logger.info('Model: {}'.format(model))
        
        feature_file = os.path.join(model_dir, model)
        conv_features = np.load(feature_file)
        
        print('Features loaded.')
        print('Features shape: {}'.format(conv_features['train'].shape))
        
        print('Determining binarization threshold...')
        best_score = 0
        best_threshold = 0
        for threshold in np.arange(0, 3, 0.2):
            clf = RidgeClassifier(alpha=1.0, fit_intercept=False)
            # clf = SGDClassifier(loss='squared_loss', penalty='elasticnet', alpha=1.0, l1_ratio=0.5, fit_intercept=False)
            # clf = ElasticNet(alpha=0.0001, l1_ratio=l1_ratio, fit_intercept=False, selection='random')
            features_train = threshold_binarize(conv_features['train'], threshold)
            features_test = threshold_binarize(conv_features['test'], threshold)

            clf.fit(features_train, labels_train)
            score = clf.score(features_test, labels_test)
            
            # score = clf_score(clf, features_test, labels_test)
            
            print('Threshold', threshold, 'Score', score)
            
            if score > best_score:
                best_score = score
                best_threshold = threshold        
        print('Finished threshold determination. Best threshold: {}. Best Score: {}.'.format(best_threshold, best_score))
        
        features_train = threshold_binarize(conv_features['train'], best_threshold)
        features_test = threshold_binarize(conv_features['test'], best_threshold)
        
        print('Comparing random features...')
        d_in = features_train.shape[1]
        
        for config in rf_configs:
            print('Computing random features for {} kernel'.format(config['name']))
            logger.info('-----------')
            logger.info('Kernel: {}'.format(config['name']))
            
            if config['name'] == 'opu':
                for degree in config['degrees']:
                    for dummy in config['dummies']:
                        module = OPUModuleNumpy(d_in + 1, d_out, initial_log_scale='auto', exponent=degree)
                        data = np.vstack([features_train, features_test])
                        data = np.hstack([data, np.ones((data.shape[0], 1)) * dummy]).astype('float32')
                        projection, raw_scale = module.get_projection_and_rawscale(data)
                        del data
                        
                        projection = projection / raw_scale
                        
                        print('Type', type(projection))
                        
                        kwargs = {
                            'kernel': config['name'],
                            'output_dim': d_out,
                            'dummy': dummy,
                            'degree': degree,
                            'model': model,
                            'dataset': dataset,
                            'threshold': best_threshold
                        }
                        
                        logger.info('Degree: {}'.format(degree))
                        logger.info('Dummy: {}'.format(dummy))
                        
                        df = evaluate_kernel(
                                df,
                                projection[:len(features_train)], labels_train,
                                projection[len(features_train):], labels_test,
                                config['scales'], config['alphas'], kwargs
                        )
                        
                        i = i + 1
                        print('Finished {} / {} kernels'.format(i, total_number_kernels))
                        # we update the dataframe after processing one set of seeds
                        df.to_csv(os.path.join('csv', save_name + '.csv'), index=False)
                    
            elif config['name'] == 'rbf':
                # determine rbf gamma values to test
                initial_gamma_str = '{:.20f}'.format(1. / d_in)
                
                # this determines the gamma range: we take the interval around 1./d_in such that we test all values on the same digit after the comma
                non_zeros = [i-1 for i in range(len(initial_gamma_str)) if (initial_gamma_str[i] != '0' and initial_gamma_str[i] != '.')]
                min_val = 10**(-non_zeros[0])
                max_val = 10**(-non_zeros[0]+1)
                
                # start, end, step
                for gamma in np.arange(min_val, max_val, min_val):
                    log_lengthscale = -0.5 * np.log(2*gamma)
                    module = RBFModuleNumpy(d_in, d_out, log_lengthscale_init=log_lengthscale)
                    data = np.vstack([features_train, features_test])
                    projection = module.forward(data)
                    
                    logger.info('Gamma: {}'.format(gamma))
                        
                    kwargs = {
                        'kernel': config['name'],
                        'output_dim': d_out,
                        'gamma': gamma,
                        'model': model,
                        'dataset': dataset,
                        'threshold': best_threshold
                    }
                        
                    df = evaluate_kernel(
                            df,
                            projection[:len(features_train)], labels_train,
                            projection[len(features_train):], labels_test,
                            config['scales'], config['alphas'], kwargs
                    )
                    
                    i = i + 1
                    print('Finished {} / {} kernels'.format(i, total_number_kernels))
                    # we update the dataframe after processing one set of seeds
                    df.to_csv(os.path.join('csv', save_name + '.csv'), index=False)
                    
            elif config['name'] == 'linear':
                data = np.vstack([features_train, features_test])
                
                kwargs = {
                    'kernel': config['name'],
                    'output_dim': d_out,
                    'model': model,
                    'dataset': dataset,
                    'threshold': best_threshold
                }
                        
                df = evaluate_kernel(
                        df,
                        data[:len(features_train)], labels_train,
                        data[len(features_train):], labels_test,
                        config['scales'], config['alphas'], kwargs
                )
                
                i = i + 1
                print('Finished {} / {} kernels'.format(i, total_number_kernels))
                # we update the dataframe after processing one set of seeds
                df.to_csv(os.path.join('csv', save_name + '.csv'), index=False)
                
print('Done!')