import numpy as np
import matplotlib.pyplot as plt
import time

import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms

import util.data

import os
os.environ["CUDA_VISIBLE_DEVICES"]="3"

feature_dir = 'conv_features'

models = [
    {'name':'vgg16_bn', 'model':models.vgg16_bn, 'layers':[13, 23, 33, 43]},
    # {'name':'resnet34', 'model':models.resnet34, 'layers':[1, 2, 3]},
    # {'name':'alexnet', 'model':models.alexnet, 'layers':[2, 5]}
]

datasets = [
    # {'name':'stl10', 'dataset':torchvision.datasets.STL10},
    {'name':'cifar10', 'dataset':torchvision.datasets.CIFAR10}
]

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x

def get_data_loaders(dataset):
    transform = transforms.Compose(
        [# transforms.Resize(224),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
         # cifar10
         # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])

    if dataset['name'] == 'stl10':
        trainset = dataset['dataset'](root='data/' + dataset['name'], split='train',
                                                    download=True, transform=transform)
        testset = dataset['dataset'](root='data/' + dataset['name'], split='test',
                                                   download=True, transform=transform)
    else:
        trainset = dataset['dataset'](root='data/' + dataset['name'], train=True,
                                                    download=True, transform=transform)
        testset = dataset['dataset'](root='data/' + dataset['name'], train=False,
                                                   download=True, transform=transform)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=100,
                                              shuffle=False, num_workers=0)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100,
                                             shuffle=False, num_workers=0)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    
    return trainloader, testloader

use_cuda = torch.cuda.is_available()

def compute_features(loader, model, layer_number, avgpool=False):
    
    model = model['model'](pretrained=True)
    model.eval()
    conv_features = []
    labels = []
    for i, (images, targets) in enumerate(loader):
        if (i*images.shape[0]) % 1000 == 0:
            print('Processed {} images.'.format(i*images.shape[0]))
        
        if use_cuda:
            model = model.cuda()
            images = images.cuda()

        with torch.no_grad():
            if isinstance(model, torchvision.models.resnet.ResNet):
                # in the case of ResNet we only remove the last layer (FC)
                model.fc = Identity()
                model.avgpool = Identity()
                
                if layer_number < 4:
                    model.layer4 = Identity()
                if layer_number < 3:
                    model.layer3 = Identity()
                if layer_number < 2:
                    model.layer2 = Identity()
                
                outputs = model.forward(images)
            
            else:
                # in the case of AlexNet and VGG we can choose to keep avg pooling
                outputs = model.features[:(layer_number+1)](images)
                if avgpool:
                    outputs = model.avgpool(outputs)
            
                
        conv_features.append(outputs.data.cpu().view(images.size(0), -1).numpy())
        labels.append(targets.numpy())
    return np.concatenate(conv_features), np.concatenate(labels)


def imagenet_norm(data, means, stds):
    data = data.astype('float32')

    for i, (mean, std) in enumerate(zip(means, stds)):
        # data has shape (n, channels, dims)
        data[:,i] = (data[:,i] - mean) / std

    return data

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_config', type=str, required=True,
                        help='Path to dataset configuration file')
    parser.add_argument('--device_config', type=str, required=True,
                        help='Path to device configuration file')

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = parse_args()

    print('Loading dataset: {}'.format(args.dataset_config))
    data = util.data.load_dataset(args.dataset_config, binarize_data=False, transform=imagenet_norm)

    print('Loading device config: {}'.format(args.device_config))
    device_config = util.data.load_device_config(args.device_config)

    for dataset in datasets:
        print('Computing features for dataset: {}'.format(dataset['name']))
        
        for model in models:
            for layer in model['layers']:
                print('Current model: {}'.format(model['name']))
                print('Current layer: {}'.format(layer))

                trainloader, testloader = get_data_loaders(dataset)

                train_conv_features, train_labels = compute_features(trainloader, model, layer, avgpool=False)
                test_conv_features, test_labels = compute_features(testloader, model, layer, avgpool=False)

                out_file = os.path.join(feature_dir, dataset['name'], model['name'] + '_' + str(layer))
                label_file = os.path.join(feature_dir, dataset['name'], 'labels.npz')

                np.savez_compressed(out_file + '.npz', train=train_conv_features, test=test_conv_features)
                np.savez_compressed(label_file, train=train_labels, test=test_labels)

                print('Saved features to: {}'.format(out_file + '.npz'))

    #             if not isinstance(model['model'], torchvision.models.resnet.ResNet):
    #                 print('Computing avgpool features...')

    #                 train_conv_features, train_labels = compute_features(trainloader, model, avgpool=True)
    #                 test_conv_features, test_labels = compute_features(testloader, model, avgpool=True)
    #                 np.savez_compressed(out_file + '_avgpool.npz', train=train_conv_features, test=test_conv_features)

    #                 print('Saved features to: {}'.format(out_file + '_avgpool.npz'))
                
    print('Done!')