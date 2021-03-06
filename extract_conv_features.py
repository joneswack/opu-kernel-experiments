import numpy as np
import time
import argparse

import util.data

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.models as models

import util.data

import os

models = [
    {'name':'vgg16_bn', 'model':models.vgg16_bn, 'layers':[13, 23, 33, 43, 'final_conv']},
    {'name':'resnet34', 'model':models.resnet34, 'layers':[1, 2, 3, 'final_conv']},
    {'name':'alexnet', 'model':models.alexnet, 'layers':[2, 5, 'final_conv']}
]

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x

def compute_features(device_config, loader, model, layer_number, avgpool=False):
    """
    We use conv features extracted from ImageNet.
    """
    model = model['model'](pretrained=True)
    model.eval()
    conv_features = []

    for i, images in enumerate(loader):
        images = images[0]
        
        if (i*images.shape[0]) % 1000 == 0:
            print('Processed {} images.'.format(i*images.shape[0]))
        
        if not device_config['use_cpu_memory']:
            model = model.to('cuda:{}'.format(device_config['active_gpus'][0]))
            images = images.to('cuda:{}'.format(device_config['active_gpus'][0]))

        with torch.no_grad():

            # 2D upsampling in case of the last conv layer
            if layer_number == 'final_conv':
                # it may make sense to choose an upsampling mode here!
                images = F.interpolate(images, size=224)

            if isinstance(model, torchvision.models.resnet.ResNet):
                # in the case of ResNet we only remove the last layer (FC)
                model.fc = Identity()

                if layer_number == 'final_conv':
                    layer_number = 4
                else:
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
                if layer_number != 'final_conv':
                    outputs = model.features[:(layer_number+1)](images)
                else:
                    outputs = model.features(images)
                if avgpool:
                    outputs = model.avgpool(outputs)
            
                
        conv_features.append(outputs.data.cpu().view(images.size(0), -1).numpy())
    return np.concatenate(conv_features)


def imagenet_norm(data):
    data = data.astype('float32')

    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]

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
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Path to the output directory for the conv features')

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = parse_args()

    print('Loading dataset: {}'.format(args.dataset_config))
    data = util.data.load_dataset(args.dataset_config, binarize_data=False, flatten=False, transform=imagenet_norm)
    data_name, train_data, test_data, train_labels, test_labels = data

    print('Loading device config: {}'.format(args.device_config))
    device_config = util.data.load_device_config(args.device_config)

    train_loader = util.data.get_dataloader(train_data, labels=None, batchsize=100, shuffle=False)
    test_loader = util.data.get_dataloader(test_data, labels=None, batchsize=100, shuffle=False)
        
    for model in models:
        for layer in model['layers']:
            print('Current model: {}'.format(model['name']))
            print('Current layer: {}'.format(layer))

            train_conv_features = compute_features(device_config, train_loader, model, layer, avgpool=False)
            test_conv_features = compute_features(device_config, test_loader, model, layer, avgpool=False)

            out_file = os.path.join(args.output_dir, data_name, model['name'] + '_' + str(layer))
            label_file = os.path.join(args.output_dir, data_name, 'labels')

            np.save(out_file + '_train', train_conv_features)
            np.save(out_file + '_test', test_conv_features)

            np.save(label_file + '_train', train_labels)
            np.save(label_file + '_test', test_labels)

            print('Saved features to: {}'.format(args.output_dir))

            # if not isinstance(model['model'], torchvision.models.resnet.ResNet):
            #     print('Computing avgpool features...')

            #     train_conv_features, train_labels = compute_features(trainloader, model, avgpool=True)
            #     test_conv_features, test_labels = compute_features(testloader, model, avgpool=True)
            #     np.savez_compressed(out_file + '_avgpool.npz', train=train_conv_features, test=test_conv_features)

            #     print('Saved features to: {}'.format(out_file + '_avgpool.npz'))
                
    print('Done!')