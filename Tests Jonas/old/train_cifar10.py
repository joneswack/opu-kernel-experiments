import numpy as np
import matplotlib.pyplot as plt

import torch
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms

from models.alexnet import alexnet
from model_trainer import ModelTrainer

import os
os.environ["CUDA_VISIBLE_DEVICES"]="3"

transform = transforms.Compose(
    [
     transforms.RandomHorizontalFlip(),
     transforms.Resize(224),
     transforms.ToTensor(),
     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])

trainset = torchvision.datasets.CIFAR10(root='data/CIFAR10', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=100,
                                          shuffle=True, num_workers=0)

testset = torchvision.datasets.CIFAR10(root='data/CIFAR10', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=100,
                                         shuffle=False, num_workers=0)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')



# Load the model
model = alexnet(pretrained=False, num_classes=10)

model_trainer = ModelTrainer('alexnet_cifar_resized', model, trainloader, testloader,
            lr=1e-3, epochs=30, use_gpu=True)

model_trainer.run()

print('Finished!')