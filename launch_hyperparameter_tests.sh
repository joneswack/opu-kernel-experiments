#!/bin/sh
python3 "hyperparameter_optimizer.py" --dataset_config "Tests Jonas/config/datasets/fashion_mnist.json" --hyperparameter_config "config/hyperparameters/fashion_mnist_sim_orf.json" --device_config "config/devices/single_gpu.json"