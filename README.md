# Kernel computations from large-scale random features obtained by Optical Processing Units
This is the repository for the experiments related to the paper "Kernel computations from large-scale random features obtained by Optical Processing Units". The paper can be found here: https://arxiv.org/abs/1910.09880

Apart from allowing one to reproduce the paper's result, the repository also contains useful code for large-scale random feature and kernel computations. This makes it possible to solving Ridge Regression problems with up to 100K random features or datapoints without exceeding 32GB of CPU RAM. In addition, multiple GPUs can be used to accelerate large matrix-matrix products and euclidean distance computations.

## Memory-efficient large-scale computations
When solving Ridge Regression for large-scale random features (more than 10K dimensions) or when computing exact kernel matrices, we make sure to reduce GPU memory usage. The two most important functions are large_matrix_matrix_product and large_pairwise_distances in util/kernels.py.

large_matrix_matrix_product computes the matrix product between X and Y. large_pairwise_distances computes the pairwise euclidean distances between the rows of X and the rows of Y.

In both methods we split Y into column-wise chunks according to the device configuration (see next section). Partial matrix products/pairwise distances are then computed between row-batches of X and column-chunks of Y. This computation can then take place on a GPU without exceeding its memory. The result is stored back to CPU memory. All partial results are combined in the end to yield the full matrix product/pairwise distance result.

The batch processing of the rows of X make use of PyTorch nn.DataParallel to enable support for multiple GPUs.

All kernels defined in util/kernels.py make use of the functions defined above. The random features in util/random_features.py use these large scale matrix products for large projection dimensions.

### Device settings
The directory config/devices allows to store GPU/CPU settings in .json format that are loaded together with each experiment. The following parameters can be set:
- "active_gpus": List of GPU ids addressed by cuda:{gpu_id}, e.g. [1,2,3] for GPUs 1, 2 and 3
- "use_cpu_memory": Flag whether to store intermediate computations in CPU memory. If set to false, data will be constantly kept in GPU. For CPU only usage, this flag should also be set to true.
- "memory_limit_gb": Memory limit for Y in GB to be stored in GPU at once without chunking
- "preferred_chunk_size_gb": Chunk size for Y in GB to be stored at once in case memory_limit_gb is exceeded
- "matrix_prod_batch_size": Number of datapoints in X to be gathered into a mini-batch

## Fashion MNIST and CIFAR10 experiments
In order to reproduce the experiments, follow the following the steps:
- Fashion MNIST: Download the data from: https://github.com/zalandoresearch/fashion-mnist
- CIFAR10: Download the data from: https://www.cs.toronto.edu/~kriz/cifar.html
- Load the data into NumPy using the website's instructions
- Save the data using np.save to the following files: train_data.npy, test_data.npy, train_labels.npy, test_labels.npy
- Fashion MNIST: Adjust config/datasets/fashion_mnist.json to include your data paths
- CIFAR10: Adjust config/datasets/cifar10.json to include your data paths
- CIFAR10: Run extract_conv_features.py to extract convolutional features from pretrained CNNs. For each feature set, create a .json dataset configuration file in config/datasets
- Run find_thresholds.py to determine the optimal binary threshold for every dataset and fill it into the respective file. E.g., we used 10 as a threshold for Fashion MNIST
- Optional: Run optimize_hyperparameters.py to obtain the optimal hyperparameters for the random projections defined in config/hyperparameter_search/your_config.json. Alternatively, you can simply use one of our config files for the next step
- Optional: Use Hyperparameters-Fashion-MNIST.ipynb to visualize the hyperparameter grids with validation scores for Fashion MNIST
- Run evaluate_features.py to obtain test scores for each feature dimension for the desired random projection configuration stored in config/hyperparameter_config/your_config.json
- Run evaluate_kernels.py to obtain test scores for each kernel configuration stored in config/hyperparameter_config/your_config.json
- Fashion MNIST: Use Plot-Fashion-MNIST.ipynb to produce the plot shown in the paper

**PLEASE NOTE**: For projection dimensions larger than 10 000 and for exact kernel evaluation, we recommend to activate the option "use_cpu_memory" in your device configuration, even if you use a GPU. This way intermediate results are stored in CPU memory and the GPUs are only used for partial results.
This is particularly important for evaluate_features.py and evaluate_kernels.py. optimize_hyperparameters.py can be run on a single GPU with "use_cpu_memory" set to false. This accelerates the grid search a lot because all results are kept in GPU memory at any time. In this case, ~50x speed up can be achieved compared to CPU.

## Repository Structure

    .
    ├── config                              # .json configuration files
    │   ├── datasets                        # Paths and binarization thresholds for every dataset
    │   ├── devices                         # GPU ids and memory constraints
    │   ├── hyperparameter_config           # Kernel hyperparameters used for feature and kernel evaluation
    │   └── hyperparameter_search           # Search ranges for hyperparameters
    ├── csv                                 # .csv output files for every experiment
    │   ├── feature_evaluation              # Evaluation of random features for different projection dimensions
    │   ├── hyperparameter_optimization     # Hyperparameter grid search output for 10K projection dimensions
    │   ├── kernel_evaluation               # Evaluation of kernels
    │   └── thresholds                      # Evaluation of thresholds
    ├── figures                             # .pdf figures to be used in the paper
    ├── logs                                # Same as csv for .log files
    │   ├── feature_evaluation
    │   ├── hyperparameter_optimization
    │   ├── kernel_evaluation
    │   └── thresholds
    ├── opu_features                        # .npy files for precomputed opu feature dumps
    ├── opu_gpu_energy                      # Everything related to the energy consumption analysis
    ├── util                                # All python code that is needed for the experiments
    │   ├── data.py                         # Helper functions for data, configs and logging
    │   ├── kernels.py                      # Memory-efficient functions to compute exact kernels
    │   ├── linear_solvers_torch.py         # Cholesky and conjugate gradient solvers in PyTorch
    │   ├── random_features_numpy.py        # Sim. optical and rbf fourier random features in NumPy (not used in the code)
    │   ├── random_features.py              # (Sim.) Optical and rbf fourier random features in PyTorch
    │   └── ridge_regression.py             # Ridge Regression model in PyTorch using solvers specified above
    ├── evaluate_features.py                # Script to carry out feature evaluation over projection dimensions
    ├── evaluate_kernels.py                 # Script to carry out evaluation for exact kernels
    ├── extract_conv_features.py            # Script to extract conv features for models pretrained on ImageNet
    ├── extract_opu_features.py             # Script to extract features using the real OPU device
    ├── find_thresholds.py                  # Script to find binarization thresholds for given datasets
    ├── Hyperparameters-Fashion-MNIST.ipynb # Visualization of hyperparameter grid search for Fashion MNIST
    ├── Plot-Fashion-MNIST.ipynb            # Plot of the Fashion MNIST projections and their test scores
    ├── optimize_hyperparameters.py         # Script to carry out hyperparameter grid search
    ├── *.sh                                # Shell script to facilitate calls to the scripts above
    └── README.md                           # This file