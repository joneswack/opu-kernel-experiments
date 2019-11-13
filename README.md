# Kernel computations from large-scale random features obtained by Optical Processing Units
Repository for the experiments related to the paper "Kernel computations from large-scale random features obtained by Optical Processing Units". The paper can be found here: https://arxiv.org/abs/1910.09880

## Device settings

## Fashion MNIST experiments
In order to reproduce the Fashion MNIST experiments, follow the following the steps:
- Download the data from: https://github.com/zalandoresearch/fashion-mnist
- Load the data into NumPy using the website's instructions
- Save the data using np.save to the following files: train_data.npy, test_data.npy, train_labels.npy, test_labels.npy
- Adjust config/datasets/fashion_mnist.json to include your data paths
- Optional: Run find_thresholds.py to determine the optimal binary threshold and fill it into config/datasets/fashion_mnist.json. We used 10 as a threshold
- Optional: Run optimize_hyperparameters.py to obtain the optimal hyperparameters for the random projections defined in config/hyperparameter_search/your_config.json. Alternatively, you can simply use one of our config files for the next step
- Optional: Use Hyperparameters-Fashion-MNIST.ipynb to visualize the hyperparameter grids with validation scores
- Run evaluate_features.py to obtain test scores for each feature dimension for the desired random projection configuration stored in config/hyperparameter_config/your_config.json
- Run evaluate_kernels.py to obtain test scores for each kernel configuration stored in config/hyperparameter_config/your_config.json
- Use Plot-Fashion_MNIST.ipynb to produce the plot shown in the paper

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
    ├── optimize_hyperparameters.py         # Script to carry out hyperparameter grid search
    ├── *.sh                                # Shell script to facilitate calls to the scripts above
    └── README.md                           # This file