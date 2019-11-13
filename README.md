# opu-kernel-experiments
Experiments related to the ICASSP submission

### Repository Structure

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