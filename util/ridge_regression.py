"""
This class solves a regression problem.

Standard method is to use a cholesky solver
So does sklearn with scipy.linalg.solve (LAPACK cholesky under the hood)
However, this only works well for small linear systems, e.g. n < 10k.

For large systems, we use conjugate gradients with GPU support.
"""
import torch
import numpy as np
import scipy
from .kernels import large_matrix_matrix_product
from .linear_solvers_torch import cholesky as cholesky_torch
from .linear_solvers_torch import cg as cg_torch

class RidgeRegression(object):
    """
    This class represents a Ridge Regression model.

    solver: The solver to solve the linear system defined by the primal/dual form.
    kernel: A kernel function k(X, Y) taking two matrices as arguments.
        If a kernel function is defined, the dual form is solved.
        Otherwise, we solve the primal form.
    tol: Relative error tolerance for conjugate gradients.
    atol: Absolute error tolerance for conjugate gradients.
    max_iterations: Maximum number of iterations for conjugate gradients.
    """

    def __init__(self, device_config, solver='cg_torch', kernel=None, tol=1e-5, atol=1e-9, max_iterations=15000):
        super(RidgeRegression, self).__init__()
        self.device_config = device_config
        self.solver = solver
        self.kernel = kernel
        self.tol = tol
        self.atol = atol
        self.max_iterations = max_iterations

    def solve(self, A, b):
        if self.solver == 'cholesky_scipy':
            if not self.device_config['use_cpu_memory']:
                raise RuntimeError("cholesky_scipy is only available when use_cpu_memory=true")
            solution = scipy.linalg.solve(A.numpy(), b.numpy(), sym_pos=True) # , lower=True
            return torch.from_numpy(solution)
        if self.solver == 'cholesky_torch':
            return cholesky_torch(self.device_config, A, b)
        if self.solver == 'cg_torch':
            solution, _, _ = cg_torch(self.device_config, A, b, tol=self.tol, atol=self.atol, max_iterations=self.max_iterations)
            return solution
        raise RuntimeError("Solver {} not available.".format(self.solver))

    def fit(self, X, y, alpha):
        if self.kernel is not None:
            # For kernel ridge, we need to solve the dual form: beta = (X X' + alpha*I)^(-1) y
            self.X_fit_ = X
            kernel = self.kernel(self.device_config, X, Y=None)

            # add alpha to the diagonal without using additional memory
            kernel.view(-1)[::len(kernel)+1] += alpha

            self.coef = self.solve(kernel + alpha_eye, y)
        else:
            # We solve the standard primal form: beta = (X' X + alpha*I)^(-1) X' y
            if self.device_config['use_cpu_memory']:
                xTx = large_matrix_matrix_product(self.device_config, X.t(), X, bias=0, p=1.)
            else:
                xTx = torch.matmul(X.t(), X)

            # add alpha to the diagonal without using additional memory
            xTx.view(-1)[::len(xTx)+1] += alpha
            
            xTy = torch.matmul(X.t(), y)
            self.coef = self.solve(xTx, xTy)

    def predict(self, X):
        if self.kernel is not None:
            kernel = self.kernel(self.device_config, X, Y=self.X_fit_)
            return torch.matmul(kernel, self.coef)
        else:
            return torch.matmul(X, self.coef)
    
    def score(self, X, y):
        y_hat = self.predict(X)
        
        n_correct = torch.argmax(y_hat, dim=1).eq(torch.argmax(y, dim=1)).sum()
        accuracy = float(n_correct.item()) / len(y)

        return accuracy
    