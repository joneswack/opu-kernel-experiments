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

    def __init__(self, solver='cg_torch', kernel=None, tol=1e-5, atol=1e-9, max_iterations=15000):
        super(RidgeRegression, self).__init__()
        self.solver = solver
        self.kernel = kernel
        self.tol = tol
        self.atol = atol
        self.max_iterations = max_iterations

    def solve(self, A, b):
        if self.solver == 'cholesky_scipy':
            return scipy.linalg.solve(A, b, sym_pos=True) # , lower=True
        if self.solver == 'cholesky_torch':
            return cholesky_torch(A, b)
        if self.solver == 'cg_torch':
            return cg_torch(A, b, tol=self.tol, atol=self.atol, max_iterations=self.max_iterations)

    def fit(self, X, y, alpha):
        if self.kernel is not None:
            self.X_fit_ = X
            # For kernel ridge, we need to solve the dual form: beta = (X X' + alpha*I)^(-1) y
            kernel = self.kernel(X, Y=None)
            alpha_eye = alpha*np.eye(len(kernel)).astype('float32')
            self.coef = self.solve(kernel + alpha_eye, y)
        else:
            # We solve the standard primal form: beta = (X' X + alpha*I)^(-1) X' y
            xTx = large_matrix_matrix_product(X.T, X, bias=0, p=1., dtype=torch.FloatTensor)
            xTy = X.T @ y
            alpha_eye = alpha*np.eye(len(xTx)).astype('float32')
            self.coef = self.solve(xTx + alpha_eye, xTy)

    def predict(self, X):
        if self.kernel is not None:
            kernel = self.kernel(X, Y=self.X_fit_)
            return kernel @ self.coef
        else:
            return X @ self.coef
    
    def score(self, X, y):
        y_hat = self.predict(X)
        accuracy = np.sum(np.equal(np.argmax(y_hat, 1), np.argmax(y, 1))) / len(X)

        return accuracy
    