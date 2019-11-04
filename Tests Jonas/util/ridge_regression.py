"""
This class solves a regression problem in its primal form
"""
import torch
import numpy as np
from kernels import large_matrix_matrix_product

class RidgeRegression(object):
    def __init__(self, solver='cg', kernel=None):
        super(RidgeRegression, self).__init__()
        self.solver = solver
        self.kernel = kernel

    def solve(self, X, y, alpha):
        if self.kernel is not None:
            self.X = X
            # For kernel ridge, we need to solve the dual problem!
            pass
        else:
            # We solve the standard primal problem: beta = (X' X + alpha*I)^(-1) X' y
            xTx = large_matrix_matrix_product(X.T, X, bias=0, p=1., dtype=torch.FloatTensor)
            xTy = X.T @ y

            self.coef = solve(xTx + alpha*np.eye(len(X)), xTy)


        return self.coef

    def predict(self, X):
        if self.kernel is not None:
            xTx = large_matrix_matrix_product(X, self.X.T, bias=0, p=1., dtype=torch.FloatTensor)
            return xTx @ self.coef
        else:
            return X @ self.coef
    
    def score(self, X, y):
        y_hat = self.predict(X)
        accuracy = np.sum(np.equal(np.argmax(y_hat, 1), np.argmax(y, 1))) / len(X)

        return accuracy
    