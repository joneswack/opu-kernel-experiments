"Adapted from sklearn"

from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.utils import check_array, check_X_y
from sklearn.utils.validation import check_is_fitted
from sklearn.linear_model import SGDRegressor
import numpy as np
from scipy import linalg
import torch
import scipy

from multiple_regression_solver import MultipleRegressionSolver

def _solve_kernel(K, y, alpha, sample_weight=None, copy=False, lr=1e-5, epochs=10):
    # dual_coef = inv(X X^t + alpha*Id) y
    n_samples = K.shape[0]
    n_targets = y.shape[1]

    if copy:
        K = K.copy()

    alpha = np.atleast_1d(alpha)
    one_alpha = (alpha == alpha[0]).all()
    has_sw = isinstance(sample_weight, np.ndarray) \
        or sample_weight not in [1.0, None]

    if has_sw:
        # Unlike other solvers, we need to support sample_weight directly
        # because K might be a pre-computed kernel.
        sw = np.sqrt(np.atleast_1d(sample_weight))
        y = y * sw[:, np.newaxis]
        K *= np.outer(sw, sw)

    if one_alpha:
        # Only one penalty, we can solve multi-target problems in one time.
        K.flat[::n_samples + 1] += alpha[0]

#         try:
#             # Note: we must use overwrite_a=False in order to be able to
#             #       use the fall-back solution below in case a LinAlgError
#             #       is raised
#             dual_coef = linalg.solve(K, y, sym_pos=True,
#                                      overwrite_a=False)
#         except np.linalg.LinAlgError:
#             warnings.warn("Singular matrix in solving dual problem. Using "
#                           "least-squares solution instead.")
#             dual_coef = linalg.lstsq(K, y)[0]

        # Solution 1:
        # dual_coef = linalg.lstsq(K, y)[0]
        
        # Solution 2:
        # solver = MultipleRegressionSolver(K, y, batch_size=128, cuda=True)
        # optimizer = torch.optim.Adam(solver.model.parameters(), lr=lr)
        # dual_coef = solver.fit(optimizer, epochs=epochs)
        
        # Solution 3:
        dual_cofs = []
        for dim in range(y.shape[1]):
            print('Running CG for dim', dim)
            coef = scipy.sparse.linalg.cg(K, y[:, dim])[0]
            dual_cofs.append(coef.reshape((-1, 1)))
            print('Current coef shape', coef.shape)
        dual_coef = np.hstack(dual_cofs)

        # K is expensive to compute and store in memory so change it back in
        # case it was user-given.
        K.flat[::n_samples + 1] -= alpha[0]

        if has_sw:
            dual_coef *= sw[:, np.newaxis]

        return dual_coef

class KernelRidge(object):
    def __init__(self, alpha=1, kernel="linear", gamma=None, degree=3, coef0=1, kernel_params=None, lr=1e-5, epochs=10):
        self.alpha = alpha
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.kernel_params = kernel_params
        
    def _get_kernel(self, X, Y=None):
        if callable(self.kernel):
            params = self.kernel_params or {}
        else:
            params = {"gamma": self.gamma,
                      "degree": self.degree,
                      "coef0": self.coef0}
        return pairwise_kernels(X, Y, metric=self.kernel,
                                filter_params=True, **params)
    
    @property
    def _pairwise(self):
        return self.kernel == "precomputed"

    def fit(self, X, y=None, sample_weight=None):
        X, y = check_X_y(X, y, accept_sparse=("csr", "csc"), multi_output=True,
                         y_numeric=True)
        if sample_weight is not None and not isinstance(sample_weight, float):
            sample_weight = check_array(sample_weight, ensure_2d=False)

        K = self._get_kernel(X)
        alpha = np.atleast_1d(self.alpha)

        ravel = False
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)
            ravel = True

        copy = self.kernel == "precomputed"
        
        # Solving cholesky has O(n^3) computation cost
        # We could use a stochastic optimizer or scipy leastsq
        # The following call is adapted
        self.dual_coef_ = _solve_kernel(K, y, alpha, sample_weight, copy, lr=1e-5, epochs=10)
        if ravel:
            self.dual_coef_ = self.dual_coef_.ravel()

        self.X_fit_ = X

        return self
    
    def predict(self, X):
        check_is_fitted(self, ["X_fit_", "dual_coef_"])
        K = self._get_kernel(X, self.X_fit_)
        return np.dot(K, self.dual_coef_)