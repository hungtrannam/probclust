import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import trapezoid

from scipy.optimize import minimize

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import trapezoid
from scipy.optimize import minimize

class Model:
    def __init__(self, l1_penalty=0.0, n_iter=1000, verbose=True):
        self.l1_penalty = l1_penalty
        self.n_iter = n_iter
        self.verbose = verbose
        self.beta = None

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def negative_log_likelihood(self, beta, X, y):
        z = X @ beta
        ll = np.sum(y * z - np.log(1 + np.exp(z)))
        penalty = self.l1_penalty * np.sum(np.abs(beta[1:]))
        loss = -ll + penalty  # dấu âm vì minimize
        if self.verbose:
            print(f"[Objective] Loss: {loss:.6f}")
        return loss

    def gradient(self, beta, X, y):
        z = X @ beta
        p = self.sigmoid(z)
        grad = X.T @ (y - p)
        grad[1:] -= self.l1_penalty * np.sign(beta[1:])
        grad = -grad  # dấu âm vì minimize
        if self.verbose:
            print(f"[Gradient] Norm: {np.linalg.norm(grad):.6f}")
        return grad

    def fit(self, X, y):
        n_features = X.shape[1]
        beta_init = np.zeros(n_features)
        res = minimize(
            fun=self.negative_log_likelihood,
            x0=beta_init,
            args=(X, y),
            jac=self.gradient,
            method='L-BFGS-B',
            options={'maxiter': self.n_iter, 'disp': self.verbose}
        )
        self.beta = res.x
        if self.verbose:
            print(f"Optimization finished. Final loss: {res.fun:.6f}")
            print(f"Intercept: {self.intercept_:.4f}")
            print(f"Coefficients: {self.coef_}")
        return self

    def predict_proba(self, X):
        return self.sigmoid(X @ self.beta)

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)

    @property
    def coef_(self):
        return self.beta[1:]

    @property
    def intercept_(self):
        return self.beta[0]


def compute_moments(pdfs, basis_functions, x_grid):
    moments = np.zeros((pdfs.shape[0], len(basis_functions)))
    for i, pdf in enumerate(pdfs):
        for j, psi in enumerate(basis_functions):
            val = trapezoid(pdf * psi, x=x_grid)
            moments[i, j] = val
    return moments


