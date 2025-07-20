import numpy as np
import matplotlib.pyplot as plt

class Model:
    def __init__(self, lr=0.1, n_iter=1000, l1_penalty=0.01, verbose=True):
        self.lr = lr
        self.n_iter = n_iter
        self.l1_penalty = l1_penalty
        self.verbose = verbose
        self.beta = None

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def compute_log_likelihood(self, X, y):
        z = X @ self.beta
        loglik = np.sum(y * z - np.log(1 + np.exp(z)))
        loglik -= self.l1_penalty * np.sum(np.abs(self.beta[1:]))
        return loglik

    def compute_gradient(self, X, y):
        z = X @ self.beta
        p = self.sigmoid(z)
        grad = X.T @ (y - p)
        grad[1:] -= self.l1_penalty * np.sign(self.beta[1:])
        return grad

    def fit(self, X, y):
        self.beta = np.zeros(X.shape[1])
        for i in range(self.n_iter):
            grad = self.compute_gradient(X, y)
            self.beta += self.lr * grad
            if self.verbose and i % 100 == 0:
                ll = self.compute_log_likelihood(X, y)
                print(f"Iteration {i}, penalized log-likelihood: {ll:.4f}")
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
            moments[i, j] = np.trapezoid(pdf * psi, x_grid)
    return moments


