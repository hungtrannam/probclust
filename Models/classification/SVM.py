import numpy as np
from scipy.optimize import minimize

class Model:
    def __init__(self, C=1.0, max_iter=1000, verbose=False):
        self.C = C
        self.max_iter = max_iter
        self.verbose = verbose
        self.alpha = None
        self.bias = 0
        self.y_train = None
        self.K_train = None

    def fit(self, K, y):
        """
        K: (n_samples, n_samples), precomputed kernel matrix
        y: (n_samples,), labels {0,1}
        """
        n_samples = len(y)
        y_signed = y * 2 - 1  # Convert {0,1} → {-1,1}
        self.y_train = y_signed
        self.K_train = K

        def objective(alpha):
            return 0.5 * np.sum((alpha * y_signed)[:, None] * (alpha * y_signed)[None, :] * K) - np.sum(alpha)

        def gradient(alpha):
            return (alpha * y_signed) @ (K * y_signed[:, None]) - np.ones_like(alpha)

        def constraint(alpha):
            return np.dot(alpha, y_signed)

        bounds = [(0, self.C) for _ in range(n_samples)]
        constraints = {'type': 'eq', 'fun': constraint}

        result = minimize(
            fun=objective,
            x0=np.zeros(n_samples),
            jac=gradient,
            constraints=constraints,
            bounds=bounds,
            method='SLSQP',
            options={'maxiter': self.max_iter, 'disp': self.verbose}
        )

        self.alpha = result.x
        sv_mask = self.alpha > 1e-5
        active_sv = np.where((self.alpha > 1e-5) & (self.alpha < self.C - 1e-5))[0]

        # Compute bias (average over non-bound SVs)
        if len(active_sv) > 0:
            self.bias = np.mean([
                y_signed[i] - np.sum(self.alpha * y_signed * K[i, :])
                for i in active_sv
            ])
        else:
            self.bias = 0

        if self.verbose:
            print(f"[SVM] Done. Support vectors: {np.sum(sv_mask)}, Bias: {self.bias:.4f}")
        return self

    def decision_function(self, K_new):
        """
        K_new: (n_samples_train, n_samples_test), kernel between train & test
        Return: raw decision values (before sign)
        """
        return np.sum((self.alpha * self.y_train)[:, None] * K_new, axis=0) + self.bias


    def predict_proba(self, K):
        decision_values = K @ self.alpha + self.bias
        probs = 1 / (1 + np.exp(-decision_values))
        return np.vstack([1 - probs, probs]).T 
    
    def predict(self, K_new):
        """
        Predict class labels {0,1} based on sign of decision_function.
        """
        return (self.decision_function(K_new) >= 0).astype(int)
