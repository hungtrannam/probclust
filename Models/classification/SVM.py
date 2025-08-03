import numpy as np
from scipy.optimize import minimize

class Model:
    """
    SVM với kernel tiền tính (precomputed kernel), huấn luyện bằng tối ưu lồi.
    Hỗ trợ dự đoán xác suất qua Platt scaling.
    """

    def __init__(self, C=1.0, max_iter=1000, tol=1e-6, probability=True, verbose=False):
        self.C = C
        self.max_iter = max_iter
        self.tol = tol
        self.probability = probability
        self.verbose = verbose

        # Tham số sau khi huấn luyện
        self.alpha = None
        self.bias = 0.0
        self.support_ = None
        self.dual_coef_ = None
        self.intercept_ = None
        self.support_vectors_ = None
        self.y_train = None
        self.K_train = None

        # Tham số sigmoid cho Platt scaling
        self.A = -1.0
        self.B = 0.0

    def fit(self, K, y):
        """
        Huấn luyện SVM với kernel đã tiền tính.
        Parameters
        ----------
        K : ndarray [n_samples, n_samples] - ma trận kernel (Gram)
        y : ndarray [n_samples] - nhãn {0,1}
        """
        n = len(y)
        y_signed = y * 2 - 1  # {0,1} → {-1,1}
        self.y_train = y_signed
        self.K_train = K

        # Q = y_i y_j K_ij
        Q = (y_signed[:, None] * y_signed[None, :]) * K

        def objective(alpha):
            return 0.5 * alpha @ Q @ alpha - np.sum(alpha)

        def gradient(alpha):
            return Q @ alpha - np.ones_like(alpha)

        def constraint(alpha):
            return np.dot(alpha, y_signed)

        bounds = [(0, self.C)] * n
        constraints = {'type': 'eq', 'fun': constraint}

        result = minimize(
            fun=objective,
            x0=np.zeros(n),
            jac=gradient,
            bounds=bounds,
            constraints=constraints,
            method='SLSQP',
            options={'maxiter': self.max_iter, 'disp': self.verbose}
        )

        alpha = result.x
        self.alpha = alpha

        # Lưu các chỉ số SV
        sv_mask = alpha > self.tol
        self.support_ = np.where(sv_mask)[0]
        self.dual_coef_ = (alpha[sv_mask] * y_signed[sv_mask])[None, :]
        self.support_vectors_ = self.support_

        # Tính bias b
        free_sv = (alpha > self.tol) & (alpha < self.C - self.tol)
        if np.any(free_sv):
            self.bias = np.mean([
                y_signed[i] - np.sum(alpha * y_signed * K[i, :])
                for i in np.where(free_sv)[0]
            ])
        else:
            i = np.argmax(alpha)
            self.bias = y_signed[i] - np.sum(alpha * y_signed * K[i, :])

        self.intercept_ = np.array([self.bias])

        # Huấn luyện sigmoid nếu cần
        if self.probability:
            f_train = self.decision_function(K)
            y_binary = (y_signed + 1) // 2
            self.A, self.B = self._fit_sigmoid(f_train, y_binary)

        if self.verbose:
            print(f"[SVM] Done. Support Vectors: {len(self.support_)} | Bias: {self.bias:.4f}")

        return self

    def _fit_sigmoid(self, f, y):
        """
        Huấn luyện sigmoid: 1 / (1 + exp(A*f + B)) theo Platt scaling
        """
        def loss(params):
            A, B = params
            probs = 1 / (1 + np.exp(A * f + B))
            probs = np.clip(probs, 1e-12, 1 - 1e-12)
            return -np.mean(y * np.log(probs) + (1 - y) * np.log(1 - probs))

        res = minimize(loss, x0=[-1.0, 0.0], method='L-BFGS-B')
        return res.x

    def decision_function(self, K_new):
        """
        Trả về f(x) = ∑ α_i y_i K(x, x_i) + b
        Parameters
        ----------
        K_new : ndarray [n_test, n_train]
        """
        return np.sum((self.alpha * self.y_train)[None, :] * K_new, axis=1) + self.bias

    def predict(self, K_new):
        """
        Dự đoán nhãn: 0 hoặc 1
        """
        return (self.decision_function(K_new) >= 0).astype(int)

    def predict_proba(self, K_new):
        """
        Trả về xác suất nhãn bằng hàm sigmoid huấn luyện tay.
        """
        if not self.probability:
            raise ValueError("Model không được huấn luyện với probability=True")
        f = self.decision_function(K_new)
        probs = 1 / (1 + np.exp(self.A * f + self.B))
        return np.vstack([1 - probs, probs]).T
