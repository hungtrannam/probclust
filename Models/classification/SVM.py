import numpy as np
from scipy.optimize import minimize


class Model:
    """
    SVM với kernel matrix tiền tính (precomputed kernel).
    Hỗ trợ phân loại nhị phân {0,1}.
    """

    def __init__(self, C=1.0, max_iter=1000, verbose=False):
        self.C = C
        self.max_iter = max_iter
        self.verbose = verbose
        self.alpha = None
        self.bias = 0
        self.y_train = None
        self.K_train = None

    # ===============================
    # TRAINING
    # ===============================
    def fit(self, K, y):
        """
        Huấn luyện SVM với kernel matrix tiền tính.

        Parameters
        ----------
        K : ndarray [n_samples, n_samples]
            Kernel matrix (symmetric, positive semidefinite).
        y : ndarray [n_samples]
            Nhãn {0,1}.
        """
        n_samples = len(y)
        y_signed = y * 2 - 1  # Chuyển {0,1} thành {-1,1}
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

        # Tính bias bằng trung bình trên support vectors không bị chặn
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

    # ===============================
    # DECISION FUNCTION
    # ===============================
    def decision_function(self, K_new):
        """
        Tính giá trị decision function trước khi threshold.

        Parameters
        ----------
        K_new : ndarray [n_train, n_test]
            Kernel matrix giữa tập train và test.

        Returns
        -------
        ndarray [n_test]
            Giá trị decision (chưa lấy sign).
        """
        return np.sum((self.alpha * self.y_train)[:, None] * K_new, axis=0) + self.bias

    # ===============================
    # PREDICT & PROBA
    # ===============================
    def predict_proba(self, K_new):
        """
        Tính xác suất ước lượng từ decision function (qua sigmoid).
        """
        decision_values = self.decision_function(K_new)
        probs = 1 / (1 + np.exp(-decision_values))
        return np.vstack([1 - probs, probs]).T

    def predict(self, K_new):
        """
        Dự đoán nhãn {0,1}.
        """
        return (self.decision_function(K_new) >= 0).astype(int)
