import numpy as np
from scipy.integrate import trapezoid
from scipy.optimize import minimize


class Model:
    """
    Logistic Regression với regularization L1 (Lasso penalty).
    Sử dụng thuật toán L-BFGS-B để tối ưu negative log-likelihood.
    """

    def __init__(self, l1_penalty=0.0, l2_penalty=0.0, n_iter=1000, verbose=True):
        self.l1_penalty = l1_penalty
        self.l2_penalty = l2_penalty 
        self.n_iter = n_iter
        self.verbose = verbose
        self.beta = None  # Vector hệ số [intercept, coef...]

    # ===============================
    # CORE FUNCTIONS
    # ===============================
    @staticmethod
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    def negative_log_likelihood(self, beta, X, y):
        z = X @ beta
        ll = np.sum(y * z - np.log1p(np.exp(z)))
        penalty = self.l1_penalty * np.sum(np.abs(beta[1:])) \
                  + 0.5 * self.l2_penalty * np.sum(beta[1:] ** 2)
        return -ll + penalty

    def gradient(self, beta, X, y):
        z = X @ beta
        p = self.sigmoid(z)
        grad = X.T @ (p - y)
        grad[1:] += self.l1_penalty * np.sign(beta[1:]) + self.l2_penalty * beta[1:]
        return grad

    # ===============================
    # TRAINING
    # ===============================
    def fit(self, X, y):
        """
        Huấn luyện logistic regression với L1 penalty.
        X: ndarray [n_samples, n_features]
        y: ndarray [n_samples]
        """
        n_features = X.shape[1]
        beta_init = np.zeros(n_features)

        res = minimize(
            fun=self.negative_log_likelihood,
            x0=beta_init,
            args=(X, y),
            jac=self.gradient,
            method="L-BFGS-B",
            options={"maxiter": self.n_iter, "disp": self.verbose},
        )

        self.beta = res.x
        if self.verbose:
            print(f"Optimization finished. Final loss: {res.fun:.6f}")
            print(f"Intercept: {self.intercept_:.4f}")
            print(f"Coefficients: {self.coef_vector_}")
        return self

    # ===============================
    # PREDICTION
    # ===============================
    def predict_proba(self, X):
        """Dự đoán xác suất (p=1)."""
        return self.sigmoid(X @ self.beta)

    def predict(self, X):
        """Dự đoán nhãn 0/1."""
        return (self.predict_proba(X) >= 0.5).astype(int)

    # ===============================
    # PROPERTIES
    # ===============================
    @property
    def coef_(self):
        return self.beta

    @property
    def coef_vector_(self):
        return self.beta[1:]

    @property
    def intercept_(self):
        return self.beta[0]


# ===============================
# COMPUTE MOMENTS
# ===============================
def compute_moments(pdfs, basis_functions, x_grid):
    """
    Tính moments (hệ số chiếu) của các pdf lên các hàm cơ sở.

    Parameters
    ----------
    pdfs : ndarray [n_samples, n_points]
        Các hàm mật độ cần tính moments.
    basis_functions : list[ndarray]
        Danh sách các hàm cơ sở (mỗi hàm cùng chiều dài x_grid).
    x_grid : ndarray
        Lưới x để tích phân.

    Returns
    -------
    ndarray [n_samples, n_basis]
        Ma trận moments.
    """
    n_samples = pdfs.shape[0]
    n_basis = len(basis_functions)
    moments = np.zeros((n_samples, n_basis))

    for i, pdf in enumerate(pdfs):
        for j, psi in enumerate(basis_functions):
            moments[i, j] = trapezoid(pdf * psi, x=x_grid)
    return moments



def generate_gaussian_basis(x_grid, M, sigma=None):
    centers = np.linspace(x_grid.min(), x_grid.max(), M)
    sigma = sigma or (x_grid.max() - x_grid.min()) / (2 * M)
    return [np.exp(-0.5 * ((x_grid - c) / sigma) ** 2) for c in centers]