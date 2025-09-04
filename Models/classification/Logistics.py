import numpy as np
from scipy.integrate import trapezoid
from scipy.optimize import minimize
from sklearn.metrics import confusion_matrix, roc_auc_score, precision_score, recall_score, f1_score
from scipy.stats import chi2

class Model:
    """
    Logistic Regression với regularization L1 (Lasso penalty).
    Sử dụng thuật toán L-BFGS-B để tối ưu negative log-likelihood.
    """

    def __init__(self, l1_penalty=0.0, l2_penalty=0.0, n_iter=1000, verbose=False):
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
        self.X_train = X
        self.y_train = y
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


    # ---------- Statistics ----------
    def summary(self, X=None, y=None):
        """
        Return a pretty-formatted string like statsmodels.summary().
        """
        if X is None:
            X = self.X_train
        if y is None:
            y = self.y_train

        y_pred = self.predict(X)
        proba   = self.predict_proba(X)

        n, k = X.shape
        nll_final = self.negative_log_likelihood(self.beta, X, y)
        ll_null   = np.sum(y*np.log(np.mean(y)+1e-12) + (1-y)*np.log(1-np.mean(y)+1e-12))
        pseudo_r2 = 1 - nll_final / (-ll_null)

        # --- metrics
        aic = 2*nll_final + 2*k
        bic = 2*nll_final + k*np.log(n)
        nonzero = np.sum(np.abs(self.beta[1:]) > 1e-6)

        cm = confusion_matrix(y, y_pred)
        acc  = np.mean(y_pred == y)
        prec = precision_score(y, y_pred, zero_division=0)
        rec  = recall_score(y, y_pred, zero_division=0)
        f1   = f1_score(y, y_pred, zero_division=0)
        auc  = roc_auc_score(y, proba) if len(np.unique(y))==2 else None

        # --- build string
        summary_str  = "="*78 + "\n"
        summary_str += "Logistic Regression Summary\n"
        summary_str += "="*78 + "\n\n"

        # top panel
        summary_str += f"{'No. Observations:':<25}{n}\n"
        summary_str += f"{'Model:':<25}Logistic (L-BFGS-B)\n"
        summary_str += f"{'Log-Likelihood:':<25}{-nll_final:.4f}\t"
        summary_str += f"{'Pseudo R-squared:':<25}{pseudo_r2:.4f}\n"
        summary_str += f"{'AIC:':<25}{aic:.4f}\t"
        summary_str += f"{'BIC:':<25}{bic:.4f}\n"
        summary_str += f"{'No. Features:':<25}{k-1}\t\t"
        summary_str += f"{'Non-zero Coeffs:':<25}{nonzero}\n"
        summary_str += f"{'Accuracy:':<25}{acc:.4f}\t\t"
        summary_str += f"{'Precision:':<25}{prec:.4f}\n"
        summary_str += f"{'Recall:':<25}{rec:.4f}\t\t"
        summary_str += f"{'F1-Score:':<25}{f1:.4f}\n"
        if auc is not None:
            summary_str += f"{'ROC-AUC:':<25}{auc:.4f}\n"
        summary_str += "\n"

        # ---------- Coefficients table ----------
        summary_str += "-" * 78 + "\n"
        summary_str += f"{'Variable':<12}{'Coef':>12}{'|Coef|':>12}{'Std.Err':>12}{'z':>10}{'p>|z|':>12}\n"
        summary_str += "-" * 78 + "\n"

        # Tính t-stats, SE, p-values
        t_stats, p_vals, se = self.wald_test(X, y)

        # Dòng Intercept
        summary_str += (f"{'Intercept':<12}"
                        f"{self.beta[0]:>12.4f}"
                        f"{abs(self.beta[0]):>12.4f}"
                        f"{se[0]:>12.4f}"
                        f"{t_stats[0]:>10.3f}"
                        f"{p_vals[0]:>12.4f}\n")

        # Các hệ số còn lại
        for i, (coef, s, t, p) in enumerate(zip(self.beta[1:], se[1:], t_stats[1:], p_vals[1:]), start=1):
            summary_str += (f"{f'X{i}':<12}"
                            f"{coef:>12.4f}"
                            f"{abs(coef):>12.4f}"
                            f"{s:>12.4f}"
                            f"{t:>10.3f}"
                            f"{p:>12.4f}\n")

        # # confusion matrix
        # summary_str += "\n" + "-"*78 + "\n"
        # summary_str += "Confusion Matrix\n"
        # summary_str += "-"*78 + "\n"
        # summary_str += f"{'':>6}{'Pred 0':>8}{'Pred 1':>8}\n"
        # summary_str += f"{'True class 0:':>6}{cm[0,0]:>8}{cm[0,1]:>8}\n"
        # summary_str += f"{'True class 1:':>6}{cm[1,0]:>8}{cm[1,1]:>8}\n"
        # summary_str += "="*78 + "\n"
        
        return summary_str
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
    
    def wald_test(self, X, y):
        from scipy.stats import norm

        proba = np.clip(self.predict_proba(X), 1e-5, 1-1e-5)
        W = np.diag(proba * (1 - proba))
        # Dùng pseudo-inverse để tránh suy biến
        ridge = 1e-8 * np.eye(X.shape[1])
        cov = np.linalg.pinv(X.T @ W @ X + ridge)
        se = np.sqrt(np.clip(np.diag(cov), 0, None))

        t_stats = np.divide(self.beta, se, out=np.zeros_like(self.beta), where=se != 0)
        p_values = 2 * (1 - norm.cdf(np.abs(t_stats)))
        return t_stats, p_values, se


# ===============================
# COMPUTE MOMENTS
# ===============================
import numpy as np
from scipy.integrate import trapezoid
from scipy.interpolate import BSpline

class FLR:
    """
    Functional Logistic Regression utilities:
    – moment projection
    – ready-to-use basis generators
    """

    # ---------- Core utility ----------
    @staticmethod
    def compute_moments(pdfs: np.ndarray,
                        basis_functions: list[np.ndarray],
                        x_grid: np.ndarray) -> np.ndarray:
        """
        Project each pdf onto the given basis.

        Parameters
        ----------
        pdfs : ndarray, shape (n_samples, n_points)
        basis_functions : list[ndarray], len = n_basis
        x_grid : ndarray, shape (n_points,)

        Returns
        -------
        ndarray, shape (n_samples, n_basis)
        """
        n_samples, n_basis = pdfs.shape[0], len(basis_functions)
        moments = np.empty((n_samples, n_basis))

        for j, psi in enumerate(basis_functions):
            moments[:, j] = trapezoid(pdfs * psi, x=x_grid, axis=1)
        return moments

    # ---------- Basis factories ----------
    @staticmethod
    def generate_gaussian_basis(x_grid: np.ndarray,
                                M: int,
                                sigma: float | None = None) -> list[np.ndarray]:
        centers = np.linspace(x_grid.min(), x_grid.max(), M)
        sigma = sigma or (np.ptp(x_grid) / (2 * M)) 
        return [np.exp(-0.5 * ((x_grid - c) / sigma) ** 2) for c in centers]

    @staticmethod
    def generate_fourier_basis(x_grid: np.ndarray, M: int) -> list[np.ndarray]:
        basis = [np.ones_like(x_grid)]
        period = np.ptp(x_grid)
        for k in range(1, M // 2 + 1):
            basis.extend([
                np.sin(2 * np.pi * k * x_grid / period),
                np.cos(2 * np.pi * k * x_grid / period)
            ])
        return basis[:M]

    @staticmethod
    def generate_bspline_basis(x_grid: np.ndarray,
                            M: int,
                            degree: int = 3) -> list[np.ndarray]:
        from scipy.interpolate import BSpline

        knots = np.linspace(x_grid.min(), x_grid.max(), M - degree + 1)
        knots = np.r_[[knots[0]] * degree, knots, [knots[-1]] * degree]

        basis = []
        for i in range(M):
            c = np.zeros(M)
            c[i] = 1.0
            spline = BSpline(knots, c, degree, extrapolate=True)  # hoặc True
            basis.append(spline(x_grid))
        return basis

    @staticmethod
    def generate_sigmoid_basis(x_grid: np.ndarray, M: int) -> list[np.ndarray]:
        centers = np.linspace(x_grid.min(), x_grid.max(), M)
        slope = np.ptp(x_grid) / M
        return [1 / (1 + np.exp(-(x_grid - c) / slope)) for c in centers]