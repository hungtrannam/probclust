from sklearn.metrics import f1_score
from scipy.optimize import minimize
import numpy as np

class Model:
    """
    Logistic Regression tối ưu trực tiếp F1-score (Okabe-style).
    """

    def __init__(self, l1_penalty=0.0, n_iter=1000, verbose=True):
        super().__init__(l1_penalty, n_iter, verbose)
        self.threshold = 0.5  # threshold mặc định

    def f1_loss(self, beta, X, y):
        """
        Hàm loss = -F1-score.
        """
        probs = self.sigmoid(X @ beta)
        preds = (probs >= 0.5).astype(int)
        score = f1_score(y, preds)
        loss = -score + self.l1_penalty * np.sum(np.abs(beta[1:]))
        if self.verbose:
            print(f"[F1 Loss] -F1: {loss:.6f}")
        return loss

    def fit(self, X, y):
        n_features = X.shape[1]
        beta_init = np.zeros(n_features)
        res = minimize(
            fun=self.f1_loss,
            x0=beta_init,
            args=(X, y),
            method="Nelder-Mead",  # Không dùng gradient vì F1 không khả vi
            options={"maxiter": self.n_iter, "disp": self.verbose}
        )
        self.beta = res.x
        return self

    def optimize_threshold(self, X_val, y_val):
        """
        Tìm threshold tốt nhất trên tập validation.
        """
        probs = self.sigmoid(X_val @ self.beta)
        best_thr, best_f1 = 0.5, 0
        for t in np.linspace(0, 1, 101):
            preds = (probs >= t).astype(int)
            score = f1_score(y_val, preds)
            if score > best_f1:
                best_f1, best_thr = score, t
        self.threshold = best_thr
        if self.verbose:
            print(f"Best threshold: {best_thr:.3f} with F1: {best_f1:.4f}")
        return self

    def predict(self, X):
        return (self.predict_proba(X) >= self.threshold).astype(int)
