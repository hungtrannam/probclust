import numpy as np
from utils.integral import int_trapz


class Kernel:
    """
    Tính các kernel giữa 2 PDF (f1, f2) theo Hellinger, L1, L2, Linear, Chi2.
    - f1, f2: np.ndarray cùng shape, 1D (M,) hoặc 2D (H, W) tuỳ Dim.
    - h: bước lưới (1D) hoặc theo định nghĩa trong int_trapz cho 2D.
    """

    def __init__(self, h: float, Dim: int | None = None):
        self.h = h
        self.Dim = Dim

    def BC(self, f1: np.ndarray, f2: np.ndarray) -> float:
        """Hellinger: K = ∫ sqrt(f1 * f2)"""
        return float(int_trapz(np.sqrt(f1 * f2), self.h, Dim=self.Dim))

    def L2(self, f1: np.ndarray, f2: np.ndarray, gamma: float = 1.0) -> float:
        """Gaussian-like (L2): K = exp(-gamma * ∫ (f1 - f2)^2)"""
        d2 = int_trapz((f1 - f2) ** 2, self.h, Dim=self.Dim)
        return float(np.exp(-gamma * d2))

    def L1(self, f1: np.ndarray, f2: np.ndarray, gamma: float = 2.0) -> float:
        """Exponential (L1): K = exp(-gamma * ∫ |f1 - f2|)"""
        d1 = int_trapz(np.abs(f1 - f2), self.h, Dim=self.Dim)
        return float(np.exp(-gamma * d1))

    def LN(self, f1: np.ndarray, f2: np.ndarray) -> float:
        """Linear: K = ∫ f1 * f2"""
        return float(int_trapz(f1 * f2, self.h, Dim=self.Dim))

    def Chi2(self, f1: np.ndarray, f2: np.ndarray, eps: float = 1e-10) -> float:
        """Chi-squared: K = ∫ 2 f1 f2 / (f1 + f2 + eps)"""
        return float(int_trapz(2.0 * f1 * f2 / (f1 + f2 + eps), self.h, Dim=self.Dim))
