import numpy as np
from scipy.integrate import trapezoid   # <-- import đúng từ SciPy

class Dist:
    """
    Distance utilities for 1-D / 2-D probability densities
    (discrete, trên cùng một lưới đã cho).
    """

    def __init__(self, Dim: int = 1, h: float | None = None, grid: np.ndarray | None = None):
        self.Dim = int(Dim)
        if h is None or grid is None:
            raise ValueError("Cần cung cấp cả h (bước) và grid.")
        self.h = float(h) if Dim == 1 else float(h**2)  # diện tích ô 2D
        self.grid = np.asarray(grid, dtype=float)
        if Dim == 1:
            self.x = self.grid.ravel()
            self.G = self.grid.shape[0]
        else:
            self.x = None
            self.G = self.grid.shape[0]

    # ===========================
    # Helpers
    # ===========================
    def _int(self, y: np.ndarray) -> float:
        if self.Dim == 1:
            from scipy.integrate import trapezoid
            return float(trapezoid(y, self.x))
        else:
            # trong 2D: tích phân ≈ tổng * diện tích ô
            return float(np.sum(y) * self.h)

    # ===========================
    # Distances
    # ===========================
    def L1(self, f1, f2):
        return self._int(np.abs(f1 - f2))

    def L2(self, f1, f2):
        return np.sqrt(self._int((f1 - f2)**2))

    def H(self, f1, f2):
        sq1, sq2 = np.sqrt(f1), np.sqrt(f2)
        return np.sqrt(0.5 * self._int((sq1 - sq2)**2))

    def M(self, f1, f2, r: int = 2):
        return self._int(np.abs(f1**(1/r) - f2**(1/r))**r)**(1/r)

    def KLinfo(self, f1, f2):
        eps = 1e-12
        a = np.clip(f1, eps, None)
        b = np.clip(f2, eps, None)
        return self._int(a * np.log(a / b))

    def KLdiv(self, f1, f2):
        return 0.5 * (self.KLinfo(f1, f2) + self.KLinfo(f2, f1))

    def BC(self, f1, f2):
        bc = self._int(np.sqrt(f1 * f2))
        bc = np.clip(bc, 1e-100, 1.0)
        return -np.log(bc)

    def CWD(self, f1, f2):
        return self._int(np.maximum(f1, f2)) - 1.0

    def OVL(self, f1, f2):
        return 1.0 - self._int(np.minimum(f1, f2))


    # ------------------------------------------------------------------
    # 2-Wasserstein
    # ------------------------------------------------------------------


    def sinkhorn(a, b, M, reg=1e-1, n_iter=50):
        """
        Sinkhorn algorithm for OT distance
        a, b: (n,) probability vectors
        M: (n,n) cost matrix
        reg: regularization λ
        """
        K = np.exp(-M / reg)  # kernel matrix
        u = np.ones_like(a)
        v = np.ones_like(b)

        for _ in range(n_iter):
            u = a / (K @ v + 1e-12)
            v = b / (K.T @ u + 1e-12)

        P = np.outer(u, v) * K
        return np.sum(P * M)
    

    def W2(self, f1, f2):
        if self.Dim ==1:
            a, b = f1, f2

            cdf_a = np.cumsum(a) * self.h
            cdf_b = np.cumsum(b) * self.h

            # bảo vệ cdf không giảm và nằm trong [0,1]
            cdf_a = np.clip(cdf_a, 0, 1)
            cdf_b = np.clip(cdf_b, 0, 1)

            # tạo trục đồng nhất
            t = np.linspace(0, 1, self.G)
            inv_a = np.interp(t, cdf_a, self.x)
            inv_b = np.interp(t, cdf_b, self.x)
            return np.sqrt(self._int((inv_a - inv_b)**2))    
        
        elif self.Dim == 2:
            a = f1.ravel()
            b = f2.ravel()
            a = a / (a.sum() + 1e-12)
            b = b / (b.sum() + 1e-12)
            return np.sqrt(self.sinkhorn(a, b, self.M))