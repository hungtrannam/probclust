import numpy as np
from scipy.integrate import trapezoid

class Dist:
    """
    Distance utilities for 1-D / 2-D probability densities
    (discrete, trên cùng một lưới đã cho).
    Mọi khoảng cách đều KHỚP công thức lý thuyết.
    """

    def __init__(self, Dim: int = 1, h: float | None = None, grid: np.ndarray | None = None):
        self.Dim = int(Dim)
        if h is None or grid is None:
            raise ValueError("Cung cấp cả h (bước/ diện tích ô) và grid.")
        self.h = float(h) if Dim == 1 else float(h ** 2)  # diện tích ô 2-D
        self.grid = np.asarray(grid, dtype=float)        # (G,) hoặc (G,2)

        # --- cache trục x cho 1-D ---
        if Dim == 1:
            self.x = self.grid.ravel()
        else:
            self.x = None
        self.G = self.grid.shape[0]

    # ------------------------------------------------------------------
    # helpers chung
    # ------------------------------------------------------------------

    def _int(self, y: np.ndarray) -> float:
        """Tích phân trên lưới đã cho."""
        if self.Dim == 1:
            return float(trapezoid(y, self.x))
        return float(trapezoid(y, dx=np.sqrt(self.h)))

    # ------------------------------------------------------------------
    # các khoảng cách riêng
    # ------------------------------------------------------------------
    def L1(self, f1, f2):
        a, b = f1, f2
        return self._int(np.abs(a - b))

    def L2(self, f1, f2):
        a, b = f1, f2
        return np.sqrt(self._int((a - b) ** 2))

    def H(self, f1, f2):
        a, b = f1, f2
        sq1, sq2 = np.sqrt(a), np.sqrt(b)
        return np.sqrt(self._int((sq1 - sq2) ** 2)) / np.sqrt(2)

    def M(self, f1, f2, r: int = 2):
        a, b = f1, f2
        return self._int(np.abs(a ** (1 / r) - b ** (1 / r)) ** r) ** (1 / r)

    def KLinfo(self, f1, f2):
        a, b = f1, f2
        eps = 1e-12
        a = np.clip(a, eps, None)
        b = np.clip(b, eps, None)
        return self._int(a * np.log(a / b))

    def KLdiv(self, f1, f2):
        return 0.5 * (self.KLinfo(f1, f2) + self.KLinfo(f2, f1))

    def BC(self, f1, f2):
        a, b = f1, f2
        bc = self._int(np.sqrt(a * b))
        return -np.log(np.clip(bc, 1e-100, None))

    def CWD(self, f1, f2):
        a, b = f1, f2
        return self._int(np.maximum(a, b)) - 1.0

    def OVL(self, f1, f2):
        a, b = f1, f2
        return 1.0 - self._int(np.minimum(a, b))

    # ------------------------------------------------------------------
    # 2-Wasserstein
    # ------------------------------------------------------------------
    def W2(self, f1, f2):
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

        return np.sqrt(trapezoid((inv_a - inv_b) ** 2, t))