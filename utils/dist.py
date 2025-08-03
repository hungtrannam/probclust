import numpy as np
from utils.integral import int_trapz

class Dist:
    """
    Class to handle distance calculations between probability distributions.
    Supports both 1D and 2D (flattened) PDFs.
    """

    def __init__(self, f1, f2, Dim=1, h=None, grid=None):
        self.Dim = Dim
        self.h = h if Dim ==1 else h**2
        self.grid = grid

        # Xử lý khi là 2D: reshape f1, f2 từ vector về (n_x, n_y)
        if self.Dim == 2:
            if grid is None or grid.ndim != 2 or grid.shape[1] != 2:
                raise ValueError("Với Dim=2, cần cung cấp grid có shape (n_x*n_y, 2)")
            
            n_x = len(np.unique(grid[:, 0]))
            n_y = len(np.unique(grid[:, 1]))
            expected_size = n_x * n_y

            if f1.size != expected_size or f2.size != expected_size:
                raise ValueError(f"Size của f1 hoặc f2 ({f1.size}) không khớp với grid ({n_x}×{n_y})")

            self.f1 = f1.reshape(n_x, n_y)
            self.f2 = f2.reshape(n_x, n_y)
            self.n_x, self.n_y = n_x, n_y
            self.x = None  # Không dùng trong 2D
        else:
            # 1D
            if len(f1) != len(f2):
                raise ValueError("Distributions must have the same length.")
            self.f1 = f1
            self.f2 = f2
            self.x = grid

    def __call__(self):
        return self.calculate_distance()

    def L1(self):
        return int_trapz(np.abs(self.f1 - self.f2), Dim=self.Dim, h=self.h)

    def L2(self):
        return np.sqrt(int_trapz((self.f1 - self.f2) ** 2, Dim=self.Dim, h=self.h))

    def H(self):
        sqrt_f1 = np.sqrt(self.f1)
        sqrt_f2 = np.sqrt(self.f2)
        return np.sqrt(0.5) * np.sqrt(int_trapz((sqrt_f1 - sqrt_f2)**2, Dim=self.Dim, h=self.h))

    def BC(self):
        bc = int_trapz(np.sqrt(self.f1 * self.f2), Dim=self.Dim, h=self.h) + 1e-100
        return - np.log(bc)

    def W2(self):
        if self.Dim == 1:
            # 1D: dùng CDF và inverse CDF
            cdf_f1 = np.cumsum(self.f1) * self.h
            cdf_f2 = np.cumsum(self.f2) * self.h
            t_vals = np.linspace(0, 1, len(cdf_f1))
            inv_f1 = np.interp(t_vals, cdf_f1, self.x)
            inv_f2 = np.interp(t_vals, cdf_f2, self.x)
            return np.sqrt(np.sum((inv_f1 - inv_f2) ** 2) * (1 / len(t_vals)))

        elif self.Dim == 2:
            # 2D: dùng POT (Python Optimal Transport)
            try:
                import ot
            except ImportError:
                raise ImportError("Cần cài đặt thư viện POT: pip install POT")

            f1_flat = self.f1.flatten()
            f2_flat = self.f2.flatten()
            f1_flat /= f1_flat.sum()
            f2_flat /= f2_flat.sum()

            M = ot.dist(self.grid, self.grid, metric='euclidean')**2  # cost matrix
            w2 = ot.emd2(f1_flat, f2_flat, M, numItermax = 1e10)
            return np.sqrt(w2)

        else:
            raise NotImplementedError("Chỉ hỗ trợ W2 cho Dim = 1 hoặc 2.")


    def OVL(self):
        return 1 - int_trapz(np.minimum(self.f1, self.f2), Dim=self.Dim, h=self.h)

    @staticmethod
    def compute_all(F_data: np.ndarray, centers: np.ndarray, metric: str, bandwidth: float):
        n_samples = F_data.shape[0]
        n_centers = centers.shape[0]
        dist_matrix = np.zeros((n_samples, n_centers))
        for i in range(n_samples):
            for j in range(n_centers):
                dist_matrix[i, j] = Dist.compute(F_data[i], centers[j], metric, bandwidth)
        return dist_matrix
