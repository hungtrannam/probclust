
# =======================================
# Author: Hung Tran-Nam
# Email: namhung34.info@gmail.com
# Repo: https://github.com/hungtrannam/probclust
# =======================================
# File: utils/dist.py
# Description: Cài đặt các hàm tính khoảng cách giữa các phân phối xác suất
# =======================================

import numpy as np
from utils.integral import int_trapz

class Dist:
    """
    Class to handle distance calculations between probability distributions.
    """

    def __init__(self, f1, f2, Dim=1, h=None, grid=None):
        self.f1 = f1
        self.f2 = f2
        self.h = h
        self.Dim = Dim
        self.x = grid

    def __call__(self):
        return self.calculate_distance()

    def L1(self):
        """
        Calculate the distance between two probability distributions.
        """
        if len(self.f1) != len(self.f2):
            raise ValueError("Distributions must have the same length.")
        
        # Using L1 norm (Euclidean distance)
        return int_trapz(np.abs(self.f1 - self.f2), Dim=self.Dim, h=self.h)
    
    def L2(self):
        """
        Calculate the L2 distance (Euclidean distance) between two probability distributions.
        """
        if len(self.f1) != len(self.f2):
            raise ValueError("Distributions must have the same length.")
        
        return np.sqrt(int_trapz((self.f1 - self.f2)**2, Dim=self.Dim, h=self.h))
    
    def H(self):
        """
        Calculate the Hellinger distance between two probability distributions.
        """
        if len(self.f1) != len(self.f2):
            raise ValueError("Distributions must have the same length.")
        
        # Hellinger distance is defined as:
        # H(P, Q) = 1/sqrt(2) * ||sqrt(P) - sqrt(Q)||
        sqrt_f1 = np.sqrt(self.f1)
        sqrt_f2 = np.sqrt(self.f2)
        return np.sqrt(0.5) * np.sqrt(int_trapz((sqrt_f1 - sqrt_f2)**2, Dim=self.Dim, h=self.h))
    
    def BC(self):
        """
        Calculate the Bhattacharyya distance between two probability distributions.
        """
        if len(self.f1) != len(self.f2):
            raise ValueError("Distributions must have the same length.")
        
        # Bhattacharyya distance is defined as:
        # D_B(P, Q) = -ln(BC(P, Q))
        # where BC(P, Q) is the Bhattacharyya coefficient
        bc = int_trapz(np.sqrt(self.f1 * self.f2), Dim=self.Dim, h=self.h) + 1e-100 # Adding epsilon to avoid log(0)
        return - np.log(bc)
    
    def W2(self):
        """
        Calculate the Wasserstein distance (Earth Mover's Distance) between two probability distributions.
        """
        if len(self.f1) != len(self.f2):
            raise ValueError("Distributions must have the same length.")
        
        # Using cumulative distribution functions (CDFs)
        cdf_f1 = np.cumsum(self.f1) * self.h
        cdf_f2 = np.cumsum(self.f2) * self.h

        # Các mốc t từ 0 đến 1
        t_vals = np.linspace(0, 1, len(cdf_f1))

        # Lấy giá trị x tại các t từ CDF nghịch đảo
        inv_f1 = np.interp(t_vals, cdf_f1, self.x)  # self.x = grid
        inv_f2 = np.interp(t_vals, cdf_f2, self.x)

        # Tính tích phân ∫ (inv_f1 - inv_f2)^2 dt
        return np.sqrt(np.sum((inv_f1 - inv_f2) ** 2) * (1 / len(t_vals)))
    
    def OVL(self):
        """
        Calculate the overlap between two probability distributions.
        """
        if len(self.f1) != len(self.f2):
            raise ValueError("Distributions must have the same length.")
        
        # Overlap is defined as the integral of the minimum of the two distributions
        return 1-int_trapz(np.minimum(self.f1, self.f2), Dim=self.Dim, h=self.h)
    

    @staticmethod
    def compute_all(F_data: np.ndarray, centers: np.ndarray, metric: str, bandwidth: float) -> np.ndarray:
        """Tính toàn bộ ma trận khoảng cách F_data vs centers."""
        n_samples = F_data.shape[0]
        n_centers = centers.shape[0]
        dist_matrix = np.zeros((n_samples, n_centers))
        for i in range(n_samples):
            for j in range(n_centers):
                dist_matrix[i, j] = Dist.compute(F_data[i], centers[j], metric, bandwidth)
        return dist_matrix
