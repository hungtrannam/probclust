import numpy as np
from utils.integral import int_trapz

class Dist:
    """
    Class to handle distance calculations between probability distributions.
    Supports both 1D and 2D (flattened) PDFs.
    """

    def __init__(self, Dim=1, h=None, grid=None):
        """
        Parameters
        ----------
        f1, f2 : np.ndarray
            Probability densities (1-D vector or 2-D flattened).
        Dim : {1, 2}
            Dimensionality.
        h : float, optional
            Grid spacing (1-D) or cell area (2-D).  For 1-D it is `dx`,
            for 2-D it is `dx*dy`.
        grid : np.ndarray
            Grid coordinates.  1-D: (n,) or (n,1);  2-D: (n_x*n_y, 2).
        """
        self.Dim = Dim
        self.h = h if Dim == 1 else h**2  # h^2 for 2-D area element
        self.grid = grid

    # ------------------------------------------------------------------
    # Individual distances
    # ------------------------------------------------------------------
    def L1(self, f1, f2):
        """L1 (total variation) distance."""
        return int_trapz(np.abs(f1 - f2), Dim=self.Dim, h=self.h)

    def L2(self, f1, f2):
        """L2 (Euclidean) distance."""
        return np.sqrt(int_trapz((f1 - f2) ** 2, Dim=self.Dim, h=self.h))

    def H(self, f1, f2):
        """Hellinger distance."""
        sqrt_f1 = np.sqrt(f1)
        sqrt_f2 = np.sqrt(f2)
        return (1/2) * np.sqrt(
            int_trapz((sqrt_f1 - sqrt_f2) ** 2, Dim=self.Dim, h=self.h)
        )

    def M(self, f1,f2,r=2):
        """Matusita distance."""
        return np.sqrt(
            int_trapz(np.abs(f1**(1/r) - f2**(1/r))**r, Dim=self.Dim, h=self.h)
        )
    
    def KLinfo(self, f1, f2):
        """Kullback–Leibler divergence (f1 || f2)."""
        eps = 1e-12
        f1_safe = np.clip(f1, eps, None)
        f2_safe = np.clip(f2, eps, None)
        return int_trapz(f1_safe * np.log(f1_safe / f2_safe), Dim=self.Dim, h=self.h)

    def KLdiv(self, f1, f2):
        """Symmetric Kullback–Leibler divergence (average of f1||f2 and f2||f1)."""
        return 0.5 * self.KLinfo(f1, f2) + 0.5 * self.KLinfo(f2, f1)

    def BC(self, f1,f2):
        """Bhattacharyya distance."""
        bc = int_trapz(np.sqrt(f1 * f2), Dim=self.Dim, h=self.h) + 1e-100
        return -np.log(bc)

    def W2(self, f1,f2):
        from scipy.integrate import trapezoid

        """2-Wasserstein distance."""
        if self.Dim == 1:
            cdf_f1 = np.cumsum(f1) * self.h
            cdf_f2 = np.cumsum(f2) * self.h
            t_vals = np.linspace(0, 1, len(cdf_f1))
            inv_f1 = np.interp(t_vals, cdf_f1, self.grid)
            inv_f2 = np.interp(t_vals, cdf_f2, self.grid)
            return np.sqrt(trapezoid((inv_f1 - inv_f2) ** 2, x=t_vals))

        elif self.Dim == 2:
            try:
                import ot
            except ImportError:
                raise ImportError("Install POT: pip install POT")

            f1_flat = f1.flatten()
            f2_flat = f2.flatten()
            f1_flat /= f1_flat.sum()
            f2_flat /= f2_flat.sum()

            M = ot.dist(self.grid, self.grid, metric="euclidean") ** 2  # cost matrix
            w2 = ot.emd2(f1_flat, f2_flat, M, numItermax=int(1e7))
            return np.sqrt(w2)

        else:
            raise NotImplementedError("W2 only supports Dim = 1 or 2.")

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------
    @staticmethod
    def compute_single(f1, f2, metric, h, grid=None, Dim=1):
        """Compute distance between two single distributions."""
        return Dist(f1, f2, Dim=Dim, h=h, grid=grid).calculate_distance(metric)

    @staticmethod
    def compute_all(F_data: np.ndarray,
                    centers: np.ndarray,
                    metric: str,
                    bandwidth: float,
                    Dim=1,
                    grid=None):
        """
        Compute pairwise distance matrix between samples and centers.

        Parameters
        ----------
        F_data : np.ndarray (n_samples, n_points)
        centers : np.ndarray (n_centers, n_points)
        metric : str
            Any implemented metric.
        bandwidth : float
            Grid spacing `h`.
        Dim : {1, 2}
        grid : np.ndarray
            Required for 2-D.

        Returns
        -------
        dist_matrix : np.ndarray (n_samples, n_centers)
        """
        n_samples = F_data.shape[0]
        n_centers = centers.shape[0]
        dist_matrix = np.zeros((n_samples, n_centers))

        for i in range(n_samples):
            for j in range(n_centers):
                dist_matrix[i, j] = Dist.compute_single(
                    F_data[i], centers[j], metric, h=bandwidth, grid=grid, Dim=Dim
                )
        return dist_matrix