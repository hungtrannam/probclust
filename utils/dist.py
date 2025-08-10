import numpy as np
from utils.integral import int_trapz

class Dist:
    """
    Class to handle distance calculations between probability distributions.
    Supports both 1D and 2D (flattened) PDFs.
    """

    def __init__(self, f1, f2, Dim=1, h=None, grid=None):
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

        # --- 2-D case --------------------------------------------------------
        if self.Dim == 2:
            if grid is None or grid.ndim != 2 or grid.shape[1] != 2:
                raise ValueError("For Dim=2, provide a grid of shape (n_x*n_y, 2).")

            n_x = len(np.unique(grid[:, 0]))
            n_y = len(np.unique(grid[:, 1]))
            expected_size = n_x * n_y

            if f1.size != expected_size or f2.size != expected_size:
                raise ValueError(
                    f"Size of f1 or f2 ({f1.size}) does not match grid ({n_x}×{n_y})."
                )

            self.f1 = f1.reshape(n_x, n_y)
            self.f2 = f2.reshape(n_x, n_y)
            self.n_x, self.n_y = n_x, n_y
            self.x = None  # not used in 2-D
        # --- 1-D case --------------------------------------------------------
        else:
            if len(f1) != len(f2):
                raise ValueError("Distributions must have the same length.")
            self.f1 = f1
            self.f2 = f2
            self.x = grid

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def __call__(self, metric="L2"):
        """Convenience wrapper."""
        return self.calculate_distance(metric)

    def calculate_distance(self, metric="L2"):
        """Compute the chosen distance."""
        metric = metric.lower()
        if metric == "l1":
            return self.L1()
        elif metric == "l2":
            return self.L2()
        elif metric == "h":
            return self.H()
        elif metric == "m":
            return self.M()
        elif metric == "kl":
            return self.KL()
        elif metric == "bc":
            return self.BC()
        elif metric == "w2":
            return self.W2()
        elif metric == "ovl":
            return self.OVL()
        else:
            raise ValueError(f"Unknown metric: {metric}")

    # ------------------------------------------------------------------
    # Individual distances
    # ------------------------------------------------------------------
    def L1(self):
        """L1 (total variation) distance."""
        return int_trapz(np.abs(self.f1 - self.f2), Dim=self.Dim, h=self.h)

    def L2(self):
        """L2 (Euclidean) distance."""
        return np.sqrt(int_trapz((self.f1 - self.f2) ** 2, Dim=self.Dim, h=self.h))

    def H(self):
        """Hellinger distance."""
        sqrt_f1 = np.sqrt(self.f1)
        sqrt_f2 = np.sqrt(self.f2)
        return np.sqrt(0.5) * np.sqrt(
            int_trapz((sqrt_f1 - sqrt_f2) ** 2, Dim=self.Dim, h=self.h)
        )

    def M(self):
        """Matusita distance."""
        sqrt_f1 = np.sqrt(self.f1)
        sqrt_f2 = np.sqrt(self.f2)
        bc = int_trapz(np.sqrt(self.f1 * self.f2), Dim=self.Dim, h=self.h)
        # Ensure bc does not exceed 1 (numerical safety)
        bc = np.clip(bc, 0.0, 1.0)
        return np.sqrt(2) * np.sqrt(1 - bc)

    def KL(self):
        """Kullback–Leibler divergence (f1 || f2)."""
        # Add small epsilon to avoid log(0)
        eps = 1e-12
        f1_safe = np.clip(self.f1, eps, None)
        f2_safe = np.clip(self.f2, eps, None)
        return int_trapz(f1_safe * np.log(f1_safe / f2_safe), Dim=self.Dim, h=self.h)

    def BC(self):
        """Bhattacharyya distance."""
        bc = int_trapz(np.sqrt(self.f1 * self.f2), Dim=self.Dim, h=self.h) + 1e-100
        return -np.log(bc)

    def W2(self):
        """2-Wasserstein distance."""
        if self.Dim == 1:
            cdf_f1 = np.cumsum(self.f1) * self.h
            cdf_f2 = np.cumsum(self.f2) * self.h
            t_vals = np.linspace(0, 1, len(cdf_f1))
            inv_f1 = np.interp(t_vals, cdf_f1, self.x)
            inv_f2 = np.interp(t_vals, cdf_f2, self.x)
            return np.sqrt(np.sum((inv_f1 - inv_f2) ** 2) * (1 / len(t_vals)))

        elif self.Dim == 2:
            try:
                import ot
            except ImportError:
                raise ImportError("Install POT: pip install POT")

            f1_flat = self.f1.flatten()
            f2_flat = self.f2.flatten()
            f1_flat /= f1_flat.sum()
            f2_flat /= f2_flat.sum()

            M = ot.dist(self.grid, self.grid, metric="euclidean") ** 2  # cost matrix
            w2 = ot.emd2(f1_flat, f2_flat, M, numItermax=int(1e7))
            return np.sqrt(w2)

        else:
            raise NotImplementedError("W2 only supports Dim = 1 or 2.")

    def OVL(self):
        """Overlapping coefficient distance = 1 - ∫ min(f1,f2)."""
        return 1 - int_trapz(np.minimum(self.f1, self.f2), Dim=self.Dim, h=self.h)

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