import numpy as np
from utils.integral import int_trapz


class Kernel:
    """
    Kernel matrix computation for PDFs using different metrics (Hellinger, L1, L2).
    """

    def __init__(self, h: float, Dim: int = None):
        self.h = h
        self.Dim = Dim

    def _compute_pairwise(self, pdfs, metric_fn):
        """
        Tính toán ma trận kernel đôi một với hàm metric cho trước.
        """
        n = pdfs.shape[0]
        K = np.zeros((n, n))
        for i in range(n):
            for j in range(i, n):
                val = metric_fn(pdfs[i], pdfs[j])
                K[i, j] = val
                K[j, i] = val
        return K

    def H(self, pdfs: np.ndarray) -> np.ndarray:
        """
        Hellinger kernel matrix.
        K(i, j) = ∫ sqrt(pdf_i * pdf_j)
        """
        return self._compute_pairwise(
            pdfs,
            lambda f1, f2: int_trapz(np.sqrt(f1 * f2), self.h, Dim=self.Dim),
        )

    def L2(self, pdfs: np.ndarray, gamma: float = 1.0) -> np.ndarray:
        """
        Gaussian-like kernel using L2 distance.
        K(i, j) = exp(-gamma * ∫ (f1 - f2)^2)
        """
        return self._compute_pairwise(
            pdfs,
            lambda f1, f2: np.exp(
                -gamma * int_trapz((f1 - f2) ** 2, self.h, Dim=self.Dim)
            ),
        )
    
    def L1(self, pdfs: np.ndarray, gamma: float = 2.0) -> np.ndarray:
        """
        Exponential kernel using L1 distance.
        K(i, j) = exp(-gamma * ∫ |f1 - f2|)
        """
        return self._compute_pairwise(
            pdfs,
            lambda f1, f2: np.exp(
                -gamma * int_trapz(np.abs(f1 - f2), self.h, Dim=self.Dim)
            ),
        )


    def LN(self, pdfs: np.ndarray) -> np.ndarray:
        """
        Exponential kernel using L1 distance.
        K(i, j) = exp(-gamma * ∫ |f1 - f2|)
        """
        return self._compute_pairwise(
            pdfs,
            lambda f1, f2: int_trapz(
               f1*f2, self.h, Dim=self.Dim
            ),
        ),

    def Chi2(self, pdfs: np.ndarray, eps: float = 1e-10) -> np.ndarray:
        """
        Chi-squared kernel matrix.
        K(i, j) = ∫ 2 * f1 * f2 / (f1 + f2 + eps) dx
        """
        return self._compute_pairwise(
            pdfs,
            lambda f1, f2: int_trapz(
                2 * f1 * f2 / (f1 + f2 + eps), self.h, Dim=self.Dim
            ),
        )

    def compute(self, pdfs: np.ndarray, kind: str = "H", gamma: float = 1.0) -> np.ndarray:
        """
        General interface to compute kernel matrix.

        Parameters
        ----------
        pdfs : np.ndarray
            Matrix of PDFs [n_pdfs, n_points].
        kind : str
            Kernel type: 'H', 'L2', 'L1'.
        gamma : float
            Kernel parameter for L1/L2.

        Returns
        -------
        np.ndarray
            Kernel matrix [n_pdfs, n_pdfs].
        """
        kind = kind.upper()
        if kind == "H":
            return self.H(pdfs)
        elif kind == "L2":
            return self.L2(pdfs, gamma)
        elif kind == "L1":
            return self.L1(pdfs, gamma)
        else:
            raise ValueError(f"Unknown kernel kind: {kind}")
