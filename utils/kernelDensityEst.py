import numpy as np


class KernelDensityEstimator:
    """
    Kernel Density Estimator (KDE) cơ bản cho dữ liệu 1D.
    Hỗ trợ nhiều loại kernel: Gaussian, Tophat, Linear, Epanechnikov, Exponential, Cosine, Logistic, Sigmoid.
    """

    def __init__(self, bandwidth: float = 1.0, kernel: str = "gaussian"):
        self.bandwidth = bandwidth
        self.kernel_name = kernel.lower()
        self.data = None
        self.kernel = self._get_kernel_function(self.kernel_name)

    # ===============================
    # FIT
    # ===============================
    def fit(self, data: np.ndarray) -> "KernelDensityEstimator":
        self.data = np.asarray(data, dtype=float)
        return self   # <-- trả về chính đối tượng


    # ===============================
    # EVALUATE
    # ===============================
    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """
        Ước lượng mật độ tại các điểm x.

        Parameters
        ----------
        x : np.ndarray
            Các điểm cần ước lượng (1D array).

        Returns
        -------
        np.ndarray
            Giá trị ước lượng mật độ tại mỗi x.
        """
        if self.data is None:
            raise ValueError("Cần gọi fit(data) trước khi evaluate().")

        x = np.asarray(x, dtype=float)
        h = self.bandwidth
        n = len(self.data)

        # Broadcasting thay vì vòng lặp
        diff = (x[:, None] - self.data[None, :]) / h  # shape [len(x), n]
        density = np.sum(self.kernel(diff), axis=1) / (n * h)
        return density

    # ===============================
    # GET KERNEL FUNCTION
    # ===============================
    def _get_kernel_function(self, name: str):
        """Trả về hàm kernel tương ứng."""
        kernels = {
            "gaussian": lambda u: (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * u**2),
            "tophat": lambda u: 0.5 * (np.abs(u) <= 1),
            "linear": lambda u: (1 - np.abs(u)) * (np.abs(u) <= 1),
            "epanechnikov": lambda u: 0.75 * (1 - u**2) * (np.abs(u) <= 1),
            "exponential": lambda u: 0.5 * np.exp(-np.abs(u)),
            "cosine": lambda u: (np.pi / 4) * np.cos((np.pi / 2) * u) * (np.abs(u) <= 1),
            "logistic": lambda u: np.exp(-u) / (1 + np.exp(-u)) ** 2 / 2,
            "sigmoid": lambda u: 1 / (np.exp(u) + np.exp(-u)),
        }
        if name not in kernels:
            raise ValueError(f"Kernel '{name}' không được hỗ trợ.")
        return kernels[name]
