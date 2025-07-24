import numpy as np

class KernelDensityEstimator:
    def __init__(self, bandwidth=1.0, kernel='gaussian'):
        self.bandwidth = bandwidth
        self.kernel_name = kernel
        self.data = None
        self.kernel = self._get_kernel_function(kernel)

    def fit(self, data):
        """Lưu trữ dữ liệu mẫu để ước lượng"""
        self.data = np.asarray(data)

    def evaluate(self, x):
        """Ước lượng mật độ tại các điểm x"""
        if self.data is None:
            raise ValueError("need fit(data) first")

        n = len(self.data)
        h = self.bandwidth
        return np.array([
            np.sum(self.kernel((x_i - self.data) / h)) / (n * h)
            for x_i in x
        ])

    def _get_kernel_function(self, name):
        """Trả về hàm kernel tương ứng"""
        kernels = {
            'gaussian': lambda u: (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * u ** 2),
            'tophat': lambda u: 0.5 * (np.abs(u) <= 1),
            'linear': lambda u: (1 - np.abs(u)) * (np.abs(u) <= 1),
            'epanechnikov': lambda u: 0.75 * (1 - u ** 2) * (np.abs(u) <= 1),
            'exponential': lambda u: 0.5 * np.exp(-np.abs(u)),
            'cosine': lambda u: (np.pi / 4) * np.cos((np.pi / 2) * u) * (np.abs(u) <= 1),
            'logistic': lambda u: np.exp(-u) / (1 + np.exp(-u)) ** 2 / 2,
            'sigmoid': lambda u: 1 / (np.exp(u) + np.exp(-u)),
        }
        if name not in kernels:
            raise ValueError(f"Kernel '{name}' không được hỗ trợ.")
        return kernels[name]
