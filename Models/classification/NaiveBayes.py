import numpy as np
from typing import Dict, List, Optional

def _mean_var_from_pdf(pdf: np.ndarray, x: np.ndarray, dx: float):
    """Tính (mean, var) rời rạc của một pdf đã chuẩn hoá trên lưới đều."""
    # đảm bảo chuẩn hoá (phòng khi pdf lệch nhỏ)
    mu = float(np.sum(x * pdf) * dx)
    var = float(np.sum((x - mu)**2 * pdf) * dx)
    return mu, var

def _gaussian_kernel_on_grid(x: np.ndarray, sigma: float, dx: float):
    """Mẫu của N(x;0,sigma^2) nhân sẵn dx để tích chập ~ tích phân."""
    if sigma <= 0:
        # suy giảm về delta (không nên xảy ra nếu Silverman > 0)
        k = np.zeros_like(x)
        k[len(x)//2] = dx
        return k
    k = (1.0 / np.sqrt(2*np.pi*sigma**2)) * np.exp(-0.5 * (x / sigma)**2)
    return k * dx  # nhân dx để conv ≈ integral

class KernelBayesGrid:
    """
    Kernel Bayes cho dữ liệu pdf trên lưới đều (1D).
    - Kernel: Gaussian chuẩn (có băng thông lớp h_k).
    - f(h|k) ≈ mean_i <conv(h, G_{h_k}), f_{k,i}>_dx
    """
    def __init__(
        self,
        x_grid: np.ndarray,
        priors: Optional[Dict[int, float]] = None,   # {label: prior}; None => proportional to freq (or uniform)
        bandwidths: Optional[Dict[int, float]] = None, # {label: h_k}; None => Silverman
        use_uniform_prior: bool = True,
        verbose: bool = False,
    ):
        self.x = np.asarray(x_grid)
        self.dx = float(self.x[1] - self.x[0])
        self.priors = priors
        self.bandwidths = bandwidths
        self.use_uniform_prior = use_uniform_prior
        self.verbose = verbose

        # learned
        self.classes_: List[int] = []
        self.data_: Dict[int, np.ndarray] = {}   # class -> [n_k, m]
        self.stats_: Dict[int, dict] = {}        # class -> {'n':, 'sigma_k':, 'h':}
        self.kernels_: Dict[int, np.ndarray] = {}# class -> kernel vector aligned with x

    def fit(self, all_pdfs: np.ndarray, y: np.ndarray):
        """
        all_pdfs: [n_samples, m] — mỗi hàng là pdf trên cùng lưới x_grid
        y:        [n_samples]     — nhãn nguyên (0/1/...)
        """
        all_pdfs = np.asarray(all_pdfs, dtype=float)
        y = np.asarray(y)

        # gom theo lớp
        self.classes_ = sorted(np.unique(y).tolist())
        self.data_.clear()
        for c in self.classes_:
            self.data_[c] = all_pdfs[y == c]  # shape [n_k, m]

        # priors
        if self.priors is None:
            if self.use_uniform_prior:
                self.priors = {c: 1.0 / len(self.classes_) for c in self.classes_}
            else:
                # theo tần suất
                N = len(y)
                self.priors = {c: float(len(self.data_[c])) / N for c in self.classes_}

        # Silverman bandwidth theo sigma_k (trung bình độ lệch chuẩn của pdf trong lớp)
        self.stats_.clear()
        self.kernels_.clear()
        for c in self.classes_:
            P = self.data_[c]          # [n_k, m]
            n_k, m = P.shape

            # ước lượng sigma cho từng pdf rồi lấy trung bình
            sigmas = []
            for i in range(n_k):
                _, var_i = _mean_var_from_pdf(P[i], self.x, self.dx)
                sigmas.append(np.sqrt(max(var_i, 1e-12)))
            sigma_k = float(np.mean(sigmas))

            # bandwidth
            if self.bandwidths is None or c not in self.bandwidths:
                h_k = 1.06 * sigma_k * (n_k ** (-1.0/5.0))
            else:
                h_k = float(self.bandwidths[c])

            # kernel vector (căn giữa tại 0): cần lưới đối xứng quanh 0 để conv 'same'
            # Tạo lưới sai phân đối xứng tương ứng độ dài m
            mid = m // 2
            x_centered = (self.x - self.x[mid])  # dịch tâm về 0
            k_vec = _gaussian_kernel_on_grid(x_centered, h_k, self.dx)

            self.stats_[c] = {'n': n_k, 'sigma_k': sigma_k, 'h': h_k}
            self.kernels_[c] = k_vec

            if self.verbose:
                print(f"[fit] class={c} n={n_k} sigma_k≈{sigma_k:.4f} h_k≈{h_k:.4f} prior={self.priors[c]:.4f}")

        return self

    def _smooth(self, h_pdf: np.ndarray, c: int) -> np.ndarray:
        """smooth_h = conv(h, G_{h_k}) ~ integral conv (đã nhân dx trong kernel)."""
        k = self.kernels_[c]
        # dùng 'same' để giữ kích thước; giả định lưới đều và k đã nhân dx
        return np.convolve(h_pdf, k, mode='same')

    def f_cond(self, h_pdf: np.ndarray, c: int) -> float:
        """
        f(h | Y=c) ≈ mean_i < smooth_h , f_{c,i} >_dx
        """
        smooth_h = self._smooth(h_pdf, c)  # [m]
        P = self.data_[c]                  # [n_k, m]
        vals = np.dot(P, smooth_h) * self.dx  # [n_k]
        return float(np.mean(vals))

    def posterior(self, h_pdf: np.ndarray) -> Dict[int, float]:
        num = {c: self.priors[c] * self.f_cond(h_pdf, c) for c in self.classes_}
        Z = sum(num.values())
        if Z <= 0:
            return {c: 1.0 / len(self.classes_) for c in self.classes_}
        return {c: v / Z for c, v in num.items()}

    def predict(self, h_pdf: np.ndarray) -> int:
        post = self.posterior(h_pdf)
        return max(post.items(), key=lambda kv: kv[1])[0]
