import numpy as np
from utils.dist import Dist


class Model:
    """
    Neutrosophic Clustering cho dữ liệu hàm mật độ xác suất (PDFs).
    Bao gồm 3 thành phần: 
        - T: membership (độ thuộc)
        - I: indeterminacy (độ không xác định)
        - F: hesitation (độ do dự)
    """

    def __init__(
        self,
        grid_x: np.ndarray,
        num_clusters: int = 3,
        fuzziness: float = 2.0,
        delta: float = 1.0,
        w1: float = 1.0,
        w2: float = 1.0,
        w3: float = 1.0,
        max_iterations: int = 100,
        tolerance: float = 1e-5,
        distance_metric: str = "L2",
        bandwidth: float = 0.01,
        seed: int = None,
        Dim: int = 1,
        verbose: bool = False,
    ):
        self.grid_x = grid_x
        self.num_clusters = num_clusters
        self.fuzziness = fuzziness
        self.delta = delta
        self.w1, self.w2, self.w3 = w1, w2, w3
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.distance_metric = distance_metric
        self.bandwidth = bandwidth
        self.seed = seed
        self.verbose = verbose
        self.Dim = Dim
        self.eps = 1e-10

        # Kết quả
        self.T = None
        self.I = None
        self.F = None
        self.Theta = None
        self.num_pdfs = None
        self.J_hist = []


    def _update_centroids(self) -> np.ndarray:
        for j in range(self.num_clusters):
            weights = (self.w1 * self.T[:, j]) ** self.fuzziness       # (N,)
            self.Theta[j] = np.sum(weights[:, None] * self.pdf_matrix, axis=0) / (np.sum(weights) + self.eps)

    def _compute_distance_matrix(self) -> np.ndarray:
            """Tính ma trận khoảng cách [num_pdfs, num_clusters]."""
            d_obj = Dist(h=self.bandwidth, Dim=self.Dim, grid=self.grid_x)

            num_pdfs = self.pdf_matrix.shape[0]
            return np.array([
                [getattr(d_obj, self.distance_metric)(self.pdf_matrix[i], self.Theta[j])**2 + 1e-10
                for j in range(self.num_clusters)]
                for i in range(num_pdfs)
            ])

    def fit(self, pdf_matrix: np.ndarray) -> None:
        """
        Huấn luyện mô hình Neutrosophic Clustering cho dữ liệu PDF.
        """
        self.pdf_matrix = pdf_matrix

        self.num_pdfs, D = self.pdf_matrix .shape
        c = self.num_clusters
        m = self.fuzziness
        delta = self.delta
        power = -2 / (m - 1)

        if self.seed is not None:
            np.random.seed(self.seed)

        # Khởi tạo
        self.Theta = self.pdf_matrix [np.random.choice(self.num_pdfs, c, replace=False)]
        self.T = np.random.dirichlet(np.ones(c), size=self.num_pdfs)        # shape (N, c)
        self.I = np.full(self.num_pdfs, 1 / (c + 2))
        self.F = np.full(self.num_pdfs, 1 / (c + 2))

        dist_obj = Dist(h=self.bandwidth, Dim=self.Dim, grid=self.grid_x)

        for iteration in range(self.max_iterations):
            Theta_prev = self.Theta.copy()

            D = self._compute_distance_matrix()


            # Tính trung tâm gần nhất (cho phần I)
            dist_I = np.zeros(self.num_pdfs)
            for i in range(self.num_pdfs):
                nearest = np.argsort(D[i])[:2]
                avg_c = (self.Theta[nearest[0]] + self.Theta[nearest[1]]) / 2
                dist_I[i] = getattr(dist_obj, self.distance_metric)(self.pdf_matrix[i], avg_c) + self.eps

            # Tính hệ số chuẩn hoá K_i
            K = 1.0 / (
                np.sum((1 / self.w1) * D ** power, axis=1) +
                (1 / self.w2) * dist_I ** power +
                (1 / self.w3) * delta ** power +
                self.eps
            )  # shape (N,)

            # Cập nhật T, I, F
            self.T = (K[:, None] / self.w1) * D ** power     # (N, c)
            self.I = (K / self.w2) * dist_I ** power         # (N,)
            self.F = (K / self.w3) * delta ** power          # (N,)

            # Chuẩn hoá T + I + F = 1 (nếu muốn)
            # total = self.T.sum(axis=1) + self.I + self.F
            # self.T /= total[:, None]
            # self.I /= total
            # self.F /= total

            # Cập nhật centroid Theta_j
            self._update_centroids()


            # Kiểm tra hội tụ
            delta_c = np.linalg.norm(self.Theta - Theta_prev)
            obj = np.sum(self.T ** m * D ** 2)
            self.J_hist.append(obj)

            if self.verbose:
                print(f"[{iteration+1}] ΔC={delta_c:.2e}, Obj={obj:.6f}")

            if delta_c < self.tolerance or np.isnan(delta_c):
                break

        # Lưu kết quả
        # self.T, self.I, self.F = T, I, F

    def predict(self, new_X: np.ndarray) -> np.ndarray:
        """
        Dự đoán membership T cho dữ liệu mới.
        Trả về shape: (c, num_samples)
        """
        if new_X.ndim == 1:
            new_X = new_X[np.newaxis, :]

        N_new = new_X.shape[0]
        c = self.num_clusters
        m = self.fuzziness
        power = -2 / (m - 1)

        dist_obj = Dist(h=self.bandwidth, Dim=self.Dim, grid=self.grid_x)
        T_new = np.zeros((c, N_new))

        for i in range(N_new):
            D = np.array([
                getattr(dist_obj, self.distance_metric)(new_X[i], self.Theta[j])**2 + self.eps
                for j in range(c)
            ])
            dist_I = np.min(D)
            K = 1.0 / (
                np.sum((1 / self.w1) * D ** power) +
                (1 / self.w2) * dist_I ** power +
                (1 / self.w3) * self.delta ** power +
                self.eps
            )
            T_new[:, i] = K / self.w1 * D ** power

        return T_new

    def get_results(self):
        """
        Trả về các kết quả: T, I, F, Theta, J_hist
        """
        return self.T.T.copy(), self.I.copy(), self.F.copy(), self.Theta.copy(), self.J_hist
