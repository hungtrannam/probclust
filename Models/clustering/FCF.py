import numpy as np
from utils.dist import Dist


class Model:
    """
    Fuzzy C-Means Clustering for probability density functions (PDFs).
    """

    def __init__(
        self,
        grid_x: np.ndarray,
        num_clusters: int = 3,
        fuzziness: float = 2.0,
        max_iterations: int = 100,
        tolerance: float = 1e-5,
        distance_metric: str = "L2",
        bandwidth: float = 0.01,
        centroid_mode: str = "mean",
        Dim=None,
        seed: int = None,
        verbose: bool = False,
    ):
        """
        Parameters
        ----------
        grid_x : np.ndarray
            Lưới x để tính tích phân và khoảng cách.
        num_clusters : int
            Số cụm.
        fuzziness : float
            Hệ số m (>1).
        max_iterations : int
            Số vòng lặp tối đa.
        tolerance : float
            Ngưỡng hội tụ.
        distance_metric : str
            Loại khoảng cách ('L1', 'L2', 'H', 'BC', 'W2').
        bandwidth : float
            Bước tích phân h.
        seed : int
            Seed random (nếu cần).
        verbose : bool
            In log nếu True.
        """
        self.grid_x = grid_x
        self.num_clusters = num_clusters
        self.fuzziness = fuzziness
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.distance_metric = distance_metric
        self.bandwidth = bandwidth
        self.seed = seed
        self.centroids_mode = centroid_mode
        self.verbose = verbose
        self.Dim = Dim if Dim is not None else 1


        self.pdf_matrix = None
        self.centroids = None
        self.membership_matrix = None
        self.objective_history = []

    @staticmethod
    def _cdf_from_pdf(pdf: np.ndarray, dx: float) -> np.ndarray:
        cdf = np.cumsum(pdf) * dx
        Z = float(cdf[-1]) if cdf.size else 1.0
        if Z <= 0:
            n = pdf.size
            return np.linspace(0.0, 1.0, n)
        return cdf / Z


    def _update_centroids_w2(self) -> None:
        """
        Cập nhật tâm cụm bằng Wasserstein-2 barycenter (1D).
        - Dựa trên tính chất: Q_k(t) = sum_j w_jk * Q_j(t), với w_jk = U_jk^m / sum_j U_jk^m.
        - Sau đó suy ra CDF F_k(x) = Q_k^{-1}(x) bằng nội suy ngược, rồi pdf = dF/dx.
        """
        assert self.Dim == 1, "W2 barycenter hiện cài cho 1D."
        assert self.pdf_matrix is not None and self.membership_matrix is not None

        dx = float(self.grid_x[1] - self.grid_x[0])
        n_samples, m = self.pdf_matrix.shape
        K = self.num_clusters

        # 1) Tiền tính các inverse-CDF (quantile) cho MỌI pdf một lần
        t_grid = np.linspace(0.0, 1.0, m)
        inv_mat = np.empty((n_samples, m), dtype=float)  # inv_mat[j, :] = Q_j(t_grid)

        for j in range(n_samples):
            cdf_j = self._cdf_from_pdf(self.pdf_matrix[j], dx)
            # nội suy Q_j(t) = F_j^{-1}(t) trên grid t_grid
            inv_mat[j] = np.interp(t_grid, cdf_j, self.grid_x)

        # 2) Trọng số mờ cho từng cụm
        W = self.membership_matrix ** self.fuzziness          # [n_samples, K]
        denom = np.sum(W, axis=0, keepdims=True) + 1e-12      # [1, K]
        W_norm = W / denom                                    # chuẩn hoá theo cột

        # 3) Với mỗi cụm: Q_k(t) = Σ_j w_jk * Q_j(t)
        centroids = np.empty((K, m), dtype=float)
        for k in range(K):
            Qk = (W_norm[:, k][:, None] * inv_mat).sum(axis=0)  # [m]

            # đảm bảo đơn điệu không giảm (tránh nhiễu số)
            Qk = np.maximum.accumulate(Qk)

            # 4) Suy ra CDF tâm: F_k(x) = Q_k^{-1}(x) (nội suy ngược)
            Fk = np.interp(self.grid_x, Qk, t_grid, left=0.0, right=1.0)

            # 5) pdf tâm = dF/dx (đạo hàm rời rạc trên lưới)
            fk = np.gradient(Fk, self.grid_x)
            fk = np.clip(fk, 0.0, None)          # tránh âm do nhiễu số

            centroids[k] = fk

        self.centroids = centroids


    def _update_centroids(self) -> None:
        """Cập nhật tâm cụm (centroid)."""
        weights = self.membership_matrix ** self.fuzziness
        self.centroids = (weights.T @ self.pdf_matrix) / (np.sum(weights.T, axis=1, keepdims=True) + 1e-12)

        

    def _compute_distance_matrix(self) -> np.ndarray:
        """Tính ma trận khoảng cách [num_pdfs, num_clusters]."""
        d_obj = Dist(h=self.bandwidth, Dim=1, grid=self.grid_x)

        num_pdfs = self.pdf_matrix.shape[0]
        return np.array([
            [getattr(d_obj, self.distance_metric)(self.pdf_matrix[i], self.centroids[j]) + 1e-10
             for j in range(self.num_clusters)]
            for i in range(num_pdfs)
        ])

    def _update_membership_matrix(self, distance_matrix: np.ndarray) -> np.ndarray:
        """Cập nhật ma trận membership U."""
        power = 2 / (self.fuzziness - 1)
        inv_distance = 1.0 / (distance_matrix + 1e-10)
        inv_distance_power = inv_distance ** power
        return inv_distance_power / np.sum(inv_distance_power, axis=1, keepdims=True)

    def fit(self, pdf_matrix: np.ndarray) -> None:
        """Huấn luyện FCM."""
        self.pdf_matrix = pdf_matrix
        self.num_pdfs, _ = pdf_matrix.shape

        if self.seed is not None:
            np.random.seed(self.seed)

        # Khởi tạo U ngẫu nhiên (mỗi hàng sum=1)
        self.membership_matrix = np.random.dirichlet(np.ones(self.num_clusters), size=self.num_pdfs)

        # Khởi tạo centroid từ dữ liệu
        init_indices = np.random.choice(self.num_pdfs, self.num_clusters, replace=False)
        self.centroids = pdf_matrix[init_indices, :]

        self.objective_history.clear()

        for it in range(self.max_iterations):
            if self.centroids_mode == "mean":
                self._update_centroids()   
            elif self.centroids_mode == "frechet":
                self._update_centroids_w2()   
                
            dist_matrix = self._compute_distance_matrix()
            new_U = self._update_membership_matrix(dist_matrix)

            # Tính hàm mục tiêu
            self.objective_value = np.sum((new_U ** self.fuzziness) * dist_matrix)
            self.objective_history.append(self.objective_value)

            # Kiểm tra hội tụ
            delta = np.linalg.norm(new_U - self.membership_matrix)
            if self.verbose:
                print(f"Iteration {it + 1}, delta = {delta:.6f}, objective = {self.objective_value:.6f}")
            if delta < self.tolerance:
                if self.verbose:
                    print("Converged.")
                break

            self.membership_matrix = new_U

    def predict(self, new_pdfs: np.ndarray) -> np.ndarray:
        """Dự đoán membership cho các PDF mới."""
        if new_pdfs.ndim == 1:
            new_pdfs = new_pdfs[np.newaxis, :]

        memberships = []
        for pdf in new_pdfs:
            distances = np.array([
                self._compute_distance(pdf, self.centroids[j]) + 1e-10
                for j in range(self.num_clusters)
            ])
            power = 2 / (self.fuzziness - 1)
            inv_distance = 1.0 / distances
            inv_distance_power = inv_distance ** power
            memberships.append(inv_distance_power / np.sum(inv_distance_power))

        return np.array(memberships)

    def get_results(self):
        """Trả về U.T, centroids, và lịch sử hàm mục tiêu."""
        return self.membership_matrix.T.copy(), self.centroids.copy(), self.objective_history.copy()

    def get_hard_assignments(self) -> np.ndarray:
        """Trả về nhãn cứng của từng PDF."""
        return np.argmax(self.membership_matrix, axis=1)


