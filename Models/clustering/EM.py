import numpy as np
from utils.dist import Dist


class Model:
    """
    EM clustering cho hàm mật độ xác suất (PDF).
    """

    def __init__(
        self,
        grid_x: np.ndarray,
        num_clusters: int = 3,
        max_iterations: int = 100,
        tolerance: float = 1e-5,
        distance_metric: str = "L2",
        bandwidth: float = 0.01,
        seed: int = None,
        Dim=None,
        centroid_mode: str = "mean",
        verbose: bool = False,
    ):
        """
        Parameters
        ----------
        grid_x : np.ndarray
            Lưới x để tính khoảng cách.
        num_clusters : int
            Số cụm.
        max_iterations : int
            Số vòng lặp tối đa của EM.
        tolerance : float
            Ngưỡng hội tụ.
        distance_metric : str
            Loại khoảng cách ('L1', 'L2', 'H', 'BC', 'W2').
        bandwidth : float
            Tham số bandwidth cho tích phân.
        seed : int
            Seed random (nếu cần).
        verbose : bool
            In log nếu True.
        """
        self.grid_x = grid_x
        self.num_clusters = num_clusters
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.distance_metric = distance_metric
        self.bandwidth = bandwidth
        self.seed = seed
        self.centroids_mode = centroid_mode
        self.Dim = Dim if Dim is not None else 1

        self.verbose = verbose

        self.pdf_matrix = None
        self.centroids = None
        self.responsibilities = None
        self.cluster_priors = None

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
        assert self.pdf_matrix is not None and self.responsibilities is not None

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
        W = self.responsibilities         # [n_samples, K]
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

    def _compute_distance_matrix(self) -> np.ndarray:
        """Tính ma trận khoảng cách [num_pdfs, num_clusters]."""
        d_obj = Dist(h=self.bandwidth, Dim=1, grid=self.grid_x)

        num_pdfs = self.pdf_matrix.shape[0]
        return np.array([
            [getattr(d_obj, self.distance_metric)(self.pdf_matrix[i], self.centroids[j])**2 + 1e-10
             for j in range(self.num_clusters)]
            for i in range(num_pdfs)
        ])

    def _update_centroids(self) -> None:
        """Cập nhật centroid dựa trên responsibilities."""
        for j in range(self.num_clusters):
            weights = self.responsibilities[:, j]
            numerator = np.sum(weights[:, np.newaxis] * self.pdf_matrix, axis=0)
            denominator = np.sum(weights) + 1e-12
            self.centroids[j, :] = numerator / denominator

    def _update_cluster_priors(self) -> None:
        """Cập nhật trọng số cụm (priors)."""
        self.cluster_priors = np.sum(self.responsibilities, axis=0) / self.pdf_matrix.shape[0]

    def _e_step(self) -> np.ndarray:
        """Bước E: Tính responsibilities mới."""
        dist_matrix = self._compute_distance_matrix()

        new_responsibilities = np.exp(-dist_matrix) * self.cluster_priors[np.newaxis, :]
        new_responsibilities /= np.sum(new_responsibilities, axis=1, keepdims=True)
        return new_responsibilities

    def fit(self, pdf_matrix: np.ndarray) -> None:
        """Huấn luyện EM."""
        
        self.pdf_matrix = pdf_matrix
        self.num_pdfs, _ = pdf_matrix.shape


        if self.seed is not None:
            np.random.seed(self.seed)

        # Khởi tạo responsibilities
        self.responsibilities = np.random.dirichlet(np.ones(self.num_clusters), size=self.num_pdfs)

        # Khởi tạo centroids từ dữ liệu
        init_indices = np.random.choice(self.num_pdfs, self.num_clusters, replace=False)
        self.centroids = pdf_matrix[init_indices, :].copy()

        # Khởi tạo cluster priors
        self.cluster_priors = np.ones(self.num_clusters) / self.num_clusters

        for iteration in range(self.max_iterations):
            # M-step
            if self.centroids_mode == "mean":
                self._update_centroids()   
            elif self.centroids_mode == "frechet":
                self._update_centroids_w2()   
            self._update_cluster_priors()

            # E-step
            new_responsibilities = self._e_step()

            # Kiểm tra hội tụ
            delta = np.linalg.norm(new_responsibilities - self.responsibilities)
            if self.verbose:
                print(f"Iteration {iteration + 1}, delta = {delta:.6f}")
            if delta < self.tolerance:
                if self.verbose:
                    print("Converged.")
                break
            self.responsibilities = new_responsibilities

    def predict(self, new_pdfs: np.ndarray) -> np.ndarray:
        """Dự đoán soft-assignments cho PDF mới."""
        if new_pdfs.ndim == 1:
            new_pdfs = new_pdfs[np.newaxis, :]

        assignments = []
        for pdf in new_pdfs:
            distances = self._compute_distance_matrix()
            probabilities = self.cluster_priors * np.exp(-distances)
            probabilities /= np.sum(probabilities)
            assignments.append(probabilities)

        return np.array(assignments)

    def get_results(self):
        """Trả về responsibilities, centroids, cluster_priors."""
        return self.responsibilities.T.copy(), self.centroids.copy(), self.cluster_priors.copy()

    def get_hard_assignments(self) -> np.ndarray:
        """Trả về nhãn cứng cho từng PDF."""
        return np.argmax(self.responsibilities, axis=1)
