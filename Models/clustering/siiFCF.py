import numpy as np
from utils.dist import Dist


class Model:
    """
    Size-Insensitive Integrity-based Fuzzy C-Means (SiiFCM)
    dành cho hàm mật độ xác suất (PDFs).
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
        centroid_mode: str = "mean",  # 'mean' hoặc 'frechet'
        Dim=None,
        seed: int = None,
        verbose: bool = False,
    ):
        self.grid_x = grid_x
        self.num_clusters = num_clusters
        self.fuzziness = fuzziness
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.distance_metric = distance_metric
        self.bandwidth = bandwidth
        self.centroids_mode = centroid_mode
        self.seed = seed
        self.verbose = verbose
        self.Dim = Dim if Dim is not None else 1

        # Dữ liệu và kết quả
        self.pdf_matrix = None
        self.centroids = None
        self.membership_matrix = None
        self.objective_history = []

    # ------------------------------------------------------------------
    # Hàm hỗ trợ
    # ------------------------------------------------------------------
    @staticmethod
    def _cdf_from_pdf(pdf: np.ndarray, dx: float) -> np.ndarray:
        cdf = np.cumsum(pdf) * dx
        Z = float(cdf[-1]) if cdf.size else 1.0
        if Z <= 0:
            n = pdf.size
            return np.linspace(0.0, 1.0, n)
        return cdf / Z

    def _update_centroids_w2(self) -> None:
        """Cập nhật tâm cụm bằng Wasserstein-2 barycenter (1D)."""
        assert self.Dim == 1, "W2 barycenter chỉ hỗ trợ 1D."
        dx = float(self.grid_x[1] - self.grid_x[0])
        n_samples, m = self.pdf_matrix.shape
        K = self.num_clusters

        t_grid = np.linspace(0.0, 1.0, m)
        inv_mat = np.empty((n_samples, m))

        for j in range(n_samples):
            cdf_j = self._cdf_from_pdf(self.pdf_matrix[j], dx)
            inv_mat[j] = np.interp(t_grid, cdf_j, self.grid_x)

        W = self.membership_matrix ** self.fuzziness
        denom = np.sum(W, axis=0, keepdims=True) + 1e-12
        W_norm = W / denom

        centroids = np.empty((K, m))
        for k in range(K):
            Qk = (W_norm[:, k][:, None] * inv_mat).sum(axis=0)
            Qk = np.maximum.accumulate(Qk)
            Fk = np.interp(self.grid_x, Qk, t_grid, left=0.0, right=1.0)
            fk = np.gradient(Fk, self.grid_x)
            centroids[k] = np.clip(fk, 0.0, None)

        self.centroids = centroids

    def _update_centroids(self) -> None:
        """Cập nhật tâm cụm bằng trung bình trọng số mờ."""
        weights = self.membership_matrix ** self.fuzziness
        self.centroids = (weights.T @ self.pdf_matrix) / (np.sum(weights.T, axis=1, keepdims=True) + 1e-12)

    def _compute_distance_matrix(self) -> np.ndarray:
        """Tính ma trận khoảng cách [num_pdfs, num_clusters]."""
        d_obj = Dist(h=self.bandwidth, Dim=self.Dim, grid=self.grid_x)
        num_pdfs = self.pdf_matrix.shape[0]
        return np.array([
            [getattr(d_obj, self.distance_metric)(self.pdf_matrix[i], self.centroids[j]) + 1e-10
             for j in range(self.num_clusters)]
            for i in range(num_pdfs)
        ])

    # ------------------------------------------------------------------
    # Tính toán chỉ số integrity (Compactness + Separation)
    # ------------------------------------------------------------------
    def _integrity_indices(self, distance_matrix: np.ndarray) -> np.ndarray:
        """Trả về I_star : (K,) theo công thức SiiFCM."""
        hard_labels = np.argmax(self.membership_matrix, axis=1)
        self.num_clusters
        Comp = np.zeros(self.num_clusters)
        P = np.zeros(self.num_clusters)

        for k in range(self.num_clusters):
            mask = hard_labels == k
            nk = mask.sum()
            sqrt_d = np.sqrt(distance_matrix[mask, k])
            mu_k = sqrt_d.sum() / (nk + 1)
            Comp[k] = 1 - np.sqrt(np.sum((sqrt_d - mu_k) ** 2) / (nk + 1))

        # Separation
        for k in range(self.num_clusters):
            tmp = np.linalg.norm(self.centroids[k] - self.centroids, axis=1)
            tmp[k] = np.inf
            j = np.argmin(tmp)
            delta = np.abs(
                np.linalg.norm(self.pdf_matrix - self.centroids[k], axis=1) -
                np.linalg.norm(self.pdf_matrix - self.centroids[j], axis=1)
            ) / (np.linalg.norm(self.centroids[k] - self.centroids[j]) + 1e-12)
            P[k] = delta[hard_labels == k].sum() / ((hard_labels == k).sum() + 1)

        I = 0.5 * (Comp + P)
        I_star = (I - I.min()) / (I.max() - I.min() + 1e-12)
        return I_star

    # ------------------------------------------------------------------
    # Cập nhật membership theo SiiFCM
    # ------------------------------------------------------------------
    def _update_membership_matrix(self, distance_matrix: np.ndarray) -> np.ndarray:
        """Tính ma trận membership U với integrity và kích thước cụm."""
        I_star = self._integrity_indices(distance_matrix)
        hard_labels = np.argmax(self.membership_matrix, axis=1)
        S = np.bincount(hard_labels, minlength=self.num_clusters) / self.pdf_matrix.shape[0]
        rho = (1 - S[hard_labels]) / (1 - S).max()
        rho = rho.reshape(-1, 1)

        # Tính p_star (separation score)
        delta = np.zeros((self.pdf_matrix.shape[0], self.num_clusters))
        for k in range(self.num_clusters):
            tmp = np.linalg.norm(self.centroids[k] - self.centroids, axis=1)
            tmp[k] = np.inf
            j = np.argmin(tmp)
            d_k = np.linalg.norm(self.pdf_matrix - self.centroids[k], axis=1)
            d_j = np.linalg.norm(self.pdf_matrix - self.centroids[j], axis=1)
            delta[:, k] = np.abs(d_k - d_j) / (np.linalg.norm(self.centroids[k] - self.centroids[j]) + 1e-12)

        p_star = np.exp((1 - I_star[None, :]) * delta)
        U_aux = p_star * rho

        # Áp dụng công thức fuzzy
        D = distance_matrix + 1e-12
        exp = 1 / (self.fuzziness - 1)
        inv = (D[:, :, None] / D[:, None, :]) ** exp
        denom = np.sum(inv, axis=2)
        U = U_aux / denom

        # Xử lý hàng toàn 0
        zero_rows = U.sum(axis=1) == 0
        if np.any(zero_rows):
            U[zero_rows] = 0
            U[zero_rows, np.argmin(D[zero_rows], axis=1)] = 1
        return U

    # ------------------------------------------------------------------
    # Huấn luyện SiiFCM
    # ------------------------------------------------------------------
    def fit(self, pdf_matrix: np.ndarray) -> None:
        self.pdf_matrix = pdf_matrix
        n_samples = pdf_matrix.shape[0]

        if self.seed is not None:
            np.random.seed(self.seed)

        # Khởi tạo membership và centroid
        self.membership_matrix = np.random.dirichlet(np.ones(self.num_clusters), size=n_samples)
        init_idx = np.random.choice(n_samples, self.num_clusters, replace=False)
        self.centroids = pdf_matrix[init_idx]

        self.objective_history.clear()

        for it in range(self.max_iterations):
            # Cập nhật tâm cụm
            if self.centroids_mode == "mean":
                self._update_centroids()
            elif self.centroids_mode == "frechet":
                self._update_centroids_w2()

            # Cập nhật membership
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

    # ------------------------------------------------------------------
    # Dự đoán và truy xuất kết quả
    # ------------------------------------------------------------------
    def predict(self, new_pdfs: np.ndarray) -> np.ndarray:
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
        return self.membership_matrix.T.copy(), self.centroids.copy(), self.objective_history.copy()

    def get_hard_assignments(self) -> np.ndarray:
        return np.argmax(self.membership_matrix, axis=1)