import numpy as np
from utils.dist import Dist

EPS = 1e-12


class Model:
    """
    K-Means clustering cho probability density functions (PDFs),
    viết theo cùng kiến trúc với FCM.
    """

    def __init__(
        self,
        grid_x: np.ndarray,
        num_clusters: int = 3,
        max_iterations: int = 100,
        tolerance: float = 1e-5,
        distance_metric: str = "L2",   # ["L1","L2","H","BC","W2"]
        bandwidth: float = 0.01,
        Dim: int = 1,
        seed: int | None = None,
        verbose: bool = False,
    ):
        self.grid_x = grid_x
        self.num_clusters = num_clusters
        self.max_iterations = max_iterations
        self.tol = tolerance
        self.distance_metric = distance_metric
        self.bandwidth = bandwidth
        self.Dim = Dim
        self.seed = seed
        self.verbose = verbose

        # distance object
        self.dist_obj = Dist(h=self.bandwidth, Dim=self.Dim, grid=self.grid_x)
        self.dist_func = getattr(self.dist_obj, self.distance_metric)

        # placeholders
        self.pdf_matrix = None
        self.Theta = None   # centroids
        self.U = None       # membership one-hot
        self.objective_history = []

    # ---------------- Core update steps ---------------- #
    def _update_centroids(self, U: np.ndarray) -> np.ndarray:
        """
        Cập nhật centroids (PDF hợp lệ) theo loại khoảng cách.
        U là ma trận one-hot (hard assignment).
        """
        centroids = []
        for k in range(self.num_clusters):
            idx = np.where(U[:, k] == 1)[0]
            if len(idx) == 0:
                # cụm rỗng → khởi tạo lại ngẫu nhiên
                centroids.append(self.pdf_matrix[np.random.choice(len(self.pdf_matrix))])
                continue

            cluster_pdfs = self.pdf_matrix[idx]

            if self.distance_metric in ["L1", "L2"]:
                # trung bình cộng PDF
                theta = np.mean(cluster_pdfs, axis=0)

            elif self.distance_metric == "H":
                # Hellinger: làm việc trên sqrt(PDF)
                sqrt_pdfs = np.sqrt(cluster_pdfs)
                theta = (np.mean(sqrt_pdfs, axis=0)) ** 2

            elif self.distance_metric == "W2":
                # Wasserstein: barycenter qua quantile
                G = len(self.grid_x)
                cdfs = np.cumsum(cluster_pdfs, axis=1) * self.bandwidth
                cdfs = np.clip(cdfs, 0, 1)
                t = np.linspace(0, 1, G)

                invs = [np.interp(t, cdfs[i], self.grid_x) for i in range(len(cluster_pdfs))]
                invs = np.stack(invs, axis=0)
                inv_cent = np.mean(invs, axis=0)  # barycenter ở space quantile

                F_cent = np.interp(self.grid_x, inv_cent, t)
                theta = np.gradient(F_cent, self.grid_x)

            else:
                raise ValueError(f"Unknown distance metric {self.distance_metric}")

            # ép PDF hợp lệ
            theta = np.clip(theta, 0, None)
            theta /= np.trapz(theta, self.grid_x)
            centroids.append(theta)

        return np.stack(centroids, axis=0)


    def _dist_matrix(self) -> np.ndarray:
        """Ma trận khoảng cách (N,K)."""
        N = self.pdf_matrix.shape[0]
        dist = np.zeros((N, self.num_clusters))
        for i in range(N):
            for j in range(self.num_clusters):
                dist[i, j] = self.dist_func(self.pdf_matrix[i], self.Theta[j]) + EPS
        return dist

    def _update_membership(self, dist_matrix: np.ndarray) -> np.ndarray:
        """Cập nhật membership one-hot (hard assignment)."""
        N = dist_matrix.shape[0]
        labels = np.argmin(dist_matrix, axis=1)
        U = np.zeros((N, self.num_clusters))
        U[np.arange(N), labels] = 1.0
        return U

    # ---------------- Public API ---------------- #
    def fit(self, pdf_matrix: np.ndarray) -> None:
        """Huấn luyện K-means."""
        self.pdf_matrix = pdf_matrix
        self.num_pdfs = pdf_matrix.shape[0]

        if self.seed is not None:
            np.random.seed(self.seed)

        # init centroids chọn ngẫu nhiên từ dữ liệu
        init_idx = np.random.choice(self.num_pdfs, self.num_clusters, replace=False)
        self.Theta = pdf_matrix[init_idx, :]

        self.objective_history.clear()

        for it in range(self.max_iterations):
            # update membership
            dist_matrix = self._dist_matrix()
            new_U = self._update_membership(dist_matrix)

            # update centroids
            new_Theta = self._update_centroids(new_U)

            # objective
            J = np.sum(new_U * dist_matrix)
            self.objective_history.append(J)

            # hội tụ
            delta = np.linalg.norm(new_Theta - self.Theta)
            if self.verbose:
                print(f"[Iter {it+1}] Δθ={delta:.3e}, J={J:.6f}")

            if delta < self.tol:
                if self.verbose:
                    print(f"Converged at iteration {it+1}")
                break

            self.Theta = new_Theta
            self.U = new_U

    def predict(self, new_pdfs: np.ndarray) -> np.ndarray:
        """Trả về nhãn cứng cho dữ liệu mới."""
        if new_pdfs.ndim == 1:
            new_pdfs = new_pdfs[None, :]

        labels = []
        for pdf in new_pdfs:
            d = np.array([self.dist_func(pdf, self.Theta[j]) + EPS for j in range(self.num_clusters)])
            labels.append(int(np.argmin(d)))
        return np.array(labels)

    def get_results(self):
        """Trả về U.T, centroids, và lịch sử hàm mục tiêu."""
        return self.U.T.copy(), self.Theta.copy(), self.objective_history.copy()

    def get_hard_assignments(self) -> np.ndarray:
        """Trả về nhãn cứng (N,)."""
        return np.argmax(self.U, axis=1)
