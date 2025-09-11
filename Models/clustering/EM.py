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
        gamma: float = 0.1,
        seed: int = None,
        Dim=None,
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
        self.Dim = Dim if Dim is not None else 1
        self.verbose = verbose
        self.gamma = gamma

        self.pdf_matrix = None
        self.Theta = None
        self.responsibilities = None
        self.cluster_priors = None

    def _compute_distance_matrix(self) -> np.ndarray:
        """Tính ma trận khoảng cách [num_pdfs, num_clusters]."""
        d_obj = Dist(h=self.bandwidth, Dim=self.Dim, grid=self.grid_x)

        num_pdfs = self.pdf_matrix.shape[0]
        return np.array([
            [getattr(d_obj, self.distance_metric)(self.pdf_matrix[i], self.Theta[j])**2 + 1e-10
             for j in range(self.num_clusters)]
            for i in range(num_pdfs)
        ])

    def _update_centroids(self) -> None:
        """Cập nhật centroid dựa trên responsibilities."""
        for j in range(self.num_clusters):
            weights = self.responsibilities[:, j]
            numerator = np.sum(weights[:, np.newaxis] * self.pdf_matrix, axis=0)
            denominator = np.sum(weights) + 1e-12
            self.Theta[j, :] = numerator / denominator

    def _update_cluster_priors(self) -> None:
        """Cập nhật trọng số cụm (priors)."""
        self.cluster_priors = np.sum(self.responsibilities, axis=0) / self.pdf_matrix.shape[0]


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
        self.Theta = pdf_matrix[init_indices, :].copy()

        # Khởi tạo cluster priors
        self.cluster_priors = np.ones(self.num_clusters) / self.num_clusters

        for iteration in range(self.max_iterations):
            if self.verbose:
                print(f"\n[Iteration {iteration + 1}]")

            # M-step
            self._update_centroids()
            self._update_cluster_priors()

            if self.verbose:
                print("  ➤ M-step ")
                print(f"    - Cluster priors: {np.round(self.cluster_priors, 4)}")
                cent_norms = [np.linalg.norm(c) for c in self.Theta]
                print(f"    - Centroid norms: {[f'{n:.4f}' for n in cent_norms]}")

            # E-step
            dist_matrix = self._compute_distance_matrix()
            new_responsibilities = np.exp(-dist_matrix / self.gamma) * self.cluster_priors[np.newaxis, :]

            # Normalize responsibilities
            new_responsibilities /= np.sum(new_responsibilities, axis=1, keepdims=True)

            if self.verbose:
                entropy = -np.sum(new_responsibilities * np.log(new_responsibilities + 1e-12)) / self.num_pdfs
                print(f"  ➤ E-step. Responsibility entropy: {entropy:.4f}")

            # Kiểm tra hội tụ
            delta = np.linalg.norm(new_responsibilities - self.responsibilities)
            if self.verbose:
                print(f"  ➤ ΔR = {delta:.6e}")
            if delta < self.tolerance:
                if self.verbose:
                    print("✔ Converged.")
                break

            self.responsibilities = new_responsibilities


    def predict(self, new_pdfs: np.ndarray) -> np.ndarray:
        if new_pdfs.ndim == 1:
            new_pdfs = new_pdfs[np.newaxis, :]

        d_obj = Dist(h=self.bandwidth, Dim=self.Dim, grid=self.grid_x)
        func = getattr(d_obj, self.distance_metric)

        soft_assignments = []
        for pdf in new_pdfs:
            dists = np.array([func(pdf, c)**2 + 1e-10 for c in self.Theta])
            probs = self.cluster_priors * np.exp(-dists)
            probs /= np.sum(probs)
            soft_assignments.append(probs)
        return np.array(soft_assignments)


    def get_results(self):
        """Trả về responsibilities, centroids, cluster_priors."""
        return self.responsibilities.T.copy(), self.Theta.copy(), self.cluster_priors.copy()

    def get_hard_assignments(self) -> np.ndarray:
        """Trả về nhãn cứng cho từng PDF."""
        return np.argmax(self.responsibilities, axis=1)
