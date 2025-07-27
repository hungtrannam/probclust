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
        self.verbose = verbose

        self.pdf_matrix = None
        self.centroids = None
        self.responsibilities = None
        self.cluster_priors = None

    def _compute_distance(self, pdf1: np.ndarray, pdf2: np.ndarray) -> float:
        """Tính khoảng cách giữa 2 PDF."""
        dist_obj = Dist(pdf1, pdf2, h=self.bandwidth, Dim=1, grid=self.grid_x)
        distance_map = {
            "L1": dist_obj.L1(),
            "L2": dist_obj.L2(),
            "H": dist_obj.H(),
            "BC": dist_obj.BC(),
            "W2": dist_obj.W2(),
        }
        return distance_map[self.distance_metric]

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
        distance_matrix = np.array([
            [self._compute_distance(self.pdf_matrix[i], self.centroids[j]) + 1e-10
             for j in range(self.num_clusters)]
            for i in range(self.pdf_matrix.shape[0])
        ])
        new_responsibilities = np.exp(-distance_matrix) * self.cluster_priors[np.newaxis, :]
        new_responsibilities /= np.sum(new_responsibilities, axis=1, keepdims=True)
        return new_responsibilities

    def fit(self, pdf_matrix: np.ndarray) -> None:
        """Huấn luyện EM."""
        self.pdf_matrix = pdf_matrix
        num_pdfs, num_points = pdf_matrix.shape

        if self.seed is not None:
            np.random.seed(self.seed)

        # Khởi tạo responsibilities
        self.responsibilities = np.random.dirichlet(np.ones(self.num_clusters), size=num_pdfs)

        # Khởi tạo centroids từ dữ liệu
        init_indices = np.random.choice(num_pdfs, self.num_clusters, replace=False)
        self.centroids = pdf_matrix[init_indices, :].copy()

        # Khởi tạo cluster priors
        self.cluster_priors = np.ones(self.num_clusters) / self.num_clusters

        for iteration in range(self.max_iterations):
            # M-step
            self._update_centroids()
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
            distances = np.array([
                self._compute_distance(pdf, self.centroids[j]) + 1e-10
                for j in range(self.num_clusters)
            ])
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
