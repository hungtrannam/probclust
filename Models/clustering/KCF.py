import numpy as np
from utils.dist import Dist


class Model:
    """
    K-Means Clustering cho hàm mật độ xác suất (PDF).
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
            Số vòng lặp tối đa.
        tolerance : float
            Ngưỡng hội tụ.
        distance_metric : str
            Loại khoảng cách ('L1', 'L2', 'H', 'BC', 'W2').
        bandwidth : float
            Tham số bandwidth cho tính khoảng cách.
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
        self.cluster_assignments = None
        self.partition_matrix = None
        self.objective_history = []

    # ------------------------------
    # TÍNH KHOẢNG CÁCH
    # ------------------------------

    def _compute_distance_matrix(self) -> np.ndarray:
        """Tính ma trận khoảng cách [num_pdfs, num_clusters]."""
        d_obj = Dist(h=self.bandwidth, Dim=1, grid=self.grid_x)

        num_pdfs = self.pdf_matrix.shape[0]
        return np.array([
            [getattr(d_obj, self.distance_metric)(self.pdf_matrix[i], self.centroids[j]) + 1e-10
             for j in range(self.num_clusters)]
            for i in range(num_pdfs)
        ])

    # ------------------------------
    # HUẤN LUYỆN (FIT)
    # ------------------------------
    def fit(self, pdf_matrix: np.ndarray) -> None:
        """Huấn luyện K-Means."""
        self.pdf_matrix = pdf_matrix
        self.num_pdfs, self.num_points = pdf_matrix.shape

        if self.seed is not None:
            np.random.seed(self.seed)

        # Khởi tạo ngẫu nhiên cluster assignments
        self.cluster_assignments = np.random.randint(0, self.num_clusters, self.num_pdfs)

        # Khởi tạo centroids
        self.centroids = np.zeros((self.num_clusters, self.num_points))

        self.objective_history.clear()
        for iteration in range(self.max_iterations):
            prev_assignments = self.cluster_assignments.copy()

            # Cập nhật centroids
            for k in range(self.num_clusters):
                member_indices = np.where(self.cluster_assignments == k)[0]
                if len(member_indices) > 0:
                    self.centroids[k] = np.mean(self.pdf_matrix[member_indices], axis=0)
                else:
                    # Cụm rỗng: chọn lại ngẫu nhiên
                    rand_idx = np.random.randint(0, self.num_pdfs)
                    self.centroids[k] = self.pdf_matrix[rand_idx]

            # Cập nhật assignments
            distance_matrix = self._compute_distance_matrix()
            self.cluster_assignments = np.argmin(distance_matrix, axis=1)

            # Tính hàm mục tiêu
            dist_matrix = self._compute_distance_matrix()

            self.objective_value = np.sum([dist_matrix])
            self.objective_history.append(self.objective_value)

            # Kiểm tra hội tụ
            num_changed = np.sum(prev_assignments != self.cluster_assignments)
            if self.verbose:
                print(f"Iteration {iteration + 1}, changed = {num_changed}, objective = {self.objective_value:.6f}")
            if num_changed == 0:
                if self.verbose:
                    print("Converged.")
                break

        # Tạo ma trận one-hot
        self.partition_matrix = np.zeros((self.num_pdfs, self.num_clusters))
        self.partition_matrix[np.arange(self.num_pdfs), self.cluster_assignments] = 1

    # ------------------------------
    # DỰ ĐOÁN
    # ------------------------------
    def predict(self, new_pdfs: np.ndarray) -> np.ndarray:
        """Dự đoán nhãn cứng cho PDF mới."""
        if new_pdfs.ndim == 1:
            new_pdfs = new_pdfs[np.newaxis, :]

        cluster_indices = []
        for pdf in new_pdfs:
            distances = np.array([
                self._compute_distance(pdf, self.centroids[k]) + 1e-10
                for k in range(self.num_clusters)
            ])
            cluster_indices.append(np.argmin(distances))

        return np.array(cluster_indices)

    # ------------------------------
    # KẾT QUẢ
    # ------------------------------
    def get_results(self):
        """Trả về (partition_matrix.T, centroids, objective_history)."""
        return self.partition_matrix.T.copy(), self.centroids.copy(), self.objective_history.copy()

    def get_hard_assignments(self) -> np.ndarray:
        """Trả về nhãn cụm của mỗi PDF."""
        return self.cluster_assignments.copy()
