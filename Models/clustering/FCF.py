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
        self.verbose = verbose
        self.Dim = Dim if Dim is not None else 1


        self.pdf_matrix = None
        self.centroids = None
        self.membership_matrix = None
        self.objective_history = []

    def _compute_distance(self, pdf1: np.ndarray, pdf2: np.ndarray) -> float:
        """Tính khoảng cách giữa 2 PDF."""
        dist_obj = Dist(pdf1, pdf2, h=self.bandwidth,Dim=self.Dim, grid=self.grid_x)
        distance_map = {
            "L1": dist_obj.L1(),
            "L2": dist_obj.L2(),
            "H": dist_obj.H(),
            "BC": dist_obj.BC(),
            "W2": dist_obj.W2(),
        }
        return distance_map[self.distance_metric]

    def _update_centroids(self) -> None:
        """Cập nhật tâm cụm (centroid)."""
        weights = self.membership_matrix ** self.fuzziness
        self.centroids = (weights.T @ self.pdf_matrix) / (np.sum(weights.T, axis=1, keepdims=True) + 1e-12)

    def _compute_distance_matrix(self) -> np.ndarray:
        """Tính ma trận khoảng cách [num_pdfs, num_clusters]."""
        num_pdfs = self.pdf_matrix.shape[0]
        return np.array([
            [self._compute_distance(self.pdf_matrix[i], self.centroids[j]) + 1e-10
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
            self._update_centroids()
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
