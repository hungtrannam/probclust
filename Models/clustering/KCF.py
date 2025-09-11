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


    def _update_centroids(self) -> None:
        """Cập nhật tâm cụm (centroid)."""
        weights = self.membership_matrix
        self.centroids = (weights.T @ self.pdf_matrix) / (np.sum(weights.T, axis=1, keepdims=True) + 1e-12)


    def _compute_distance_pair(self, x: np.ndarray, y: np.ndarray) -> float:
        """Khoảng cách 2 PDF theo self.distance_metric từ self._dist_obj."""
        return float(getattr(self._dist_obj, self.distance_metric)(x, y))
        

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
        cluster_indices = np.argmin(distance_matrix, axis=1)

        # khởi tạo ma trận one-hot
        U = np.zeros_like(distance_matrix, dtype=float)
        U[np.arange(len(cluster_indices)), cluster_indices] = 1.0
        return U


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
            self.objective_value = np.sum((new_U) * dist_matrix)
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
        out = []
        for pdf in new_pdfs:
            d = [self._compute_distance_pair(pdf, self.centroids[k]) for k in range(self.num_clusters)]
            out.append(int(np.argmin(d)))
        return np.array(out, dtype=int)


    def get_results(self):
        """Trả về U.T, centroids, và lịch sử hàm mục tiêu."""
        return self.membership_matrix.T.copy(), self.centroids.copy(), self.objective_history.copy()

    def get_hard_assignments(self) -> np.ndarray:
        """Trả về nhãn cứng của từng PDF."""
        return np.argmax(self.membership_matrix, axis=1)


