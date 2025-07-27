import numpy as np
from utils.dist import Dist


class Model:
    """
    Kernel Fuzzy C-Means clustering cho hàm mật độ xác suất (PDF).
    """

    def __init__(
        self,
        grid_x: np.ndarray,
        num_clusters: int = 3,
        fuzziness: float = 2.0,
        max_iterations: int = 100,
        tolerance: float = 1e-5,
        kernel_type: str = "L2",
        gamma: float = 1.0,
        bandwidth: float = 0.01,
        seed: int = None,
        verbose: bool = False,
    ):
        self.grid_x = grid_x
        self.num_clusters = num_clusters
        self.fuzziness = fuzziness
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.kernel_type = kernel_type
        self.gamma = gamma
        self.bandwidth = bandwidth
        self.seed = seed
        self.verbose = verbose

        self.pdf_matrix = None
        self.membership_matrix = None
        self.centroids = None
        self.kernel_matrix = None
        self.objective_history = []

    # ===============================
    # KERNEL FUNCTION
    # ===============================
    def _kernel_function(self, pdf1: np.ndarray, pdf2: np.ndarray) -> float:
        """Hàm kernel dựa trên khoảng cách."""
        d_obj = Dist(pdf1, pdf2, h=self.bandwidth, Dim=1, grid=self.grid_x)
        if self.kernel_type == "L1":
            return np.exp(-self.gamma * d_obj.L1() ** 2)
        elif self.kernel_type == "L2":
            return np.exp(-self.gamma * d_obj.L2() ** 2)
        elif self.kernel_type == "BC":
            return np.exp(-d_obj.BC())
        else:
            raise ValueError(f"Unsupported kernel type: {self.kernel_type}")

    def _compute_kernel_matrix(self) -> np.ndarray:
        """Tính ma trận kernel giữa các PDF."""
        K = np.zeros((self.num_pdfs, self.num_pdfs))
        for i in range(self.num_pdfs):
            for j in range(i, self.num_pdfs):
                val = self._kernel_function(self.pdf_matrix[i], self.pdf_matrix[j])
                K[i, j] = K[j, i] = val
        return K

    # ===============================
    # FIT
    # ===============================
    def fit(self, pdf_matrix: np.ndarray) -> None:
        self.pdf_matrix = pdf_matrix
        self.num_pdfs = pdf_matrix.shape[0]

        if self.seed is not None:
            np.random.seed(self.seed)

        # Khởi tạo membership matrix U
        self.membership_matrix = np.random.dirichlet(np.ones(self.num_clusters), size=self.num_pdfs)

        # Khởi tạo centroids
        init_idx = np.random.choice(self.num_pdfs, self.num_clusters, replace=False)
        self.centroids = pdf_matrix[init_idx]

        # Tiền tính kernel matrix
        self.kernel_matrix = self._compute_kernel_matrix()

        eps = 1e-100
        self.objective_history.clear()

        for iteration in range(self.max_iterations):
            U_prev = self.membership_matrix.copy()

            # Step 1: Tính kernel đến centroids
            kernel_to_centroids = np.zeros((self.num_pdfs, self.num_clusters))
            for i in range(self.num_pdfs):
                for j in range(self.num_clusters):
                    kernel_to_centroids[i, j] = self._kernel_function(self.pdf_matrix[i], self.centroids[j])

            # Step 2: Cập nhật membership matrix
            if self.kernel_type == "BC":
                distances = 1 - kernel_to_centroids
            else:
                distances = np.sqrt(2 * (1 - kernel_to_centroids))

            power = -1 / (self.fuzziness - 1)
            for i in range(self.num_pdfs):
                inv_dist = (distances[i] + eps) ** power
                self.membership_matrix[i] = inv_dist / np.sum(inv_dist)

            # Step 3: Cập nhật centroids
            for j in range(self.num_clusters):
                weights = self.membership_matrix[:, j] ** self.fuzziness
                weighted_kernel = weights * kernel_to_centroids[:, j]
                numerator = np.sum(weighted_kernel[:, None] * self.pdf_matrix, axis=0)
                self.centroids[j] = numerator / (np.sum(weighted_kernel) + eps)

            # Step 4: Tính hàm mục tiêu và kiểm tra hội tụ
            self.objective_value = np.sum((self.membership_matrix ** self.fuzziness) * distances)
            self.objective_history.append(self.objective_value)

            delta = np.linalg.norm(self.membership_matrix - U_prev)
            if self.verbose:
                print(f"Iteration {iteration + 1}, delta = {delta:.6f}, objective = {self.objective_value:.6f}")
            if delta < self.tolerance:
                if self.verbose:
                    print("Converged.")
                break

    # ===============================
    # PREDICT
    # ===============================
    def predict(self, new_pdfs: np.ndarray) -> np.ndarray:
        """Dự đoán membership cho các PDF mới."""
        if new_pdfs.ndim == 1:
            new_pdfs = new_pdfs[np.newaxis, :]

        memberships = []
        eps = 1e-100
        power = -1 / (self.fuzziness - 1)

        for pdf in new_pdfs:
            kernel_to_pdfs = np.array([self._kernel_function(pdf, self.pdf_matrix[i]) for i in range(self.num_pdfs)])
            kernel_to_centroids = np.zeros(self.num_clusters)
            for j in range(self.num_clusters):
                weights = self.membership_matrix[:, j] ** self.fuzziness
                kernel_to_centroids[j] = np.sum(weights * kernel_to_pdfs) / (np.sum(weights) + eps)

            if self.kernel_type == "BC":
                distances = 1 - kernel_to_centroids
            else:
                distances = np.sqrt(2 * (1 - kernel_to_centroids))

            inv_dist = distances ** power
            memberships.append(inv_dist / np.sum(inv_dist))

        return np.array(memberships)

    # ===============================
    # GET RESULTS
    # ===============================
    def get_results(self):
        """Trả về (U.T, centroids, objective_history)."""
        return self.membership_matrix.T.copy(), self.centroids.copy(), self.objective_history.copy()

    def get_hard_assignments(self):
        """Trả về nhãn cứng cho mỗi PDF."""
        return np.argmax(self.membership_matrix, axis=1)
