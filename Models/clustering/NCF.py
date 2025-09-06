import numpy as np
from utils.dist import Dist


class Model:
    """
    Neutrosophic Clustering cho dữ liệu hàm mật độ xác suất (PDF).
    Bao gồm 3 thành phần: 
    - T (membership - độ thuộc)
    - I (indeterminacy - độ không xác định)
    - F (hesitation - độ do dự)
    """

    def __init__(
        self,
        grid_x: np.ndarray,
        num_clusters: int = 3,
        m: float = 2.0,
        delta: float = 1.0,
        w1: float = 1.0,
        w2: float = 1.0,
        w3: float = 1.0,
        max_iterations: int = 100,
        tolerance: float = 1e-5,
        distance_metric: str = "L2",
        bandwidth: float = 0.01,
        seed: int = None,
        verbose: bool = False,
    ):
        self.grid_x = grid_x
        self.c = num_clusters
        self.m = m
        self.delta = delta
        self.w1, self.w2, self.w3 = w1, w2, w3
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.distance_metric = distance_metric
        self.bandwidth = bandwidth
        self.seed = seed
        self.verbose = verbose
        self.eps = 1e-10

        self.T, self.I, self.F, self.C = None, None, None, None

    # ===============================
    # DISTANCE
    # ===============================

    def _compute_distance(self,f1,f2) -> np.ndarray:
        """Tính ma trận khoảng cách [num_pdfs, num_clusters]."""
        d_obj = Dist(h=self.bandwidth, Dim=1, grid=self.grid_x)

        return np.array([
            [getattr(d_obj, self.distance_metric)(f1,f2) + 1e-10
             for j in range(self.num_clusters)]
            for i in range(self.num_pdfs)
        ])

    # ===============================
    # FIT
    # ===============================
    def fit(self, X: np.ndarray) -> None:
        """Huấn luyện mô hình NCF."""
        N, D = X.shape
        c, m, delta = self.c, self.m, self.delta

        if self.seed is not None:
            np.random.seed(self.seed)

        # Khởi tạo
        C = X[np.random.choice(N, c, replace=False)]
        T = np.full((c, N), 1 / c)
        I = np.full(N, 1 / (c + 2))
        F = np.full(N, 1 / (c + 2))
        power = -2 / (m - 1)

        for iteration in range(self.max_iterations):
            C_prev = C.copy()

            # Step 1: tính c_i_max (trung bình 2 cụm gần nhất)
            c_i_max = np.zeros((N, D))
            dist_matrix = np.zeros((N, c))
            for i in range(N):
                dists = [self._compute_distance(X[i], C[j]) for j in range(c)]
                dist_matrix[i] = dists
                p, q = np.argsort(dists)[:2]
                c_i_max[i] = (C[p] + C[q]) / 2

            # Step 2: tính khoảng cách dist_I
            dist_I = np.array([self._compute_distance(X[i], c_i_max[i]) for i in range(N)])
            dist_F = max(delta, self.eps)

            # Step 3: tính K_i
            K = 1.0 / (
                np.sum((1 / self.w1) * dist_matrix ** power, axis=1)
                + (1 / self.w2) * dist_I ** power
                + (1 / self.w3) * dist_F ** power
                + self.eps
            )

            # Step 4: cập nhật T, I, F
            for i in range(N):
                T[:, i] = K[i] / self.w1 * dist_matrix[i] ** power
                I[i] = K[i] / self.w2 * dist_I[i] ** power
                F[i] = K[i] / self.w3 * dist_F ** power

            # Step 5: cập nhật C_j
            for j in range(c):
                weights = (self.w1 * T[j]) ** m
                C[j] = np.sum(weights[:, None] * X, axis=0) / (np.sum(weights) + self.eps)

            # Step 6: kiểm tra hội tụ
            delta_c = np.linalg.norm(C - C_prev)
            if self.verbose:
                print(f"Iteration {iteration + 1}, delta = {delta_c:.6e}")
            if delta_c < self.tolerance or np.isnan(delta_c):
                if self.verbose:
                    print("Converged or stopped.")
                break

        self.T, self.I, self.F, self.C = T, I, F, C

    # ===============================
    # PREDICT
    # ===============================
    def predict(self, new_X: np.ndarray) -> np.ndarray:
        """Dự đoán membership cho dữ liệu mới."""
        if new_X.ndim == 1:
            new_X = new_X[np.newaxis, :]

        num_new = new_X.shape[0]
        memberships = np.zeros((self.c, num_new))
        power = -2 / (self.m - 1)

        for i in range(num_new):
            dists = np.array([self._compute_distance(new_X[i], self.C[j]) for j in range(self.c)])
            sum_T = np.sum((1 / self.w1) * dists ** power)
            sum_I = (1 / self.w2) * np.min(dists) ** power
            sum_F = (1 / self.w3) * self.delta ** power
            K_i = 1.0 / (sum_T + sum_I + sum_F + self.eps)
            memberships[:, i] = K_i / self.w1 * dists ** power

        return memberships

    # ===============================
    # GET RESULTS
    # ===============================
    def get_results(self):
        """Trả về ma trận T, I, F và các centroid C."""
        return self.T.copy(), self.I.copy(), self.F.copy(), self.C.copy()
