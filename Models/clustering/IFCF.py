import numpy as np
from utils.dist import Dist

class Model:
    """
    Improved Fuzzy C-Means (IFCM) cho PDF rời rạc.
    - Công thức mục tiêu:
        J(U, Θ) = sum_{i=1}^N sum_{j=1}^K (1/f_j) u_{ij}^m ||X_i - θ_j||^2
      với f_j = (1/N) * sum_{i} u_{ij}.
    - Cập nhật:
        (35)  u_{ij} = 1 / sum_{k=1}^K ( ( (||x_i-θ_j||^2 / f_j) / (||x_i-θ_k||^2 / f_k) )^(1/(m-1)) )
        (37)  θ_j    = (∑ u_{ij}^m X_i) / (∑ u_{ij}^m)
    - KHÔNG có δ_i, KHÔNG có D*.
    - Quy ước: U shape (K, N), D2 shape (K, N).
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
        init: str = "random",
        Dim: int | None = None,
        seed: int | None = None,
        verbose: bool = False,
        eps: float = 1e-12,
    ):
        assert fuzziness > 1.0, "m (fuzziness) phải > 1"
        self.grid_x = grid_x
        self.num_clusters = int(num_clusters)
        self.fuzziness = float(fuzziness)
        self.maxit = int(max_iterations)
        self.tol = float(tolerance)
        self.init = str(init)
        self.distance_metric = str(distance_metric)
        self.bandwidth = float(bandwidth)
        self.Dim = 1 if Dim is None else int(Dim)
        self.seed = seed
        self.verbose = verbose
        self.eps = float(eps)

        # runtime
        self.pdf_matrix = None
        self.num_pdfs = None
        self.num_points = None
        self.U = None       # (K, N)
        self.Theta = None   # (K, G)
        self.obj_hist = []


    # ---------- khởi tạo centroids bằng KMeans++ ----------
    def _init_centroids_kmeanspp(self, pdf_matrix):
        N = pdf_matrix.shape[0]
        K = self.num_clusters

        # đối tượng khoảng cách
        dobj = Dist(h=self.bandwidth, Dim=self.Dim, grid=self.grid_x)
        func = getattr(dobj, self.distance_metric)

        # Nếu chỉ cần 2 centroid -> chọn 2 điểm xa nhau nhất
        if K == 2:
            # chọn ngẫu nhiên centroid đầu tiên
            idx0 = np.random.randint(N)
            d2 = np.array([
                func(pdf_matrix[i], pdf_matrix[idx0]) ** 2
                for i in range(N)
            ])
            idx1 = int(np.argmax(d2))
            indices = [idx0, idx1]
            return pdf_matrix[indices, :].copy()

        # Trường hợp K > 2: dùng KMeans++
        indices = [np.random.randint(N)]  # chọn ngẫu nhiên centroid đầu tiên

        for _ in range(1, K):
            # tính khoảng cách bình phương nhỏ nhất tới centroids đã chọn
            d2 = np.array([
                min((func(pdf_matrix[i], pdf_matrix[j]) ** 2) for j in indices)
                for i in range(N)
            ])
            # chuẩn hoá thành xác suất
            probs = d2 / (d2.sum() + 1e-12)
            # chọn theo phân phối
            next_idx = np.random.choice(N, p=probs)
            indices.append(next_idx)

        return pdf_matrix[indices, :].copy()

    # ---------- khoảng cách bình phương ----------
    def _dist2_matrix_to(self, Theta):
        dobj = Dist(h=self.bandwidth, Dim=self.Dim, grid=self.grid_x)
        func = getattr(dobj, self.distance_metric)
        N, K = self.num_pdfs, self.num_clusters
        D2 = np.empty((K, N), dtype=float)
        for j in range(K):
            tj = Theta[j]
            for i in range(N):
                d = func(self.pdf_matrix[i], tj)
                D2[j, i] = d**2 + 1e-30
        return D2

    # ---------- fuzzy size ----------
    def _fuzzy_sizes(self, U: np.ndarray) -> np.ndarray:
        f = U.sum(axis=1) #/ self.num_pdfs
        f = np.clip(f, self.eps, None)
        return f

    # ---------- cập nhật centroids ----------
    def _update_centroids(self) -> np.ndarray:
        W = self.U ** self.fuzziness
        num = W @ self.pdf_matrix
        den = np.sum(W, axis=1, keepdims=True)
        return num / (den + self.eps)

    # ---------- cập nhật U ----------
    def _update_U(self, omega_prev: np.ndarray, D2: np.ndarray) -> np.ndarray:
        """
        Cập nhật U cho IFCM:
        u_ij = [ (1 / (f_j * D2_ij))^(1/(m-1)) ] / sum_k [...]
        
        Parameters
        ----------
        omega_prev : np.ndarray
            f_j (fuzzy size) với shape (K,)
        D2 : np.ndarray
            khoảng cách bình phương từ mẫu tới tâm, shape (K, N)
        """
        m = self.fuzziness
        p = 1.0 / (m - 1.0)
        K, N = D2.shape
        U_new = np.zeros((K, N))

        for i in range(N):
            denom_total = 0.0
            numerators = []
            for j in range(K):
                # đúng công thức: 1 / (f_j * D2_ij)
                numer = (omega_prev[j] * (D2[j, i] + self.eps)**p)
                numerators.append(numer)
                denom_total += numer
            for j in range(K):
                U_new[j, i] = numerators[j] / (denom_total + self.eps)

        return U_new


    # ---------- hàm mục tiêu ----------
    def _objective(self, U: np.ndarray, omega: np.ndarray, D2: np.ndarray) -> float:
        U_m = U ** self.fuzziness
        term = np.sum(U_m * (D2 / (omega[:, None] + self.eps)))
        return float(term)

    # ---------- fit ----------
    def fit(self, pdf_matrix: np.ndarray):
        X = np.asarray(pdf_matrix, dtype=float)
        self.pdf_matrix = X
        self.num_pdfs, self.num_points = X.shape
        N, K = self.num_pdfs, self.num_clusters
        rng = np.random.default_rng(self.seed)

        # init U
        self.U = rng.random((K, N))
        self.U /= self.U.sum(axis=0, keepdims=True) + self.eps

        # init Theta
        if self.init == "random":
            indices = rng.choice(N, size=K, replace=False)
            self.Theta = X[indices, :].copy()
        else:
            # có thể thêm kmeans++ giống bên EM-IFCM
            self.Theta = self._init_centroids_kmeanspp(X)


        self.obj_hist = []
        J_prev = None

        for it in range(1, self.maxit + 1):
            Theta_tm1 = self.Theta.copy()

            # khoảng cách tới Θ^{t-1}
            D2 = self._dist2_matrix_to(Theta_tm1)
            omega_prev = self._fuzzy_sizes(self.U)

            # U^t
            U_new = self._update_U(omega_prev, D2)

            # Θ^t
            self.Theta = self._update_centroids()

            # mục tiêu
            omega_curr = self._fuzzy_sizes(U_new)
            J = self._objective(U_new, omega_curr, D2)
            self.obj_hist.append(J)

            dU = float(np.linalg.norm(U_new - self.U))
            dTheta = float(np.linalg.norm(self.Theta - Theta_tm1))
            dJ = abs(J - J_prev) if J_prev is not None else np.inf

            if self.verbose:
                print(f"[IFCM] it={it:03d} | dU={dU:.3e} | dTheta={dTheta:.3e} | ΔJ={dJ:.3e} | J={J:.6e} | clusterwidth={omega_prev}")

            self.U = U_new
            J_prev = J

            if dJ < self.tol:
                if self.verbose:
                    print("Converged by ΔJ.")
                break

        return self

    def predict(self, new_pdfs: np.ndarray) -> np.ndarray:
        Xn = np.asarray(new_pdfs, dtype=float)
        Nn = Xn.shape[0]
        omega_prev = self._fuzzy_sizes(self.U)

        dobj = Dist(h=self.bandwidth, Dim=self.Dim, grid=self.grid_x)
        func = getattr(dobj, self.distance_metric)
        K = self.num_clusters
        D2 = np.empty((K, Nn), dtype=float)
        for j in range(K):
            for i in range(Nn):
                d = func(Xn[i], self.Theta[j]) + 1e-30
                D2[j, i] = d**2

        # cập nhật U
        return self._update_U(omega_prev, D2)

    def get_results(self):
        return self.U.copy(), self.Theta.copy(), list(self.obj_hist)

    def get_hard_assignments(self):
        return np.argmax(self.U, axis=0)
