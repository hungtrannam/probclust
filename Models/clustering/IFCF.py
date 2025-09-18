import numpy as np
from utils.dist import Dist  # khoảng cách cho PDF

class Model:
    """
    Improved Fuzzy C-Means (IFCM) cho PDF rời rạc (Liu et al. 2017).
    - Mục tiêu:
        J(U, Θ) = sum_{i,j} (u_{ij}^m / f_j) * ||X_i - θ_j||^2
          với f_j = sum_i u_{ij}.
    - Cập nhật:
        u_{ij} ∝ (f_j / ||X_i - θ_j||^2)^(1/(m-1))
        θ_j    = (∑ u_{ij}^m X_i) / (∑ u_{ij}^m)
    - Không có δ_i, không có D*.
    - U có shape (K, N).
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
        self.K = int(num_clusters)
        self.m = float(fuzziness)
        self.maxit = int(max_iterations)
        self.tol = float(tolerance)
        self.init = init
        self.distance_metric = distance_metric
        self.bandwidth = bandwidth
        self.Dim = 1 if Dim is None else int(Dim)
        self.seed = seed
        self.verbose = verbose
        self.eps = eps

        self.pdf_matrix = None
        self.N = None
        self.G = None
        self.U = None       # (K,N)
        self.Theta = None   # (K,G)
        self.obj_hist = []

    # --------- khoảng cách bình phương (K,N) ---------
    def _dist2_matrix_to(self, Theta):
        dobj = Dist(h=self.bandwidth, Dim=self.Dim, grid=self.grid_x)
        func = getattr(dobj, self.distance_metric)
        D2 = np.empty((self.K, self.N))
        for j in range(self.K):
            for i in range(self.N):
                d = func(self.pdf_matrix[i], Theta[j])
                D2[j, i] = d**2 + self.eps
        return D2

    # --------- f_j (fuzzy size) ---------
    # def _fuzzy_sizes(self, U):
    #     """
    #     Tính f_j (ω_j) theo Liu 2017 nhưng dùng hard assignment.
    #     - U: (K,N)
    #     - Trả về: ω (K,)
    #     """
    #     # Hard assignment
    #     labels = np.argmax(U, axis=0)   # (N,)
    #     nj = np.array([(labels == j).sum() for j in range(self.K)], dtype=float)  # (K,)
    #     n = nj.sum()

    #     # Tỉ lệ cụm
    #     ratio = nj / (n + self.eps)   # (K,)

    #     # Công thức ω_j
    #     denom = 1.0 - np.min(ratio)
    #     omega = (1.0 - ratio) / (denom + self.eps)

    #     return omega
    def _fuzzy_sizes(self, U: np.ndarray) -> np.ndarray: # U: (K, N) -> f: (K,) 
        f = U.sum(axis=1) / (sum(U.sum(axis=1)) + self.eps)  # (K,)
        f = np.clip(f, self.eps, None, out=f)  # tránh chia 0
        return f

    # --------- cập nhật θ ---------
    def _update_centroids(self, U):
        W = U ** self.m
        num = W @ self.pdf_matrix
        den = np.sum(W, axis=1, keepdims=True)
        return num / (den + self.eps)

    # --------- cập nhật U ---------
    def _update_U(self, f, D2):
        U_new = np.zeros((self.K, self.N))
        p = 1.0 / (self.m - 1.0)
        for i in range(self.N):
            # đúng công thức: (ω_i / D2_{ij})^p
            vals = (f / D2[:, i]) ** p
            U_new[:, i] = vals / (vals.sum() + self.eps)
        return U_new



    # --------- mục tiêu ---------
    def _objective(self, U, f, D2):
        return float(np.sum((U ** self.m) * (D2 / f[:, None])))

    # --------- khởi tạo θ ---------
    def _init_centroids_kmeanspp(self, X):
        N = X.shape[0]
        dobj = Dist(h=self.bandwidth, Dim=self.Dim, grid=self.grid_x)
        func = getattr(dobj, self.distance_metric)

        indices = [np.random.randint(N)]
        for _ in range(1, self.K):
            d2 = np.array([min((func(X[i], X[j])**2) for j in indices) for i in range(N)])
            probs = d2 / (d2.sum() + self.eps)
            indices.append(np.random.choice(N, p=probs))
        return X[indices, :].copy()

    # --------- fit ---------
    def fit(self, pdf_matrix):
        X = np.asarray(pdf_matrix, dtype=float)
        self.pdf_matrix = X
        self.N, self.G = X.shape
        rng = np.random.default_rng(self.seed)

        # init U
        self.U = rng.random((self.K, self.N))
        self.U /= self.U.sum(axis=0, keepdims=True) + self.eps

        # init Θ
        if self.init == "random":
            indices = rng.choice(self.N, size=self.K, replace=False)
            self.Theta = X[indices, :].copy()
        else:
            self.Theta = self._init_centroids_kmeanspp(X)

        import matplotlib.pyplot as plt
        plt.figure()
        plt.title("Khởi tạo tâm kmeans++")
        for j in range(self.K):
            plt.plot(self.grid_x, self.Theta[j], label=f'Centroid {j}')
        plt.show()

        self.obj_hist = []
        J_prev = None

        for it in range(1, self.maxit + 1):
            Theta_tm1 = self.Theta.copy()

            # bước chính
            D2 = self._dist2_matrix_to(Theta_tm1)
            f = self._fuzzy_sizes(self.U)
            U_new = self._update_U(f, D2)
            self.Theta = self._update_centroids(U_new)
            J = self._objective(U_new, f, D2)

            self.obj_hist.append(J)
            dU = np.linalg.norm(U_new - self.U)
            dTheta = np.linalg.norm(self.Theta - Theta_tm1)
            dJ = abs(J - J_prev) if J_prev is not None else np.inf

            if self.verbose:
                print(f"[IFCM] it={it:03d} | dU={dU:.3e} | dTheta={dTheta:.3e} "
                      f"| ΔJ={dJ:.3e} | J={J:.6e} | f={f}")

            self.U = U_new
            J_prev = J

            if dJ < self.tol:
                if self.verbose: print("Converged by ΔJ.")
                break

        return self

    # --------- predict ---------
    def predict(self, new_pdfs):
        Xn = np.asarray(new_pdfs, dtype=float)
        Nn = Xn.shape[0]
        f = self._fuzzy_sizes(self.U)

        dobj = Dist(h=self.bandwidth, Dim=self.Dim, grid=self.grid_x)
        func = getattr(dobj, self.distance_metric)
        D2 = np.empty((self.K, Nn))
        for j in range(self.K):
            for i in range(Nn):
                d = func(Xn[i], self.Theta[j]) + 1e-30
                D2[j, i] = d**2
        return self._update_U(f, D2)

    # --------- tiện ích ---------
    def get_results(self):
        return self.U.copy(), self.Theta.copy(), list(self.obj_hist)

    def get_hard_assignments(self):
        return np.argmax(self.U, axis=0)
