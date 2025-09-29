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



    
    # ------------------------- f_j (fuzzy size) ------------------------- 
    def _fuzzy_sizes(self, U_t_1: np.ndarray) -> np.ndarray:
        f = U_t_1.sum(axis=1) / self.N  # (K,)
        return f

    # --------------------------- cập nhật θ theo (37) --------------------------- 
    def _update_centroids(self, U) -> np.ndarray:
        W = U ** self.m  # (K, N)

        centroids = []
        for k in range(self.K):
            weights = W[k][:, None, None] if self.Dim == 2 else W[k][:, None]  # (N,1,1) or (N,1)
            den = np.sum(weights) + self.eps


            if self.distance_metric == 'H':
                if self.Dim == 2:
                    num = np.sum(weights * np.sqrt(self.pdf_matrix), axis=0)  # (h,w)
                else:
                    num = np.sum(weights * np.sqrt(self.pdf_matrix), axis=0)  # (G,)

                theta = (num / den) ** 2
            elif self.distance_metric in ['L2', 'L1']:
                if self.Dim == 2:
                    num = np.sum(weights * self.pdf_matrix, axis=0)  # (h,w)
                else:
                    num = np.sum(weights * self.pdf_matrix, axis=0)  # (G,)
                    
                theta = num / den
            else:
                if self.Dim == 2:
                    raise NotImplementedError(f"Khoảng cách {self.distance_metric} chưa được hỗ trợ cho IFCF 2D.")
                else:
                    # xử lý Wasserstein / BC
                    G = len(self.grid_x)
                    cdfs = np.cumsum(self.pdf_matrix, axis=1) * self.bandwidth
                    cdfs = np.clip(cdfs, 0, 1)
                    t = np.linspace(0, 1, G)
                    weights = (U[k] ** self.m)[:, None]
                    invs = []
                    for i in range(self.pdf_matrix.shape[0]):
                        inv_f = np.interp(t, cdfs[i], self.grid_x)
                        invs.append(inv_f)
                    invs = np.stack(invs, axis=0)
                    inv_cent = np.sum(weights * invs, axis=0) / (np.sum(weights) + self.eps)
                    F_cent = np.interp(self.grid_x, inv_cent, t)
                    theta = np.gradient(F_cent, self.grid_x)
                    theta = np.clip(theta, 0, None)
                    theta /= np.trapz(theta, self.grid_x)

            centroids.append(theta)

        return np.stack(centroids, axis=0)  # (K,h,w) hoặc (K,G)




    # --------- cập nhật U ---------
    def _update_U(self, f, D2):
        # D2 shape: (K, N)  -> khoảng cách cụm i đến điểm j
        U_new = np.zeros((self.K, self.N))
        for j in range(self.N):          # qua từng điểm
            for i in range(self.K):      # qua từng cụm
                num = (f[i] / D2[i, j]) ** (1.0 / (self.m - 1))
                U_new[i, j] = num / np.sum((f[:] / D2[:, j]) ** (1.0 / (self.m - 1)))

        return U_new




    # --------- mục tiêu ---------
    def _objective(self, U, f, D2):
        return float(np.sum((U ** self.m) * (D2 / f[:, None])))

    # --------- fit ---------
    def fit(self, pdf_matrix):
        self.pdf_matrix = np.asarray(pdf_matrix, dtype=float)

        if self.pdf_matrix.ndim == 3:   # (N,h,w)
            self.N, h, w = self.pdf_matrix.shape
            self.pdf_shape = (h, w)
        elif self.pdf_matrix.ndim == 2: # (N,G) 1D
            self.N, self.G = self.pdf_matrix.shape
            self.pdf_shape = (self.G,)
        else:
            raise ValueError("pdf_matrix phải (N,h,w) hoặc (N,G)")


        rng = np.random.default_rng(self.seed)

        # init U
        self.U = rng.random((self.K, self.N))
        self.U /= self.U.sum(axis=0, keepdims=True) + self.eps

        # init Θ
        if self.init == "random":
            indices = rng.choice(self.N, size=self.K, replace=False)
            if self.pdf_matrix.ndim == 2:
                self.Theta = self.pdf_matrix[indices, :].copy()
            else:
                self.Theta = self.pdf_matrix[indices,:].copy()
        else:
            from utils.init import init_centroids_kmeanspp
            self.Theta = init_centroids_kmeanspp(self.pdf_matrix, self.K, self.bandwidth,
                                                 self.distance_metric, self.Dim, self.grid_x)


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
