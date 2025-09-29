import numpy as np
from utils.dist import Dist  # các hàm khoảng cách cho PDF


class Model:

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
        delta_prime: float = 0.05,     
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
        self.delta_prime = float(delta_prime)
        self.eps = float(eps)

        # runtime
        self.pdf_matrix = None              
        self.num_pdfs = None
        self.num_points = None
        self.U_t = None               # U^(t)
        self.delta_i_t = None         # δ_i^(t)
        self.Theta_t = None           # Θ^(t)
        self.obj_hist = []

    # ------------------------- D^2 tới các tâm Θ -------------------------
    def _dist2_matrix_to(self, Theta_t):
        dobj = Dist(h=self.bandwidth, Dim=self.Dim, grid=self.grid_x)
        func = getattr(dobj, self.distance_metric)
        D2 = np.empty((self.num_clusters, self.num_pdfs), dtype=float)
        for j in range(self.num_clusters):
            for i in range(self.num_pdfs):
                d = func(self.pdf_matrix[i], Theta_t[j])
                D2[j, i] = d + self.eps
        return D2**2  # (K, N)

    # --------------------------- D* theo (27) ---------------------------
    def _D_star(self, D2_t_1: np.ndarray) -> np.ndarray:
        return D2_t_1 / D2_t_1.sum(axis=0, keepdims=True) 
    
    # ------------------------- f_j (fuzzy size) ------------------------- 
    def _fuzzy_sizes(self, U_t_1: np.ndarray) -> np.ndarray:
        f = U_t_1.sum(axis=1) / self.num_pdfs  # (K,)
        return f


    # ------------------------- cập nhật δ_i^(t) theo (38) -------------------------
    def _update_delta_i(
        self,
        omega_t_1: np.ndarray,   # f_j^(t-1)
        D2_t_1: np.ndarray,      # ||x_i - θ_j^(t-1)||^2
        Dstar_t_1: np.ndarray,   # D*_{ij}^(t-1)
        delta_i_t_1: np.ndarray  # δ_i^(t-1)
    ) -> np.ndarray:

        m = self.fuzziness
        eps = self.eps
        K, N = D2_t_1.shape
        delta_i_t = np.zeros(N)

        for i in range(N):
            acc = 0.0
            for j in range(K):
                numer = (1.0 - self.delta_prime * Dstar_t_1[j, i]) * omega_t_1[j]
                denom = m * (D2_t_1[j, i] - omega_t_1[j] * delta_i_t_1[i] * Dstar_t_1[j, i] + eps)
                term = (numer / denom) ** (1.0 / (m - 1.0))
                acc += term
            delta_i_t[i] = self.delta_prime * (acc ** (1 - m))

        # Bound theo (41)-(42)
        bound = D2_t_1 / (omega_t_1[:, None] * Dstar_t_1 + eps)
        bound = np.where(np.isfinite(bound), bound, np.inf)
        max_delta = np.min(bound, axis=0)
        max_delta = np.clip(max_delta, eps, 1e12)

        delta_i_t = np.clip(delta_i_t, 0.0, max_delta)
        return delta_i_t

    # --------------------------- cập nhật Θ^(t) theo (37) --------------------------- 
    def _update_centroids(self, U) -> np.ndarray:
        W = U ** self.fuzziness  # (K, N)
        centroids = []
        for k in range(self.num_clusters):
            weights = W[k][:, None, None] if self.Dim == 2 else W[k][:, None]
            den = np.sum(weights) + self.eps

            if self.distance_metric == 'H':
                if self.Dim == 2:
                    num = np.sum(weights * np.sqrt(self.pdf_matrix), axis=0)
                else:
                    num = np.sum(weights * np.sqrt(self.pdf_matrix), axis=0)
                theta = (num / den) ** 2

            elif self.distance_metric in ['L2', 'L1']:
                if self.Dim == 2:
                    num = np.sum(weights * self.pdf_matrix, axis=0)
                else:
                    num = np.sum(weights * self.pdf_matrix, axis=0)
                theta = num / den

            else:
                # xử lý Wasserstein / BC
                G = len(self.grid_x)
                cdfs = np.cumsum(self.pdf_matrix, axis=1) * self.bandwidth
                cdfs = np.clip(cdfs, 0, 1)
                t = np.linspace(0, 1, G)
                weights = (U[k] ** self.fuzziness)[:, None]
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

        return np.stack(centroids, axis=0)

    # ------------------------- cập nhật U^(t) theo (35) -------------------------
    def _update_U(
        self,
        omega_t_1: np.ndarray,   # f_j^(t-1)
        D2_t: np.ndarray,      # ||X_i - θ_j^(t-1)||^2
        delta_i_t: np.ndarray,   # δ_i^(t)
        Dstar_t_1: np.ndarray    # D*_{ij}^(t-1)
    ) -> np.ndarray:

        p = 1.0 / (self.fuzziness - 1.0)
        U_t = np.zeros((self.num_clusters, self.num_pdfs))

        for i in range(self.num_pdfs):
            numerators = []
            denom_total = 0.0
            for j in range(self.num_clusters):
                numer = (1.0 - self.delta_prime * Dstar_t_1[j, i]) * omega_t_1[j]
                denom = D2_t[j, i] - omega_t_1[j] * delta_i_t[i] * Dstar_t_1[j, i]
                term = (numer / (denom + self.eps)) ** p
                numerators.append(term)
                denom_total += term
            for j in range(self.num_clusters):
                U_t[j, i] = numerators[j] / (denom_total + self.eps)

        return U_t

    # ---------------------------- J^(t) theo (26) ----------------------------
    def _objective(
        self,
        U_t: np.ndarray,
        omega_t: np.ndarray,
        D2_t_1: np.ndarray,
        Dstar_t_1: np.ndarray,
        delta_i_t: np.ndarray
    ) -> float:
        term_ifcm = np.sum((U_t**self.fuzziness) * (D2_t_1 / (omega_t[:, None] + self.eps)))
        term_edge = np.sum(delta_i_t[None, :] * U_t * (1.0 - U_t ** (self.fuzziness - 1.0)) * Dstar_t_1)
        return float(term_ifcm + term_edge), term_ifcm, term_edge

    # --------- fit ---------
    def fit(self, pdf_matrix):
        X = np.asarray(pdf_matrix, dtype=float)
        self.pdf_matrix = X
        self.num_pdfs = self.pdf_matrix.shape[0]   # <<< thêm dòng này
        
        if self.pdf_matrix.ndim == 3:   # (N,h,w)
            _, h, w = self.pdf_matrix.shape
            self.pdf_shape = (h, w)
        elif self.pdf_matrix.ndim == 2: # (N,G)
            _, G = self.pdf_matrix.shape
            self.pdf_shape = (G,)
        else:
            raise ValueError("pdf_matrix phải (N,h,w) hoặc (N,G)")


        rng = np.random.default_rng(self.seed)

        # init U
        self.U = rng.random((self.num_clusters, self.num_pdfs))
        self.U /= self.U.sum(axis=0, keepdims=True) + self.eps

        # init Θ
        if self.init == "random":
            indices = rng.choice(self.num_pdfs, size=self.num_clusters, replace=False)
            if self.pdf_matrix.ndim == 2:
                Theta_0 = self.pdf_matrix[indices, :].copy()
            else:
                Theta_0 = self.pdf_matrix[indices,:].copy()
        else:
            from utils.init import init_centroids_kmeanspp
            Theta_0 = init_centroids_kmeanspp(self.pdf_matrix, self.num_clusters, self.bandwidth,
                                                 self.distance_metric, self.Dim, self.grid_x)



        # --- δ_i^0 = 0 ---
        delta_i_0 = np.zeros(self.num_pdfs, dtype=float)

        # --- U^0 từ Θ^0 ---
        D2_0    = self._dist2_matrix_to(Theta_0)
        Dstar_0 = self._D_star(D2_0)
        omega_0 = np.ones(self.num_clusters) / self.num_clusters
        U_0     = self._update_U(omega_0, D2_0, delta_i_0, Dstar_0)

        self.obj_hist = []
        J_prev = np.inf

        for it in range(1, self.maxit + 1):
            # 1. f^0
            omega_0 = self._fuzzy_sizes(U_0)

            # 2. δ^new từ (U^0, D2^0, D*^0)
            delta_new = self._update_delta_i(omega_0, D2_0, Dstar_0, delta_i_0)

            # 3. U^new từ (δ^new, D2^0, D*^0)
            U_new = self._update_U(omega_0, D2_0, delta_new, Dstar_0)

            # 4. Θ^new từ U^new
            Theta_new = self._update_centroids(U_new)

            # 5. D2^new, D*^new
            D2_new    = self._dist2_matrix_to(Theta_new)
            Dstar_new = self._D_star(D2_new)


            # 7. J^new
            J_new, term_ifcm, term_edge = self._objective(U_new, omega_0, D2_0, Dstar_0, delta_new)
            self.obj_hist.append(J_new)

            # --- Kiểm tra hội tụ ---
            dU  = float(np.linalg.norm(U_new - U_0))
            dTh = float(np.linalg.norm(Theta_new - Theta_0))
            dJ  = abs(J_new - J_prev) if np.isfinite(J_prev) else np.inf

            if self.verbose:
                print(f"[EM-IFCM] it={it:03d} | dU={dU:.3e} | dΘ={dTh:.3e} "
                    f"| ΔJ={dJ:.3e} | J={J_new:.6e} "
                    f"| edge={term_edge:.3e} | IFCM={term_ifcm:.3e} | f={omega_0}")

            # --- Cập nhật: new → 0 ---
            U_0, Theta_0, delta_i_0, D2_0, Dstar_0, J_prev = \
                U_new, Theta_new, delta_new, D2_new, Dstar_new, J_new

            if dJ < self.tol:
                if self.verbose: print("Converged by ΔJ.")
                break

        # Kết quả cuối
        self.U_t = U_new
        self.Theta_t = Theta_new
        self.delta_i_t = delta_new
        return self




    # -------------------------------- predict --------------------------------
    def predict(self, new_pdfs: np.ndarray) -> np.ndarray:
        Xn = np.asarray(new_pdfs, dtype=float)
        Nn = Xn.shape[0]

        omega_t = self._fuzzy_sizes(self.U_t)
        dobj = Dist(h=self.bandwidth, Dim=self.Dim, grid=self.grid_x)
        func = getattr(dobj, self.distance_metric)
        K = self.num_clusters
        D2 = np.empty((K, Nn), dtype=float)
        for j in range(K):
            for i in range(Nn):
                d = func(Xn[i], self.Theta_t[j]) + 1e-30
                D2[j, i] = d * d

        Dstar = self._D_star(D2)

        m, dp = self.fuzziness, self.delta_prime
        p = 1.0 / (m - 1.0)
        one_minus = np.clip(1.0 - dp * Dstar, 1e-9, None)
        num = one_minus * omega_t[:, None]
        denom = np.clip(D2, 1e-9, None)
        term = (num / denom) ** p
        s = term.sum(axis=0) + self.eps
        delta_step = dp * (s ** (1.0 - m))

        bound = D2 / (omega_t[:, None] * Dstar + 1e-12)
        bound = np.where(np.isfinite(bound), bound, np.inf)
        max_delta = np.min(bound, axis=0)
        max_delta = np.clip(max_delta, 1e-9, 1e12)
        delta_step = np.clip(delta_step, 0.0, max_delta)

        core = np.clip(D2 - (omega_t[:, None] * delta_step[None, :] * Dstar), 1e-9, None)
        numU = (one_minus * omega_t[:, None] / core) ** p
        denU = numU.sum(axis=0, keepdims=True) + self.eps
        U_pred = numU / denU
        return U_pred

    # ------------------------------- tiện ích -------------------------------
    def get_results(self):
        return self.U_t.copy(), self.Theta_t.copy(), list(self.obj_hist)

    def get_hard_assignments(self):
        return np.argmax(self.U_t, axis=0)
