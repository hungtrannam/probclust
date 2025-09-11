import numpy as np
from utils.dist import Dist  # các hàm khoảng cách cho PDF của bạn


class Model:
    """
    EM-IFCM (Edge-Modification Improved FCM) cho PDF rời rạc.
    - ĐÚNG công thức paper:
        (26)  J(U, Θ) = sum_{i,j} (1/f_j) u_{ij}^m ||X_i-θ_j||^2
                      + sum_i δ_i sum_j u_{ij}(1 - u_{ij}^{m-1}) D*_{ij}
        (27)  D*_{ij} = ||X_i-θ_j^{t-1}||^2 / sum_s ||X_i-θ_s^{t-1}||^2
        (35)  u_{ij} cập nhật đóng dạng
        (37)  θ_j  = (∑ u_{ij}^m X_i) / (∑ u_{ij}^m)
        (38)  δ_i  cập nhật đóng dạng (dùng δ' thay vì n tham số δ_i)
    - CHÚ Ý: Trong toàn bộ lớp này, U có shape (K, N) cho nhất quán với định nghĩa của bạn:
        K = số cụm, N = số mẫu (PDF).
      Các ma trận khoảng cách D2, D* vẫn giữ quy ước (N, K).
    - MỌI KHOẢNG CÁCH ĐỀU BÌNH PHƯƠNG trong J, D*, cập nhật U, δ_i.
    """

    def __init__(
        self,
        grid_x: np.ndarray,
        num_clusters: int = 3,
        fuzziness: float = 2.0,
        max_iterations: int = 100,
        tolerance: float = 1e-5,
        distance_metric: str = "L2",   # tên method trong Dist: L1, L2, H, BC, KL, ...
        bandwidth: float = 0.01,
        Dim: int | None = None,
        seed: int | None = None,
        verbose: bool = False,
        delta_prime: float = 0.05,     # δ' (nên nhỏ: 0.005 ~ 0.2)
        eps: float = 1e-12,
    ):
        assert fuzziness > 1.0, "m (fuzziness) phải > 1"
        self.grid_x = grid_x
        self.num_clusters = int(num_clusters)
        self.fuzziness = float(fuzziness)
        self.maxit = int(max_iterations)
        self.tol = float(tolerance)
        self.distance_metric = str(distance_metric)
        self.bandwidth = float(bandwidth)
        self.Dim = 1 if Dim is None else int(Dim)
        self.seed = seed
        self.verbose = verbose
        self.delta_prime = float(delta_prime)
        self.eps = float(eps)

        # runtime
        self.pdf_matrix = None              # (N, G) ma trận PDF (rời rạc trên grid_x)
        self.num_pdfs = None
        self.num_points = None
        self.U = None              # (K, N)
        self.delta_i = None        # (N,)
        self.Theta = None          # (K, G)
        self.obj_hist = []

    # ------------------------- DIST^2 tới các tâm -------------------------
    


    # def _init_centroids_farthest(self):
    #     dobj = Dist(h=self.bandwidth, Dim=self.Dim, grid=self.grid_x)
    #     func = getattr(dobj, self.distance_metric)

    #     N = self.num_pdfs; K = self.num_clusters
    #     idx0 = np.random.randint(N)
    #     centers = [idx0]

    #     def d2_to_center(x, c):
    #         d = func(x, self.pdf_matrix[c]) + 1e-30
    #         return d*d

    #     # khoảng cách bình phương tới tập center hiện tại
    #     d2_min = np.array([d2_to_center(self.pdf_matrix[i], idx0) for i in range(N)])
    #     for _ in range(1, K):
    #         idx = int(np.argmax(d2_min))
    #         centers.append(idx)
    #         # cập nhật d2_min
    #         d2_new = np.array([d2_to_center(self.pdf_matrix[i], idx) for i in range(N)])
    #         d2_min = np.minimum(d2_min, d2_new)

    #     self.Theta = self.pdf_matrix[centers].copy()

    # ------------------------------------------------------------------
    # Hàm hỗ trợ cho W2 (1D)
    # ------------------------------------------------------------------
    # @staticmethod
    # def _cdf_from_pdf(pdf: np.ndarray, dx: float) -> np.ndarray:
    #     cdf = np.cumsum(pdf) * dx
    #     Z = float(cdf[-1]) if cdf.size else 1.0
    #     if Z <= 0:
    #         n = pdf.size
    #         return np.linspace(0.0, 1.0, n)
    #     return cdf / Z

    # def _update_centroids_w2(self) -> None:
    #     """Cập nhật tâm cụm bằng Wasserstein-2 barycenter (1D)."""
    #     assert self.Dim == 1, "W2 barycenter chỉ hỗ trợ 1D."
    #     dx = float(self.grid_x[1] - self.grid_x[0])
    #     n_samples, m = self.pdf_matrix.shape
    #     K = self.num_clusters

    #     t_grid = np.linspace(0.0, 1.0, m)
    #     inv_mat = np.empty((n_samples, m))  # (N, m)

    #     for j in range(n_samples):
    #         cdf_j = self._cdf_from_pdf(self.pdf_matrix[j], dx)
    #         inv_mat[j] = np.interp(t_grid, cdf_j, self.grid_x)

    #     W = self.U ** self.fuzziness              # (K, N)
    #     denom = np.sum(W, axis=1, keepdims=True) + 1e-12  # (K, 1)
    #     W_norm = W / denom                         # (K, N)

    #     Theta = np.empty((K, m))
    #     for k in range(K):
    #         Qk = (W_norm[k][:, None] * inv_mat).sum(axis=0)     # (m,)
    #         Qk = np.maximum.accumulate(Qk)
    #         Fk = np.interp(self.grid_x, Qk, t_grid, left=0.0, right=1.0)
    #         fk = np.gradient(Fk, self.grid_x)
    #         Theta[k] = np.clip(fk, 0.0, None)

    #     self.Theta = Theta  # (K, m)

    # --------------------------- cập nhật θ theo (37) ---------------------------
    def _update_centroids(self) -> None:
        """Cập nhật tâm cụm bằng trung bình trọng số mờ (U có shape (K, N))."""
        W = self.U ** self.fuzziness                  # (K, N)
        num = W @ self.pdf_matrix                     # (K, G)
        den = np.sum(W, axis=1, keepdims=True) + 1e-12  # (K, 1)
        return num / den                        # (K, G)

        # ------------------------- DIST^2 tới các tâm: (K, N) -------------------------
    def _dist2_matrix_to(self, Theta):
        dobj = Dist(h=self.bandwidth, Dim=self.Dim, grid=self.grid_x)
        func = getattr(dobj, self.distance_metric)
        N, K = self.num_pdfs, self.num_clusters
        D2 = np.empty((K, N), dtype=float)  # (K, N) = (số cụm, số mẫu)
        for j in range(K):
            tj = Theta[j]
            for i in range(N):
                d = func(self.pdf_matrix[i], tj) + 1e-30
                D2[j, i] = d**2  # dùng khoảng cách bình phương
        return D2  # (K, N)

    # --------------------------- D* theo (27): (K, N) ---------------------------
    def _D_star(self, D2_prev: np.ndarray) -> np.ndarray:
        # Chuẩn hoá theo từng mẫu i: tổng theo K cho mỗi cột (mỗi i)
        denom = D2_prev.sum(axis=0, keepdims=True) + self.eps  # (1, N)
        return D2_prev / denom  # (K, N)
    
    # ------------------------- f_j (fuzzy size, từ U^{t-1}) ------------------------- 
    def _fuzzy_sizes(self, U: np.ndarray) -> np.ndarray: # U: (K, N) -> f: (K,) 
        f = U.sum(axis=1) / float(self.num_pdfs) 
        return f

    # ------------------------- cập nhật δ_i theo (38) -------------------------
    def _update_delta_i(
        self,
        omega_prev: np.ndarray,      # (K,)
        D2: np.ndarray,              # (K, N)  ||x_i-θ_j^(t)||^2
        Dstar: np.ndarray            # (K, N)  D*_{ij} @ θ^(t-1)
    ) -> None:
        m, dp = self.fuzziness, self.delta_prime
        p = 1.0 / (m - 1.0)

        # (K, N): chú ý broadcast theo (K,1) và (1,N)
        denom = m * np.clip(
            D2 - omega_prev[:, None] * self.delta_i[None, :] * Dstar,
            self.eps, None
        )
        numer = np.clip(1.0 - dp * Dstar, self.eps, None) * omega_prev[:, None]

        S = (numer / denom) ** p            # (K, N)
        S = S.sum(axis=0) + self.eps        # (N,)  — tổng theo K cho từng mẫu i

        delta_new = dp * (S ** (1.0 - m))   # (N,)

        # bound theo (41)-(42), lấy min theo K cho từng i
        bound = D2 / (omega_prev[:, None] * Dstar + self.eps)  # (K, N)
        bound = np.where(np.isfinite(bound), bound, np.inf)
        max_delta = np.min(bound, axis=0)                      # (N,)
        max_delta = np.clip(max_delta, self.eps, 1e12)

        self.delta_i = np.clip(delta_new, 0.0, max_delta)

    # --------------------------- cập nhật θ theo (37) --------------------------- 
    def _update_centroids(self) -> None: 
        """Cập nhật tâm cụm bằng trung bình trọng số mờ (U có shape (K, N)).""" 
        W = self.U ** self.fuzziness # (K, N) 
        num = W @ self.pdf_matrix # (K, G) 
        den = np.sum(W, axis=1, keepdims=True) + 1e-12 # (K, 1) 
        return num / den # (K, G)

    # ------------------------- cập nhật U theo (35) -------------------------
    def _update_U(
        self,
        omega_prev: np.ndarray,   # (K,)
        D2: np.ndarray,           # (K, N)
        Dstar: np.ndarray         # (K, N)
    ) -> np.ndarray:
        """Trả về U_new (K, N) đã chuẩn hoá theo từng cột (mỗi mẫu i)."""
        m, dp = self.fuzziness, self.delta_prime
        p = 1.0 / (m - 1.0)

        numer = np.clip(1.0 - dp * Dstar, self.eps, None) * omega_prev[:, None]  # (K, N)
        denom = np.clip(
            D2 - omega_prev[:, None] * self.delta_i[None, :] * Dstar,
            self.eps, None
        )  # (K, N)

        base = (numer / denom) ** p                                           # (K, N)
        base_norm = base / (base.sum(axis=0, keepdims=True)** p + self.eps)       # (K, N)
        return base_norm

    # ---------------------------- mục tiêu J theo (26) ----------------------------
    def _objective(
        self,
        U_curr: np.ndarray,          # (K, N)
        omega_curr: np.ndarray,      # (K,)
        D2_now: np.ndarray,          # (K, N)
        Dstar_prev: np.ndarray,      # (K, N)
    ) -> float:
        m = self.fuzziness
        U_m = U_curr ** m  # (K, N)

        # IFCM term: sum_{i,j} (1/f_j) u_{ij}^m ||x_i-θ_j||^2
        term_ifcm = np.sum(U_m * (D2_now / (omega_curr[:, None] + self.eps)))

        # Edge term: sum_i δ_i sum_j u_{ij}(1 - u_{ij}^{m-1}) D*_{ij}
        term_edge = np.sum(self.delta_i[None, :] * U_curr * (1.0 - U_curr ** (m - 1.0)) * Dstar_prev)

        return float(term_ifcm + term_edge)

    # --------------------------------- fit (đồng bộ K,N) ---------------------------------
    def fit(self, pdf_matrix: np.ndarray):
        self.pdf_matrix = np.asarray(pdf_matrix, dtype=float)
        self.num_pdfs, self.num_points = self.pdf_matrix.shape

        if self.seed is not None:
            np.random.seed(self.seed)

        # init
        self.U = np.random.dirichlet(np.ones(self.num_clusters), size=self.num_pdfs).T  # (N, K)
        init_indices = np.random.choice(self.num_pdfs, self.num_clusters, replace=False)
        self.Theta = pdf_matrix[init_indices, :].copy()  # (K, G)
        self.delta_i = np.zeros(self.num_pdfs, dtype=float)
        self.obj_hist = []
        J_prev = None

        for it in range(1, self.maxit + 1):
            Theta_prev = self.Theta.copy()

            # (37)
            self.Theta = self._update_centroids()  # dùng U^{t-1} -> Θ^{t}

            # D* dùng Θ^{t-1}
            D2_prev = self._dist2_matrix_to(Theta_prev)  # (K, N)
            Dstar   = self._D_star(D2_prev)              # (K, N)

            # khoảng cách tới Θ^{t}
            D2_now  = self._dist2_matrix_to(self.Theta)  # (K, N)

            # f (omega) từ U^{t-1}
            omega_prev = self._fuzzy_sizes(self.U)       # (K,)

            # (38) δ_i
            self._update_delta_i(omega_prev, D2_now, Dstar)

            # (35) U^{t}
            U_new = self._update_U(omega_prev, D2_now, Dstar)  # (K, N)

            # J^{t} với f từ U^{t}
            omega_curr = self._fuzzy_sizes(U_new)        # (K,)
            J = self._objective(U_new, omega_curr, D2_now, Dstar)
            self.obj_hist.append(J)

            dU = np.linalg.norm(U_new - self.U)
            dTheta = np.linalg.norm(self.Theta - Theta_prev)
            dJ = abs(J - J_prev) if J_prev is not None else np.inf

            if self.verbose:
                print(f"[EM-IFCM] it={it:03d} | dU={dU:.3e} | dTheta={dTheta:.3e} | ΔJ={dJ:.3e} | J={J:.6e}")

            self.U = U_new
            J_prev = J

            if dJ < self.tol:
                if self.verbose:
                    print("Converged by ΔJ.")
                break

        return self

    # -------------------------------- predict (đồng bộ K,N) --------------------------------
    def predict(self, new_pdfs: np.ndarray) -> np.ndarray:
        Xn = np.asarray(new_pdfs, dtype=float)
        Nn = Xn.shape[0]

        omega_prev = self._fuzzy_sizes(self.U)  # (K,)

        # D2: (K, Nn)
        dobj = Dist(h=self.bandwidth, Dim=self.Dim, grid=self.grid_x)
        func = getattr(dobj, self.distance_metric)
        K = self.num_clusters
        D2 = np.empty((K, Nn), dtype=float)
        for j in range(K):
            for i in range(Nn):
                d = func(Xn[i], self.Theta[j]) + 1e-30
                D2[j, i] = d * d

        # D* xấp xỉ dùng tâm hiện tại
        Dstar = self._D_star(D2)  # (K, Nn)

        # một bước δ_i từ 0
        m, dp = self.fuzziness, self.delta_prime
        p = 1.0 / (m - 1.0)
        one_minus = np.clip(1.0 - dp * Dstar, 1e-9, None)  # (K, Nn)
        num = one_minus * omega_prev[:, None]               # (K, Nn)
        denom = np.clip(D2, 1e-9, None)                    # (K, Nn)
        term = (num / denom) ** p                           # (K, Nn)
        s = term.sum(axis=0) + self.eps                     # (Nn,)
        delta_step = dp * (s ** (1.0 - m))                  # (Nn,)

        # chặn δ_i
        bound = D2 / (omega_prev[:, None] * Dstar + 1e-12)  # (K, Nn)
        bound = np.where(np.isfinite(bound), bound, np.inf)
        max_delta = np.min(bound, axis=0)                   # (Nn,)
        max_delta = np.clip(max_delta, 1e-9, 1e12)
        delta_step = np.clip(delta_step, 0.0, max_delta)    # (Nn,)

        # U theo (35)
        core = np.clip(D2 - (omega_prev[:, None] * delta_step[None, :] * Dstar), 1e-9, None)  # (K, Nn)
        numU = (one_minus * omega_prev[:, None] / core) ** p                                   # (K, Nn)
        denU = numU.sum(axis=0, keepdims=True) + self.eps                                      # (1, Nn)
        U_pred = numU / denU                                                                    # (K, Nn)
        return U_pred

    # ------------------------------- tiện ích -------------------------------
    def get_results(self):
        return self.U.copy(), self.Theta.copy(), list(self.obj_hist)

    def get_hard_assignments(self):
        # U: (K, N) -> argmax theo K cho từng mẫu (trục 0)
        return np.argmax(self.U, axis=0)  # (N,)
