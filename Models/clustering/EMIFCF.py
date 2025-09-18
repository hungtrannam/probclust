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
        init: str = "random",  # khởi tạo tâm: random, kmeans++
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
        self.init = str(init)
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

        # ------------------------- DIST^2 tới các tâm: (K, N) -------------------------
    def _dist2_matrix_to(self, Theta):
        dobj = Dist(h=self.bandwidth, Dim=self.Dim, grid=self.grid_x)
        func = getattr(dobj, self.distance_metric)
        N, K = self.num_pdfs, self.num_clusters
        D2 = np.empty((K, N), dtype=float)  # (K, N) = (số cụm, số mẫu)
        for j in range(K):
            tj = Theta[j]
            for i in range(N):
                d = func(self.pdf_matrix[i], tj) 
                D2[j, i] = d**2 + self.eps
        return D2  # (K, N)

    # --------------------------- D* theo (27): (K, N) ---------------------------
    def _D_star(self, D2_prev: np.ndarray) -> np.ndarray:
        # Chuẩn hoá theo từng mẫu i: tổng theo K cho mỗi cột (mỗi i)
        denom = D2_prev.sum(axis=0, keepdims=True) + self.eps  # (1, N)
        return D2_prev / denom  # (K, N)
    
    # ------------------------- f_j (fuzzy size, từ U^{t-1}) ------------------------- 
    def _fuzzy_sizes(self, U: np.ndarray) -> np.ndarray: # U: (K, N) -> f: (K,) 
        f = U.sum(axis=1) / (sum(U.sum(axis=1)) + self.eps)  # (K,)
        f = np.clip(f, self.eps, None, out=f)  # tránh chia 0
        return f

    # ------------------------- cập nhật δ_i theo (38) -------------------------
    def _update_delta_i(
        self,
        omega_prev: np.ndarray,   # (K,)   f_j
        D2: np.ndarray,           # (K, N)  ||x_i - θ_j^(t)||^2
        Dstar: np.ndarray         # (K, N)  D*_{ij} @ θ^(t-1)
    ) -> None:
        """
        Cập nhật delta_i theo công thức (38).
        omega_prev: f_j (fuzzy size) (K,)
        D2: khoảng cách bình phương (K, N)
        Dstar: D*_{ij} (K, N)
        """
        m = self.fuzziness
        K, N = D2.shape
        eps = self.eps
        dp = self.delta_prime

        delta_new = np.zeros(N)

        for i in range(N):   # duyệt từng mẫu
            acc = 0.0
            for j in range(K):  # duyệt từng cụm
                # Tử số: (1 - δ' * D*_{ij}) * f_j
                numer = (1.0 - dp * Dstar[j, i]) * omega_prev[j]
                numer = max(numer, eps)

                # Mẫu số: m( ||X_i - θ_j||^2 - f_j * δ_i^(t-1) * D*_{ij} )
                denom = m * (D2[j, i] - omega_prev[j] * self.delta_i[i] * Dstar[j, i])
                denom = max(denom, eps)

                # Term cho cụm j
                term = (numer / denom) ** (1.0 / (m - 1.0))
                acc += term

            # Tổng theo j, rồi mũ (1-m)
            delta_new[i] = dp * (acc ** (1-m))   # instead of (1.0 - m)

        # Áp dụng bound theo (41)-(42)
        bound = D2 / (omega_prev[:, None] * Dstar + eps)  # (K, N)
        bound = np.where(np.isfinite(bound), bound, np.inf)
        max_delta = np.min(bound, axis=0)  # (N,)
        max_delta = np.clip(max_delta, eps, 1e12)

        # Cập nhật
        self.delta_i = np.clip(delta_new, 0.0, max_delta)


    # --------------------------- cập nhật θ theo (37) --------------------------- 
    def _update_centroids(self) -> np.ndarray:
        """Cập nhật tâm cụm bằng trung bình trọng số mờ (U: (K,N), X: (N,G))."""
        W = self.U ** self.fuzziness            # (K, N)
        num = W @ self.pdf_matrix               # (K, G)
        den = np.sum(W, axis=1, keepdims=True)  # (K, 1)
        return num / (den + self.eps)           # (K, G)


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



    # ------------------------- cập nhật U theo (35) -------------------------
    def _update_U(
        self,
        omega_prev: np.ndarray,   # (K,)
        D2: np.ndarray,           # (K, N) = ||X_i - θ_j||^2
        Dstar: np.ndarray         # (K, N) = D*_{ij}
    ) -> np.ndarray:
        """Trả về U_new (K, N), tính tường minh theo từng mẫu i."""
        m, dp = self.fuzziness, self.delta_prime
        p = 1.0 / (m - 1.0)
        K, N = D2.shape

        U_new = np.zeros((K, N))

        for i in range(N):
            denom_total = 0.0
            numerators = []
            for j in range(K):
                # tử số: (1 - δ' * D*_{ij}) * f_j
                numer = (1.0 - dp * Dstar[j, i]) * omega_prev[j]
                # mẫu số: ||X_i - θ_j||^2 - f_j * δ_i * D*_{ij}
                denom = D2[j, i] - omega_prev[j] * self.delta_i[i] * Dstar[j, i]
                value = (numer / denom) ** p
                numerators.append(value)
                denom_total += value
            # chuẩn hóa lại
            for j in range(K):
                U_new[j, i] = numerators[j] / (denom_total + self.eps)

        return U_new


    # ---------------------------- mục tiêu J theo (26) ----------------------------
    def _objective(
        self,
        U_curr: np.ndarray,          # (K, N)
        omega_curr: np.ndarray,      # (K,)
        D2_now: np.ndarray,          # (K, N)
        Dstar_prev: np.ndarray,      # (K, N)
    ) -> float:

        # IFCM term: sum_{i,j} (1/f_j) u_{ij}^m ||x_i-θ_j||^2
        term_ifcm = np.sum((U_curr**self.fuzziness) * (D2_now / (omega_curr[:, None] + self.eps)))

        # Edge term: sum_i δ_i sum_j u_{ij}(1 - u_{ij}^{m-1}) D*_{ij}
        term_edge = np.sum(self.delta_i[None, :] * U_curr * (1.0 - U_curr ** (self.fuzziness - 1.0)) * Dstar_prev)

        return float(term_ifcm + term_edge), term_ifcm, term_edge


    # --------------------------------- fit (K,N) ---------------------------------
    def fit(self, pdf_matrix: np.ndarray):
        X = np.asarray(pdf_matrix, dtype=float)
        self.pdf_matrix = X
        self.num_pdfs, self.num_points = X.shape
        N, K = self.num_pdfs, self.num_clusters
        eps = self.eps

        rng = np.random.default_rng(self.seed)

        # Step 1–2: init δ_i = 0, init U (normalize over clusters per sample), init Θ^0 from U^0
        self.delta_i = np.zeros(N, dtype=float)
        self.U = rng.random((K, N), dtype=float)
        self.U /= (self.U.sum(axis=0, keepdims=True) + eps)

        if self.init == "random":
            self.Theta = X[rng.choice(N, size=K, replace=False), :].copy()
        else:
            self.Theta = self._init_centroids_kmeanspp(X)
        import matplotlib.pyplot as plt
        plt.figure()
        plt.title("Khởi tạo tâm kmeans++")
        for j in range(K):
            plt.plot(self.grid_x, self.Theta[j], label=f'Centroid {j}')
        plt.show()

        # replace initial Θ by centroid update from U^0 to strictly follow Θ^0 = argmin J(·|U^0)
        self.Theta = self._update_centroids()

        self.obj_hist = []
        J_prev = np.inf

        for it in range(1, self.maxit + 1):
            # --- E-step : U^{t-1}  →  Θ^t ---
            Theta_new = self._update_centroids()          # (K,G)  (dùng U^{t-1})

            # D* theo Θ^{t-1} (Eq.27)  – cần cache Θ_prev trước khi ghi đè
            D2_prev = self._dist2_matrix_to(self.Theta)   # (K,N)
            Dstar_prev = self._D_star(D2_prev)            # (K,N)

            # Khoảng cách mới với Θ^t
            D2_now = self._dist2_matrix_to(Theta_new)     # (K,N)

            # Cập nhật δ_i^t  (dùng Θ^t và D*^{t-1})
            f_prev = self._fuzzy_sizes(self.U)            # (K,)  từ U^{t-1}
            self._update_delta_i(f_prev, D2_now, Dstar_prev)

            # --- M-step : U^t ---
            U_new = self._update_U(f_prev, D2_now, Dstar_prev)  # (K,N)

            # --- Tính đúng J(U^t, Θ^t) ---
            f_curr = self._fuzzy_sizes(U_new)             # (K,)  từ U^t
            J, term_ifcm, term_edge = self._objective(U_new, f_curr, D2_now, Dstar_prev)

            self.obj_hist.append(J)

            # --- Đánh giá hội tụ ---
            dU   = float(np.linalg.norm(U_new - self.U))
            dTh  = float(np.linalg.norm(Theta_new - self.Theta))
            dJ   = abs(J - J_prev) if np.isfinite(J_prev) else np.inf

            if self.verbose:
                print(f"[EM-IFCM] it={it:03d} | dU={dU:.3e} | dΘ={dTh:.3e} | "
                      f"ΔJ={dJ:.3e} | J={J:.6e} | edge={term_edge:.3e} | IFCM={term_ifcm:.3e}")

            # Cam kết bước t
            self.U, self.Theta = U_new, Theta_new
            J_prev = J

            if dJ < self.tol:
                if self.verbose:
                    print("Converged by ΔJ.")
                break


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
