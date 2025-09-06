import numpy as np
from utils.kernel import Kernel


class Model:
    """
    Kernel Fuzzy C-Means (KFCM) dùng kernel-trick hoàn toàn.
    - Không chuyển K -> d qua các công thức heuristic (1-K, sqrt(2(1-K)), ...).
    - Không cần centroid trong không gian gốc; mọi thứ tính từ K và U.

    pdf_matrix: (N, M) cho 1D hoặc (N, H, W) cho 2D (tuỳ utils.kernel.Kernel & int_trapz).
    """

    def __init__(
        self,
        grid_x: np.ndarray,
        num_clusters: int = 3,
        fuzziness: float = 2.0,
        max_iterations: int = 100,
        tolerance: float = 1e-5,
        kernel_type: str = "L2",   # "H" (BC), "L2", "L1", "LN", "Chi2"
        gamma: float = 1.0,
        bandwidth: float = 0.01,
        Dim: int = 1,
        seed: int | None = None,
        verbose: bool = False,
        compute_prototypes_view: bool = False,  # chỉ để xem/plot: lấy trung bình có trọng số trong không gian gốc
    ):
        self.grid_x = grid_x
        self.num_clusters = num_clusters
        self.m = fuzziness
        self.T = max_iterations
        self.tol = tolerance
        self.kernel_type = kernel_type
        self.gamma = gamma
        self.h = bandwidth
        self.Dim = Dim
        self.seed = seed
        self.verbose = verbose
        self.compute_prototypes_view = compute_prototypes_view

        # will be set in fit()
        self.X = None              # pdf_matrix
        self.U = None              # (N, C)
        self.K = None              # (N, N)
        self.K_diag = None         # (N,)
        self.J_hist = []
        self.R2 = None             # (N, C) trong mỗi iteration
        self.prototypes_view = None  # (C, M) nếu compute_prototypes_view=True

        # cache cho predict
        self._w = None             # (N, C)  w_ij = u_ij^m
        self._S = None             # (C,)
        self._wKw = None           # (C,)

    # ---------- Kernel pair ----------
    def _kpair(self, f1: np.ndarray, f2: np.ndarray) -> float:
        kobj = Kernel(h=self.h, Dim=self.Dim)
        func = getattr(kobj, self.kernel_type)
        if self.kernel_type in ("L2", "L1"):
            return float(func(f1, f2, gamma=self.gamma))
        return float(func(f1, f2))

    # ---------- K, K_diag ----------
    def _compute_kernel_matrix(self) -> None:
        N = self.num_pdfs
        K = np.empty((N, N), dtype=float)   # <- dùng ndarray thay vì list

        for i in range(N):
            K[i, i] = self._kpair(self.X[i], self.X[i])
            for j in range(i + 1, N):
                v = self._kpair(self.X[i], self.X[j])
                K[i, j] = K[j, i] = v

        self.K = K
        self.K_diag = np.diag(K)
        return K


    # ---------- R^2(i,j) theo kernel-trick ----------
    def _compute_R2(self) -> np.ndarray:
        """
        R2[i, j] = ||Phi(x_i) - v_j||^2 tính hoàn toàn bằng K, U^m.
        """
        eps = 1e-100
        U_m = self.U ** self.m                      # (N, C)
        K = self.K

        R2 = np.empty((self.num_pdfs, self.num_clusters), dtype=float)
        wKw = np.empty(self.num_clusters, dtype=float)
        S = np.sum(U_m, axis=0) + eps               # (C,)

        for j in range(self.num_clusters):
            w = U_m[:, j]                           # (N,)
            Kw = K @ w                               # (N,)
            wKw[j] = float(w @ Kw)                  # scalar
            # R2[:, j] = K(xi,xi) - 2/S * sum_p w_p K(xi,xp) + 1/S^2 * sum_pq w_p w_q K(xp,xq)
            R2[:, j] = self.K_diag - 2.0 * Kw / S[j] + (wKw[j] / (S[j] * S[j]))

        # cache cho predict
        self._w = U_m
        self._S = S
        self._wKw = wKw
        return np.maximum(0.0, R2)

    # ---------- cập nhật U từ R2 ----------
    def _update_U(self, R2: np.ndarray) -> None:
        eps = 1e-100
        N, C = R2.shape
        power = 1.0 / (self.m - 1.0)

        for i in range(N):
            r = R2[i] + eps
            # xử lý điểm trùng kernel-prototype: nếu có R2≈0, gán membership 1 cho cluster nhỏ nhất
            j_min = int(np.argmin(r))
            if r[j_min] <= 1e-14:
                row = np.zeros(C, dtype=float)
                row[j_min] = 1.0
                self.U[i] = row
            else:
                denom = np.sum((r / r[:, None]) ** power, axis=0)  # sai chiều; làm chuẩn:
                # công thức chuẩn: u_ij = 1 / sum_k (r_ij / r_ik)^{1/(m-1)}
                # viết lại hiệu quả:
                inv = r ** (-power)
                self.U[i] = inv / np.sum(inv)

    # ---------- mục tiêu ----------
    def _objective(self, R2: np.ndarray) -> float:
        return float(np.sum((self.U ** self.m) * R2))

    # ---------- (tuỳ chọn) prototype để xem/plot trong không gian gốc ----------
    def _update_prototypes_view(self) -> None:
        if not self.compute_prototypes_view:
            return
        eps = 1e-100
        U_m = self.U ** self.m          # (N, C)
        S = np.sum(U_m, axis=0) + eps
        # trung bình có trọng số trong không gian gốc (chỉ để trực quan hoá)
        # shape: (C, ...) giống 1 sample PDF
        self.prototypes_view = (U_m.T @ self.X) / S[:, None]

    # ---------- fit ----------
    def fit(self, pdf_matrix: np.ndarray) -> None:
        self.X = np.asarray(pdf_matrix, dtype=float)
        
        self.pdf_matrix = pdf_matrix
        self.num_pdfs, self.num_points = pdf_matrix.shape


        if self.seed is not None:
            np.random.seed(self.seed)

        # Khởi tạo U ngẫu nhiên (mỗi hàng là phân phối xác suất)
        self.U = np.random.dirichlet(np.ones(self.num_clusters), size=self.num_pdfs)

        # Tính K 1 lần
        self._compute_kernel_matrix()

        self.J_hist = []
        for t in range(self.T):
            U_prev = self.U.copy()

            # R^2 theo kernel-trick
            self.R2 = self._compute_R2()

            # cập nhật U
            self._update_U(self.R2)

            # mục tiêu
            J = self._objective(self.R2)
            self.J_hist.append(J)

            # (tuỳ chọn) centroid để xem/plot (không ảnh hưởng tối ưu)
            self._update_prototypes_view()

            # hội tụ
            delta = np.linalg.norm(self.U - U_prev)
            if self.verbose:
                print(f"[Iter {t+1:03d}] ΔU={delta:.6e}  J={J:.6f}")
            if delta < self.tol:
                if self.verbose:
                    print("Converged.")
                break

    # ---------- predict ----------
    def predict(self, new_pdfs: np.ndarray) -> np.ndarray:
        """
        Tính membership cho PDF mới bằng kernel-trick:
        R^2(x,c_j) = K(x,x) - 2/S_j * Σ_i w_ij K(x,x_i) + (w^T K w)/S_j^2,
        với w_ij = u_ij^m lấy từ tập train.
        """
        if new_pdfs.ndim == 1:
            new_pdfs = new_pdfs[None, ...]

        if self.K is None or self.U is None:
            raise RuntimeError("Gọi fit() trước khi predict().")

        eps = 1e-100
        power = 1.0 / (self.m - 1.0)

        memberships = []
        for x in new_pdfs:
            # K(x, x_i) cho toàn bộ i
            K_xi = np.array([self._kpair(x, self.X[i]) for i in range(self.num_pdfs)], dtype=float)  # (N,)
            K_xx = float(self._kpair(x, x))

            R2 = np.empty(self.num_clusters, dtype=float)
            for j in range(self.num_clusters):
                w = self._w[:, j]                    # (N,)
                S = self._S[j] + eps
                wKw = self._wKw[j]
                Kw = float(K_xi @ w)
                R2[j] = K_xx - 2.0 * Kw / S + (wKw / (S * S))
            R2 = np.maximum(0.0, R2)

            # cập nhật u(x) theo công thức FCM
            r = R2 + eps
            j_min = int(np.argmin(r))
            if r[j_min] <= 1e-14:
                u = np.zeros(self.num_clusters, dtype=float)
                u[j_min] = 1.0
            else:
                inv = r ** (-power)
                u = inv / np.sum(inv)
            memberships.append(u)

        return np.vstack(memberships)

    # ---------- tiện ích ----------
    def get_results(self):
        """
        Trả về:
        - U.T: (C, N) membership
        - prototypes_view: (C, M) nếu compute_prototypes_view=True, else None
        - J_hist: list các giá trị mục tiêu
        """
        return self.U.T.copy(), (None if self.prototypes_view is None else self.prototypes_view.copy()), list(self.J_hist)

    def get_hard_assignments(self):
        return np.argmax(self.U, axis=1)
