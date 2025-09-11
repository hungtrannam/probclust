import numpy as np
from utils.dist import Dist  # Khoảng cách giữa các hàm mật độ

class Model:
    def __init__(
            self, 
            grid_x, 
            num_clusters=3, 
            fuzziness=2.0, 
            tolerance=1e-5, 
            max_iterations=100, 
            bandwidth=0.01,
            distance_metric="L2", 
            Dim=None,
            seed=None, 
            verbose=False
        ):
        
        self.grid_x = grid_x
        self.num_clusters = num_clusters
        self.fuzziness = fuzziness
        self.tolerance = tolerance
        self.max_iterations = max_iterations
        self.bandwidth = bandwidth
        self.metric = distance_metric
        self.seed = seed
        self.verbose = verbose
        self.Dim = Dim if Dim is not None else 1  # mặc định 1D PDF

        self.pdf_matrix = None
        self.Theta = None  # centroids
        self.U = None      # soft membership
        self.W = None      # fuzzy weights
        self.Comp = None   # integrity
        self.P = None      # separation
        self.rho = None
        self.I_star = None
        self.obj = []

    def _dist_matrix(self):
        """Tính khoảng cách bình phương giữa từng điểm và centroid: D_ik = ||x_i - θ_k||²"""
        d_obj = Dist(h=self.bandwidth, Dim=self.Dim, grid=self.grid_x)
        func = getattr(d_obj, self.metric)

        N = self.pdf_matrix.shape[0]
        D = np.zeros((self.num_pdfs, self.num_clusters))
        for i in range(N):
            for k in range(self.num_clusters):
                D[i, k] = func(self.pdf_matrix[i], self.Theta[k])**2 + 1e-10
        return D

    def _update_membership(self, D, U):
        """Cập nhật ma trận membership W dựa trên D và U."""
        self.num_pdfs, self.num_clusters = D.shape
        W = np.zeros_like(D)
        for k in range(self.num_clusters):
            denom = np.zeros(self.num_pdfs)
            for k1 in range(self.num_clusters):
                denom += (D[:, k] / D[:, k1]) ** (1 / (self.fuzziness - 1))
            W[:, k] = U[:, k] * (1.0 / (denom + 1e-10))
        return W

    def fit(self, pdf_matrix: np.ndarray):
        self.pdf_matrix = pdf_matrix
        self.num_pdfs = pdf_matrix.shape[0]
        N, K = self.num_pdfs, self.num_clusters
        m = self.fuzziness

        if self.seed is not None:
            np.random.seed(self.seed)

        # Khởi tạo U = ones, Theta ngẫu nhiên, W ban đầu
        self.U = np.ones((N, K))
        self.Theta = pdf_matrix[np.random.choice(N, K, replace=False)].copy()
        D = self._dist_matrix()
        self.W = self._update_membership(D, self.U)

        for it in range(self.max_iterations):
            # Cập nhật nhãn cứng
            idx = np.argmax(self.W, axis=1)

            # === Tính μ_k và Comp_k ===
            Mu = np.zeros(K)
            Comp = np.zeros(K)
            for k in range(K):
                mask = (idx == k)
                nk = np.sum(mask)
                if nk == 0:
                    continue
                sqrt_d = np.sqrt(D[mask, k])
                Mu[k] = np.mean(sqrt_d)
                Comp[k] = 1.0 - np.sqrt(np.mean((sqrt_d - Mu[k])**2))
            self.Comp = Comp.copy()

            # === Tính p_{ki} ===
            p = np.zeros((K, N))
            for k in range(K):
                min_dist = np.inf
                j_closest = -1
                for j in range(K):
                    if j == k:
                        continue
                    dist = np.linalg.norm(self.Theta[k] - self.Theta[j])
                    if dist < min_dist:
                        min_dist = dist
                        j_closest = j
                d_kj = np.linalg.norm(self.Theta[k] - self.Theta[j_closest]) + 1e-10
                for i in range(N):
                    dki = np.linalg.norm(self.pdf_matrix[i] - self.Theta[k])
                    dji = np.linalg.norm(self.pdf_matrix[i] - self.Theta[j_closest])
                    p[k, i] = np.abs(dki - dji) / d_kj

            # === Tính P_k và I_star ===
            P = np.zeros(K)
            for k in range(K):
                mask = (idx == k)
                nk = np.sum(mask)
                if nk == 0:
                    continue
                P[k] = np.sum(p[k, mask]) / (nk + 1e-10)
            I = 0.5 * (Comp + P)
            I_star = (I - I.min()) / (I.max() - I.min() + 1e-10)
            self.I_star = I_star.copy()

            # === Tính rho(i) ===
            freq = np.bincount(idx, minlength=K) / N
            rho = np.array([(1 - freq[idx[i]]) / (1 - freq).max() for i in range(N)])
            self.rho = rho.copy()

            # === Cập nhật U(i,k) theo công thức đóng dạng ===
            for k in range(K):
                p_star = np.exp((1 - I_star[k]) * p[k])
                self.U[:, k] = p_star * self.rho

            # === Cập nhật Theta(k) ===
            for k in range(K):
                wk = self.W[:, k] ** m
                denom = np.sum(wk)
                if denom == 0:
                    continue
                self.Theta[k] = np.sum(wk[:, None] * self.pdf_matrix, axis=0) / denom

            # === Cập nhật D và W mới ===
            D = self._dist_matrix()
            self.W = self._update_membership(D, self.U)

            # === Kiểm tra hội tụ ===
            if it > 0:
                delta = np.linalg.norm(self.Theta - old_Theta, ord='fro') / (np.linalg.norm(self.Theta, ord='fro') + 1e-10)
                if delta < self.tolerance:
                    if self.verbose:
                        print(f"[SiiFCM] ✅ Converged at iteration {it+1}")
                    break

            # === Ghi nhận obj và lưu lại Theta để kiểm tra hội tụ ===
            old_Theta = self.Theta.copy()
            obj = np.sum((self.W ** m) * D)
            self.obj.append(obj)

            if self.verbose:
                print(f"[SiiFCM] Iter {it+1} | Obj = {obj:.6f} | Comp = {np.round(self.Comp, 3)}")

    
    def predict(self, new_pdfs):
        if new_pdfs.ndim == 1:
            new_pdfs = new_pdfs[None, :]
        d_obj = Dist(h=self.bandwidth, Dim=self.Dim, grid=self.grid_x)
        func = getattr(d_obj, self.metric)
        D = np.array([
            [func(pdf, theta)**2 + 1e-10 for theta in self.Theta]
            for pdf in new_pdfs
        ])
        return np.argmax(self._update_membership(D, np.ones_like(D)), axis=1)

    def get_results(self):
        return self.W.T.copy(), self.Theta.copy(), self.obj.copy()

    def get_hard_assignments(self):
        return np.argmax(self.W, axis=1)
