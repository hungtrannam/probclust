import numpy as np
from utils.dist import Dist   # your distance utility


class Model:
    """
    Cluster-size Insensitive Fuzzy C-Means for probability density functions.
    Compatible interface with the former FCM version.
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
        centroid_mode: str = "mean",  # 'mean' or 'frechet'
        Dim=None,
        seed: int = None,
        verbose: bool = False,
    ):
        self.grid_x = grid_x
        self.num_clusters = num_clusters
        self.m = fuzziness
        self.maxit = max_iterations
        self.tol = tolerance
        self.metric = distance_metric
        self.h = bandwidth
        self.mode = centroid_mode
        self.seed = seed
        self.verbose = verbose
        self.Dim = Dim if Dim is not None else 1

        # runtime attributes
        self.pdf_matrix = None
        self.Theta = None
        self.U = None          # membership matrix (N, K)
        self.rho = None        # cluster-size penalty (N, 1)
        self.obj_hist = []
        self.rho = []

    def _update_centroids(self):
        """Standard weighted mean (with fuzzy weights)."""
        W = self.U ** self.m
        self.Theta = (W.T @ self.pdf_matrix) / (W.sum(axis=0)[:, None] + 1e-12)

    # ---------- distance ----------
    def _dist_matrix(self):
        """Return (N, K) distance matrix."""
        dobj = Dist(h=self.h, Dim=self.Dim, grid=self.grid_x)
        func = getattr(dobj, self.metric)

        self.num_pdfs = self.pdf_matrix.shape[0]  # num_pdfs
        self.num_clusters = self.num_clusters         # num_clusters
        dist_matrix = np.zeros((self.num_pdfs, self.num_clusters))

        for i in range(self.num_pdfs):
            for j in range(self.num_clusters):
                distance = func(self.pdf_matrix[i], self.Theta[j])
                dist_matrix[i, j] = distance**2 + 1e-10
        return dist_matrix

    # ---------- membership with size penalty ----------
    # Instead of rho[i] for each sample, use rho_k[j] for each cluster
    def _update_membership(self, D, rho_k):
        """
        D : (N, K) distances
        rho_k : (K,) penalties per cluster
        returns U : (N, K) memberships
        """
        D = D + 1e-12
        exp = 1. / (self.m - 1)
        # Multiply distance per cluster with its penalty
        D_weighted = D * rho_k[None, :]  # (N, K)
        inv = (D_weighted[:, :, None] / D_weighted[:, None, :]) ** exp  # (N,K,K)
        tmp = np.sum(inv, axis=2)  # (N,K)
        U = 1. / tmp
        return U

    def fit(self, pdf_matrix: np.ndarray):
        self.pdf_matrix = pdf_matrix
        self.num_pdfs = pdf_matrix.shape[0]

        if self.seed is not None:
            np.random.seed(self.seed)

        # Khởi tạo mềm
        self.U = np.random.rand(self.num_pdfs, self.num_clusters)
        self.U = self.U / self.U.sum(axis=1, keepdims=True)
        
        # Khởi tạo centroid
        self.Theta = pdf_matrix[np.random.choice(self.num_pdfs, self.num_clusters, replace=False)]

        self.obj_hist.clear()
        eps = 1e-6

        for it in range(self.maxit):
            # ---- Cập nhật tâm cụm ----
            self._update_centroids()
            
            # ---- Khoảng cách ----
            D = self._dist_matrix()
            
            # ---- Kích thước cụm và penalty ----
            labels = self.U.argmax(axis=1)
            S = np.bincount(labels, minlength=self.num_clusters) / self.num_pdfs
            rho_k = (1 - S + eps) / (1 - S + eps).max()

            
            # ---- Cập nhật membership ----
            new_U = self._update_membership(D, rho_k)

            # ---- Hàm mục tiêu (objective) ----
            D_weighted = D * rho_k[None, :]  # dùng đúng cách cập nhật khoảng cách
            J = np.sum((new_U ** self.m) * D_weighted)
            self.rho.append(rho_k.copy())
            self.obj_hist.append(J)

            # ---- Kiểm tra hội tụ ----
            delta = np.linalg.norm(new_U - self.U)
            if self.verbose:
                print(f"[CSI-FCM] Iteration {it+1:3d} | ΔU = {delta:.6e} | J = {J:.6f}")
                print(f"           Cluster sizes: {np.round(S, 4)} | rho_k: {np.round(rho_k, 3)}")
                print(f"           Max change in U: {np.max(np.abs(new_U - self.U)):.4e}")

            if delta < self.tol:
                if self.verbose:
                    print(f"[CSI-FCM] ✅ Converged after {it+1} iterations (ΔU < tol={self.tol}).")
                break
            
            self.U = new_U
        else:
            if self.verbose:
                print(f"[CSI-FCM] ⚠️ Not converged after {self.maxit} iterations.")

        

    def predict(self, new_pdfs: np.ndarray):
        """Membership for new pdfs."""
        if new_pdfs.ndim == 1:
            new_pdfs = new_pdfs[None, :]
        dobj = Dist(h=self.h, Dim=self.Dim, grid=self.grid_x)
        func = getattr(dobj, self.metric)
        memberships = []
        for pdf in new_pdfs:
            d = np.array([func(pdf, c) + 1e-10 for c in self.Theta])
            exp = 1. / (self.m - 1)
            inv = (d / d[:, None]) ** exp
            u = 1. / inv.sum(axis=1)
            u /= u.sum()
            memberships.append(u)
        return np.array(memberships)

    def get_results(self):
        return self.U.T.copy(), self.Theta.copy(), self.obj_hist.copy()

    def get_hard_assignments(self):
        return self.U.argmax(axis=1)