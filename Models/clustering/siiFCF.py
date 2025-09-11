import numpy as np
from utils.dist import Dist


class Model:
    """
    SiiFCM: Size-Insensitive Integrity-based Fuzzy C-Means Clustering
    Áp dụng cho hàm mật độ xác suất (PDFs) theo Lin et al. (2014).
    """

    def __init__(
        self,
        grid_x: np.ndarray,
        num_clusters: int = 3,
        fuzziness: float = 2.0,
        max_iterations: int = 100,
        tolerance: float = 1e-5,
        distance_metric: str = "L1",
        bandwidth: float = 0.01,
        Dim: int = 1,
        seed: int = None,
        verbose: bool = False,
    ):
        self.grid_x = grid_x
        self.num_clusters = num_clusters
        self.fuzziness = fuzziness
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.distance_metric = distance_metric
        self.bandwidth = bandwidth
        self.seed = seed
        self.verbose = verbose
        self.Dim = Dim

        self.pdf_matrix = None
        self.Theta = None
        self.U = None
        self.objective_history = []

    def _l1(self, a, b):
        dist = Dist(h=self.bandwidth, Dim=self.Dim, grid=self.grid_x)
        return dist.L1(a, b) + 1e-10

    def _dist_matrix(self):
        dist = Dist(h=self.bandwidth, Dim=self.Dim, grid=self.grid_x)
        func = getattr(dist, self.distance_metric)
        N, K = self.pdf_matrix.shape[0], self.num_clusters
        D2 = np.zeros((N, K))
        for i in range(N):
            for j in range(K):
                D2[i, j] = func(self.pdf_matrix[i], self.Theta[j]) ** 2 + 1e-10
        return D2

    def _update_centroids(self):
        Um = self.U ** self.fuzziness
        self.Theta = (Um.T @ self.pdf_matrix) / (np.sum(Um.T, axis=1, keepdims=True) + 1e-12)

    def _compute_compactness(self, A_idx, Theta_j):
        if len(A_idx) == 0:
            return 0.0
        dists = [self._l1(self.pdf_matrix[i], Theta_j) for i in A_idx]
        return 1.0 - np.std(dists)

    def _compute_purity(self, i, A_idx):
        purity = []
        for idx in A_idx:
            xi = self.pdf_matrix[idx]
            vi = self.Theta[i]
            other_idx = [j for j in range(self.num_clusters) if j != i]
            vj = self.Theta[min(other_idx, key=lambda j: self._l1(vi, self.Theta[j]))]
            d1 = self._l1(xi, vi)
            d2 = self._l1(xi, vj)
            dp = self._l1(vi, vj)
            pij = np.abs(d1 - d2) / (dp + 1e-10)
            purity.append(pij)
        return np.mean(purity) if len(purity) > 0 else 0.0

    def _compute_integrity(self, clusters):
        C, P = [], []
        for i in range(self.num_clusters):
            A_idx = clusters[i]
            C.append(self._compute_compactness(A_idx, self.Theta[i]))
            P.append(self._compute_purity(i, A_idx))
        I = [(c + p) / 2 for c, p in zip(C, P)]
        I_min, I_max = min(I), max(I)
        I_norm = [(ii - I_min) / (I_max - I_min + 1e-10) for ii in I]
        return I_norm, P

    def _compute_condition_W(self, D2, clusters):
        N, K = D2.shape
        cluster_sizes = np.array([len(clusters[k]) for k in range(K)])
        S = cluster_sizes / np.sum(cluster_sizes)
        F = (1 - S) / (1 - np.min(S) + 1e-10)
        I_norm, _ = self._compute_integrity(clusters)

        W = np.zeros((N, K))
        for i in range(N):
            for j in range(K):
                if j not in clusters or len(clusters[j]) == 0:
                    W[i, j] = F[j]
                else:
                    vi = self.Theta[j]
                    other = min(
                        [jj for jj in range(K) if jj != j],
                        key=lambda t: self._l1(vi, self.Theta[t])
                    )
                    d1 = self._l1(self.pdf_matrix[i], vi)
                    d2 = self._l1(self.pdf_matrix[i], self.Theta[other])
                    dp = self._l1(vi, self.Theta[other])
                    pij = np.abs(d1 - d2) / (dp + 1e-10)
                    W[i, j] = F[j] * np.exp((1 - I_norm[j]) * pij)
        return W

    def _update_membership(self, D2, W):
        m = self.fuzziness
        N, K = D2.shape
        U_new = np.zeros((N, K))
        for i in range(N):
            for j in range(K):
                denom = sum([(D2[i, j] / D2[i, k]) ** (1 / (m - 1)) for k in range(K)])
                U_new[i, j] = W[i, j] / (denom + 1e-10)
        return U_new

    def fit(self, pdf_matrix: np.ndarray):
        self.pdf_matrix = pdf_matrix
        N, _ = pdf_matrix.shape
        K = self.num_clusters
        if self.seed is not None:
            np.random.seed(self.seed)

        self.U = np.random.dirichlet(np.ones(K), size=N)
        idx_init = np.random.choice(N, K, replace=False)
        self.Theta = pdf_matrix[idx_init]

        for it in range(self.max_iterations):
            self._update_centroids()
            D2 = self._dist_matrix()
            hard_labels = np.argmax(self.U, axis=1)
            clusters = {k: np.where(hard_labels == k)[0].tolist() for k in range(K)}

            W = self._compute_condition_W(D2, clusters)
            U_new = self._update_membership(D2, W)

            obj = np.sum((U_new ** self.fuzziness) * D2)
            self.objective_history.append(obj)

            delta = np.linalg.norm(U_new - self.U)
            if self.verbose:
                print(f"[{it:02d}] ΔU = {delta:.5f}, Obj = {obj:.5f}")
            if delta < self.tolerance:
                break

            self.U = U_new

    def predict(self, pdfs: np.ndarray) -> np.ndarray:
        dist = Dist(h=self.bandwidth, Dim=self.Dim, grid=self.grid_x)
        func = getattr(dist, self.distance_metric)

        if pdfs.ndim == 1:
            pdfs = pdfs[None, :]

        N, K = pdfs.shape[0], self.num_clusters
        D2 = np.zeros((N, K))
        for i in range(N):
            for j in range(K):
                D2[i, j] = func(pdfs[i], self.Theta[j]) ** 2 + 1e-10

        U_pred = np.zeros((N, K))
        m = self.fuzziness
        for i in range(N):
            denom = sum([(D2[i, j] / D2[i, k]) ** (1 / (m - 1)) for k in range(K)])
            for j in range(K):
                U_pred[i, j] = 1.0 / (denom + 1e-10)
        return U_pred

    def get_results(self):
        return self.U.T.copy(), self.Theta.copy(), self.objective_history.copy()

    def get_hard_assignments(self) -> np.ndarray:
        return np.argmax(self.U, axis=1)
