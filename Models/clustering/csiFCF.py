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
        self.centroids = None
        self.U = None          # membership matrix (N, K)
        self.rho = None        # cluster-size penalty (N, 1)
        self.obj_hist = []

    def _update_centroids(self):
        """Standard weighted mean (with fuzzy weights)."""
        W = self.U ** self.m
        self.centroids = (W.T @ self.pdf_matrix) / (W.sum(axis=0)[:, None] + 1e-12)

    # ---------- distance ----------
    def _dist_matrix(self):
        """Return (N, K) distance matrix."""
        dobj = Dist(h=self.h, Dim=self.Dim, grid=self.grid_x)
        func = getattr(dobj, self.metric)
        return np.array([[func(self.pdf_matrix[i], self.centroids[j])**2 + 1e-10
                          for j in range(self.num_clusters)]
                         for i in range(self.pdf_matrix.shape[0])])

    # ---------- membership with size penalty ----------
    def _update_membership(self, D):
        """
        D : (N, K) distances
        rho : (N, 1) penalties
        returns U : (N, K) memberships
        """
        D = D + 1e-12
        exp = 1. / (self.m - 1)
        inv = (D[:, :, None] / D[:, None, :]) ** exp  # (N,K,K)
        tmp = np.sum(inv, axis=2)                     # (N,K)
        U = self.rho * (1. / tmp)                     # (N,K)
        return U

    # ---------- public API ----------
    def fit(self, pdf_matrix: np.ndarray):
        self.pdf_matrix = pdf_matrix
        self.num_pdfs = pdf_matrix.shape[0]

        if self.seed is not None:
            np.random.seed(self.seed)

        # init membership (Dirichlet)
        self.U = np.random.dirichlet(np.ones(self.num_clusters), size=self.num_pdfs)
        # init centroids
        self.centroids = pdf_matrix[np.random.choice(self.num_pdfs, self.num_clusters, replace=False)]

        self.obj_hist.clear()
        self.rho = np.ones((self.num_pdfs, 1))

        for it in range(self.maxit):
            # # ----- centroid step -----
            # if self.mode == "frechet":
            #     self._update_centroids_w2()
            # else:
            self._update_centroids()

            # ----- distance & membership -----
            D = self._dist_matrix()
            # compute hard labels for size estimation
            labels = self.U.argmax(axis=1)
            S = np.bincount(labels, minlength=self.num_clusters) / self.num_pdfs      # (K,)
            self.rho = (1 - S[labels]) / (1 - S).max()    # (N,)
            self.rho = self.rho.reshape(-1, 1)

            new_U = self._update_membership(D)

            # ----- objective (for history only) -----
            J = np.sum((new_U ** self.m) * D)
            self.obj_hist.append(J)

            # ----- convergence -----
            delta = np.linalg.norm(new_U - self.U)
            if self.verbose:
                print(f"it {it+1}: delta={delta:.6f}, J={J:.6f}")
            if delta < self.tol:
                if self.verbose:
                    print("Converged.")
                break
            self.U = new_U
        else:
            if self.verbose:
                print(f"csiFCM did not converge in {self.maxit} iterations.")

    def predict(self, new_pdfs: np.ndarray):
        """Membership for new pdfs."""
        if new_pdfs.ndim == 1:
            new_pdfs = new_pdfs[None, :]
        dobj = Dist(h=self.h, Dim=self.Dim, grid=self.grid_x)
        func = getattr(dobj, self.metric)
        memberships = []
        for pdf in new_pdfs:
            d = np.array([func(pdf, c) + 1e-10 for c in self.centroids])
            exp = 1. / (self.m - 1)
            inv = (d / d[:, None]) ** exp
            u = 1. / inv.sum(axis=1)
            u /= u.sum()
            memberships.append(u)
        return np.array(memberships)

    def get_results(self):
        return self.U.T.copy(), self.centroids.copy(), self.obj_hist.copy()

    def get_hard_assignments(self):
        return self.U.argmax(axis=1)