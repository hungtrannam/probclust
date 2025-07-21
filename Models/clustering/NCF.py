import numpy as np
from utils.dist import Dist

class Model:
    def __init__(self, grid_x, num_clusters=3, m=2, delta=1,
                 w1=1, w2=1, w3=1, max_iterations=100, tolerance=1e-5,
                 distance_metric='L2', bandwidth=0.01,seed= None):
        self.grid_x = grid_x
        self.c = num_clusters
        self.m = m
        self.delta = delta
        self.w1, self.w2, self.w3 = w1, w2, w3
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.distance_metric = distance_metric
        self.bandwidth = bandwidth
        self.eps = 1e-10
        self.seed = seed

    def _compute_distance(self, f1, f2):
        dist_obj = Dist(f1, f2, h=self.bandwidth, Dim=1, grid=self.grid_x)
        dist = {
            'L1': dist_obj.L1(),
            'L2': dist_obj.L2(),
            'H': dist_obj.H(),
            'BC': dist_obj.BC(),
            'W2': dist_obj.W2()
        }[self.distance_metric]
        return max(dist, self.eps)  # avoid zero

    def fit(self, X, verbose=True):
        N, D = X.shape
        c, m, δ = self.c, self.m, self.delta

        # Initialize cluster centers
        if self.seed is not None:
            np.random.seed(self.seed)

        C = X[np.random.choice(N, c, replace=False), :]
        T = np.full((c, N), 1 / c)
        I = np.full(N, 1 / (c + 2))
        F = np.full(N, 1 / (c + 2))

        pow_exp = -2 / (m - 1)

        for iteration in range(self.max_iterations):
            C_prev = C.copy()

            # Step 1: compute c_i_max (average of two closest centers)
            c_i_max = np.zeros((N, D))
            for i in range(N):
                dists = np.array([self._compute_distance(X[i], C[j]) for j in range(c)])
                p, q = np.argsort(dists)[:2]
                c_i_max[i] = (C[p] + C[q]) / 2

            # Precompute distances with eps clip
            dist_T = np.zeros((N, c))
            dist_I = np.zeros(N)
            for i in range(N):
                dist_T[i] = np.array([max(self._compute_distance(X[i], C[j]), self.eps) for j in range(c)])
                dist_I[i] = max(self._compute_distance(X[i], c_i_max[i]), self.eps)
            dist_F = max(δ, self.eps)

            # Step 2: compute K_i
            K = 1.0 / (
                np.sum((1 / self.w1) * dist_T ** pow_exp, axis=1)
                + (1 / self.w2) * dist_I ** pow_exp
                + (1 / self.w3) * dist_F ** pow_exp
                + self.eps
            )

            # Step 3: update T, I, F
            for i in range(N):
                for j in range(c):
                    T[j, i] = K[i] / self.w1 * dist_T[i, j] ** pow_exp
                I[i] = K[i] / self.w2 * dist_I[i] ** pow_exp
                F[i] = K[i] / self.w3 * dist_F ** pow_exp

            # Step 4: update C_j
            for j in range(c):
                weights = (self.w1 * T[j]) ** m
                numerator = np.sum(weights[:, np.newaxis] * X, axis=0)
                denominator = np.sum(weights) + self.eps
                C[j] = numerator / denominator

            # Step 5: check convergence
            delta = np.linalg.norm(C - C_prev)
            if verbose:
                print(f"Iteration {iteration + 1}, delta = {delta:.6e}")
            if delta < self.tolerance or np.isnan(delta):
                if verbose:
                    print("Converged or stopped.")
                break

        self.T, self.I, self.F, self.C = T, I, F, C


    def get_results(self):
        return self.T, self.I, self.F, self.C

    def predict(self, new_X):
        if new_X.ndim == 1:
            new_X = new_X[np.newaxis, :]
        num_new = new_X.shape[0]
        memberships = np.zeros((self.c, num_new))
        for i in range(num_new):
            dists = np.array([self._compute_distance(new_X[i], self.C[j]) for j in range(self.c)])
            sum_T = np.sum([(1 / self.w1) * d ** (-2 / (self.m - 1)) for d in dists])
            sum_I = (1 / self.w2) * np.min(dists) ** (-2 / (self.m - 1))  # approx c_i_max
            sum_F = (1 / self.w3) * self.delta ** (-2 / (self.m - 1))
            K_i = 1 / (sum_T + sum_I + sum_F + self.eps)
            for j in range(self.c):
                memberships[j, i] = K_i / self.w1 * dists[j] ** (-2 / (self.m - 1))
        return memberships
