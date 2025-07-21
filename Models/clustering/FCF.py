import numpy as np
from utils.dist import Dist

class Model:
    def __init__(self, grid_x, num_clusters=3, fuzziness=2, max_iterations=100,
                 tolerance=1e-5, distance_metric='L2', bandwidth=0.01):
        """
        Fuzzy C-Means Clustering for probability density functions.

        Parameters:
        - pdf_matrix: np.ndarray, shape [num_pdfs, num_points]
        - grid_x: grid points for integration / distance
        - num_clusters: number of clusters
        - fuzziness: fuzziness coefficient m (>1)
        - max_iterations: maximum number of iterations
        - tolerance: convergence threshold (epsilon)
        - distance_metric: distance type ('L1', 'L2', 'H', 'BC', 'W2')
        - bandwidth: bandwidth parameter for distance calculation
        """
        self.grid_x = grid_x
        self.num_clusters = num_clusters
        self.fuzziness = fuzziness
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.distance_metric = distance_metric
        self.bandwidth = bandwidth

    def _compute_distance(self, pdf1, pdf2):
        dist_obj = Dist(pdf1, pdf2, h=self.bandwidth, Dim=1, grid=self.grid_x)
        distance_map = {
            'L1': dist_obj.L1(),
            'L2': dist_obj.L2(),
            'H': dist_obj.H(),
            'BC': dist_obj.BC(),
            'W2': dist_obj.W2()
        }
        return distance_map.get(self.distance_metric, None)

    def fit(self, pdf_matrix, verbose=True):
        self.pdf_matrix = pdf_matrix  # [num_pdfs, num_points]
        self.num_pdfs, self.num_points = pdf_matrix.shape

        # Initialize fuzzy membership matrix U [num_pdfs, num_clusters]
        self.membership_matrix = np.random.dirichlet(np.ones(self.num_clusters), size=self.num_pdfs)

        # Initialize centroids (cluster prototypes) [num_clusters, num_points]
        self.centroids = np.zeros((self.num_clusters, self.num_points))

        # Randomly pick initial centroids from data points
        init_indices = np.random.choice(self.num_pdfs, self.num_clusters, replace=False)
        for j, idx in enumerate(init_indices):
            self.centroids[j, :] = self.pdf_matrix[idx, :]


        for iteration in range(self.max_iterations):
            # Update centroids
            for j in range(self.num_clusters):
                weights = self.membership_matrix[:, j] ** self.fuzziness  # [num_pdfs]
                numerator = np.sum(weights[:, np.newaxis] * self.pdf_matrix, axis=0)  # [num_points]
                denominator = np.sum(weights)
                self.centroids[j, :] = numerator / (denominator + 1e-10)

            # Update distances
            distance_matrix = np.array([
                [self._compute_distance(self.pdf_matrix[i, :], self.centroids[j, :]) + 1e-10
                 for j in range(self.num_clusters)]
                for i in range(self.num_pdfs)
            ])  # [num_pdfs, num_clusters]

            # Update membership matrix U
            new_membership_matrix = np.zeros((self.num_pdfs, self.num_clusters))
            for i in range(self.num_pdfs):
                for j in range(self.num_clusters):
                    ratio = distance_matrix[i, j] / (distance_matrix[i, :] + 1e-10)
                    new_membership_matrix[i, j] = 1.0 / np.sum(ratio ** (2 / (self.fuzziness - 1)))

            # Check convergence
            delta = np.linalg.norm(new_membership_matrix - self.membership_matrix)
            if verbose:
                print(f"Iteration {iteration + 1}, delta = {delta:.6f}")
            if delta < self.tolerance:
                if verbose:
                    print("Converged.")
                break

            self.membership_matrix = new_membership_matrix

    def predict(self, new_pdfs):
        """
        Predict fuzzy membership for new pdf(s).

        Parameters:
        - new_pdfs: np.ndarray, shape [num_new_pdfs, num_points]

        Returns:
        - memberships: np.ndarray, shape [num_new_pdfs, num_clusters]
        """
        if new_pdfs.ndim == 1:
            new_pdfs = new_pdfs[np.newaxis, :]  # [1, num_points]

        num_new = new_pdfs.shape[0]
        memberships = np.zeros((num_new, self.num_clusters))

        for idx in range(num_new):
            distances = np.array([
                self._compute_distance(new_pdfs[idx, :], self.centroids[j, :]) + 1e-10
                for j in range(self.num_clusters)
            ])
            for j in range(self.num_clusters):
                ratio = distances[j] / (distances + 1e-10)
                memberships[idx, j] = 1.0 / np.sum(ratio ** (2 / (self.fuzziness - 1)))

        return memberships

    def get_results(self):
        """
        Returns:
        - membership_matrix.T: np.ndarray, shape [num_clusters, num_pdfs]
        - centroids: np.ndarray, shape [num_clusters, num_points]
        """
        return self.membership_matrix.copy().T, self.centroids.copy()

    def get_hard_assignments(self):
        """
        Returns:
        - hard_assignments: np.ndarray, shape [num_pdfs], cluster index per sample
        """
        return np.argmax(self.membership_matrix, axis=1)
