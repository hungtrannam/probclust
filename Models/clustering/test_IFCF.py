import numpy as np
from utils.dist import Dist

class Model:
    def __init__(self, grid_x, num_clusters=3, fuzziness=2, max_iterations=100,
                 tolerance=1e-5, distance_metric='L2', bandwidth=0.01):
        """
        Improved Fuzzy C-Means Clustering (IFCM) for probability density functions.
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
                [self._compute_distance(self.pdf_matrix[i, :], self.centroids[j, :]) + 1e-100
                 for j in range(self.num_clusters)]
                for i in range(self.num_pdfs)
            ])  # [num_pdfs, num_clusters]

            # Calculate fci (omega)
            hard_assignments = np.argmax(self.membership_matrix, axis=1)  # [num_pdfs]
            counts = np.array([np.sum(hard_assignments == j) for j in range(self.num_clusters)])  # [num_clusters]
            Sj = counts / self.num_pdfs  # [num_clusters]
            self.fci = np.zeros(self.num_clusters)
            self.fci = (1 - Sj) / (1 - np.min(Sj) + 1e-100)  # [num_clusters]

            # Update membership matrix U (with fci)
            new_membership_matrix = np.zeros((self.num_pdfs, self.num_clusters))
            for i in range(self.num_pdfs):
                denom = np.sum(self.fci / (distance_matrix[i, :] ** (1 / (self.fuzziness - 1))))
                for j in range(self.num_clusters):
                    new_membership_matrix[i, j] = (self.fci[j] / (distance_matrix[i, j] ** (1 / (self.fuzziness - 1)))) / (denom + 1e-10)

            # Check convergence (based on centroids)
            centroid_change = np.linalg.norm(self.centroids - np.array([
                np.sum((new_membership_matrix[:, j] ** self.fuzziness)[:, np.newaxis] * self.pdf_matrix, axis=0) / 
                (np.sum(new_membership_matrix[:, j] ** self.fuzziness) + 1e-100)
                for j in range(self.num_clusters)
            ]), ord=1)

            if verbose:
                print(f"Iteration {iteration + 1}, centroid change = {centroid_change:.6e}")
                print(f"Objective function value: {np.sum(new_membership_matrix ** self.fuzziness)}")

            if centroid_change < self.tolerance:
                if verbose:
                    print("Converged.")
                break

            self.membership_matrix = new_membership_matrix

    def predict(self, new_pdfs):
        """
        Predict fuzzy membership for new pdf(s).
        """
        if new_pdfs.ndim == 1:
            new_pdfs = new_pdfs[np.newaxis, :]  # [1, num_points]

        num_new = new_pdfs.shape[0]
        memberships = np.zeros((num_new, self.num_clusters))
        m = self.fuzziness

        for idx in range(num_new):
            distances = np.array([
                self._compute_distance(new_pdfs[idx, :], self.centroids[j, :]) + 1e-10
                for j in range(self.num_clusters)
            ])
            denom = np.sum(self.fci / (distances ** (1 / (m - 1))))
            for j in range(self.num_clusters):
                memberships[idx, j] = (self.fci[j] / (distances[j] ** (1 / (m - 1)))) / (denom + 1e-100)

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
