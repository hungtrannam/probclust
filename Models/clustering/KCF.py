import numpy as np
from utils.dist import Dist

class Model:
    def __init__(self, grid_x, num_clusters=3, max_iterations=100, tolerance=1e-5,
                 distance_metric='L2', bandwidth=0.01, seed = None):
        """
        Classic K-Means Clustering for probability density functions.

        Parameters:
        - pdf_matrix: np.ndarray of shape [num_pdfs, num_points]
        - grid_x: grid points for integration / distance
        - num_clusters: number of clusters
        - max_iterations: maximum number of iterations
        - tolerance: convergence threshold
        - distance_metric: distance type ('L1', 'L2', 'H', 'BC', 'W2')
        - bandwidth: bandwidth parameter for distance calculation
        """
        self.grid_x = grid_x
        self.num_clusters = num_clusters
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.distance_metric = distance_metric
        self.bandwidth = bandwidth
        self.seed = seed

    def _compute_distance(self, pdf1, pdf2):
        distance_obj = Dist(pdf1, pdf2, h=self.bandwidth, Dim=1, grid=self.grid_x)
        distance_map = {
            'L1': distance_obj.L1(),
            'L2': distance_obj.L2(),
            'H': distance_obj.H(),
            'BC': distance_obj.BC(),
            'W2': distance_obj.W2()
        }
        return distance_map.get(self.distance_metric, None)

    def fit(self,pdf_matrix, verbose=True):
        self.pdf_matrix = pdf_matrix  # [num_pdfs, num_points]
        self.num_pdfs, self.num_points = pdf_matrix.shape

        if self.seed is not None:
            np.random.seed(self.seed)

        # Initialize random cluster assignments
        self.cluster_assignments = np.random.randint(0, self.num_clusters, self.num_pdfs)

        # Initialize cluster centroids (prototypes)
        self.centroids = np.zeros((self.num_clusters, self.num_points))  # [num_clusters, num_points]

        for iteration in range(self.max_iterations):
            prev_assignments = self.cluster_assignments.copy()

            # Update centroids (cluster prototypes)
            for cluster_idx in range(self.num_clusters):
                member_indices = np.where(self.cluster_assignments == cluster_idx)[0]
                if len(member_indices) > 0:
                    self.centroids[cluster_idx, :] = np.mean(self.pdf_matrix[member_indices, :], axis=0)
                else:
                    # Reinitialize empty cluster centroid randomly
                    random_idx = np.random.randint(0, self.num_pdfs)
                    self.centroids[cluster_idx, :] = self.pdf_matrix[random_idx, :]

            # Update cluster assignments (hard assignment)
            distance_matrix = np.array([
                [self._compute_distance(self.pdf_matrix[pdf_idx, :], self.centroids[cluster_idx, :]) + 1e-100
                 for cluster_idx in range(self.num_clusters)]
                for pdf_idx in range(self.num_pdfs)
            ])  # [num_pdfs, num_clusters]

            self.cluster_assignments = np.argmin(distance_matrix, axis=1)

            # Calculate objective function
            objective_value = np.sum([
                self._compute_distance(self.pdf_matrix[pdf_idx, :], self.centroids[self.cluster_assignments[pdf_idx], :])
                for pdf_idx in range(self.num_pdfs)
            ])

            # Check convergence
            num_changed = np.sum(prev_assignments != self.cluster_assignments)
            if verbose:
                print(f"Iteration {iteration + 1}, changed assignments = {num_changed}, objective = {objective_value:.6f}")
            if num_changed == 0:
                if verbose:
                    print("Converged.")
                break

        # Create one-hot partition matrix
        self.partition_matrix = np.zeros((self.num_pdfs, self.num_clusters))
        self.partition_matrix[np.arange(self.num_pdfs), self.cluster_assignments] = 1

    def predict(self, new_pdfs):
        """
        Predict hard cluster assignment for new pdfs.

        Parameters:
        - new_pdfs: np.ndarray of shape [num_new_pdfs, num_points]

        Returns:
        - cluster_indices: np.ndarray of shape [num_new_pdfs,]
        """
        if new_pdfs.ndim == 1:
            new_pdfs = new_pdfs[np.newaxis, :]  # [1, num_points]
        num_new = new_pdfs.shape[0]
        predicted_clusters = np.zeros(num_new, dtype=int)

        for idx in range(num_new):
            distances = np.array([
                self._compute_distance(new_pdfs[idx, :], self.centroids[cluster_idx, :]) + 1e-10
                for cluster_idx in range(self.num_clusters)
            ])
            predicted_clusters[idx] = np.argmin(distances)

        return predicted_clusters

    def get_results(self):
        """
        Returns:
        - partition_matrix: np.ndarray [num_clusters, num_pdfs] (one-hot transposed)
        - centroids: np.ndarray [num_clusters, num_points]
        """
        return self.partition_matrix.T.copy(), self.centroids.copy()

    def get_cluster_assignments(self):
        """
        Returns:
        - cluster_assignments: np.ndarray [num_pdfs]
        """
        return self.cluster_assignments.copy()
