import numpy as np
from utils.dist import Dist

class EMClustering:
    def __init__(self, pdf_matrix, grid_x, num_clusters=3, max_iterations=100, tolerance=1e-5,
                 distance_metric='L2', bandwidth=0.01):
        """
        EM clustering for probability density functions.

        Parameters:
        - pdf_matrix: np.ndarray, shape [num_points, num_pdfs]
        - grid_x: np.ndarray, grid points for distance calculations
        - num_clusters: int, number of clusters
        - max_iterations: int, maximum number of EM iterations
        - tolerance: float, convergence threshold
        - distance_metric: str, type of distance ('L1', 'L2', 'H', 'BC', 'W2')
        - bandwidth: float, integration bandwidth parameter
        """
        self.pdf_matrix = pdf_matrix  # [num_points, num_pdfs]
        self.grid_x = grid_x
        self.num_clusters = num_clusters
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.distance_metric = distance_metric
        self.bandwidth = bandwidth

        self.num_points, self.num_pdfs = pdf_matrix.shape

        # Initialize soft responsibility matrix (gamma): [num_pdfs, num_clusters]
        self.responsibilities = np.random.dirichlet(np.ones(num_clusters), size=self.num_pdfs)

        # Initialize cluster prototypes (centroids): [num_points, num_clusters]
        self.centroids = np.zeros((self.num_points, self.num_clusters))
        init_indices = np.random.choice(self.num_pdfs, self.num_clusters, replace=False)
        for j, idx in enumerate(init_indices):
            self.centroids[:, j] = self.pdf_matrix[:, idx]

        # Initialize cluster priors (weights)
        self.cluster_priors = np.ones(self.num_clusters) / self.num_clusters

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

    def fit(self, verbose=True):
        for iteration in range(self.max_iterations):
            # M-step: update centroids (prototypes)
            for j in range(self.num_clusters):
                weights = self.responsibilities[:, j]
                numerator = np.sum(weights * self.pdf_matrix, axis=1)
                denominator = np.sum(weights)
                self.centroids[:, j] = numerator / (denominator + 1e-10)

            # M-step: update cluster priors (weights)
            self.cluster_priors = np.sum(self.responsibilities, axis=0) / self.num_pdfs

            # E-step: compute distances
            distance_matrix = np.array([
                [self._compute_distance(self.pdf_matrix[:, i], self.centroids[:, j]) + 1e-10
                 for j in range(self.num_clusters)]
                for i in range(self.num_pdfs)
            ])

            # E-step: update responsibilities (soft assignments)
            new_responsibilities = np.array([
                [self.cluster_priors[j] * np.exp(-distance_matrix[i, j])
                 for j in range(self.num_clusters)]
                for i in range(self.num_pdfs)
            ])
            new_responsibilities /= np.sum(new_responsibilities, axis=1, keepdims=True)

            # Check convergence
            delta = np.linalg.norm(new_responsibilities - self.responsibilities)
            if verbose:
                print(f"Iteration {iteration + 1}, delta = {delta:.6f}")
            if delta < self.tolerance:
                if verbose:
                    print("Converged.")
                break
            self.responsibilities = new_responsibilities

    def predict(self, new_pdfs):
        """
        Predict soft cluster assignments for new pdfs.

        Parameters:
        - new_pdfs: np.ndarray, shape [num_points,] or [num_points, num_new_pdfs]

        Returns:
        - soft_assignments: np.ndarray, shape [num_new_pdfs, num_clusters]
        """
        if new_pdfs.ndim == 1:
            new_pdfs = new_pdfs[:, np.newaxis]
        num_new = new_pdfs.shape[1]
        soft_assignments = np.zeros((num_new, self.num_clusters))

        for idx in range(num_new):
            distances = np.array([
                self._compute_distance(new_pdfs[:, idx], self.centroids[:, j]) + 1e-10
                for j in range(self.num_clusters)
            ])
            probabilities = self.cluster_priors * np.exp(-distances)
            probabilities /= np.sum(probabilities)
            soft_assignments[idx, :] = probabilities

        return soft_assignments

    def get_results(self):
        """
        Returns:
        - responsibilities.T: np.ndarray, shape [num_clusters, num_pdfs]
        - centroids: np.ndarray, shape [num_points, num_clusters]
        - cluster_priors: np.ndarray, shape [num_clusters,]
        """
        return self.responsibilities.copy().T, self.centroids.copy(), self.cluster_priors.copy()

    def get_hard_assignments(self):
        """
        Returns:
        - hard_assignments: np.ndarray, shape [num_pdfs], cluster index per sample
        """
        return np.argmax(self.responsibilities, axis=1)
