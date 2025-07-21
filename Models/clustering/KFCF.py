import numpy as np
from utils.dist import Dist

class Model:
    def __init__(self, grid_x, num_clusters=3, fuzziness=2, max_iterations=100,
                 tolerance=1e-5, kernel_type='L2', gamma=1.0, bandwidth=0.01, seed= None):
        """
        Kernel Fuzzy C-Means clustering for probability density functions.

        Parameters:
        - pdf_matrix: np.ndarray, shape [num_points, num_pdfs]
        - grid_x: np.ndarray, grid points
        - num_clusters: int, number of clusters
        - fuzziness: float, fuzziness coefficient (m > 1)
        - max_iterations: int, maximum number of iterations
        - tolerance: float, convergence threshold
        - kernel_type: str, 'L1' (Laplacian) or 'L2' (Gaussian)
        - gamma: float, kernel parameter
        - bandwidth: float, integration bandwidth
        """
        self.grid_x = grid_x
        self.num_clusters = num_clusters
        self.fuzziness = fuzziness
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.kernel_type = kernel_type
        self.gamma = gamma
        self.bandwidth = bandwidth
        self.seed= seed

    def _kernel_function(self, pdf1, pdf2):
        dist_obj = Dist(pdf1, pdf2, h=self.bandwidth, Dim=1, grid=self.grid_x)
        if self.kernel_type == 'L1':
            l1 = dist_obj.L1()
            return np.exp(-self.gamma * l1 ** 2)
        elif self.kernel_type == 'L2':
            l2 = dist_obj.L2()
            return np.exp(-self.gamma * l2 ** 2)
        else:
            raise ValueError(f"Unsupported kernel type: {self.kernel_type}")

    def _compute_kernel_matrix(self):
        K = np.zeros((self.num_pdfs, self.num_pdfs))
        for i in range(self.num_pdfs):
            for j in range(i, self.num_pdfs):
                K_val = self._kernel_function(self.pdf_matrix[i], self.pdf_matrix[j])
                K[i, j] = K[j, i] = K_val
        return K

    def fit(self, pdf_matrix, verbose=True):
        self.pdf_matrix = pdf_matrix
        self.num_pdfs = self.pdf_matrix.shape[0]
        if self.seed is not None:
            np.random.seed(self.seed)

        self.membership_matrix = np.random.dirichlet(np.ones(self.num_clusters), size=self.num_pdfs)  # [num_pdfs, num_clusters]

        # Initialize cluster centroids (prototypes) as random pdfs
        initial_indices = np.random.choice(self.num_pdfs, self.num_clusters, replace=False)
        self.centroids = self.pdf_matrix[initial_indices]  # [num_clusters, num_points]

        # Precompute kernel matrix
        self.kernel_matrix = self._compute_kernel_matrix()
        
        eps_small = 1e-100

        for iteration in range(self.max_iterations):
            previous_U = self.membership_matrix.copy()

            # Step 1: Compute K(x_k, v_j)
            kernel_to_centroids = np.zeros((self.num_pdfs, self.num_clusters))
            for i in range(self.num_pdfs):
                for j in range(self.num_clusters):
                    kernel_to_centroids[i, j] = self._kernel_function(self.pdf_matrix[i], self.centroids[j])

            # Step 2: Update membership matrix U
            distances = np.sqrt(2 * (1 - kernel_to_centroids))
            for k in range(self.num_pdfs):
                denom = np.sum((distances[k, :] + eps_small) ** (-1 / (self.fuzziness - 1)))
                for j in range(self.num_clusters):
                    self.membership_matrix[k, j] = (distances[k, j] + eps_small) ** (-1 / (self.fuzziness - 1)) / denom

            # Step 3: Update centroids
            for j in range(self.num_clusters):
                weights_m = self.membership_matrix[:, j] ** self.fuzziness
                weights_kernel = weights_m * kernel_to_centroids[:, j]
                numerator = np.sum(weights_kernel[:, None] * self.pdf_matrix, axis=0)
                denominator = np.sum(weights_kernel) + eps_small
                self.centroids[j] = numerator / denominator

            # Check convergence
            delta = np.linalg.norm(self.membership_matrix - previous_U)
            if verbose:

                objective_value = np.sum((self.membership_matrix ** self.fuzziness) * (2 * (1 - kernel_to_centroids)))
                print(f"Iteration {iteration + 1}, delta = {delta:.6f}, objective = {objective_value:.6f}")  

            if delta < self.tolerance:
                print("Converged.")
                break

    def predict(self, new_pdfs):
        """
        Predict fuzzy membership for new pdf(s).

        Parameters:
        - new_pdfs: np.ndarray, shape [num_points,] or [num_points, num_new_pdfs]

        Returns:
        - memberships: np.ndarray, shape [num_new_pdfs, num_clusters]
        """
        if new_pdfs.shape[1] == 1:
            new_pdfs = new_pdfs.T
        num_new = new_pdfs.shape[0]
        memberships = np.zeros((num_new, self.num_clusters))
        eps_small = 1e-100

        for i in range(num_new):
            # Compute kernel between f_new[i] and all pdfs
            kernel_to_pdfs = np.array([self._kernel_function(new_pdfs[i], self.pdf_matrix[k]) for k in range(self.num_pdfs)])

            # Compute kernel between f_new[i] and each cluster centroid
            kernel_to_centroids = np.zeros(self.num_clusters)
            for j in range(self.num_clusters):
                weights = self.membership_matrix[:, j] ** self.fuzziness
                kernel_to_centroids[j] = np.sum(weights * kernel_to_pdfs) / (np.sum(weights) + eps_small)

            # Compute distance and fuzzy membership
            distances = np.sqrt(2 * (1 - kernel_to_centroids))
            denom = np.sum(distances ** (-1 / (self.fuzziness - 1)))
            memberships[i, :] = (distances ** (-1 / (self.fuzziness - 1))) / denom

        return memberships

    def get_results(self):
        """
        Returns:
        - membership_matrix.T: np.ndarray, shape [num_clusters, num_pdfs]
        - centroids.T: np.ndarray, shape [num_points, num_clusters]
        """
        return self.membership_matrix.copy().T, self.centroids
