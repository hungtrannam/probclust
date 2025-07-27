import numpy as np
from utils.dist import Dist

class Model:
    """
    Mini-Batch Online K-Clustering cho hàm mật độ xác suất (PDF).
    """

    def __init__(
        self,
        grid_x: np.ndarray,
        num_clusters: int = 3,
        max_epochs: int = 10,
        batch_size: int = 16,
        eta0: float = 0.5,
        distance_metric: str = "L2",
        bandwidth: float = 0.01,
        seed: int = None,
        verbose: bool = False,
    ):
        """
        Parameters
        ----------
        grid_x : np.ndarray
            Lưới x để tính khoảng cách.
        num_clusters : int
            Số cụm.
        max_epochs : int
            Số epoch.
        batch_size : int
            Kích thước mini-batch.
        eta0 : float
            Learning rate ban đầu.
        distance_metric : str
            Loại khoảng cách ('L1', 'L2', 'H', 'BC', 'W2').
        bandwidth : float
            Tham số bandwidth.
        seed : int
            Seed random.
        verbose : bool
            In log nếu True.
        """
        self.grid_x = grid_x
        self.num_clusters = num_clusters
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.eta0 = eta0
        self.distance_metric = distance_metric
        self.bandwidth = bandwidth
        self.seed = seed
        self.verbose = verbose

        self.centroids = None
        self.cluster_assignments = None
        self.objective_history = []

    def _compute_distance(self, pdf1, pdf2):
        """Tính khoảng cách giữa 2 PDF."""
        d_obj = Dist(pdf1, pdf2, h=self.bandwidth, Dim=1, grid=self.grid_x)
        return {
            "L1": d_obj.L1(),
            "L2": d_obj.L2(),
            "H": d_obj.H(),
            "BC": d_obj.BC(),
            "W2": d_obj.W2(),
        }[self.distance_metric]

    def _compute_distances_to_centroids(self, pdf):
        """Tính khoảng cách pdf đến tất cả centroids."""
        return np.array([self._compute_distance(pdf, self.centroids[k])
                         for k in range(self.num_clusters)])

    def fit(self, pdf_matrix):
        """Huấn luyện Mini-Batch Online K-Clustering."""
        num_pdfs, num_points = pdf_matrix.shape
        if self.seed is not None:
            np.random.seed(self.seed)

        # Khởi tạo centroids ngẫu nhiên
        indices = np.random.choice(num_pdfs, self.num_clusters, replace=False)
        self.centroids = pdf_matrix[indices].copy()

        step = 0
        self.objective_history.clear()

        for epoch in range(self.max_epochs):
            perm = np.random.permutation(num_pdfs)
            for batch_start in range(0, num_pdfs, self.batch_size):
                batch_idx = perm[batch_start: batch_start + self.batch_size]
                batch = pdf_matrix[batch_idx]

                for pdf in batch:
                    distances = self._compute_distances_to_centroids(pdf)
                    k_star = np.argmin(distances)

                    # Learning rate giảm dần theo step
                    eta = self.eta0 / (1 + step / (num_pdfs * self.max_epochs))
                    self.centroids[k_star] += eta * (pdf - self.centroids[k_star])
                    step += 1

            # Tính objective cuối epoch
            obj = np.sum([
                self._compute_distance(pdf_matrix[i],
                                       self.centroids[np.argmin([
                                           self._compute_distance(pdf_matrix[i], self.centroids[k])
                                           for k in range(self.num_clusters)
                                       ])])
                for i in range(num_pdfs)
            ])
            self.objective_history.append(obj)
            if self.verbose:
                print(f"Epoch {epoch + 1}, objective = {obj:.6f}")

        # Gán nhãn cuối cùng
        self.cluster_assignments = np.array([
            np.argmin([self._compute_distance(pdf_matrix[i], self.centroids[k])
                       for k in range(self.num_clusters)])
            for i in range(num_pdfs)
        ])

    def predict(self, new_pdfs):
        """Dự đoán nhãn cứng cho PDF mới."""
        if new_pdfs.ndim == 1:
            new_pdfs = new_pdfs[np.newaxis, :]
        return np.array([
            np.argmin([self._compute_distance(pdf, self.centroids[k])
                       for k in range(self.num_clusters)])
            for pdf in new_pdfs
        ])

    def get_results(self):
        """Trả về (centroids, assignments, objective_history)."""
        return self.centroids.copy(), self.cluster_assignments.copy(), self.objective_history.copy()
