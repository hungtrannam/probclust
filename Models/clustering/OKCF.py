import numpy as np
from utils.dist import Dist

class Model:
    """
    Mini-Batch Online K-Clustering (OKC) cho hàm mật độ xác suất (PDF).
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
        self.grid_x = grid_x
        self.num_clusters = num_clusters
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.eta0 = eta0
        self.distance_metric = distance_metric
        self.bandwidth = bandwidth
        self.seed = seed
        self.verbose = verbose
        self.Dim = 1  # chỉ dùng cho PDF 1D

        self.Theta = None              # centroids
        self.pdf_matrix = None        # dữ liệu
        self.assignments = None       # nhãn cụm
        self.objective_history = []   # history loss

    def _compute_distance_matrix(self) -> np.ndarray:
        """Tính ma trận khoảng cách [num_pdfs, num_clusters]."""
        d_obj = Dist(h=self.bandwidth, Dim=self.Dim, grid=self.grid_x)
        num_pdfs = self.pdf_matrix.shape[0]
        return np.array([
            [getattr(d_obj, self.distance_metric)(self.pdf_matrix[i], self.Theta[j])**2 + 1e-10
             for j in range(self.num_clusters)]
            for i in range(num_pdfs)
        ])

    def fit(self, pdf_matrix: np.ndarray):
        """
        Huấn luyện mô hình OKC bằng mini-batch online update.
        """
        self.pdf_matrix = pdf_matrix
        num_pdfs = pdf_matrix.shape[0]

        if self.seed is not None:
            np.random.seed(self.seed)

        # Khởi tạo centroids ngẫu nhiên
        indices = np.random.choice(num_pdfs, self.num_clusters, replace=False)
        self.Theta = pdf_matrix[indices].copy()

        step = 0
        self.objective_history.clear()

        for epoch in range(self.max_epochs):
            perm = np.random.permutation(num_pdfs)
            for batch_start in range(0, num_pdfs, self.batch_size):
                batch_idx = perm[batch_start: batch_start + self.batch_size]
                batch = pdf_matrix[batch_idx]

                for pdf in batch:
                    # Tính khoảng cách đến từng centroid
                    d_obj = Dist(h=self.bandwidth, Dim=self.Dim, grid=self.grid_x)
                    distances = np.array([
                        getattr(d_obj, self.distance_metric)(pdf, self.Theta[k])**2 + 1e-10
                        for k in range(self.num_clusters)
                    ])
                    k_star = np.argmin(distances)

                    # Cập nhật centroid theo learning rate
                    eta = self.eta0 / (1 + step / (num_pdfs * self.max_epochs))
                    self.Theta[k_star] += eta * (pdf - self.Theta[k_star])
                    step += 1

            # Tính objective (loss)
            D = self._compute_distance_matrix()
            obj = np.sum([D[i, np.argmin(D[i])] for i in range(num_pdfs)])
            self.objective_history.append(obj)

            if self.verbose:
                print(f"Epoch {epoch + 1}, objective = {obj:.6f}")

        # Gán nhãn cứng cuối cùng
        D = self._compute_distance_matrix()
        self.assignments = np.argmin(D, axis=1)

    def predict(self, new_pdfs: np.ndarray):
        """Dự đoán nhãn cứng cho PDF mới."""
        if new_pdfs.ndim == 1:
            new_pdfs = new_pdfs[np.newaxis, :]
        d_obj = Dist(h=self.bandwidth, Dim=self.Dim, grid=self.grid_x)
        return np.array([
            np.argmin([
                getattr(d_obj, self.distance_metric)(pdf, self.Theta[k])**2 + 1e-10
                for k in range(self.num_clusters)
            ]) for pdf in new_pdfs
        ])

    def get_results(self):
        """Trả về (centroids, assignments, objective_history)."""
        return self.Theta.copy(), self.assignments.copy(), self.objective_history.copy()
