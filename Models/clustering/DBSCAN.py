import numpy as np
from utils.dist import Dist

class Model:
    def __init__(self, grid_x, eps=0.5, min_samples=2, distance_metric='L2', bandwidth=0.01, verbose=False):
        """
        DBSCAN clustering cho các hàm mật độ xác suất.

        Parameters:
        - grid_x: np.ndarray, lưới tích phân
        - eps: float, bán kính lân cận
        - min_samples: int, số điểm tối thiểu để tạo cụm
        - distance_metric: str, loại khoảng cách ('L1', 'L2', 'H', 'BC', 'W2')
        - bandwidth: float, bước tích phân
        """
        self.grid_x = grid_x
        self.eps = eps
        self.min_samples = min_samples
        self.distance_metric = distance_metric
        self.bandwidth = bandwidth
        self.verbose = verbose

    def _compute_distance(self, pdf1, pdf2):
        d_obj = Dist(h=self.bandwidth, Dim=1, grid=self.grid_x)
        return getattr(d_obj, self.distance_metric)(pdf1,pdf2)**2

    def _region_query(self, idx):
        neighbors = []
        for j in range(self.num_pdfs):
            if idx == j:
                continue
            d = self._compute_distance(self.pdf_matrix[idx], self.pdf_matrix[j])
            if d <= self.eps:
                neighbors.append(j)
        return neighbors

    def _expand_cluster(self, idx, neighbors, cluster_id):
        self.labels[idx] = cluster_id
        queue = list(neighbors)
        while queue:
            current = queue.pop()
            if not self.visited[current]:
                self.visited[current] = True
                current_neighbors = self._region_query(current)
                if len(current_neighbors) >= self.min_samples:
                    queue.extend(current_neighbors)
            if self.labels[current] == -1:
                self.labels[current] = cluster_id

    def fit(self, pdf_matrix):
        self.pdf_matrix = pdf_matrix
        self.num_pdfs = pdf_matrix.shape[0]
        self.labels = -np.ones(self.num_pdfs, dtype=int)
        self.visited = np.zeros(self.num_pdfs, dtype=bool)

        cluster_id = 0
        for i in range(self.num_pdfs):
            if self.visited[i]:
                continue
            self.visited[i] = True
            neighbors = self._region_query(i)
            if len(neighbors) < self.min_samples:
                self.labels[i] = -1  # 
            else:
                self._expand_cluster(i, neighbors, cluster_id)
                if self.verbose:
                    print(f"Tạo cụm {cluster_id} từ điểm {i}")
                cluster_id += 1


    def get_hard_assignments(self):
        """
        Trả về:
        - hard_assignments: np.ndarray, shape [num_pdfs], cluster index per sample
        """
        return self.labels.copy()


    def get_results(self):
        """
        Trả về:
        - U: ma trận phân vùng nhị phân (num_clusters + 1, num_pdfs)
        Hàng cuối cùng (= hàng nhiễu) tương ứng nhãn -1.
        """
        unique_clusters = sorted(set(self.labels) - {-1})  # loại -1 ra
        num_clusters = len(unique_clusters)

        # Khởi tạo ma trận 0-1: (số cụm + 1 hàng nhiễu) x số pdf
        U = np.zeros((num_clusters + 1, self.num_pdfs), dtype=int)

        # Gán 1 cho các điểm thuộc cụm
        for i, label in enumerate(self.labels):
            if label != -1:
                cluster_idx = unique_clusters.index(label)
                U[cluster_idx, i] = 1
            else:  # label == -1  -> nhiễu
                U[-1, i] = 1  # hàng cuối cùng là hàng nhiễu

        return U