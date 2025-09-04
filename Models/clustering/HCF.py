import numpy as np
from utils.dist import Dist


class Model:
    """
    Hierarchical Clustering cho các hàm mật độ xác suất (PDF).
    """

    def __init__(
        self,
        grid_x: np.ndarray,
        max_depth: int = 7,
        linkage: str = "average",
        min_cluster_size: int = 1,
        distance_metric: str = "L2",
        bandwidth: float = 0.01,
    ):
        """
        Parameters
        ----------
        grid_x : np.ndarray
            Lưới x để tính khoảng cách.
        max_depth : int
            Độ sâu tối đa của cây phân cụm.
        linkage : str
            Kiểu liên kết ('single', 'complete', 'average', 'centroid', 'ward').
        min_cluster_size : int
            Kích thước cụm nhỏ nhất.
        distance_metric : str
            Loại khoảng cách ('L1', 'L2', 'H', 'BC', 'W2').
        bandwidth : float
            Bước tích phân (bandwidth).
        """
        self.grid_x = grid_x
        self.max_depth = max_depth
        self.min_cluster_size = min_cluster_size
        self.linkage = linkage
        self.distance_metric = distance_metric
        self.bandwidth = bandwidth
        self.tree = {}

    # ------------------------------
    # TÍNH TOÁN KHOẢNG CÁCH
    # ------------------------------
    def _compute_distance_matrix(self, pdf_matrix: np.ndarray) -> np.ndarray:
        """Tính ma trận khoảng cách giữa các PDF."""
        n_pdfs = pdf_matrix.shape[0]
        dist_matrix = np.zeros((n_pdfs, n_pdfs))
        for i in range(n_pdfs):
            for j in range(i + 1, n_pdfs):
                d_obj = Dist(h=self.bandwidth, Dim=1, grid=self.grid_x)
                dist_matrix[i, j] = dist_matrix[j, i] = getattr(d_obj, self.distance_metric)(pdf_matrix[i], pdf_matrix[j])
        return dist_matrix

    def _calculate_linkage_distance(
        self, cluster_a: list, cluster_b: list, dist_matrix: np.ndarray, pdf_matrix: np.ndarray
    ) -> float:
        """Tính khoảng cách giữa hai cụm theo kiểu liên kết."""
        if self.linkage in ["single", "complete", "average"]:
            dists = [dist_matrix[i, j] for i in cluster_a for j in cluster_b]
            if self.linkage == "single":
                return np.min(dists)
            elif self.linkage == "complete":
                return np.max(dists)
            else:
                return np.mean(dists)

        elif self.linkage in ["centroid", "ward"]:
            centroid_a = np.mean(pdf_matrix[cluster_a], axis=0)
            centroid_b = np.mean(pdf_matrix[cluster_b], axis=0)
            d_obj = Dist(h=self.bandwidth, Dim=1, grid=self.grid_x)
            centroid_dist = getattr(d_obj, self.distance_metric)(centroid_a, centroid_b)
            if self.linkage == "ward":
                size_a, size_b = len(cluster_a), len(cluster_b)
                return (size_a * size_b) / (size_a + size_b) * centroid_dist ** 2
            return centroid_dist

        else:
            raise ValueError(f"Unsupported linkage: {self.linkage}")

    # ------------------------------
    # CHIA CỤM
    # ------------------------------
    def _split_cluster(self, pdf_matrix: np.ndarray) -> tuple:
        """Tách cụm hiện tại thành 2 cụm con."""
        dist_matrix = self._compute_distance_matrix(pdf_matrix)
        clusters = [[i] for i in range(pdf_matrix.shape[0])]

        while len(clusters) > 2:
            min_dist = np.inf
            merge_pair = None
            for i in range(len(clusters)):
                for j in range(i + 1, len(clusters)):
                    dist_ij = self._calculate_linkage_distance(clusters[i], clusters[j], dist_matrix, pdf_matrix)
                    if dist_ij < min_dist:
                        min_dist = dist_ij
                        merge_pair = (i, j)
            i, j = merge_pair
            clusters[i].extend(clusters[j])
            del clusters[j]

        return np.array(clusters[0]), np.array(clusters[1])

    # ------------------------------
    # ĐỆ QUY XÂY CÂY
    # ------------------------------
    def _fit_recursive(self, pdf_matrix: np.ndarray, indices: np.ndarray, depth: int, node_name: str) -> None:
        """Xây cây phân cụm đệ quy."""
        n_samples = pdf_matrix.shape[0]
        if depth >= self.max_depth or n_samples <= self.min_cluster_size:
            self.tree[node_name] = {"indices": indices.tolist(), "children": {}}
            return

        cluster_0_idx, cluster_1_idx = self._split_cluster(pdf_matrix)
        if len(cluster_0_idx) < self.min_cluster_size or len(cluster_1_idx) < self.min_cluster_size:
            self.tree[node_name] = {"indices": indices.tolist(), "children": {}}
            return

        self.tree[node_name] = {"indices": indices.tolist(), "children": {}}

        # Gọi đệ quy cho cluster 0
        self._fit_recursive(pdf_matrix[cluster_0_idx], indices[cluster_0_idx], depth + 1, node_name + "0")
        # Gọi đệ quy cho cluster 1
        self._fit_recursive(pdf_matrix[cluster_1_idx], indices[cluster_1_idx], depth + 1, node_name + "1")

    # ------------------------------
    # FIT MODEL
    # ------------------------------
    def fit(self, pdf_matrix: np.ndarray) -> None:
        """Huấn luyện cây phân cụm."""
        if pdf_matrix.shape[1] != len(self.grid_x):
            raise ValueError(
                f"pdf_matrix.shape[1] ({pdf_matrix.shape[1]}) must match grid_x length ({len(self.grid_x)})"
            )
        indices = np.arange(pdf_matrix.shape[0])
        self.dist_matrix = self._compute_distance_matrix(pdf_matrix)
        self._fit_recursive(pdf_matrix, indices, depth=0, node_name="root")

    # ------------------------------
    # IN CÂY
    # ------------------------------
    def print_tree(self, node: str = "root", level: int = 0) -> None:
        """In cấu trúc cây phân cụm."""
        if node not in self.tree:
            return
        indent = "│   " * level
        indices = self.tree[node]["indices"]
        print(f"{indent}├── {node} ({len(indices)} samples): {indices}")

        for child in [node + "0", node + "1"]:
            if child in self.tree:
                self.print_tree(child, level + 1)
