import numpy as np
from utils.dist import Dist

class Model:
    def __init__(self, grid_x, max_depth=3, linkage='average', min_cluster_size=1,
                 distance_metric='L2', bandwidth=0.01):
        self.grid_x = grid_x
        self.max_depth = max_depth
        self.min_cluster_size = min_cluster_size
        self.linkage = linkage
        self.distance_metric = distance_metric
        self.bandwidth = bandwidth
        self.tree = {}

    def _compute_distance_matrix(self, pdf_matrix):
        n_pdfs = pdf_matrix.shape[0]
        dist_matrix = np.zeros((n_pdfs, n_pdfs))
        for i in range(n_pdfs):
            for j in range(i + 1, n_pdfs):
                d_obj = Dist(pdf_matrix[i, :], pdf_matrix[j, :], h=self.bandwidth, Dim=1, grid=self.grid_x)
                dist = getattr(d_obj, self.distance_metric)()
                dist_matrix[i, j] = dist_matrix[j, i] = dist
        return dist_matrix

    def _calculate_linkage_distance(self, cluster_a, cluster_b, dist_matrix, pdf_matrix):
        if self.linkage in ['single', 'complete', 'average']:
            dists = [dist_matrix[i, j] for i in cluster_a for j in cluster_b]
            if self.linkage == 'single':
                return np.min(dists)
            elif self.linkage == 'complete':
                return np.max(dists)
            else:
                return np.mean(dists)
        elif self.linkage in ['centroid', 'ward']:
            centroid_a = np.mean(pdf_matrix[cluster_a, :], axis=0)
            centroid_b = np.mean(pdf_matrix[cluster_b, :], axis=0)
            d_obj = Dist(centroid_a, centroid_b, h=self.bandwidth, Dim=1, grid=self.grid_x)
            centroid_dist = getattr(d_obj, self.distance_metric)()
            if self.linkage == 'ward':
                size_a, size_b = len(cluster_a), len(cluster_b)
                return (size_a * size_b) / (size_a + size_b) * centroid_dist ** 2
            else:
                return centroid_dist
        else:
            raise ValueError(f"Unsupported linkage: {self.linkage}")

    def _split_cluster(self, pdf_matrix):
        dist_matrix = self._compute_distance_matrix(pdf_matrix)
        n = pdf_matrix.shape[0]
        clusters = [[i] for i in range(n)]

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

    def _fit_recursive(self, pdf_matrix, indices, depth, node_name):
        n_samples = pdf_matrix.shape[0]
        if depth >= self.max_depth or n_samples <= self.min_cluster_size:
            self.tree[node_name] = {'indices': indices.tolist(), 'children': {}}
            return

        cluster_0_idx, cluster_1_idx = self._split_cluster(pdf_matrix)
        if len(cluster_0_idx) < self.min_cluster_size or len(cluster_1_idx) < self.min_cluster_size:
            self.tree[node_name] = {'indices': indices.tolist(), 'children': {}}
            return

        self.tree[node_name] = {'indices': indices.tolist(), 'children': {}}

        # Recursive on cluster 0
        self._fit_recursive(pdf_matrix[cluster_0_idx, :], indices[cluster_0_idx], depth + 1, node_name + '0')

        # Recursive on cluster 1
        self._fit_recursive(pdf_matrix[cluster_1_idx, :], indices[cluster_1_idx], depth + 1, node_name + '1')

    def fit(self, pdf_matrix):
        if pdf_matrix.shape[1] != len(self.grid_x):
            raise ValueError(f"pdf_matrix.shape[1] ({pdf_matrix.shape[1]}) must match grid_x length ({len(self.grid_x)})")
        n_pdfs = pdf_matrix.shape[0]
        indices = np.arange(n_pdfs)
        self.dist_matrix = self._compute_distance_matrix(pdf_matrix)
        self._fit_recursive(pdf_matrix, indices, depth=0, node_name='root')

    def print_tree(self, node='root', level=0):
        if node not in self.tree:
            return
        indent = '│   ' * level
        indices = self.tree[node]['indices']
        print(f"{indent}├── {node} ({len(indices)} samples): {indices}")

        child0, child1 = node + '0', node + '1'
        if child0 in self.tree:
            self.print_tree(child0, level + 1)
        if child1 in self.tree:
            self.print_tree(child1, level + 1)

# ===========================================================
