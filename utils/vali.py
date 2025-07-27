import numpy as np
from sklearn.metrics import normalized_mutual_info_score
from utils.dist import Dist


class CVI:
    """
    Đánh giá kết quả phân cụm cho dữ liệu hàm mật độ xác suất [n_pdf, n_point].
    Bao gồm cả chỉ số nội bộ (silhouette, Dunn, Davies-Bouldin)
    và chỉ số bên ngoài (Rand Index, Adjusted Rand Index, NMI).
    """

    def __init__(self, distance_metric='L2', bandwidth=0.01, grid=None):
        self.distance_metric = distance_metric
        self.bandwidth = bandwidth
        self.grid = grid

    # ===============================
    # MA TRẬN KHOẢNG CÁCH
    # ===============================
    def _compute_distance_matrix(self, F_data):
        n_pdf = F_data.shape[0]
        D = np.zeros((n_pdf, n_pdf))
        for i in range(n_pdf):
            for j in range(i + 1, n_pdf):
                d = Dist(F_data[i], F_data[j], h=self.bandwidth, grid=self.grid)
                D[i, j] = getattr(d, self.distance_metric)()
                D[j, i] = D[i, j]
        return D

    # ===============================
    # CHỈ SỐ NỘI BỘ
    # ===============================
    def silhouette_index(self, F_data, labels):
        """
        Silhouette Index: mức độ gắn kết và tách biệt của các cụm.
        """
        D = self._compute_distance_matrix(F_data)
        n = F_data.shape[0]
        sil = np.zeros(n)
        clusters = np.unique(labels)

        for i in range(n):
            same_cluster = labels == labels[i]
            a_i = np.mean(D[i, same_cluster]) if np.sum(same_cluster) > 1 else 0
            b_i = np.min(
                [np.mean(D[i, labels == c]) for c in clusters if c != labels[i]]
            ) if len(clusters) > 1 else 0
            sil[i] = (b_i - a_i) / max(a_i, b_i) if max(a_i, b_i) > 0 else 0
        return np.mean(sil)

    def dunn_index(self, F_data, labels):
        """
        Dunn Index: khoảng cách nhỏ nhất giữa các cụm / đường kính lớn nhất trong cụm.
        """
        D = self._compute_distance_matrix(F_data)
        clusters = np.unique(labels)
        min_inter, max_intra = np.inf, 0

        for i in clusters:
            points_i = np.where(labels == i)[0]
            if len(points_i) > 1:
                intra_dist = np.max(D[np.ix_(points_i, points_i)])
                max_intra = max(max_intra, intra_dist)

            for j in clusters:
                if i < j:
                    points_j = np.where(labels == j)[0]
                    inter_dist = np.min(D[np.ix_(points_i, points_j)])
                    min_inter = min(min_inter, inter_dist)
        return min_inter / max_intra if max_intra > 0 else 0

    def davies_bouldin_index(self, F_data, labels):
        """
        Davies-Bouldin Index: trung bình tỷ lệ phân tán trong cụm / khoảng cách giữa các tâm cụm.
        """
        clusters = np.unique(labels)
        k = len(clusters)
        centroids = []
        scatters = []

        for c in clusters:
            idx = np.where(labels == c)[0]
            centroid = np.mean(F_data[idx], axis=0)
            centroids.append(centroid)
            scatters.append(
                np.mean([Dist(f, centroid, h=self.bandwidth, grid=self.grid).L2() for f in F_data[idx]])
            )

        DB = 0
        for i in range(k):
            max_ratio = max(
                (scatters[i] + scatters[j]) /
                getattr(Dist(centroids[i], centroids[j], h=self.bandwidth, grid=self.grid), self.distance_metric)()
                for j in range(k) if i != j
            )
            DB += max_ratio
        return DB / k

    # ===============================
    # CHỈ SỐ BÊN NGOÀI
    # ===============================
    @staticmethod
    def rand_index(labels_true, labels_pred):
        """
        Rand Index (RI).
        """
        n = len(labels_true)
        a = b = c = d = 0
        for i in range(n):
            for j in range(i + 1, n):
                same_true = labels_true[i] == labels_true[j]
                same_pred = labels_pred[i] == labels_pred[j]
                if same_true and same_pred:
                    a += 1
                elif same_true and not same_pred:
                    b += 1
                elif not same_true and same_pred:
                    c += 1
                else:
                    d += 1
        return (a + d) / (a + b + c + d)

    @staticmethod
    def adjusted_rand_index(labels_true, labels_pred):
        """
        Adjusted Rand Index (ARI).
        """
        labels_true, labels_pred = np.asarray(labels_true), np.asarray(labels_pred)
        classes_true, classes_pred = np.unique(labels_true), np.unique(labels_pred)

        contingency = np.array([
            [(labels_pred[labels_true == c_true] == c_pred).sum() for c_pred in classes_pred]
            for c_true in classes_true
        ])

        sum_comb = lambda x: (x * (x - 1)) // 2
        sum_squares = (contingency * (contingency - 1) // 2).sum()
        sum_c = sum_comb(contingency.sum(axis=1)).sum()
        sum_k = sum_comb(contingency.sum(axis=0)).sum()
        total_pairs = sum_comb(len(labels_true))

        expected_index = (sum_c * sum_k) / total_pairs
        max_index = 0.5 * (sum_c + sum_k)
        return (sum_squares - expected_index) / (max_index - expected_index)

    @staticmethod
    def nmi(labels_true, labels_pred):
        """
        Normalized Mutual Information.
        """
        return normalized_mutual_info_score(labels_true, labels_pred)


def find_elbow_k(k_values, scores):
    """
    Tìm số k tối ưu bằng phương pháp Elbow (điểm xa nhất so với đường nối đầu-cuối).
    """
    k_values = np.array(k_values)
    scores = np.array(scores)
    p1, p2 = np.array([k_values[0], scores[0]]), np.array([k_values[-1], scores[-1]])
    line_vec = p2 - p1
    line_vec_norm = line_vec / np.linalg.norm(line_vec)

    distances = [
        np.linalg.norm(np.array([k_values[i], scores[i]]) -
                       (p1 + np.dot(np.array([k_values[i], scores[i]]) - p1, line_vec_norm) * line_vec_norm))
        for i in range(len(k_values))
    ]
    return k_values[np.argmax(distances)]
