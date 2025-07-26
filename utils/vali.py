import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from utils.dist import Dist

class CVI:
    """
    Đánh giá kết quả phân tích chùm cho dữ liệu hàm mật độ xác suất [n_pdf, n_point].
    """

    def __init__(self, distance_metric='L2', bandwidth=0.01, grid=None):
        """
        Parameters
        ----------
        distance_metric : str
            Loại khoảng cách ('L1', 'L2', 'H', 'BC', 'W2').
        bandwidth : float
            Bước tích phân cho tính toán khoảng cách.
        grid : ndarray
            Lưới x dùng cho tính toán W2.
        """
        self.distance_metric = distance_metric
        self.bandwidth = bandwidth
        self.grid = grid

    # ==========
    # MA TRẬN KHOẢNG CÁCH
    # ==========
    def _compute_distance_matrix(self, F_data):
        n_pdf = F_data.shape[0]
        D = np.zeros((n_pdf, n_pdf))
        for i in range(n_pdf):
            for j in range(i + 1, n_pdf):
                d = Dist(F_data[i], F_data[j], h=self.bandwidth, grid=self.grid)
                D[i, j] = getattr(d, self.distance_metric)()
                D[j, i] = D[i, j]
        return D

    # =================
    # ĐÁNH GIÁ NỘI BỘ
    # =================
    def silhouette_index(self, F_data, labels):
        """
        Silhouette Index = mức độ gắn kết và tách biệt của các chùm.
        """
        D = self._compute_distance_matrix(F_data)
        n = F_data.shape[0]
        sil = np.zeros(n)
        clusters = np.unique(labels)

        for i in range(n):
            same_cluster = labels == labels[i]
            a_i = np.mean(D[i, same_cluster]) if np.sum(same_cluster) > 1 else 0
            b_i = np.inf
            for c in clusters:
                if c != labels[i]:
                    b_i = min(b_i, np.mean(D[i, labels == c]))
            sil[i] = (b_i - a_i) / max(a_i, b_i) if max(a_i, b_i) > 0 else 0
        return np.mean(sil)

    def dunn_index(self, F_data, labels):
        """
        Dunn Index = khoảng cách nhỏ nhất giữa các chùm / đường kính lớn nhất trong chùm.
        """
        D = self._compute_distance_matrix(F_data)
        clusters = np.unique(labels)
        min_inter = np.inf
        max_intra = 0

        for i in clusters:
            points_i = np.where(labels == i)[0]
            # Đường kính cụm i
            for p in points_i:
                for q in points_i:
                    if p < q:
                        max_intra = max(max_intra, D[p, q])
            # Khoảng cách với cụm khác
            for j in clusters:
                if i < j:
                    points_j = np.where(labels == j)[0]
                    d_ij = np.min(D[np.ix_(points_i, points_j)])
                    min_inter = min(min_inter, d_ij)
        return min_inter / max_intra if max_intra > 0 else 0

    def davies_bouldin_index(self, F_data, labels):
        """
        Davies-Bouldin Index = trung bình tỉ lệ phân tán trong cụm / khoảng cách giữa tâm cụm.
        """
        clusters = np.unique(labels)
        k = len(clusters)
        centroids = []
        scatters = []

        # Tính tâm cụm (trung bình các pdf) và scatter
        for c in clusters:
            idx = np.where(labels == c)[0]
            centroid = np.mean(F_data[idx], axis=0)
            centroids.append(centroid)
            scatters.append(
                np.mean([Dist(f, centroid, h=self.bandwidth, grid=self.grid).L2() for f in F_data[idx]])
            )

        DB = 0
        for i in range(k):
            max_ratio = 0
            for j in range(k):
                if i != j:
                    dist = Dist(centroids[i], centroids[j], h=self.bandwidth, grid=self.grid)
                    dist_val = getattr(dist, self.distance_metric)()
                    ratio = (scatters[i] + scatters[j]) / dist_val
                    max_ratio = max(max_ratio, ratio)
            DB += max_ratio
        return DB / k

    # ==================
    # ĐÁNH GIÁ BÊN NGOÀI
    # ==================
    def rand_index(self, labels_true, labels_pred):
        """
        Rand Index.
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

    def adjusted_rand_index(labels_true, labels_pred):

        def pair_confusion_matrix(labels_true, labels_pred):
            labels_true = np.asarray(labels_true)
            labels_pred = np.asarray(labels_pred)
            n_samples = labels_true.shape[0]

            # Xác định tất cả nhãn
            classes_true = np.unique(labels_true)
            classes_pred = np.unique(labels_pred)
            n_true = classes_true.shape[0]
            n_pred = classes_pred.shape[0]

            # Ma trận contingency
            contingency = np.zeros((n_true, n_pred), dtype=np.int64)
            for i, c_true in enumerate(classes_true):
                mask_true = labels_true == c_true
                for j, c_pred in enumerate(classes_pred):
                    contingency[i, j] = np.sum(labels_pred[mask_true] == c_pred)

            # Tính toán
            n_c = contingency.sum(axis=1)
            n_k = contingency.sum(axis=0)
            sum_comb = lambda x: (x * (x - 1)) // 2

            sum_squares = (contingency * (contingency - 1) // 2).sum()
            sum_c = sum_comb(n_c).sum()
            sum_k = sum_comb(n_k).sum()
            total_pairs = sum_comb(n_samples)

            C = np.zeros((2, 2), dtype=np.int64)
            C[1, 1] = sum_squares
            C[0, 1] = sum_k - sum_squares
            C[1, 0] = sum_c - sum_squares
            C[0, 0] = total_pairs - C[1, 1] - C[0, 1] - C[1, 0]
            return C, sum_c, sum_k, total_pairs, sum_squares
        C, sum_c, sum_k, total_pairs, sum_squares = pair_confusion_matrix(labels_true, labels_pred)

        # ARI
        index = sum_squares
        expected_index = (sum_c * sum_k) / total_pairs
        max_index = 0.5 * (sum_c + sum_k)
        ARI = (index - expected_index) / (max_index - expected_index)
        return ARI


    def nmi(self, labels_true, labels_pred):
        return normalized_mutual_info_score(labels_true, labels_pred)


def find_elbow_k(k_values, scores):
    """
    Tìm k tối ưu bằng phương pháp Elbow (điểm xa nhất so với đường nối đầu-cuối).
    """
    k_values = np.array(k_values)
    scores = np.array(scores)

    # Tạo vector (x,y)
    p1 = np.array([k_values[0], scores[0]])
    p2 = np.array([k_values[-1], scores[-1]])

    # Vector đoạn thẳng
    line_vec = p2 - p1
    line_vec_norm = line_vec / np.linalg.norm(line_vec)

    # Khoảng cách vuông góc từ mỗi điểm đến đoạn thẳng
    distances = []
    for i in range(len(k_values)):
        p = np.array([k_values[i], scores[i]])
        proj = p1 + np.dot((p - p1), line_vec_norm) * line_vec_norm
        dist = np.linalg.norm(p - proj)
        distances.append(dist)

    return k_values[np.argmax(distances)]