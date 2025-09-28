import numpy as np
from utils.dist import Dist


def init_centroids_kmeanspp(pdf_matrix, num_clusters: int, bandwidth: float,
                            distance_metric: str, Dim: int, grid_x: np.ndarray) -> np.ndarray:
    """
    Khởi tạo centroid bằng KMeans++ cho các phân phối xác suất.
    - pdf_matrix: (N, G) cho 1D hoặc (N, h, w) cho 2D.
    - num_clusters: số cụm (K).
    - bandwidth, distance_metric, Dim, grid_x: tham số cho Dist.
    """
    # ------------------------------
    # Flatten dữ liệu về (N, G) để tính toán
    # ------------------------------
    if pdf_matrix.ndim == 3:         # (N, h, w)
        N, h, w = pdf_matrix.shape
        G = h * w
        flat_data = pdf_matrix.reshape(N, G)
        pdf_shape = (h, w)
    elif pdf_matrix.ndim == 2:       # (N, G)
        N, G = pdf_matrix.shape
        flat_data = pdf_matrix
        pdf_shape = (G,)
    else:
        raise ValueError("pdf_matrix phải có dạng (N,G) hoặc (N,h,w)")

    K = num_clusters

    # đối tượng khoảng cách
    dobj = Dist(h=bandwidth, Dim=Dim, grid=grid_x)
    func = getattr(dobj, distance_metric)

    # ------------------------------
    # Nếu chỉ cần 2 centroid -> chọn 2 điểm xa nhau nhất
    # ------------------------------
    if K == 2:
        idx0 = np.random.randint(N)
        d2 = np.array([
            func(flat_data[i], flat_data[idx0])
            for i in range(N)
        ])
        idx1 = int(np.argmax(d2))
        indices = [idx0, idx1]
        chosen = flat_data[indices, :].copy()
    else:
        # ------------------------------
        # Trường hợp K > 2: dùng KMeans++
        # ------------------------------
        indices = [np.random.randint(N)]  # chọn ngẫu nhiên centroid đầu tiên
        for _ in range(1, K):
            d2 = np.array([
                min((func(flat_data[i], flat_data[j])) for j in indices)
                for i in range(N)
            ])
            probs = d2 / (d2.sum() + 1e-12)
            next_idx = np.random.choice(N, p=probs)
            indices.append(next_idx)
        chosen = flat_data[indices, :].copy()

    # ------------------------------
    # Reshape về 2D nếu cần
    # ------------------------------
    if len(pdf_shape) == 2:  # (h, w)
        return chosen.reshape(K, *pdf_shape)
    else:
        return chosen  # (K, G)
