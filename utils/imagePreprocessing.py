import numpy as np
from utils.kernelDensityEst import KernelDensityEstimator  # lớp bạn đã cung cấp

def image_to_pdf(image: np.ndarray, bandwidth: float = 2.0, kernel: str = "gaussian") -> KernelDensityEstimator:
    """
    Chuyển ảnh xám thành hàm mật độ xác suất (PDF) bằng KDE.

    Parameters
    ----------
    image : np.ndarray
        Ảnh xám 2D với giá trị pixel trong khoảng [0, 255].
    bandwidth : float
        Tham số trơn (smoothing) cho KDE.
    kernel : str
        Loại kernel sử dụng (mặc định: "gaussian").

    Returns
    -------
    KernelDensityEstimator
        Đối tượng KDE đã được huấn luyện trên dữ liệu pixel.
    """
    if image.ndim != 2:
        raise ValueError("Ảnh phải là ảnh xám (2D).")

    # Flatten ảnh thành mảng 1D các giá trị pixel
    pixel_values = image.flatten()

    # Tạo và huấn luyện KDE
    kde = KernelDensityEstimator(bandwidth=bandwidth, kernel=kernel)
    kde.fit(pixel_values)

    return kde

import numpy as np
from typing import Tuple

def image_to_pdf_rgb(image: np.ndarray, bandwidth: float = 2.0, kernel: str = "gaussian") -> Tuple[KernelDensityEstimator, KernelDensityEstimator, KernelDensityEstimator]:
    """
    Chuyển ảnh màu RGB thành 3 hàm mật độ xác suất (PDF) riêng biệt cho từng kênh màu bằng KDE.

    Parameters
    ----------
    image : np.ndarray
        Ảnh màu RGB có dạng (H, W, 3) với giá trị pixel trong khoảng [0, 255].
    bandwidth : float
        Tham số trơn cho KDE (áp dụng cho cả 3 kênh).
    kernel : str
        Loại kernel sử dụng.

    Returns
    -------
    Tuple[KernelDensityEstimator, KernelDensityEstimator, KernelDensityEstimator]
        (kde_r, kde_g, kde_b) – KDE cho các kênh Red, Green, Blue.
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Ảnh phải là ảnh màu RGB với dạng (H, W, 3).")

    kde_list = []
    for ch in range(3):
        pixel_values = image[:, :, ch].flatten()
        kde = KernelDensityEstimator(bandwidth=bandwidth, kernel=kernel)
        kde.fit(pixel_values)
        kde_list.append(kde)

    return tuple(kde_list)


