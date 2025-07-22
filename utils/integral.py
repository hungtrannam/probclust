# =======================================
# Author: Hung Tran-Nam
# Email: namhung34.info@gmail.com
# Repo: https://github.com/hungtrannam/probclust
# =======================================
# File: utils/int.py
# Description: Hàm tích phân xấp xỉ bằng phương pháp hình thang
# =======================================

import numpy as np
from scipy.integrate import trapezoid

def grid(h, start=-20, end=20):
    """
    Tạo lưới giá trị từ start đến end với bước h.
    """
    return np.arange(start, end + h, h), h

def int_trapz(fv, h, Dim):
    """
    Tích phân xấp xỉ dùng hình thang (trapezoidal rule).
    """
    if Dim == 1:
        sol = trapezoid(fv, dx=h)
    elif Dim == 2:
        sol = trapezoid(trapezoid(fv, dx=h, axis=1), dx=h, axis=0)
    elif Dim == 3:
        sol = trapezoid(trapezoid(trapezoid(fv, dx=h, axis=2), dx=h, axis=1), dx=h, axis=0)
    else:
        raise ValueError("Dimension not supported. Only 1D, 2D, and 3D integrals are implemented.")
    return sol

# =======================================
from scipy.integrate import simpson

def int_simps(fv, h, Dim):
    """
    Tích phân chính xác đa chiều dùng SciPy (Simpson's rule).
    """
    if Dim == 1:
        sol = simpson(fv, dx=h)
    elif Dim == 2:
        sol = simpson(simpson(fv, dx = h, axis=1), dx=h, axis=0)
    elif Dim == 3:
        sol = simpson(simpson(simpson(fv, dx = h, axis=2), dx=h, axis=1), dx=h, axis=0)
    else:
        raise ValueError("Dimension not supported. Only 1D, 2D, and 3D integrals are implemented.")
    return sol
