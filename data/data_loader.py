# =======================================
# Author: Hung Tran-Nam
# Email: namhung34.info@gmail.com
# Repo: https://github.com/hungtrannam/probclust
# =======================================
# File: data/data_loader.py
# Description: Tạo các hàm mật độ xác suất (PDF) và lưu vào file npz
# =======================================

import numpy as np
from scipy.stats import norm
from scipy.stats import uniform
import os

def generateGauss(mu_list, sigma_list, grid, savefile=None):
    """
    Tạo nhiều hàm mật độ Gauss (PDF) và lưu ra file npz.
    """
    # Nếu là số đơn, bọc thành list
    if isinstance(mu_list, (int, float)):
        mu_list = [mu_list]
    if isinstance(sigma_list, (int, float)):
        sigma_list = [sigma_list]
    # Nếu là chuỗi, chuyển đổi thành list float
    if isinstance(mu_list, str):
        mu_list = [float(mu) for mu in mu_list.split(',')]
    if isinstance(sigma_list, str):
        sigma_list = [float(sigma) for sigma in sigma_list.split(',')] 

    pdfs = []
    for mu, sigma in zip(mu_list, sigma_list):
        pdf = norm.pdf(grid, loc=mu, scale=sigma)
        pdfs.append(pdf)

    pdfs = np.array(pdfs)  # shape: (n_pdfs, len(grid))

    if savefile:
        os.makedirs('dataset', exist_ok=True)
        np.savez(savefile, pdfs=pdfs, mu_list=mu_list, sigma_list=sigma_list, grid=grid)

    return pdfs


# =======================================

def generateUniform(a_list, b_list, grid, savefile=None):
    """
    Tạo nhiều hàm mật độ đều (Uniform PDF) và lưu ra file npz.
    """
    # Nếu là số đơn, bọc thành list
    if isinstance(a_list, (int, float)):
        a_list = [a_list]
    if isinstance(b_list, (int, float)):
        b_list = [b_list]
    # Nếu là chuỗi, chuyển đổi thành list float
    if isinstance(a_list, str):
        a_list = [float(a) for a in a_list.split(',')]
    if isinstance(b_list, str):
        b_list = [float(b) for b in b_list.split(',')]

    pdfs = []
    for a, b in zip(a_list, b_list):
        width = b - a
        pdf = uniform.pdf(grid, loc=a, scale=width)
        pdfs.append(pdf)

    pdfs = np.array(pdfs).T  # shape: (len(grid), n_pdfs)

    if savefile:
        os.makedirs('dataset', exist_ok=True)
        np.savez(savefile, pdfs=pdfs, a_list=a_list, b_list=b_list, grid=grid)

    return pdfs
