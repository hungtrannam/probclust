# 
# =======================================
# Author: Hung Tran-Nam
# Email: namhung34.info@gmail.com
# Repo: https://github.com/hungtrannam/probclust
# =======================================
# File: vis_dist.py
# Description: Visualize distances between probability distributions
# =======================================


import numpy as np
from scipy.stats import norm
from utils import dist
import matplotlib.pyplot as plt
from utils.vis import temp 
import os

# Tạo grid
h = 0.01
x = np.arange(-50, 50, h)
f1 = norm.pdf(x, loc=0, scale=1)

# Tạo danh sách các giá trị mu
mu_list = np.linspace(-10, 20, 1000)

# Khởi tạo mảng lưu khoảng cách
L1_list, L2_list, H_list = [], [], [], []

for mu in mu_list:
    f2 = norm.pdf(x, loc=mu, scale=1)
    D = dist.Dist(f1, f2, h=h, Dim=1, grid=x)
    L1_list.append(D.L1())
    L2_list.append(D.L2())
    H_list.append(D.H())
    

# Chuyển sang numpy array
L1_list = np.array(L1_list)
L2_list = np.array(L2_list)
H_list = np.array(H_list)

# Tạo thư mục figs
os.makedirs('figs', exist_ok=True)

plt.figure(figsize=(8, 6))
temp(fontsize=24)

plt.plot(mu_list, L1_list, label=r'$\mathcal{L}_1$', linestyle='--', lw = 4, color = 'gray')           # solid
plt.plot(mu_list, L2_list, label=r'$\mathcal{L}_2$', linestyle='-.', lw=4, color = 'gray')          # dashed
plt.plot(mu_list, H_list, label='Hellinger', linestyle='-', lw=4, color = 'gray')                   # dotted

plt.xlabel(r'$-10\leq\mu_2\leq20$')
plt.ylabel('Khoảng cách')
plt.legend()
plt.tight_layout()
plt.savefig('figs/dist.pdf')
