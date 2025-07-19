import numpy as np
import os
from data.data_loader import generateGauss, generateUniform
from utils.dist import Dist
from utils.integral import int, grid
from utils.vis import *
from Models.clustering import FCF, EM, KCF, KFCF

# Tạo grid
h = 0.01
x, _ = grid(h, start=-10, end=20)

mu = np.array([0.3, 4.0, 9.1, 1.0, 5.5, 8.0, 4.8])
sig = np.ones_like(mu)
print("sigma:", sig)

# Sinh Gaussian
F_data = generateGauss(mu, sig, x, savefile='dataset/data.npz')

# Vẽ các PDF ban đầu
plotPDF(x, F_data, savefile='figs/pdfs.pdf')

# Tham số clustering
c_clust = 3
max_iter = 100
eps = 1e-6


# Khởi tạo và huấn luyện
clusterer = KFCF.Model(F_data, x, 
                     kernel_type='L2',
                    # distance_metric='L2',
                    num_clusters=c_clust,
                    max_iterations=max_iter,
                    tolerance=eps,
                    bandwidth=h
)
clusterer.fit()
U, Theta = clusterer.get_results()

print("Cluster labels:", U)

# Vẽ heatmap nhãn (hard labels)
plotHeatmap_U(U, savefile="figs/U.pdf")

# Vẽ trung tâm cụm
plotPDF_Theta(x, F_data, theta=Theta, savefile='figs/V.pdf')

# Dự đoán nhãn cho dữ liệu mới
f_new = generateGauss([4], sig, x)

U_new = clusterer.predict(f_new)
print("Predicted labels for new data:", U_new)


