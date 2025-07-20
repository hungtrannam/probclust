import numpy as np

from Models.classification.Logistics import compute_moments
from Models.classification import Logistics

from utils.integral import grid
from data.data_loader import generateGauss
from scipy.stats import norm
from utils.vis import plot_beta_function
from scipy.stats import norm as normal_dist


# Tạo grid
h = 0.01
x_grid, _ = grid(h, start=-15, end=25)

mu_A = [4, 4.2, 4.5, 4.4]
mu_B = [1, 2, 3, 7]

sig_A = [1, 1, 1, 1]
sig_B = [2, 2, 2, 2]


# Sinh Gaussian
class1_data = generateGauss(mu_A, sig_A, x_grid, savefile='dataset/data1.npz')
class2_data = generateGauss(mu_B, sig_B, x_grid, savefile='dataset/data2.npz')

# Chọn hàm cơ sở Gauss
M = 3
basis_centers = np.linspace(0, 10, M)
basis_width = 1.0
basis_functions = [norm(c, basis_width).pdf(x_grid) for c in basis_centers]

# Tính moments
X_A = compute_moments(class1_data, basis_functions, x_grid)
X_B = compute_moments(class2_data, basis_functions, x_grid)
X = np.vstack([X_A, X_B])
y = np.array([0]*X_A.shape[0] + [1]*X_B.shape[0])

# Thêm cột bias
X_design = np.hstack([np.ones((X.shape[0],1)), X])

# ======= HUẤN LUYỆN CHÍNH =======

model = Logistics.Model(lr=0.1, n_iter=1000, l1_penalty=0.001, verbose=False)
model.fit(X_design, y)

# ======= TÍNH CHÍNH XÁC =======
y_pred = model.predict(X_design)
accuracy = np.mean(y_pred == y)
print(f"Accuracy: {accuracy * 100:.2f}%")

# ======= BOOTSTRAP KIỂM ĐỊNH =======

# B = 1000  # số lần bootstrap
# n_samples = len(y)
# beta_hat = model.beta  # vector [intercept, coef_1, coef_2, ...]

# beta_bootstrap = np.zeros((B, len(beta_hat)))

# for b in range(B):
#     idx = np.random.choice(n_samples, size=n_samples, replace=True)
#     X_b = X_design[idx]
#     y_b = y[idx]
#     model_b = Logistics.Model(lr=0.1, n_iter=1000, l1_penalty=0.001, verbose=False)
#     model_b.fit(X_b, y_b)
#     beta_bootstrap[b] = model_b.beta

# # Tính độ lệch chuẩn (SE) cho từng hệ số
# se_beta = np.std(beta_bootstrap, axis=0)

# # Tính z-score và p-value
# z_scores = beta_hat / (se_beta + 1e-10)  # thêm nhỏ tránh chia 0
# p_values = 2 * (1 - normal_dist.cdf(np.abs(z_scores)))

# # In kết quả
# print("\nBootstrap significance test:")
# print("Coef        Estimate       SE         z        p-value")
# for i, (b, se, z, p) in enumerate(zip(beta_hat, se_beta, z_scores, p_values)):
#     name = f'beta_{i}' if i > 0 else 'intercept'
#     print(f"{name:10} {b:12.4f} {se:10.4f} {z:10.4f} {p:10.4f}")

# ======= VẼ BETA FUNCTION =======

beta_func_hat = sum(model.coef_[j] * basis_functions[j] for j in range(M))
plot_beta_function(x_grid, beta_func_hat, savefile="figs/beta.pdf")



import matplotlib.pyplot as plt
import matplotlib

# ======= TÍNH TOÀN BỘ PDF VÀ XÁC SUẤT =======
all_pdfs = np.vstack([class1_data, class2_data])  # [n_samples, n_x]
all_probs = model.predict_proba(X_design)         # [n_samples]
n_A = class1_data.shape[0]
n_B = class2_data.shape[0]

# ======= TẠO PLOT =======
fig, ax = plt.subplots(figsize=(5, 4))
cmap = plt.cm.coolwarm
norm = matplotlib.colors.Normalize(vmin=0, vmax=1)

# Vẽ lớp A
for i in range(n_A):
    color = cmap(norm(all_probs[i]))
    ax.plot(x_grid, all_pdfs[i], color=color, alpha=1, linewidth=2)

# Vẽ lớp B
for i in range(n_A, n_A + n_B):
    color = cmap(norm(all_probs[i]))
    ax.plot(x_grid, all_pdfs[i], color=color, alpha=1, linewidth=2, linestyle='--')

# Thêm colorbar
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax)
cbar.set_label('Predicted Probability', labelpad=12)

# Setup plot detail
plt.tight_layout()
plt.savefig('figs/pdfs_with_probs.pdf', bbox_inches='tight')
