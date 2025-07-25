import numpy as np
import matplotlib.pyplot as plt
from Tree import TreePDF
from data.data_loader import generateGauss
from utils.integral import grid

# ==== 1. Sinh dữ liệu PDF (Gaussian) ====
bandwidth = 0.01
grid_x    = grid(bandwidth, start=-5, end=15)
mu        = np.array([0.3, 4.0, 9.1, 1.0, 5.5, 8.0, 4.8])
sig       = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
sig = np.ones_like(mu)
pdfs      = generateGauss(mu, sig, grid_x)

# ==== 2. Huấn luyện TreePDF ====
tree = TreePDF(k=3, grid_x=grid_x, max_leaves=7, distance_metric='L2',
               bandwidth=bandwidth, verbose=1)
labels = tree.fit_predict(pdfs)

# ==== 3. Hàm lấy danh sách split threshold ====
def get_splits(node):
    splits = []
    if not node.is_leaf():
        splits.append(node.value)  # threshold tại node
        splits.extend(get_splits(node.left))
        splits.extend(get_splits(node.right))
    return splits

split_positions = get_splits(tree.tree)
print("Vị trí split:", split_positions)

# ==== 4. Vẽ cây ====
try:
    tree.plot(filename="pdf_tree.pdf", view=True)
except Exception as e:
    print("Không vẽ được cây, lỗi:", e)

# ==== 5. Vẽ các PDF và vị trí split ====
plt.figure(figsize=(8, 5))
for i, pdf in enumerate(pdfs):
    plt.plot(grid_x, pdf, label=f"PDF {i} (C{labels[i]})")

# Vẽ các đường split
for s in split_positions:
    plt.axvline(x=s, color='k', linestyle='--', linewidth=1.2, alpha=0.7)

plt.title("10 Gaussian PDFs với nhãn cụm và vị trí split từ TreePDF")
plt.xlabel("x")
plt.ylabel("Density")
plt.legend()
plt.tight_layout()
plt.show()
