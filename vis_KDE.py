import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# === Định nghĩa các kernel thủ công ===
def gaussian(u):
    return (1/np.sqrt(2*np.pi)) * np.exp(-0.5 * u**2)

def tophat(u):
    return 0.5 * (np.abs(u) <= 1)

def linear(u):
    return (1 - np.abs(u)) * (np.abs(u) <= 1)

def epanechnikov(u):
    return 0.75 * (1 - u**2) * (np.abs(u) <= 1)

def exponential(u):
    return 0.5 * np.exp(-np.abs(u))

def cosine(u):
    return (np.pi/4) * np.cos((np.pi/2)*u) * (np.abs(u) <= 1)

def logistic(u):
    return np.exp(-u) / (1 + np.exp(-u))**2 / 2

def sigmoid(u):
    return 1 / (np.exp(u) + np.exp(-u))

kernel_funcs = {
    "gaussian": gaussian,
    # "tophat": tophat,
    # "linear": linear,
    "logistic": logistic,
    "epanechnikov": epanechnikov,

}

# === KDE thủ công ===
def kde_manual(x, data, h, kernel):
    n = len(data)
    return np.array([
        np.sum(kernel((x_i - data) / h)) / (n * h)
        for x_i in x
    ])

# === Dữ liệu và grid ===
np.random.seed(0)
N = 100
data = np.hstack([
    np.random.normal(loc=0, scale=1, size=N//3),
    np.random.normal(loc=5, scale=1, size=2*N//3)
])
x_grid = np.linspace(-6, 15, 500)
h = 1.06 * np.std(data) * N**(-1/5)

# === Phân phối thực tế ===
true_pdf = 1/3 * norm.pdf(x_grid, loc=0, scale=1) + 2/3 * norm.pdf(x_grid, loc=5, scale=1)

# === Vẽ ===
from utils.vis import temp
plt.figure(figsize=(6, 4))
temp(20)
plt.fill_between(x_grid, true_pdf, color='gray', alpha=0.3)

kernel_names = list(kernel_funcs.keys())

for name in kernel_names:
    y_kde = kde_manual(x_grid, data, h, kernel_funcs[name])
    
    if name in ['gaussian', ]:
        ls = '-'
        lw = 3
    elif name in ['epanechnikov']:
        ls = '--'
        lw = 3
    elif name in ['logistic']:
        ls = ':'
        lw = 1.5

    plt.plot(x_grid, y_kde, color='black', alpha=.75, linestyle=ls, linewidth=lw, label=name)


# Định dạng biểu đồ
plt.legend(loc='upper right',fontsize=13)
plt.ylim(0, 0.3)
plt.tight_layout()

plt.savefig("figs/kde_all_kernels.pdf", dpi=300)
