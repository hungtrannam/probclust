# =======================================
# Author: Hung Tran-Nam
# Email: namhung34.info@gmail.com
# Repo: https://github.com/hungtrannam/probclust
# =======================================
# File: utils/vis.py
# Description: Cài đặt các plot Matplotlib
# =======================================

import matplotlib.pyplot as plt
import seaborn as sns
import os

# =======================================
# Cài đặt các plot Matplotlib
# =======================================

def temp(fontsize=20, w=10, h=8, u='centimeters'):
    if u == 'centimeters':
        w = w / 2.54
        h = h / 2.54

    plt.rcParams.update({
        'font.size': fontsize,
        'font.family': 'serif',
        'font.serif': ['Times New Roman'],
        'mathtext.fontset': 'stix',
        'mathtext.rm': 'Times New Roman',
        'axes.titlesize': fontsize,
        'axes.labelsize': fontsize,
        'axes.labelweight': 'bold',
        'xtick.labelsize': fontsize - 1,
        'ytick.labelsize': fontsize - 1,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.minor.visible': True,
        'ytick.minor.visible': False,
        'axes.linewidth': 1,
        # 'grid.color': (0.8, 0.8, 0.8),
        # 'grid.alpha': 0.3,
        'axes.grid': False,
        'figure.dpi': 300,
        'figure.figsize': (w, h),
        'figure.facecolor': 'white',
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.format': 'pdf',
        'savefig.transparent': False,
        'savefig.pad_inches': 0.1,
        'lines.linewidth': 2,
        'lines.markersize': 6,
        'legend.fontsize': fontsize - 2,
        'legend.frameon': True,
        'legend.loc': 'upper right',
        'axes.prop_cycle': plt.cycler(color=sns.color_palette("colorblind"))
    })

# =======================================


def plotPDF(grid, pdfs, savefile=None):
    """
    Vẽ nhiều hàm mật độ (pdfs) trên cùng một grid.
    """
    plt.figure()
    temp(fontsize=18, w=12, h=8, u='centimeters')

    n_pdfs = pdfs.shape[1]
    
    for i in range(n_pdfs):
        plt.plot(grid, pdfs[:, i], lw = 2, color='black')
    plt.tight_layout()

    if savefile:
        os.makedirs('figs', exist_ok=True)
        plt.savefig(savefile, bbox_inches='tight')

# =============================

def plotPDF_Theta(grid, pdfs, theta, savefile=None):
    """
    Vẽ nhiều hàm mật độ (pdfs) và prototype (theta) trên cùng một grid.
    """
    plt.figure()
    temp(fontsize=18, w=12, h=8, u='centimeters')

    n_pdfs = pdfs.shape[1]
    n_clusters = theta.shape[1]

    # Vẽ các pdfs
    for i in range(n_pdfs):
        plt.plot(grid, pdfs[:, i], lw=2, color='gray', linestyle='--', alpha=0.6)

    # Vẽ các prototype
    for j in range(n_clusters):
        plt.plot(grid, theta[:, j], lw=2, linestyle='-', color='black')

    plt.tight_layout()

    if savefile:
        os.makedirs('figs', exist_ok=True)
        plt.savefig(savefile, bbox_inches='tight')

# ==========================================

def plotHeatmap_U(U, savefile=None):
    """
    Vẽ heatmap cho ma trận phân vùng fuzzy clustering.
    """
    plt.figure(figsize=(8, 6))
    temp(fontsize=20)
    
    sns.heatmap(
        U,
        annot=True,
        fmt=".2f",
        cmap='Greys',   # Thang xám
        cbar=True,
        linewidths=0.5,
        linecolor='white'
    )
    
    plt.ylabel('Cluster')
    plt.xlabel('PDF Index')
    plt.tight_layout()
    
    if savefile:
        os.makedirs('figs', exist_ok=True)
        plt.savefig(savefile, bbox_inches='tight')
