# =======================================
# Author: Hung Tran-Nam
# Email: namhung34.info@gmail.com
# Repo: https://github.com/hungtrannam/probclust
# =======================================
# File: utils/vis.py
# Description: Cài đặt các plot Matplotlib
# =======================================

import matplotlib.pyplot as plt
import numpy as np
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

    n_pdfs = pdfs.shape[0]
    
    for i in range(n_pdfs):
        plt.plot(grid, pdfs[i,:], lw = 2, color='black')
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

    n_pdfs = pdfs.shape[0]
    n_clusters = theta.shape[0]

    # Vẽ các pdfs
    for i in range(n_pdfs):
        plt.plot(grid, pdfs[i, :], lw=2, color='gray', linestyle='--', alpha=0.6)

    # Vẽ các prototype
    for j in range(n_clusters):
        plt.plot(grid, theta[j, :], lw=2, linestyle='-', color='black')

    plt.tight_layout()

    if savefile:
        os.makedirs('figs', exist_ok=True)
        plt.savefig(savefile, bbox_inches='tight', dpi=300)
    plt.close()

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
        plt.savefig(savefile, bbox_inches='tight', dpi=300)
    plt.close()

# ==========================================

def plot_beta_function(x_grid, beta_func_hat, savefile = None):
    """
    Vẽ hàm beta ước lượng.
    """
    temp(fontsize=20)
    plt.figure(figsize=(7,6))
    plt.plot(x_grid, beta_func_hat, linewidth=3, color='black')
    plt.xlabel('x')
    plt.ylabel(r'$\widehat{\beta}(x)$')
    plt.axhline(0, color='black', linestyle='--', linewidth=2)

    plt.tight_layout()
    
    if savefile:
        os.makedirs('figs', exist_ok=True)
        plt.savefig(savefile, bbox_inches='tight')
    plt.close()


# =======================================


def plot_tree(tree, dist_matrix, verbose=False, savefile=None):

    """ Vẽ cây phân cụm với heatmap tam giác trên.
    """
    def get_leaf_order(node='root'):
        """
        Lấy thứ tự lá của cây phân cụm.
        """
        if node + '0' not in tree and node + '1' not in tree:
            return tree[node]['indices']
        order = []
        if node + '0' in tree:
            order.extend(get_leaf_order(node + '0'))
        if node + '1' in tree:
            order.extend(get_leaf_order(node + '1'))
        return order

    def compute_positions(node, leaf_x, positions, heights):
        """
        Tính toán vị trí (x, y) cho mỗi node trong cây.
        """
        if node + '0' not in tree and node + '1' not in tree:
            leaf = tree[node]['indices'][0]
            x = leaf_x[leaf]
            positions[node] = x
            heights[node] = 0
            return x, 0
        child_xs, child_ys = [], []
        if node + '0' in tree:
            cx0, cy0 = compute_positions(node + '0', leaf_x, positions, heights)
            child_xs.append(cx0)
            child_ys.append(cy0)
        if node + '1' in tree:
            cx1, cy1 = compute_positions(node + '1', leaf_x, positions, heights)
            child_xs.append(cx1)
            child_ys.append(cy1)
        avg_x = sum(child_xs) / len(child_xs)
        height = max(child_ys) + 1
        positions[node] = avg_x
        heights[node] = height
        return avg_x, height

    def draw_dendrogram(ax, node, positions, heights):
        """ Vẽ dendrogram cho node hiện tại.
        """
        if node + '0' not in tree and node + '1' not in tree:
            return
        x0, y0 = positions[node], heights[node]
        child_xs = []
        for child in [node + '0', node + '1']:
            if child in tree:
                x1, y1 = positions[child], heights[child]
                child_xs.append(x1)
                ax.plot([x1, x1], [y1, y0], color='black', linewidth=1.5)
                draw_dendrogram(ax, child, positions, heights)
        if child_xs:
            ax.plot([min(child_xs), max(child_xs)], [y0, y0], color='black', linewidth=1.5)

    # Lấy thứ tự lá
    leaf_order = get_leaf_order()
    unique_leaf_order = []
    for leaf in leaf_order:
        if leaf not in unique_leaf_order:
            unique_leaf_order.append(leaf)
    leaf_x = {leaf: i for i, leaf in enumerate(unique_leaf_order)}

    # Tính (x, y)
    positions, heights = {}, {}
    compute_positions('root', leaf_x, positions, heights)

    # reorder matrix
    reorder_idx = [leaf for leaf in unique_leaf_order]
    dist_reordered = dist_matrix[np.ix_(reorder_idx, reorder_idx)]

    # Vẽ figure chia 2 phần: trên (dendrogram) và dưới (heatmap)
    fig = plt.figure(figsize=(4, 4))
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 2], hspace=0.08)

    # === Dendrogram ===
    ax_dendro = fig.add_subplot(gs[0])
    draw_dendrogram(ax_dendro, 'root', positions, heights)
    for leaf, x in leaf_x.items():
        ax_dendro.plot(x, 0, 'o', color='black', markersize=4)
        ax_dendro.text(x, -0.3, str(leaf), ha='center', va='top', fontsize=12)
    ax_dendro.set_ylim(-0.5, max(heights.values()) + 0.5)
    ax_dendro.set_xlim(-0.5, len(leaf_x) - 0.5)

    ax_dendro.axis('off')

    # === Heatmap tam giác trên ===
    ax_heat = fig.add_subplot(gs[1])

    n = dist_reordered.shape[0]
    # Flip matrix
    dist_flipped = np.flipud(dist_reordered)

    # Grid
    X, Y = np.meshgrid(np.arange(n + 1) - 0.5, np.arange(n + 1) - 0.5)

    # Plot
    pcm = ax_heat.pcolormesh(
        X, Y, dist_flipped,
        cmap='gray', #'bone'Greys_rGreys_r
        shading='flat',
        edgecolors='none'
    )

    # Ticks
    ax_heat.set_xticks([])
    ax_heat.set_xticklabels([])
    ax_heat.set_yticks(np.arange(n))
    ax_heat.set_yticklabels(unique_leaf_order[::-1], fontsize=12)  # reverse labels

    # Annotate
    if verbose:
        for i in range(n):
            for j in range(n):
                value = dist_reordered[n - 1 - i, j]
                ax_heat.text(j, i, f"{value:.2f}", ha='center', va='center', fontsize=6, color='white')

    # Colorbar
    fig.colorbar(pcm, ax=ax_heat, orientation='horizontal', fraction=0.08, pad=0.05).ax.tick_params(labelsize=9)

    # Lưu file
    if savefile:
        os.makedirs('figs', exist_ok=True)
        plt.savefig(savefile, bbox_inches='tight', dpi=300)
    plt.close(fig)
