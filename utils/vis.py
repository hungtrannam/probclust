# =======================================
# Author: Hung Tran-Nam
# Email: namhung34.info@gmail.com
# Repo: https://github.com/hungtrannam/probclust
# =======================================
# File: utils/vis.py
# Description: Cài đặt các plot Matplotlib
# =======================================

import matplotlib.pyplot as plt
import matplotlib
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
        print(f"Saved plot to {savefile}")
    plt.close()

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
        print(f"Saved plot to {savefile}")
    
# ==========================================

def plotHeatmap_U(membership_matrix, savefile=None):
    """
    Vẽ heatmap cho ma trận phân vùng fuzzy clustering.
    """
    plt.figure(figsize=(10, 5))
    temp(fontsize=20)
    
    sns.heatmap(
        membership_matrix,
        annot=True,
        fmt=".1f",
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
        print(f"Saved plot to {savefile}")

def plot_OF(obj_hist, savefile= None):
    """
    Vẽ heatmap cho ma trận phân vùng fuzzy clustering.
    """
    plt.figure(figsize=(6, 5))
    temp(fontsize=20)
    
    plt.plot(obj_hist, lw = 3,color = 'black', marker='o')
    plt.xlabel("Iteration")
    plt.ylabel("Objective Function")
    plt.tight_layout()
    
    if savefile:
        os.makedirs('figs', exist_ok=True)
        plt.savefig(savefile, bbox_inches='tight', dpi=300)
        print(f"Saved plot to {savefile}")


# ==========================================

def plot_log_function(x_grid, beta_func_hat, savefile = None):
    """
    Vẽ hàm beta ước lượng.
    """
    temp(fontsize=20)
    plt.figure(figsize=(5,4))
    plt.plot(x_grid, beta_func_hat, linewidth=3, color='black')
    plt.ylabel(r'$\widehat{\beta}(x)$')
    plt.axhline(0, color='black', linestyle='--', linewidth=2)

    plt.tight_layout()
    
    if savefile:
        os.makedirs('figs', exist_ok=True)
        plt.savefig(savefile, bbox_inches='tight')
        print(f"Saved plot to {savefile}")


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
    temp(20)
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 2], hspace=0.08)

    # === Dendrogram ===
    ax_dendro = fig.add_subplot(gs[0])
    draw_dendrogram(ax_dendro, 'root', positions, heights)
    for leaf, x in leaf_x.items():
        ax_dendro.plot(x, 0, 'o', color='black', markersize=4)
        ax_dendro.text(x, -0.3, str(leaf+1), ha='center', va='top', fontsize=12)
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
    ax_heat.set_yticklabels((np.array(unique_leaf_order[::-1]) + 1).tolist(), fontsize=12)



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
        print(f"Saved plot to {savefile}")
    

# =============================


def plot_decision(x_grid, all_pdfs, proba, n_A, n_B, 
                  savefile=None, title=None):
    """
    Plot PDFs with color encoded by proba or decision value.
    Supports both continuous [0,1] and discrete {0,1}.
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    cmap = plt.cm.coolwarm

    # --- Check if discrete 0/1 ---
    unique_vals = np.unique(np.round(proba, 4))
    is_discrete = np.all(np.isin(unique_vals, [0, 1]))

    if is_discrete:
        color_map_func = lambda p: cmap(0.0) if p == 0 else cmap(1.0)
    else:
        color_map_func = lambda p: cmap(p)

    # --- Plot class 0 ---
    for i in range(n_A):
        ax.plot(x_grid, all_pdfs[i], color=color_map_func(proba[i]), alpha=0.6, linewidth=2)

    # --- Plot class 1 ---
    for i in range(n_A, n_A + n_B):
        ax.plot(x_grid, all_pdfs[i], color=color_map_func(proba[i]), alpha=0.6, linewidth=2, linestyle='--')

    # --- Colorbar only if continuous ---
    if not is_discrete:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=matplotlib.colors.Normalize(vmin=0, vmax=1))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label('Decision Value')

    if title:
        ax.set_title(title)

    plt.tight_layout()
    if savefile:
        os.makedirs(os.path.dirname(savefile), exist_ok=True)
        plt.savefig(savefile, bbox_inches='tight', dpi=300)
        print(f"Saved plot to {savefile}")
        plt.close()
    else:
        plt.show()


# ========================================



def plot_silhouette_values(F_data, labels, distance_metric='L2', bandwidth=0.01, grid=None, savefile=None):
    """
    Vẽ biểu đồ giá trị Silhouette cho từng hàm mật độ.
    """
    from utils.vali import CVI
    evaluator = CVI(distance_metric=distance_metric, bandwidth=bandwidth, grid=grid)
    D = evaluator._compute_distance_matrix(F_data)
    
    n = F_data.shape[0]
    clusters = np.unique(labels)
    silhouette_vals = np.zeros(n)

    for i in range(n):
        same_cluster = labels == labels[i]
        a_i = np.mean(D[i, same_cluster]) if np.sum(same_cluster) > 1 else 0
        b_i = np.inf
        for c in clusters:
            if c != labels[i]:
                b_i = min(b_i, np.mean(D[i, labels == c]))
        silhouette_vals[i] = (b_i - a_i) / max(a_i, b_i) if max(a_i, b_i) > 0 else 0

    # Vẽ biểu đồ
    fig, ax = plt.subplots(figsize=(6, 4))
    y_lower = 0
    for c in clusters:
        cluster_vals = silhouette_vals[labels == c]
        cluster_vals.sort()
        size_c = cluster_vals.shape[0]
        ax.fill_betweenx(np.arange(y_lower, y_lower + size_c),
                         0, cluster_vals, alpha=0.7, label=f'Cluster {c}')
        y_lower += size_c

    ax.axvline(np.mean(silhouette_vals), color="red", linestyle="--", label='Mean Silhouette')
    ax.set_xlabel("Silhouette")
    ax.legend()
    plt.tight_layout()
    if savefile:
        os.makedirs(os.path.dirname(savefile), exist_ok=True)
        plt.savefig(savefile, bbox_inches='tight', dpi=300)
        print(f"Saved plot to {savefile}")
        plt.close()
    else:
        plt.show()

    return silhouette_vals

def plot_CVI_with_k(num_clusters_range, silhouette_scores, dunn_scores, dbi_scores, savefile=None):
    
    """
    Vẽ biểu đồ CVI (Silhouette, Dunn, Davies-Bouldin) với số cụm k.
    """

    from utils.vali import find_elbow_k
    k_sil = find_elbow_k(list(num_clusters_range), silhouette_scores)
    k_dunn = find_elbow_k(list(num_clusters_range), dunn_scores)
    k_dbi = find_elbow_k(list(num_clusters_range), [-d for d in dbi_scores]) 

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))

    # --- Frame 1: Silhouette & Dunn (Lớn là tốt hơn)
    ax1.plot(num_clusters_range, silhouette_scores, 'o-', color='black',
             label=f'Silhouette (k = {k_sil})', lw=3, markersize=9)
    ax1.axvline(k_sil, color='black', linestyle='-', alpha=0.5, lw=3)

    ax1.plot(num_clusters_range, dunn_scores, 's--', color='black',
             label=f'Dunn Index (k = {k_dunn})', lw=3, markersize=9)
    ax1.axvline(k_dunn, color='black', linestyle='--', alpha=0.5, lw=3)

    ax1.set_ylabel("CVI")
    ax1.grid(axis='x', linestyle='--')
    ax1.legend()

    # --- Frame 2: DBI (Bé là tốt hơn)
    ax2.plot(num_clusters_range, dbi_scores, 'o-.', color='black',
             label=f'Davies-Bouldin (k = {k_dbi})', lw=3, markersize=9)
    ax2.axvline(k_dbi, color='black', linestyle='-.', alpha=0.5, lw=3)

    ax2.set_xlabel("Số chùm")
    ax2.set_ylabel("CVI")
    ax2.grid(axis='x', linestyle='--')
    ax2.legend()

    plt.tight_layout()
    if savefile:
        os.makedirs(os.path.dirname(savefile), exist_ok=True)
        plt.savefig(savefile, bbox_inches='tight', dpi=300)
        print(f"Saved plot to {savefile}")
        plt.close()
    else:
        plt.show()

