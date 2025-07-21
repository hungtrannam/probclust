import numpy as np
import os
import argparse
from data.data_loader import generateGauss
from utils.integral import grid
from utils.vis import plotPDF, plotHeatmap_U, plotPDF_Theta
from Models.clustering import NCF

def main(args):
    # Tạo grid
    h = args.bandwidth
    x, _ = grid(h, start=-10, end=25)

    mu = np.array([0.3, 4.0, 9.1, 1.0, 5.5, 8.0, 4.8, 4])
    sig = np.array([1,1,1,1,1,1,1, 3])
    print("sigma:", sig)

    # Sinh Gaussian
    F_data = generateGauss(mu, sig, x, savefile='dataset/data.npz')

    # Vẽ các PDF ban đầu
    plotPDF(x, F_data, savefile='figs/pdfs.pdf')

    # Khởi tạo và huấn luyện
    cluster = NCF.Model(
        grid_x=x,
        num_clusters=args.num_clusters,
        max_iterations=args.max_iter,
        distance_metric=args.distance_metric,
        tolerance=args.tolerance,
        delta=args.delta,
        w1=args.w1,
        w2=args.w2,
        w3=args.w3,
        bandwidth=h,
        seed=args.seed
    )
    cluster.fit(F_data)

    T, I, F, V = cluster.get_results()

    # Vẽ heatmap T
    plotHeatmap_U(T, savefile=os.path.join(args.save_dir, "T.pdf"))

    # Vẽ heatmap tổng hợp
    compile = np.vstack([T, I[np.newaxis, :], F[np.newaxis, :]])
    plotHeatmap_U(compile, savefile=os.path.join(args.save_dir, "compile.pdf"))

    # Vẽ trung tâm cụm
    plotPDF_Theta(x, F_data, theta=V, savefile=os.path.join(args.save_dir, 'V.pdf'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run NCF clustering on generated Gaussian PDFs.")
    parser.add_argument('--num_clusters', type=int, default=3, help='Number of clusters.')
    parser.add_argument('--max_iter', type=int, default=100, help='Maximum number of iterations.')
    parser.add_argument('--tolerance', type=float, default=1e-5, help='Convergence tolerance.')
    parser.add_argument('--distance_metric', type=str, default='L2', choices=['L1', 'L2', 'H', 'BC', 'W2'],
                        help='Distance metric to use.')
    parser.add_argument('--delta', type=float, default=1, help='Delta parameter for falsity.')
    parser.add_argument('--w1', type=float, default=1, help='Weight w1 for truth membership.')
    parser.add_argument('--w2', type=float, default=1, help='Weight w2 for indeterminacy.')
    parser.add_argument('--w3', type=float, default=1, help='Weight w3 for falsity.')
    parser.add_argument('--bandwidth', type=float, default=0.01, help='Bandwidth for integration.')
    parser.add_argument('--save_dir', type=str, default='figs', help='Directory to save output figures.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility.')

    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    main(args)
