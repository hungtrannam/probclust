import numpy as np
import os
import argparse
from data.data_loader import generateGauss
from utils.integral import grid
from utils.vis import *
from Models.clustering import KCF, KFCF, HCF, FCF, EM, DBSCAN, NCF

def main(args):
    # Tạo grid
    x, _ = grid(args.bandwidth, start=-4, end=15)

    # mu = np.array([-0.3,  0.,   0.2,  3.5,  8.,   8.2, 12.5])
    # sig = np.array([.3,.4,.5, 0.3, 0.5, 0.4, 0.4])
    mu = np.array([0,0.25,0.5,0.75,1, 5.5, 9.0, 9.25, 9.5, 9.75, 10])
    sig = np.ones(11)

    # Sinh Gaussian
    F_data = generateGauss(mu, sig, x, savefile='dataset/data.npz')

    # Vẽ các PDF ban đầu
    plotPDF(x, F_data, savefile='figs/pdfs.pdf')

    # Khởi tạo và huấn luyện
    cluster = HCF.Model(
        grid_x=x,
        distance_metric=args.distance_metric,
        linkage=args.linkage, 
        bandwidth=args.bandwidth,
        # seed=args.seed
    )
    # cluster = DBSCAN.Model(
    #     grid_x=x,
    #     eps=args.eps,
    #     distance_metric=args.distance_metric,
    #     bandwidth=args.bandwidth,
    #     min_samples=args.min_samples,
    #     verbose=args.verbose
    # )

    # cluster = NCF.Model(
    #     grid_x=x,
    #     num_clusters=args.num_clusters,
    #     tolerance=args.tolerance,
    #     delta=args.delta,
    #     w1=args.w1,
    #     w2=args.w2,
    #     w3=args.w3,
    #     bandwidth=args.bandwidth,
    #     seed=args.seed
    # )

    cluster.fit(F_data)
    cluster.print_tree()
    plot_tree(cluster.tree, cluster.dist_matrix, savefile=f"figs/tree_{args.distance_metric}_{args.linkage}.pdf")

    # U, I, F, V = cluster.get_results()
    # print(U)
    # U = cluster.get_results()
    
    # Vẽ heatmap T
    # plotHeatmap_U(U, savefile=os.path.join(args.save_dir, "U.pdf"))
# 
    # # Vẽ heatmap tổng hợp
    # compile = np.vstack([T, I[np.newaxis, :], F[np.newaxis, :]])
    # plotHeatmap_U(compile, savefile=os.path.join(args.save_dir, "compile.pdf"))

    # Vẽ trung tâm cụm
    # plotPDF_Theta(x, F_data, theta=V, savefile=os.path.join(args.save_dir, 'V.pdf'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run NCF clustering on generated Gaussian PDFs.")
    parser.add_argument('--num_clusters', type=int, default=2, help='Number of clusters.')
    parser.add_argument('--max_iter', type=int, default=100, help='Maximum number of iterations.')
    parser.add_argument('--tolerance', type=float, default=1e-5, help='Convergence tolerance.')
    parser.add_argument('--distance_metric', type=str, default='W2', choices=['L1', 'L2', 'H', 'BC', 'W2'],
                        help='Distance metric to use.')
    parser.add_argument('--delta', type=float, default=1, help='Delta parameter for falsity.')
    parser.add_argument('--w1', type=float, default=1, help='Weight w1 for truth membership.')
    parser.add_argument('--w2', type=float, default=1, help='Weight w2 for indeterminacy.')
    parser.add_argument('--w3', type=float, default=1, help='Weight w3 for falsity.')
    parser.add_argument('--bandwidth', type=float, default=0.01, help='Bandwidth for integration.')
    parser.add_argument('--save_dir', type=str, default='figs', help='Directory to save output figures.')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility.')
    parser.add_argument('--linkage', default='complete')
    parser.add_argument('--eps', type=float, default=1)
    parser.add_argument('--min_samples', type=int, default=1)
    parser.add_argument('--verbose', default=False)



    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    main(args)
