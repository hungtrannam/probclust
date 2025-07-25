import numpy as np
import pandas as pd
from collections import Counter
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

BASE_TREE = ['IMM', 'NONE']

LEAF_DATA_KEY_X_DATA = 'X_DATA_KEY'
LEAF_DATA_KEY_Y = 'Y_KEY'
LEAF_DATA_KEY_X_CENTER_DOT = 'X_CENTER_DOT'
LEAF_DATA_KEY_SPLITTER = 'SPLITTER_KEY'


class TreePDF:
    def __init__(self, k, grid_x, max_leaves=None, verbose=0, light=True, base_tree='IMM',
                 n_jobs=None, random_state=None, distance_metric='L2', bandwidth=0.01):
        """
        TreePDF: Explainable K-means Tree cho dữ liệu hàm mật độ xác suất (PDF).

        :param k: số cụm KMeans.
        :param grid_x: mảng trục x (n_point).
        :param max_leaves: số lá tối đa.
        :param verbose: mức log.
        :param light: chế độ light (không lưu dữ liệu trong lá).
        :param base_tree: 'IMM' hoặc 'NONE'.
        :param n_jobs: số job song song.
        :param random_state: seed.
        :param distance_metric: 'L2' hoặc 'H'.
        :param bandwidth: bước lưới (tích phân).
        """
        self.k = k
        self.grid_x = np.array(grid_x, dtype=float)
        self.tree = None
        self._leaves_data = {}
        self.max_leaves = k if max_leaves is None else max_leaves
        self.random_state = random_state
        if self.max_leaves < k:
            raise Exception(f'max_leaves must be >= k [{self.max_leaves} < {k}]')
        self.verbose = verbose
        self.light = light
        if base_tree not in BASE_TREE:
            raise Exception(f'{base_tree} is not a supported base tree')
        self.base_tree = base_tree
        self.n_jobs = n_jobs if n_jobs is not None else 1
        self._feature_importance = None
        self.distance_metric = distance_metric
        self.bandwidth = bandwidth

    # === Distance for PDFs ===
    def _pdf_cost(self, data, center):
        """
        Tính tổng khoảng cách giữa mỗi PDF trong data và center.
        Sử dụng khoảng cách L1, L2, H, BC, hoặc W2 dựa trên self.distance_metric.
        """
        from utils.dist import Dist

        cost = 0.0
        for pdf in data:
            dist = Dist(pdf, center, Dim=1, h=self.bandwidth, grid=self.grid_x)
            if self.distance_metric == 'L1':
                cost += dist.L1()
            elif self.distance_metric == 'L2':
                cost += dist.L2()
            elif self.distance_metric == 'H':
                cost += dist.H()
            elif self.distance_metric == 'BC':
                cost += dist.BC()
            elif self.distance_metric == 'W2':
                cost += dist.W2()
            else:
                raise ValueError(f"Unsupported distance metric: {self.distance_metric}")
        return cost

    # === Build Tree ===
    def _build_tree(self, x_data, y, valid_centers, valid_cols):
        node = Node()
        if x_data.shape[0] == 0:
            node.value = 0
            return node
        elif valid_centers.sum() == 1:
            node.value = np.argmax(valid_centers)
            return node
        else:
            if np.unique(y).shape[0] == 1:
                node.value = y[0]
                return node
            else:
                cut = self._min_mistake_cut(x_data, y, valid_centers, valid_cols)
                if cut is None:
                    node.value = np.argmax(valid_centers)
                else:
                    col = cut["col"]
                    threshold = cut["threshold"]
                    node.set_condition(col, threshold)

                    # Split dựa trên cột col (giá trị PDF tại vị trí grid_x[col])
                    left_mask = x_data[:, col] <= np.mean(x_data[:, col])
                    matching_centers_mask = self.all_centers[:, col][y] <= np.mean(x_data[:, col])
                    mistakes_mask = left_mask != matching_centers_mask

                    left_valid_centers = np.zeros(valid_centers.shape, dtype=np.int32)
                    right_valid_centers = np.zeros(valid_centers.shape, dtype=np.int32)
                    left_valid_centers[valid_centers.astype(bool)] = 1
                    right_valid_centers[valid_centers.astype(bool)] = 1

                    node.left = self._build_tree(x_data[left_mask & ~mistakes_mask],
                                                 y[left_mask & ~mistakes_mask],
                                                 left_valid_centers,
                                                 valid_cols)
                    node.right = self._build_tree(x_data[~left_mask & ~mistakes_mask],
                                                  y[~left_mask & ~mistakes_mask],
                                                  right_valid_centers,
                                                  valid_cols)
                return node
            from utils.dist import Dist

    def _min_mistake_cut(self, x_data, y, valid_centers, valid_cols, mode="mistake"):
        """
        mode: 
            - "mistake": chọn split dựa trên số nhầm lẫn (ExKMC style).
            - "cost": chọn split dựa trên tổng cost (L2/Hellinger) của từng nhánh.
        """
        best_cut = None
        best_score = np.inf
        cols = np.where(valid_cols > 0)[0]

        for c in cols:
            threshold = self.grid_x[c]

            # Chia dữ liệu thành hai nhóm trái / phải
            left_mask = x_data[:, c] <= np.mean(x_data[:, c])
            right_mask = ~left_mask

            if left_mask.sum() == 0 or right_mask.sum() == 0:
                continue

            if mode == "mistake":
                # === Minimize Mistake Count ===
                left_labels = y[left_mask]
                right_labels = y[right_mask]
                mistakes = self._calc_mistakes(left_labels, right_labels)
                score = mistakes

            elif mode == "cost":
                # === Minimize Cost ===
                left_center = x_data[left_mask].mean(axis=0)
                right_center = x_data[right_mask].mean(axis=0)
                score = self._pdf_cost(x_data[left_mask], left_center) + \
                        self._pdf_cost(x_data[right_mask], right_center)

            else:
                raise ValueError(f"Unsupported mode: {mode}")

            if score < best_score:
                best_score = score
                best_cut = {"col": c, "threshold": threshold}

        return best_cut


    def _calc_mistakes(self, left_y, right_y):
        def mistakes(y):
            return 0 if len(y) == 0 else len(y) - np.bincount(y).max()
        return mistakes(left_y) + mistakes(right_y)

    # === Fit & Predict ===
    def fit(self, x_data, kmeans=None):
        x_data = convert_input(x_data)
        if kmeans is None:
            if self.verbose > 0:
                print(f'Finding {self.k}-means with KCF')
            from Models.clustering.KCF import Model  # hoặc import class Model bạn viết ở trên

            kmeans = Model(grid_x=self.grid_x, num_clusters=self.k,
                        max_iterations=100, tolerance=1e-5,
                        distance_metric=self.distance_metric,
                        bandwidth=self.bandwidth, seed=self.random_state)
            kmeans.fit(x_data, verbose=self.verbose > 0)
            y = kmeans.get_hard_assignments()
            self.all_centers = kmeans.centroids
        else:
            # Nếu truyền một model bên ngoài (chưa cần)
            y = kmeans.get_hard_assignments()
            self.all_centers = kmeans.centroids


        y = np.array(kmeans.get_hard_assignments())
        self.all_centers = np.array(kmeans.centroids, dtype=np.float64)

        if self.base_tree == "IMM":
            self.tree = self._build_tree(
                x_data, y,
                np.ones(self.all_centers.shape[0], dtype=np.int32),
                np.ones(self.all_centers.shape[1], dtype=np.int32))
            leaves = self.k
        else:
            self.tree = Node()
            self.tree.value = 0
            leaves = 1

        self._feature_importance = np.zeros(x_data.shape[1])
        self.__fill_stats__(self.tree, x_data, y)
        return self

    def predict(self, x_data):
        x_data = convert_input(x_data)
        return self._predict_subtree(self.tree, x_data)

    def fit_predict(self, x_data, kmeans=None):
        """
        Huấn luyện cây và trả về nhãn cụm dự đoán.
        :param x_data: dữ liệu PDF [n_pdf, n_point]
        :param kmeans: mô hình KMeans đã train (nếu có).
        """
        self.fit(x_data, kmeans)
        return self.predict(x_data)

    

    def _predict_subtree(self, node, x_data):
        if node.is_leaf():
            return np.full(x_data.shape[0], node.value)
        else:
            left_mask = x_data[:, node.feature] <= np.mean(x_data[:, node.feature])
            ans = np.zeros(x_data.shape[0])
            ans[left_mask] = self._predict_subtree(node.left, x_data[left_mask])
            ans[~left_mask] = self._predict_subtree(node.right, x_data[~left_mask])
            return ans
        
    def plot(self, filename="tree_plot.png", view=True):
        import matplotlib.pyplot as plt
        from matplotlib.patches import FancyBboxPatch
        from utils.vis import temp

        if self.tree is None:
            raise Exception("Tree is empty. Run fit() first.")

        nodes = []
        edges = []

        # === Thu thập node & edge ===
        def traverse(node, depth=0, parent=None):
            idx = len(nodes)
            if node.is_leaf():
                label = (
                    f"Cluster: {node.value}\n"
                    rf"#pdf(s): {node.samples}"
                    f"\nMistake(s): {node.mistakes}"
                )
            else:
                gx = self.grid_x[node.feature]
                label = (
                    rf"Split: $x \leq {gx:.2f}$"
                    f"\n#pdf(s): {node.samples}"
                )


            nodes.append((idx, label, depth, node))
            if parent is not None:
                edges.append((parent, idx))
            if not node.is_leaf():
                traverse(node.left, depth + 1, idx)
                traverse(node.right, depth + 1, idx)

        traverse(self.tree)

        # === Tính tọa độ X theo số lá ===
        def assign_x_positions(node, depth=0, x_start=0):
            if node.is_leaf():
                return [(node, x_start, depth)], x_start + 1
            else:
                left_positions, next_x = assign_x_positions(node.left, depth + 1, x_start)
                right_positions, next_x = assign_x_positions(node.right, depth + 1, next_x)
                center_x = (left_positions[0][1] + right_positions[-1][1]) / 2
                return [(node, center_x, depth)] + left_positions + right_positions, next_x

        positions, _ = assign_x_positions(self.tree)

        # Map node object -> index
        node_index_map = {n: i for i, (_, _, _, n) in enumerate(nodes)}
        coords = {node_index_map[n]: (x, -depth) for n, x, depth in positions}

        # === Vẽ cây ===
        plt.figure(figsize=(10, 6))
        temp(20)
        ax = plt.gca()

        # Vẽ cạnh nối
        for parent, child in edges:
            x1, y1 = coords[parent]
            x2, y2 = coords[child]
            plt.plot([x1, x2], [y1, y2], 'k-', linewidth=3,zorder=0)

        # Vẽ node
        box_width, box_height = 0.45, 0.4
        for idx, label, _, node in nodes:
            x, y = coords[idx]
            color = "#f0f0f0" if node.is_leaf() else "#979797"
            rect = FancyBboxPatch(
                (x - box_width / 2, y - box_height / 2),
                box_width, box_height,
                boxstyle="round,pad=0.05,rounding_size=0.05",
                linewidth=1, facecolor=color, edgecolor="black"
            )
            ax.add_patch(rect)
            plt.text(x, y, label, ha='center', va='center', fontsize=18)

        plt.axis('off')
        plt.tight_layout()
        plt.savefig(filename, dpi=300)
        if view:
            plt.show()




    def __fill_stats__(self, node, x_data, y):
        node.samples = x_data.shape[0]
        if not node.is_leaf():
            self._feature_importance[node.feature] += 1
            left_mask = x_data[:, node.feature] <= np.mean(x_data[:, node.feature])
            self.__fill_stats__(node.left, x_data[left_mask], y[left_mask])
            self.__fill_stats__(node.right, x_data[~left_mask], y[~left_mask])
        else:
            node.mistakes = len([cluster for cluster in y if cluster != node.value])


class Node:
    def __init__(self):
        self.feature = None
        self.value = None
        self.samples = None
        self.mistakes = None
        self.left = None
        self.right = None

    def is_leaf(self):
        return (self.left is None) and (self.right is None)

    def set_condition(self, feature, value):
        self.feature = feature
        self.value = value


def convert_input(data):
    if isinstance(data, list):
        data = np.array(data, dtype=np.float64)
    elif isinstance(data, np.ndarray):
        data = data.astype(np.float64, copy=False)
    elif isinstance(data, pd.DataFrame):
        data = data.values.astype(np.float64, copy=False)
    else:
        raise Exception(str(type(data)) + ' is not supported type')
    return data
