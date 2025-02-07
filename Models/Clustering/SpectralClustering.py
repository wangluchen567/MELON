import warnings
import numpy as np
from KMeans import KMeans
from Models.Utils import plot_cluster, run_points_cluster, run_circle_cluster


class SpectralClustering():
    # 定义相似度矩阵构建方式类型
    NEIGHBORS = 0
    POLY = 1
    RBF = GAUSSIAN = 2
    SIGMOID = 3

    def __init__(self, X=None, n_clusters=None, affinity=RBF, n_neighbors=10, mode='connect',
                 gamma=1.0, degree=3.0, const=1.0, num_train=10, num_iter=300, tol=1e-4, show=False):
        self.X = None  # 需要聚类的数据
        self.set_data(X)  # 设置数据
        self.labels = None  # 聚类后解结果
        self.n_clusters = n_clusters  # 聚类中心个数
        self.affinity = affinity  # 相似度矩阵的构建方式
        self.n_neighbors = n_neighbors  # 使用邻居策略时近邻数量
        self.mode = mode  # 使用邻居策略时构建相似度矩阵的模式
        self.gamma = gamma  # 核函数的系数（乘数项）
        self.degree = degree  # 核函数的系数（指数项）
        self.const = const  # 核函数的系数（常数项）
        self.num_train = num_train  # 训练次数(提升稳定性)(kmeans)
        self.num_iter = num_iter  # 最大迭代次数(kmeans)
        self.tol = tol  # 收敛的容忍度，若两次变化小于tol则说明已收敛(kmeans)
        self.show = show  # 是否展示迭代过程

    def set_data(self, X):
        """给定训练数据"""
        if X is not None:
            if self.X is not None:
                warnings.warn("Training data will be overwritten")
            self.X = X.copy()

    def set_parameters(self, n_clusters=None, affinity=None, n_neighbors=None, mode=None,
                       gamma=None, degree=None, const=None, num_train=None, num_iter=None, tol=None):
        """重新修改相关参数"""
        parameters = ['n_clusters', 'affinity', 'n_neighbors', 'mode',
                      'gamma', 'degree', 'const', 'num_train', 'num_iter', 'tol']
        values = [n_clusters, affinity, n_neighbors, mode, gamma, degree, const, num_train, num_iter, tol]
        for param, value in zip(parameters, values):
            if value is not None and getattr(self, param) is not None:
                warnings.warn(f"Parameter '{param}' will be overwritten")
                setattr(self, param, value)

    def train(self, X=None, n_clusters=None, affinity=None, n_neighbors=None, mode=None,
              gamma=None, degree=None, const=None, num_train=None, num_iter=None, tol=None):
        """对数据进行聚类"""
        self.set_data(X)
        self.set_parameters(n_clusters, affinity, n_neighbors, mode, gamma, degree, const, num_train, num_iter, tol)
        # 计算相似度矩阵
        simi_mat = self.cal_simi_mat()
        # 计算度矩阵
        degree_mat = np.diag(np.sum(simi_mat, axis=1))
        # 计算拉普拉斯矩阵
        lap_mat = degree_mat - simi_mat
        # 对拉普拉斯矩阵进行特征分解
        eigen_values, eigen_vectors = np.linalg.eigh(lap_mat)
        # 选择前 n_clusters 个最小特征值对应的特征向量
        sorted_indices = np.argsort(eigen_values)
        selected_vectors = eigen_vectors[:, sorted_indices[:self.n_clusters]]
        # 然后使用K-means方法对其进行聚类
        kmeans = KMeans(selected_vectors, n_clusters=self.n_clusters, num_train=self.num_train,
                        num_iter=self.num_iter, tol=self.tol, show=False)  # 无法展示过程
        kmeans.train()  # 进行训练
        # 得到最终结果
        self.labels = kmeans.labels
        return self.labels

    def cal_simi_mat(self):
        """计算数据集的相似度矩阵"""
        if self.affinity == self.NEIGHBORS:
            return self.k_neighbors_mat(self.X, self.n_neighbors, self.mode)
        elif self.affinity == self.POLY:
            return self.poly_kernel_mat(self.X, self.X, self.gamma, self.degree, self.const)
        elif self.affinity == self.RBF or self.affinity == self.GAUSSIAN:
            return self.rbf_kernel_mat(self.X, self.X, self.gamma)
        elif self.affinity == self.SIGMOID:
            return self.sigmoid_kernel_mat(self.X, self.X, self.const)
        else:
            raise ValueError(f"Unsupported affinity: {self.affinity}")

    @staticmethod
    def k_neighbors_mat(X, k, mode):
        """计算k近邻相似度矩阵"""
        # 数据集大小
        num_data = len(X)
        # 计算距离矩阵
        dist_mat = np.linalg.norm(X[:, None] - X, axis=-1)
        # 获取每个点的 k 个最近邻的索引
        k_neighbors_indices = np.argsort(dist_mat, axis=1)[:, 1:k + 1]
        # 初始化相似度矩阵
        simi_mat = np.zeros((num_data, num_data))
        if mode == 'connect':
            # 权重为1
            simi_mat[np.arange(num_data)[:, None], k_neighbors_indices] = 1
            simi_mat[k_neighbors_indices, np.arange(num_data)[:, None]] = 1
        elif mode == 'distance':
            # 权重为距离的倒数
            distances = np.take_along_axis(dist_mat, k_neighbors_indices, axis=1)
            simi_mat[np.arange(num_data)[:, None], k_neighbors_indices] = 1 / (distances + 1e-8)
            simi_mat[k_neighbors_indices, np.arange(num_data)[:, None]] = 1 / (distances + 1e-8)
        else:
            raise ValueError("Mode must be 'connect' or 'distance'")
        return simi_mat

    @staticmethod
    def poly_kernel_mat(X, Y, gamma=None, degree=3.0, const=1.0):
        """计算多项式核函数矩阵"""
        # 如果 gamma未指定，则设置为 1/特征数量
        if gamma is None:
            gamma = 1.0 / X.shape[1]
        # 使用多项式核函数公式计算核函数矩阵
        kernel_mat = (gamma * (X @ Y.T) + const) ** degree
        return kernel_mat

    @staticmethod
    def rbf_kernel_mat(X, Y, gamma=None):
        """计算径向基核（高斯核）函数矩阵"""
        # 如果 gamma 未指定，则设置为 1/特征数量
        if gamma is None:
            gamma = 1.0 / X.shape[1]
        # 使用 NumPy 的广播机制，计算核函数矩阵
        kernel_mat = np.exp(-gamma * np.sum((X[:, np.newaxis, :] - Y[np.newaxis, :, :]) ** 2, axis=2))
        return kernel_mat

    @staticmethod
    def sigmoid_kernel_mat(X, Y, gamma=None, const=1.0):
        """计算sigmoid核函数矩阵"""
        # 如果 gamma 未指定，则设置为 1/特征数量
        if gamma is None:
            gamma = 1.0 / X.shape[1]
        # 使用Sigmoid核函数公式计算核函数矩阵
        kernel_mat = np.tanh(gamma * (X @ Y.T) + const)
        return kernel_mat

    def plot_cluster(self, pause=False, n_iter=None, pause_time=0.15):
        plot_cluster(self.X, self.labels, None, pause, n_iter, pause_time)


if __name__ == '__main__':
    np.random.seed(100)
    model = SpectralClustering(n_clusters=5, affinity=SpectralClustering.NEIGHBORS)
    run_points_cluster(model)
    model = SpectralClustering(n_clusters=2, affinity=SpectralClustering.NEIGHBORS)
    run_circle_cluster(model)
