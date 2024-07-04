"""
K-均值聚类
K-Means Cluster
"""
import warnings
import numpy as np
from Models.Utils import plot_cluster, random_generate_cluster

class KMeans():
    def __init__(self, X=None, k=None, num_iter=None):
        self.X = None  # 需要聚类的数据
        self.set_data(X)
        self.labels = None  # 聚类后解结果
        self.centers = None  # 聚类中心点坐标
        self.k = k  # 聚类中心个数
        self.num_iter = num_iter  # 迭代次数

    def set_data(self, X):
        """给定训练数据"""
        if X is not None:
            if self.X is not None:
                warnings.warn("Training data will be overwritten")
            self.X = X.copy()

    def set_parameters(self, k, num_iter):
        """重新修改相关参数"""
        if self.k is not None and k is not None:
            warnings.warn("Parameter 'k' be overwritten")
            self.k = k
        if self.num_iter is not None and num_iter is not None:
            warnings.warn("Parameter 'num_iter' be overwritten")
            self.num_iter = num_iter

    def train(self, X=None, k=None, num_iter=None):
        """对数据进行聚类"""
        self.set_data(X)
        self.set_parameters(k, num_iter)
        # 初始化聚类结果
        self.labels = np.zeros(shape=len(self.X))
        # 随机初始化聚类中心
        self.centers = np.random.uniform(np.min(self.X), np.max(self.X), size=(self.k, self.X.shape[1]))
        # 计算每个数据点到每个聚类中心的距离 (每个数据归类到最近的聚类中心的那一类)
        distances = np.linalg.norm(self.X[:, np.newaxis, :] - self.centers[np.newaxis, :, :], axis=2)
        # 找到每个数据点最近的聚类中心
        self.labels = np.argmin(distances, axis=1)
        # 画图展示聚类效果
        self.plot_cluster(pause=True, n_iter=0)
        # 再根据当前聚类结果计算新的聚类中心
        for i in range(self.num_iter):
            # 创建簇掩码矩阵
            mask = np.zeros((self.X.shape[0], self.k))
            # 该掩码矩阵为one-hot形式
            mask[np.arange(self.X.shape[0]), self.labels] = 1
            # 计算每个簇的数据点数量
            num_points = mask.sum(axis=0)
            # 计算新的聚类中心 (取最大是防止某类中数据点数量为0)
            self.centers = (mask.T @ self.X) / np.maximum(num_points[:, None], 1)
            # 计算每个数据点到每个聚类中心的距离 (每个数据归类到最近的聚类中心的那一类)
            distances = np.linalg.norm(self.X[:, np.newaxis, :] - self.centers[np.newaxis, :, :], axis=2)
            # 找到每个数据点最近的聚类中心
            self.labels = np.argmin(distances, axis=1)
            self.plot_cluster(pause=True, n_iter=i+1)
        return self.labels, self.centers

    def plot_cluster(self, pause=False, n_iter=None, pause_time=0.15):
        plot_cluster(self.X, self.labels, self.centers, pause, n_iter, pause_time)


if __name__ == '__main__':
    # X = np.random.uniform(0, 1, (100, 2))
    X, Y = random_generate_cluster(X_size=300, X_feat=2, k=3)
    model = KMeans(X, k=3, num_iter=20)
    model.train()
    model.plot_cluster()
