"""
Copyright (c) 2023 LuChen Wang
MELON is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan
PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY
KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
NON-INFRINGEMENT, MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details.
"""
import numpy as np
from Models import Model
from Models.Utils import plot_cluster, run_blobs_cluster, run_circle_cluster, run_moons_cluster


class DBSCAN(Model):
    def __init__(self, X_train=None, eps=0.5, min_samples=5, metric='minkowski', p=2, show=False):
        """
        基于密度的噪声应用空间聚类
        Density-Based Spatial Clustering of Applications with Noise (DBSCAN)
        :param X_train: 需要聚类的数据
        :param eps: 两个样本之间的最大距离，以确定一个样本是否在另一个样本的邻域内
        :param min_samples: 一个点的邻域中必须存在的最小样本数(包含自身)
        :param metric: 计算距离时的距离度量(minkowski:闵可夫斯基距离, manhattan:曼哈顿距离, euclidean:欧几里得距离)
        :param p: 计算minkowski距离的幂次，(p=1:曼哈顿距离，p=2:欧式距离)
        :param show: 是否展示迭代过程
        """
        super().__init__(X_train, None)
        self.labels = None  # 聚类后的结果
        self.eps = eps  # 两个样本之间的最大距离，以确定一个样本是否在另一个样本的邻域内
        self.min_samples = min_samples  # 一个点的邻域中必须存在的最小样本数(包含自身)
        self.metric = metric  # 计算距离时的距离度量
        self.p = p  # 计算minkowski距离的幂次，(p=1)为曼哈顿距离，(p=2)为欧式距离
        self.show = show  # 是否展示迭代过程

    def train(self, X_train=None):
        """对数据进行聚类"""
        self.set_train_data(X_train, None)
        num_data = len(self.X_train)
        # 初始化所有点为噪声点（-1）
        self.labels = np.full(num_data, -1)
        # 第一个簇的下标
        cluster_id = 0
        # 遍历所有数据点
        for idx in range(num_data):
            # 如果点已经被访问过，则跳过
            if self.labels[idx] != -1:
                continue
            # 获取该数据点领域内的所有点
            neighbors = self.get_neighbors(idx)
            if len(neighbors) < self.min_samples:
                # 如果不是核心点，则标记为噪声
                self.labels[idx] = -1
            else:
                # 否则扩展簇
                self.expand_cluster(idx, neighbors, cluster_id)
                cluster_id += 1
        return self.labels

    def get_neighbors(self, idx):
        """获取指定点的邻域内的所有点下标"""
        if (self.metric == 'manhattan') or (self.metric == 'minkowski' and self.p == 1):
            distances = np.sum(np.abs(self.X_train - self.X_train[idx]), axis=1)
        elif (self.metric == 'euclidean') or (self.metric == 'minkowski' and self.p == 2):
            distances = np.sqrt(np.sum((self.X_train - self.X_train[idx]) ** 2, axis=1))
        else:
            raise ValueError(f"Unsupported metric: {self.metric}")
        neighbors = np.where(distances <= self.eps)[0]
        return neighbors

    def expand_cluster(self, idx, neighbors, cluster_id):
        """从核心点开始扩展簇"""
        self.labels[idx] = cluster_id
        # 使用集合避免重复运算
        neighbors = set(neighbors)
        i = 0
        while neighbors:
            # 每次从集合中弹出一个点
            neighbor = neighbors.pop()
            if self.labels[neighbor] == -1:
                # 如果是噪声点，则加入当前簇
                self.labels[neighbor] = cluster_id
            elif self.labels[neighbor] != -1:
                # 如果已经属于某个簇，则跳过
                i += 1
                continue
            # 获取该数据点领域内的所有点
            neighbor_neighbors = self.get_neighbors(neighbor)
            if len(neighbor_neighbors) >= self.min_samples:
                # 如果是核心点，将邻域内的点加入搜索集合
                neighbors.update(neighbor_neighbors)
                self.labels[neighbor] = cluster_id
            i += 1
            if self.show:
                self.plot_cluster(pause=True)

    def plot_cluster(self, pause=False, n_iter=None, pause_time=0.01):
        plot_cluster(self.X_train, self.labels, None, pause, n_iter, pause_time)


if __name__ == '__main__':
    np.random.seed(100)
    model = DBSCAN(eps=1.0, min_samples=10, show=True)
    run_blobs_cluster(model)
    model = DBSCAN(eps=0.2, min_samples=10, show=True)
    run_circle_cluster(model)
    run_moons_cluster(model)
