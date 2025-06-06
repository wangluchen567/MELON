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


class KMeans(Model):
    def __init__(self, X_train=None, n_clusters=None, init_func='k-means++',
                 num_train=10, max_iter=1000, tol=1e-4, show=False):
        """
        K-均值聚类
        :param X_train: 需要聚类的数据
        :param n_clusters: 聚类中心个数
        :param init_func: 中心初始化方法(k-means++/random)
        :param num_train: 训练次数(提升稳定性)
        :param max_iter: 最大迭代次数
        :param tol: 收敛的容忍度，若两次变化小于tol则说明已收敛
        :param show: 是否展示迭代过程
        """
        super().__init__(X_train, None)
        self.labels = None  # 聚类后的结果
        self.centers = None  # 聚类中心点坐标
        self.inertia = None  # 聚类后的惯性指标
        self.n_clusters = n_clusters  # 聚类中心个数
        self.init_func = init_func  # 中心初始化方法
        self.num_train = num_train  # 训练次数(提升稳定性)
        self.max_iter = max_iter  # 最大迭代次数
        self.tol = tol  # 收敛的容忍度，若两次变化小于tol则说明已收敛
        self.show = show  # 是否展示迭代过程

    def train(self, X_train=None):
        """对数据进行聚类"""
        self.set_train_data(X_train, None)
        self.inertia = np.inf
        # 训练num_train次以获取最佳结果
        for i in range(self.num_train):
            labels, centers, inertia = self.train_one()
            # 取最佳聚类结果
            if inertia < self.inertia:
                self.labels = labels
                self.centers = centers
                self.inertia = inertia
        return self.labels

    def train_one(self):
        """训练一次得到结果"""
        # 初始化聚类中心
        centers = self.init_centers()
        # 计算每个数据点到每个聚类中心的距离 (每个数据归类到最近的聚类中心的那一类)
        distances = np.linalg.norm(self.X_train[:, np.newaxis, :] - centers[np.newaxis, :, :], axis=2)
        # 找到每个数据点最近的聚类中心
        labels = np.argmin(distances, axis=1)
        # 创建簇的掩码矩阵
        mask = np.zeros((self.X_train.shape[0], self.n_clusters), dtype=int)
        # 簇的掩码矩阵以one-hot编码
        mask[np.arange(self.X_train.shape[0]), labels] = 1
        # 计算惯性值（即总距离的和，以判别聚类质量）
        inertia = np.sum(distances * mask)
        # 画图展示聚类效果
        if self.show:
            self.plot_cluster(labels, centers, pause=True, n_iter=0)
        # 再根据当前聚类结果计算新的聚类中心
        for i in range(self.max_iter):
            # 计算每个簇的数据点数量
            num_points = mask.sum(axis=0)
            # 计算新的聚类中心 (取最大是防止某类中数据点数量为0)
            centers = (mask.T @ self.X_train) / np.maximum(num_points[:, None], 1)
            # 计算每个数据点到每个聚类中心的距离 (每个数据归类到最近的聚类中心的那一类)
            distances = np.linalg.norm(self.X_train[:, np.newaxis, :] - centers[np.newaxis, :, :], axis=2)
            # 找到每个数据点最近的聚类中心（更新标签）
            labels = np.argmin(distances, axis=1)
            # 重置簇的掩码矩阵
            mask = np.zeros((self.X_train.shape[0], self.n_clusters), dtype=int)
            # 更新掩码矩阵，簇的掩码矩阵以one-hot编码
            mask[np.arange(self.X_train.shape[0]), labels] = 1
            # 计算新的惯性值（即总距离的和，以判别聚类质量）
            new_inertia = np.sum(distances * mask)
            # 画图展示聚类效果
            if self.show:
                self.plot_cluster(labels, centers, pause=True, n_iter=i + 1)
            # 若两次迭代惯性值的差小于容忍值则说明已经收敛
            if np.abs(inertia - new_inertia) < self.tol:
                break
            else:
                inertia = new_inertia
        return labels, centers, inertia

    def init_centers(self):
        """初始化聚类中心"""
        if self.init_func == 'k-means++':
            # 使用k-means++初始化聚类中心
            # 随机选择第一个簇中心
            centers = [self.X_train[np.random.choice(self.X_train.shape[0])]]
            for _ in range(1, self.n_clusters):
                # 计算每个点到已选择簇中心的最短距离
                dist_matrix = np.linalg.norm(self.X_train[:, np.newaxis] - centers, axis=2)
                min_distances = np.min(dist_matrix, axis=1)
                # 计算选择下一个簇中心的概率（以距离的平方计算）
                probabilities = min_distances ** 2
                probabilities /= probabilities.sum()
                # 根据概率选择下一个簇中心（计算累积概率）
                cumulative_probabilities = np.cumsum(probabilities)
                r = np.random.rand()
                # 找到该随机值应该插入到已排序数组（累积概率向量）中的位置
                next_center_idx = np.searchsorted(cumulative_probabilities, r)
                centers.append(self.X_train[next_center_idx])
            centers = np.array(centers)
        elif self.init_func == 'random':
            # 随机初始化聚类中心
            centers = np.random.uniform(np.min(self.X_train), np.max(self.X_train),
                                        size=(self.n_clusters, self.X_train.shape[1]))
        else:
            raise ValueError(f"Unsupported init_func: {self.init_func}, there is no such init_func type")
        return centers

    def plot_cluster(self, labels=None, centers=None, pause=False, n_iter=None, pause_time=0.1):
        if labels is None or centers is None:
            labels = self.labels
            centers = self.centers
        plot_cluster(self.X_train, labels, centers, pause, n_iter, pause_time)


if __name__ == '__main__':
    np.random.seed(100)
    model = KMeans(n_clusters=5, show=True)
    run_blobs_cluster(model)
    model = KMeans(n_clusters=2, show=True)
    run_circle_cluster(model)
    run_moons_cluster(model)
