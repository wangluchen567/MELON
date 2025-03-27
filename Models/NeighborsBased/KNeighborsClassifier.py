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
from Models.Utils import plot_2dim_classification, run_uniform_classification, run_double_classification


class KNeighborsClassifier(Model):
    def __init__(self, X_train=None, Y_train=None, n_neighbors=5, weights='uniform', metric='minkowski', p=2):
        """
        K-近邻分类模型
        :param X_train: 训练数据
        :param Y_train: 真实标签
        :param n_neighbors: 计算最近邻时数量
        :param weights: 计算邻居时的权重类型(uniform/distance)
        :param metric: 计算最近邻时的距离度量(minkowski:闵可夫斯基距离, manhattan:曼哈顿距离, euclidean:欧几里得距离)
        :param p: 计算minkowski距离的幂次，(p=1:曼哈顿距离，p=2:欧式距离)
        """
        super().__init__(X_train, Y_train)
        self.n_neighbors = n_neighbors  # 计算最近邻时数量
        self.weights = weights  # 计算邻居时的权重类型(uniform/distance)
        self.metric = metric  # 计算最近邻时的距离度量
        self.p = p  # 计算minkowski距离的幂次，(p=1:曼哈顿距离，p=2:欧式距离)

    def train(self, X_train=None, Y_train=None):
        """使用数据集训练模型"""
        self.set_train_data(X_train, Y_train)

    def predict(self, X_data):
        """模型对测试集进行预测"""
        num_data = len(X_data)
        # 初始化预测值
        Y_data = np.zeros((num_data, 1), dtype=self.Y_train.dtype)
        for i in range(num_data):
            # 计算每个样本与所有训练样本之间的距离
            dists = self.cal_dist(X_data[i])
            # 得到距离最近的n_neighbors个样本的索引
            nearest_indices = np.argsort(dists)[:self.n_neighbors]
            # 获取这些最近邻的标签
            nearest_labels = self.Y_train[nearest_indices]
            # 根据权重参数得到最终标签结果
            if self.weights == 'uniform':
                # 均匀权重：简单投票
                # 使用 np.unique 统计每个标签的出现次数，并找到出现次数最多的标签
                labels, counts = np.unique(nearest_labels, return_counts=True)
                Y_data[i] = labels[np.argmax(counts)]
            elif self.weights == 'distance':
                # 距离加权：加权投票
                weights = 1.0 / (dists[nearest_indices] + 1e-9)  # 防止除以零
                unique_labels, inverse_indices = np.unique(nearest_labels, return_inverse=True)
                weighted_counts = np.bincount(inverse_indices, weights=weights)
                Y_data[i] = unique_labels[np.argmax(weighted_counts)]
            else:
                raise ValueError(f"Unsupported weights: {self.weights}")
        return Y_data

    def cal_dist(self, x):
        """计算当前样本与储存的训练样本之间的距离"""
        if (self.metric == 'manhattan') or (self.metric == 'minkowski' and self.p == 1):
            return np.sum(np.abs(self.X_train - x), axis=1)
        elif (self.metric == 'euclidean') or (self.metric == 'minkowski' and self.p == 2):
            return np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))
        else:
            raise ValueError(f"Unsupported metric: {self.metric}")

    def plot_2dim(self, X_test=None, Y_test=None, Truth=None, pause=False):
        """为二维分类数据集和结果画图"""
        plot_2dim_classification(self.X_train, self.Y_train, None, X_test, Y_test, Truth=Truth, pause=pause)


if __name__ == '__main__':
    np.random.seed(100)
    model = KNeighborsClassifier()
    run_uniform_classification(model)
    run_double_classification(model)
