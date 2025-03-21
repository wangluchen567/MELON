"""
Copyright (c) 2023 LuChen Wang
[Software Name] is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan
PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY
KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
NON-INFRINGEMENT, MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details.
"""
import warnings
import numpy as np
from Models.Utils import softmax, plot_2dim_classification, run_uniform_classification, run_double_classification


class LinearDiscriminantAnalysis():
    def __init__(self, X_train=None, Y_train=None, priors=None, n_components=None):
        """
        线性判别分析
        :param X_train: 训练数据
        :param Y_train: 真实标签
        :param priors: 每类数据的先验概率
        :param n_components: 降维后的特征数量
        """
        self.X_train = None  # 训练数据
        self.Y_train = None  # 真实标签
        self.set_train_data(X_train, Y_train)
        self.priors = priors  # 每类数据的先验概率
        self.n_components = n_components  # 降维后的特征数量
        self.classes = None  # 保存训练数据的类别
        self.projection = None  # 投影参数
        self.shared_cov = None  # 共享协方差矩阵
        self.class_means = None  # 每类的均值
        self.priors = None  # 每类的先验概率

    def set_train_data(self, X_train, Y_train):
        """给定训练数据集和标签数据"""
        if X_train is not None:
            if self.X_train is not None:
                warnings.warn("Training data will be overwritten")
            self.X_train = X_train.copy()
        if Y_train is not None:
            if self.Y_train is not None:
                warnings.warn("Training label will be overwritten")
            self.Y_train = Y_train.copy()

    def set_parameters(self, **kwargs):
        """重新修改相关参数"""
        for param, value in kwargs.items():
            if hasattr(self, param):  # 检查对象是否有该属性
                if getattr(self, param) is not None:
                    warnings.warn(f"Parameter '{param}' will be overwritten")
                setattr(self, param, value)
            else:
                warnings.warn(f"Parameter '{param}' is not a valid parameter for this model")

    def train(self, X_train=None, Y_train=None):
        """使用数据集训练模型"""
        self.set_train_data(X_train, Y_train)
        self.classes = np.unique(self.Y_train)  # 获取类别
        num_data, num_feat = self.X_train.shape  # 数据与特征数量
        num_class = len(self.classes)  # 类别数量
        # 初始化降维后的特征数量
        if self.n_components is None:
            self.n_components = min(num_class - 1, num_feat)
        # 初始化类内散度矩阵 S_W 和类间散度矩阵 S_B
        S_W = np.zeros((num_feat, num_feat))
        S_B = np.zeros((num_feat, num_feat))
        # 计算总体的均值
        mean_all = np.mean(self.X_train, axis=0)
        # 计算类内散度矩阵 S_W 和类间散度矩阵 S_B
        for k in self.classes:
            X_k = self.X_train[np.array(self.Y_train == k).flatten()]
            mean_k = np.mean(X_k, axis=0)
            S_W += (X_k - mean_k).T @ (X_k - mean_k)
            mean_diff = (mean_k - mean_all).reshape(-1, 1)
            S_B += len(X_k) * mean_diff @ mean_diff.T
        # 求解广义特征值问题得到投影方向
        S_W_inv_S_B = np.linalg.inv(S_W) @ S_B
        # 特征值分解得到投影方向
        eigen_vals, eigen_vecs = np.linalg.eig(S_W_inv_S_B)
        # 提取前n个特征作为投影(判别)方向
        if self.n_components == 1:
            # 只有一个判别方向
            self.projection = eigen_vecs[:, 0].reshape(-1, 1)
            # 计算投影后的类别均值和共享协方差矩阵
            X_proj = self.X_train @ self.projection
            # 此时协方差矩阵是一个标量（方差），需要转换为1x1矩阵
            self.shared_cov = np.array([[np.cov(X_proj, rowvar=False)]])
        else:
            # 提取前 K-1 个特征向量
            self.projection = eigen_vecs[:, :self.n_components]
            # 计算投影后的类别均值和共享协方差矩阵
            X_proj = self.X_train @ self.projection
            # 计算共享协方差矩阵
            self.shared_cov = np.cov(X_proj, rowvar=False)
        # 计算每类的均值
        self.class_means = np.array([np.mean(X_proj[np.array(self.Y_train == k).flatten()], axis=0)
                                     for k in self.classes])
        # 计算类别的先验概率
        self.priors = np.array([np.mean(np.array(self.Y_train == k).flatten()) for k in self.classes])

    def predict(self, X_data):
        """模型对测试集进行预测"""
        if X_data.ndim == 2:
            pass
        elif X_data.ndim == 1:
            X_data = X_data.reshape(1, -1)
        else:
            raise ValueError("Cannot handle data with a shape of 3 dimensions or more")
        # 对测试数据进行预测概率
        Y_data_prob = self.predict_prob(X_data)
        # 选择概率最大的类别作为输出类别
        Y_data = self.classes[np.argmax(Y_data_prob, axis=1)]
        Y_data = Y_data.reshape(-1, 1)
        return Y_data

    def predict_prob(self, X_data):
        """模型对测试集进行预测(概率)"""
        if X_data.ndim == 2:
            pass
        elif X_data.ndim == 1:
            X_data = X_data.reshape(1, -1)
        else:
            raise ValueError("Cannot handle data with a shape of 3 dimensions or more")
        # 将数据投影到判别方向
        X_proj = X_data @ self.projection
        # 计算每个类别的对数后验概率
        log_posteriors = []
        for i, mean in enumerate(self.class_means):
            diff = X_proj - mean
            inv_cov = np.linalg.inv(self.shared_cov)
            log_likelihood = -0.5 * np.sum(diff @ inv_cov * diff, axis=1)  # 计算对数似然
            log_posterior = log_likelihood + np.log(self.priors[i])  # 计算对数后验
            log_posteriors.append(log_posterior)
        log_posteriors = np.array(log_posteriors).T
        # 使用softmax归一化后验概率
        Y_data_prob = softmax(log_posteriors)
        return Y_data_prob

    def plot_2dim(self, X_test=None, Y_test=None, Truth=None, pause=False):
        """为二维分类数据集和结果画图"""
        plot_2dim_classification(self.X_train, self.Y_train, None, X_test, Y_test, Truth=Truth, pause=pause)


if __name__ == '__main__':
    np.random.seed(100)
    model = LinearDiscriminantAnalysis(n_components=2)
    run_uniform_classification(model)
    run_double_classification(model)
