"""
Fisher(线性)判别分析(二分类)
Fisher (Linear) Discriminant Analysis
"""
import warnings
import numpy as np
from Models.Utils import sigmoid, plot_2dim_classification, run_uniform_classification, run_double_classification


class FisherDiscriminantAnalysis():
    def __init__(self, X_train=None, Y_train=None):
        """
        :param X_train: 训练数据
        :param Y_train: 真实标签
        """
        self.X_train = None  # 训练数据
        self.Y_train = None  # 真实标签
        self.set_train_data(X_train, Y_train)
        self.Weights = None  # 模型参数

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

    def train(self, X_train=None, Y_train=None):
        """使用数据集训练模型"""
        self.set_train_data(X_train, Y_train)
        # 标签展开，方便取值
        y_flatten = self.Y_train.flatten()
        # 获取两类样本的个数
        num_pos, num_neg = sum(y_flatten == 1), sum(y_flatten == -1)
        # 求两类样本的均值
        mu_pos = np.mean(self.X_train[y_flatten == 1], axis=0)
        mu_neg = np.mean(self.X_train[y_flatten == -1], axis=0)
        # 求两类样本的协方差
        cov_pos = (self.X_train[y_flatten == 1] - mu_pos).T.dot((self.X_train[y_flatten == 1] - mu_pos))
        cov_neg = (self.X_train[y_flatten == -1] - mu_neg).T.dot((self.X_train[y_flatten == -1] - mu_neg))
        # 计算两类的共享协方差矩阵(无偏估计)
        sigma = (cov_pos + cov_neg) / (num_pos + num_neg - 2)
        # 求共享协方差矩阵的逆矩阵
        sigma_inv = np.linalg.inv(sigma)
        # 求最优投影方向
        vector = sigma_inv.dot(mu_pos - mu_neg)
        # 求判别函数阈值(直接取中点)
        Bias = -vector.dot((num_pos * mu_pos + num_neg * mu_neg)) / (num_pos + num_neg)
        self.Weights = np.concatenate((vector, np.array([Bias]))).reshape(-1, 1)

    def predict(self, X_data):
        """模型对测试集进行预测"""
        if X_data.ndim == 2:
            pass
        elif X_data.ndim == 1:
            X_data = X_data.reshape(1, -1)
        else:
            raise ValueError("Cannot handle data with a shape of 3 dimensions or more")
        X_B = np.concatenate((X_data, np.ones((len(X_data), 1))), axis=1)
        Y_data = np.ones((len(X_data), 1), dtype=int)
        Y_data[X_B.dot(self.Weights) < 0] = -1
        return Y_data

    def predict_prob(self, X_data):
        """模型对测试集进行预测(概率)"""
        if X_data.ndim == 2:
            pass
        elif X_data.ndim == 1:
            X_data = X_data.reshape(1, -1)
        else:
            raise ValueError("Cannot handle data with a shape of 3 dimensions or more")
        X_B = np.concatenate((X_data, np.ones((len(X_data), 1))), axis=1)
        Y_data_prob = sigmoid(X_B.dot(self.Weights))
        return Y_data_prob

    def plot_2dim(self, X_test=None, Y_test=None, Truth=None, pause=False):
        """为二维分类数据集和结果画图"""
        plot_2dim_classification(self.X_train, self.Y_train, self.Weights, X_test, Y_test, Truth=Truth, pause=pause)


if __name__ == '__main__':
    np.random.seed(100)
    model = FisherDiscriminantAnalysis()
    run_uniform_classification(model)
    run_double_classification(model)
