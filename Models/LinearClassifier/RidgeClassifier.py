"""
岭回归分类器
Ridge Classifier
"""
import warnings
import numpy as np
from Models.Utils import sigmoid, plot_2dim_classification, run_uniform_classification, run_double_classification


class RidgeClassifier():
    def __init__(self, X_train=None, Y_train=None, alpha=1.0):
        """
        :param X_train: 训练数据
        :param Y_train: 真实标签
        :param alpha: 正则化系数
        """
        self.X_train = None  # 训练数据
        self.Y_train = None  # 真实标签
        self.set_train_data(X_train, Y_train)
        self.alpha = alpha  # 正则化系数
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

    def set_parameters(self, alpha=None):
        """重新修改相关参数"""
        parameters = ['alpha']
        values = [alpha]
        for param, value in zip(parameters, values):
            if value is not None and getattr(self, param) is not None:
                warnings.warn(f"Parameter '{param}' will be overwritten")
                setattr(self, param, value)

    def train(self, X_train=None, Y_train=None, alpha=None):
        """使用数据集训练模型"""
        self.set_train_data(X_train, Y_train)
        self.set_parameters(alpha)
        # 在数据最后一列添加一列单位矩阵作为偏置b
        X_B = np.concatenate((self.X_train, np.ones((len(self.X_train), 1))), axis=1)
        # 使用公式计算参数
        # 公式: W = (XT @ X + alpha * I) ^ -1 @ XT @ Y
        self.Weights = np.linalg.inv(X_B.T.dot(X_B) + self.alpha * np.eye(X_B.shape[1])).dot(X_B.T).dot(self.Y_train)

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
        """模型对测试集进行预测"""
        if X_data.ndim == 2:
            pass
        elif X_data.ndim == 1:
            X_data = X_data.reshape(1, -1)
        else:
            raise ValueError("Cannot handle data with a shape of 3 dimensions or more")
        X_B = np.concatenate((X_data, np.ones((len(X_data), 1))), axis=1)
        Y_data_prob = sigmoid(X_B.dot(self.Weights))
        return Y_data_prob

    def plot_2dim(self, X_test=None, Y_test=None, Truth=None, pause=False, n_iter=None):
        """为二维分类数据集和结果画图"""
        plot_2dim_classification(self.X_train, self.Y_train, self.Weights, X_test, Y_test,
                                 Truth=Truth, pause=pause, n_iter=n_iter)

if __name__ == '__main__':
    np.random.seed(100)
    model = RidgeClassifier()
    run_uniform_classification(model)
    run_double_classification(model)
