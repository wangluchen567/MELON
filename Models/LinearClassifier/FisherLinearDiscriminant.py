"""
Fisher线性判别分析
Fisher Linear Discriminant
"""
import warnings
import numpy as np
from Models.Utils import plat_2dim_classification, run_uniform_classification, run_double_classification


class FisherLinearDiscriminant():
    def __init__(self, X_train=None, Y_train=None):
        self.X_train = None  # 训练数据
        self.Y_train = None  # 真实标签
        self.set_train_data(X_train, Y_train)
        self.Weights = None  # 模型参数

    def set_train_data(self, X_train, Y_train):
        """给定训练数据集和标签数据"""
        if any(var is not None for var in [self.X_train, self.Y_train]):
            warnings.warn("Training data will be overwritten")
        if X_train is not None:
            self.X_train = X_train.copy()
        if Y_train is not None:
            self.Y_train = Y_train.copy()

    def train(self, X_train=None, Y_train=None):
        """使用数据集训练模型"""
        self.set_train_data(X_train, Y_train)
        # 标签展开，方便取值
        Y_F = self.Y_train.flatten()
        # 求两类样本的个数
        N1, N2 = sum(Y_F == 1), sum(Y_F == -1)
        # 求两类样本的均值
        M1 = np.mean(self.X_train[Y_F == 1], axis=0)
        M2 = np.mean(self.X_train[Y_F == -1], axis=0)
        # 求两类样本的协方差
        S1 = (self.X_train[Y_F == 1] - M1).T.dot((self.X_train[Y_F == 1] - M1))
        S2 = (self.X_train[Y_F == -1] - M2).T.dot((self.X_train[Y_F == -1] - M2))
        # 求最优投影方向
        Vec = np.linalg.inv(S1 + S2).dot(M1 - M2)
        # 求判别函数阈值
        T = Vec.dot((N1 * M1 + N2 * M2)) / (N1 + N2)
        Bias = -T  # 判别阈值在判别时移项后变号
        self.Weights = np.concatenate((Vec, np.array([Bias]))).reshape(-1, 1)

    def predict(self, X_data):
        """模型对测试集进行预测"""
        if X_data.ndim == 2:
            pass
        elif X_data.ndim == 1:
            X_data = X_data.reshape(1, -1)
        else:
            raise ValueError("Cannot handle data with a shape of 3 dimensions or more")
        X_B = np.concatenate((X_data, np.ones((len(X_data), 1))), axis=1)
        Y_data = np.ones((len(X_data), 1))
        Y_data[X_B.dot(self.Weights) < 0] = -1
        return Y_data

    def plat_2dim(self, X_data=None, Y_data=None, Truth=None, pause=False):
        """为二维分类数据集和结果画图"""
        plat_2dim_classification(self.X_train, self.Y_train, self.Weights, X_data, Y_data, Truth=Truth, pause=pause)


if __name__ == '__main__':
    model = FisherLinearDiscriminant()
    run_uniform_classification(model, num_predict=10)
    run_double_classification(model, num_predict=10)
