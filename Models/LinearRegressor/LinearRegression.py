"""
线性回归
Linear Regression
"""
import numpy as np
import matplotlib.pyplot as plt


class LinearRegression():
    def __init__(self, X_train=None, Y_train=None, Lambda=0, mode=0, grad_type='Adam'):
        self.X_train = X_train  # 训练数据
        self.Y_train = Y_train  # 真实标签
        self.Lambda = Lambda  # 正则化系数
        self.Weights = None  # 模型参数
        # 求解模式，默认是直接求解
        self.mode = mode
        # 梯度法类型
        self.grad_type = grad_type

    def get_data(self, X_train, Y_train):
        self.X_train = X_train
        self.Y_train = Y_train

    def train(self, X_train, Y_train, Lambda=0, mode=0, grad_type='Adam'):
        self.X_train = X_train
        self.Y_train = Y_train
        self.Lambda = Lambda
        self.mode = mode
        self.grad_type = grad_type
        if self.mode:
            pass
        else:
            self.train_direct()

    def train_direct(self):
        # 在数据最后一列添加一列单位矩阵作为转置b
        X_B = np.concatenate((self.X_train, np.ones((len(self.X_train), 1))), axis=1)
        if self.Lambda:
            # 加入正则化项，求模型参数
            # 公式: (XT .* X + lambda * I) ^ -1 .* XT .* Y
            self.Weights = np.linalg.inv(X_B.T.dot(X_B) + self.Lambda * np.eye(X_B.shape[1])).dot(X_B.T).dot(
                self.Y_train)
        else:
            # 若没有正则化项，则直接利用伪逆求
            # 公式: (XT .* X) ^ -1 .* XT .* Y
            self.Weights = np.linalg.pinv(X_B).dot(self.Y_train)

    @staticmethod
    def random_generate(X_size, X_feat=1, X_lower=0, X_upper=20, lower=-1, upper=1, loc=0, scale=1):
        """
        随机生成数据
        :param X_size: 数据集大小
        :param X_feat: 数据集特征数
        :param X_lower: 随机生成的数据的下界
        :param X_upper: 随机生成的数据的上界
        :param lower: 随机生成的参数范围最小值
        :param upper: 随机生成的参数范围最大值
        :param loc: 扰动期望
        :param scale: 扰动方差
        :return: 训练数据与真实参数
        """
        X_train = np.random.uniform(X_lower, X_upper, size=(X_size, X_feat))
        # 在数据最后一列添加一列单位矩阵作为转置b
        X_B = np.concatenate((X_train, np.ones((len(X_train), 1))), axis=1)
        # 随机生成的模型参数
        Truth_Weights = np.random.uniform(lower, upper, size=(X_feat + 1, 1))
        # 计算输出值
        Y_train = X_B.dot(Truth_Weights)
        # 加入扰动，正态分布扰动
        Y_train += np.random.normal(loc, scale, size=Y_train.shape)
        return X_train, Y_train, Truth_Weights

    def plat_2D(self, Truth=None):
        X = self.X_train
        Y = self.Y_train
        Predict = self.Weights
        plt.figure()
        plt.scatter(X, Y, c='blue')
        if Truth is not None:
            # 绘制真实的参数
            PX, PU = self.get_PXU(X, Truth)
            plt.plot(PX, PU, c='orange', linewidth=5)
        if Predict is not None:
            # 绘制预测的参数
            PX, PU = self.get_PXU(X, Predict)
            plt.plot(PX, PU, c='red', linewidth=2)
        plt.show()

    def get_PXU(self, X, Weights, ratio=0.1, step=0.1):
        """
        获取画图使用的X和其他未知变量
        :param X: 要画图的已知变量
        :param Weights: 要画图的权重
        :param ratio: 两边伸展的额外比例
        :param step: 采样频率
        :return:
        """
        gap = max(X) - min(X)
        PX = np.arange(min(X) - ratio * gap, max(X) + ratio * gap, step)
        PX_B = np.concatenate((PX.reshape(-1, 1), np.ones((len(PX), 1))), axis=1)
        PU = PX_B.dot(Weights)
        return PX, PU



if __name__ == '__main__':
    model = LinearRegression()
    X_train, Y_train, Truth_Weights = model.random_generate(X_size=100)
    model.get_data(X_train, Y_train)
    model.train(X_train, Y_train)
    print(Truth_Weights)
    print(model.Weights)
    model.plat_2D(Truth=Truth_Weights)
