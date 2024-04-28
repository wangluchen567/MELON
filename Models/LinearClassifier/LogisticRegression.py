"""
逻辑回归
Logistic Regression
"""
import numpy as np
import matplotlib.pyplot as plt
from Models.GradientOptimizer.Optimizer import *


class LogisticRegression():
    def __init__(self, X_train=None, Y_train=None, mode=0, epochs=30, lr=0.01, grad_type='Adam'):
        self.X_train = X_train  # 训练数据
        self.Y_train = Y_train  # 真实标签
        self.Weights = None  # 模型参数
        self.Grad = 0  # 模型梯度
        # 求解模式，0为默认直接求解，否则使用梯度法
        self.mode = mode
        # 若使用梯度方法则记录参数变化情况
        self.history = []
        # 使用梯度方法求解时需要指定优化步数
        self.epochs = epochs
        # 指定优化的学习率
        self.lr = lr
        # 梯度法类型
        self.grad_type = grad_type

    def get_data(self, X_train, Y_train):
        self.X_train = X_train
        self.Y_train = Y_train

    def get_optimizer(self):
        """获取优化器"""
        dict = {'GD': GradientDescent, 'Momentum': Momentum, 'AdaGrad': AdaGrad, 'RMSProp': RMSProp, 'Adam': Adam}
        self.optimizer = dict[self.grad_type](self, self.lr)

    def init_weights(self):
        X_feat = self.X_train.shape[1]
        self.Weights = np.random.uniform(-1, 1, size=(X_feat + 1, 1))

    def train(self, X_train, Y_train, mode=0, epochs=30, lr=0.01, grad_type='Adam'):
        self.X_train = X_train
        self.Y_train = Y_train
        self.mode = mode
        self.epochs = epochs
        self.lr = lr
        self.grad_type = grad_type
        self.train_grad()

    def cal_grad(self):
        """计算梯度值"""
        # 在数据最后一列添加一列单位矩阵作为转置b
        X_B = np.concatenate((self.X_train, np.ones((len(self.X_train), 1))), axis=1)
        self.Grad = X_B.T @ (self.sigmoid(X_B @ self.Weights) - self.Y_train) / len(self.X_train)

    def train_grad(self):
        """使用梯度下降方法进行优化"""
        self.init_weights()
        self.get_optimizer()
        for i in range(self.epochs):
            self.cal_grad()
            self.optimizer.step()
            self.history.append(self.Weights)
            self.plat_2D(pause=True, iter=i + 1)

    @staticmethod
    def sigmoid(x):
        # 在求解指数过大的指数函数时防止溢出
        indices_pos = np.nonzero(x >= 0)
        indices_neg = np.nonzero(x < 0)
        y = np.zeros_like(x)
        # y = 1 / (1 + exp(-x)), x >= 0
        # y = exp(x) / (1 + exp(x)), x < 0
        y[indices_pos] = 1 / (1 + np.exp(-x[indices_pos]))
        y[indices_neg] = np.exp(x[indices_neg]) / (1 + np.exp(x[indices_neg]))
        return y

    @staticmethod
    def random_generate(X_size, X_feat=2, X_lower=-1, X_upper=1, lower=-1, upper=1):
        """
        随机生成数据
        :param X_size: 数据集大小
        :param X_feat: 数据集特征数
        :param X_lower: 随机生成的数据的下界
        :param X_upper: 随机生成的数据的上界
        :param lower: 随机生成的范围最小值
        :param upper: 随机生成的范围最大值
        :return: 训练数据和真实参数
        """
        X_train = np.random.uniform(X_lower, X_upper, size=(X_size, X_feat))
        X_mids = (np.max(X_train, axis=0) + np.min(X_train, axis=0)) / 2
        TruthWeights = np.random.uniform(lower, upper, size=(X_feat, 1))
        Bias = - X_mids.reshape(1, -1) @ TruthWeights
        TruthWeights = np.concatenate((TruthWeights, Bias), axis=0)
        X_B = np.concatenate((X_train, np.ones((len(X_train), 1))), axis=1)
        Y_train = np.ones((len(X_train), 1))
        """注意！！！使用逻辑回归时负类标签为 0 ！"""
        # Y_train[X_B.dot(TruthWeights) < 0] = -1
        Y_train[X_B.dot(TruthWeights) < 0] = 0
        return X_train, Y_train, TruthWeights

    def plat_2D(self, Truth=None, pause=False, iter=None):
        X = self.X_train
        Y = self.Y_train
        Predict = self.Weights
        if not pause: plt.figure()
        plt.clf()
        plt.scatter(X[Y.flatten() == 1, 0], X[Y.flatten() == 1, 1], c='red')
        """注意！！！使用逻辑回归时负类标签为 0 ！"""
        # plt.scatter(X[Y.flatten() == -1, 0], X[Y.flatten() == -1, 1], c='blue')
        plt.scatter(X[Y.flatten() == 0, 0], X[Y.flatten() == 0, 1], c='blue')
        if Truth is not None:
            # 绘制真实的参数
            PX, PU = self.get_PXU(X, Truth)
            plt.plot(PX, PU, c='orange', linewidth=5)
        if Predict is not None:
            # 绘制预测的参数
            PX, PU = self.get_PXU(X, Predict)
            plt.plot(PX, PU, c='red', linewidth=2)
            plt.xlim([-1, 1])
            plt.ylim([-1, 1])
        if pause:
            if iter:
                plt.title("iter: " + str(iter))
            plt.pause(0.3)
        else:
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
        gap = max(X[:, 0]) - min(X[:, 0])
        PX = np.arange(min(X[:, 0]) - ratio * gap, max(X[:, 0]) + ratio * gap, step)
        PX_B = np.concatenate((PX.reshape(-1, 1), np.ones((len(PX), 1))), axis=1)
        W = np.concatenate((Weights[:-2, :], Weights[-1, :].reshape(-1, 1)), axis=1) / -Weights[-2, :]
        PU = PX_B.dot(W.T)
        return PX, PU


if __name__ == '__main__':
    model = LogisticRegression()
    X_train, Y_train, TruthWeights = model.random_generate(X_size=100)
    model.get_data(X_train, Y_train)
    model.train(X_train, Y_train, epochs=50, lr=0.1, grad_type='Adam')
    print("Truth Weights: ", TruthWeights.T)
    print("Predict Weights: ", model.Weights.T)
    model.plat_2D(TruthWeights)
