"""
感知机
Perceptron
"""
import numpy as np
import matplotlib.pyplot as plt

class Perceptron():
    def __init__(self, X_train=None, Y_train=None, grad_type='SGD'):
        self.X_train = X_train  # 训练数据
        self.Y_train = Y_train  # 真实标签
        self.Weights = None  # 模型参数
        # 梯度法类型
        self.grad_type = grad_type

    def get_data(self, X_train, Y_train):
        self.X_train = X_train
        self.Y_train = Y_train

    def train(self, X_train, Y_train, epochs=30):
        self.X_train = np.concatenate((X_train, np.ones((len(X_train), 1))), axis=1)
        self.Y_train = Y_train
        learning_rate = 1
        self.Weights = np.random.uniform(-1, 1, size=(1, self.X_train.shape[1]))
        for i in range(epochs):
            # 分类错误的才有损失，先取出分类错误的
            ErrorPos = self.Y_train * self.X_train.dot(self.Weights.T) < 0
            grad = np.sum((-1 * self.Y_train * self.X_train)[ErrorPos.flatten()], axis=0) / len(ErrorPos)
            self.Weights -= learning_rate * grad
            self.plat_2D(pause=True)

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
        :return: 训练数据
        """
        X_train = np.random.uniform(X_lower, X_upper, size=(X_size, X_feat))
        X_mids = (np.max(X_train, axis=0) + np.min(X_train, axis=0)) / 2
        TruthWeights = np.random.uniform(lower, upper, size=(1, X_feat))
        Bias = -TruthWeights.dot(X_mids.reshape(-1, 1))
        TruthWeights = np.concatenate((TruthWeights, Bias), axis=1)
        X_B = np.concatenate((X_train, np.ones((len(X_train), 1))), axis=1)
        Y_train = np.ones((len(X_train), 1))
        Y_train[X_B.dot(TruthWeights.T) < 0] = -1
        return X_train, Y_train, TruthWeights

    def plat_2D(self, Truth=None, pause=False):
        X = self.X_train
        Y = self.Y_train
        Predict = self.Weights
        if not pause: plt.figure()
        plt.clf()
        plt.scatter(X[Y.flatten() == 1, 0], X[Y.flatten() == 1, 1], c='red')
        plt.scatter(X[Y.flatten() == -1, 0], X[Y.flatten() == -1, 1], c='blue')
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
        W = np.concatenate((Weights[:, :-2], Weights[:, -1].reshape(-1, 1)), axis=1) / -Weights[:, -2]
        PU = PX_B.dot(W.T)
        return PX, PU


if __name__ == '__main__':
    model = Perceptron()
    X_train, Y_train, TruthWeights = model.random_generate(100)
    model.get_data(X_train, Y_train)
    model.train(X_train, Y_train)
    print(TruthWeights)
    print(model.Weights)
    model.plat_2D(TruthWeights)