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

    def train(self, X_train, Y_train, epochs=100):
        self.X_train = X_train
        self.Y_train = Y_train
        learning_rate = 1
        self.Weights = np.random.uniform(-1, 1, size=(X_train.shape[1], 1))
        # 分类错误的才有损失，先取出分类错误的
        ErrorPos = Y_train * X_train.dot(self.Weights) < 0
        for i in range(epochs):
            grad = np.sum((-1 * Y_train * X_train)[ErrorPos.flatten()], axis=0) / len(X_train)
            self.Weights -= learning_rate * grad.reshape(-1, 1)
            print(self.Weights)

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
        TruthWeights = np.random.uniform(lower, upper, size=(1, X_feat - 1))
        bias = X_mids[-1] - TruthWeights.dot(X_mids[:-1].reshape(-1, 1))
        # bias = X_mids[-1] - X_mids[:-1].reshape(1, -1).dot(TruthWeights)
        TruthWeights = np.concatenate((TruthWeights, bias), axis=1)
        X_B = np.concatenate((X_train[:, :-1], np.ones((len(X_train[:, :-1]), 1))), axis=1)
        Y_train = X_train[:, -1].reshape(-1, 1) - X_B.dot(TruthWeights.T) > 0

        return X_train, Y_train, TruthWeights

    def plat_2D(self, Truth=None):
        X = self.X_train
        Y = self.Y_train
        W = self.Weights
        plt.figure()
        plt.scatter(X[Y.flatten() == 0, 0], X[Y.flatten() == 0, 1], c='red')
        plt.scatter(X[Y.flatten() == 1, 0], X[Y.flatten() == 1, 1], c='blue')

        if Truth is not None:
            # 绘制真实的参数
            gap = max(X[:, 0]) - min(X[:, 0])
            PX = np.arange(min(X[:, 0]) - 0.1 * gap, max(X[:, 0]) + 0.1 * gap, 0.1)
            PX = PX.reshape(-1, 1)
            PX_B = np.concatenate((PX, np.ones((len(PX), 1))), axis=1)
            PY = PX_B.dot(Truth.T)
            plt.plot(PX, PY, c='orange', linewidth=6)
        if W is not None:
            # 绘制预测的参数
            gap = max(X) - min(X)
            PX = np.arange(min(X) - 0.1 * gap, max(X) + 0.1 * gap, 0.1)
            PX = PX.reshape(-1, 1)
            PX_B = np.concatenate((PX, np.ones((len(PX), 1))), axis=1)
            PY = PX_B.dot(W)
            plt.plot(PX, PY, c='orange', linewidth=5)
        plt.show()


if __name__ == '__main__':
    model = Perceptron()
    X_train, Y_train, TruthWeights = model.random_generate(100)
    model.get_data(X_train, Y_train)
    model.train(X_train, Y_train)
    print(TruthWeights)
    model.plat_2D(TruthWeights)