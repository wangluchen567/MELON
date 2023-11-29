import numpy as np
import matplotlib.pyplot as plt


class FisherLinearDiscriminant():
    def __init__(self, X_train=None, Y_train=None):
        self.X_train = X_train  # 训练数据
        self.Y_train = Y_train  # 真实标签
        self.Weights = None  # 模型参数

    def get_data(self, X_train, Y_train):
        self.X_train = X_train
        self.Y_train = Y_train

    def train(self, X_train, Y_train):
        self.X_train = X_train
        self.Y_train = Y_train
        # 标签展开，方便取值
        Y_F = self.Y_train.flatten()
        # 求两类样本的个数
        N1, N2 = len(Y_F == 1), len(Y_F == -1)
        # 求两类样本的均值
        M1 = np.mean(self.X_train[Y_F == 1], axis=0)
        M2 = np.mean(self.X_train[Y_F == -1], axis=0)
        # 求两类样本的协方差
        S1 = (self.X_train[Y_F == 1] - M1).T.dot((self.X_train[Y_F == 1] - M1))
        S2 = (self.X_train[Y_F == -1] - M2).T.dot((self.X_train[Y_F == -1] - M2))
        # 求最优投影方向
        Vec = (S1 + S2).T.dot(M1 - M2)
        # 求判别函数阈值
        T = Vec.dot((N1 * M1 + N2 * M2)) / (N1 + N2)
        self.Weights = np.concatenate((Vec, np.array([T]))).reshape(1, -1)
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

    @staticmethod
    def get_PXU(X, Weights, ratio=0.1, step=0.1):
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
    model = FisherLinearDiscriminant()
    X_train, Y_train, TruthWeights = model.random_generate(100)
    model.get_data(X_train, Y_train)
    model.train(X_train, Y_train)
    print(TruthWeights)
    print(model.Weights)
    model.plat_2D(TruthWeights)
