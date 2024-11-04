"""
感知机
Perceptron
"""
import warnings
import numpy as np
from Models.GradientOptimizer.Optimizer import GradientDescent, Momentum, AdaGrad, RMSProp, Adam
from Models.Utils import sigmoid, plot_2dim_classification, run_uniform_classification, run_double_classification


class Perceptron():
    def __init__(self, X_train=None, Y_train=None, epochs=50, lr=0.01, grad_type='Adam', show=True):
        self.X_train = None  # 训练数据
        self.Y_train = None  # 真实标签
        self.set_train_data(X_train, Y_train)
        self.Weights = None  # 模型参数
        self.Grad = 0  # 模型梯度
        self.optimizer = None  # 初始化优化器
        # 若使用梯度方法则记录参数变化情况
        self.history = []
        # 使用梯度方法求解时需要指定优化步数
        self.epochs = epochs
        # 指定优化的学习率
        self.lr = lr
        # 梯度法类型
        self.grad_type = grad_type
        # 是否展示迭代过程
        self.show = show

    def set_train_data(self, X_train, Y_train):
        """给定训练数据集和标签数据"""
        if X_train is not None:
            if self.X_train is not None:
                warnings.warn("Training data will be overwritten")
            self.X_train = X_train.copy()
        if Y_train is not None:
            if self.Y_train is not None:
                warnings.warn("Training data will be overwritten")
            self.Y_train = Y_train.copy()

    def set_parameters(self, epochs=None, lr=None, grad_type=None):
        """重新修改相关参数"""
        if self.epochs is not None and epochs is not None:
            warnings.warn("Parameter 'epochs' be overwritten")
            self.epochs = epochs
        if self.lr is not None and lr is not None:
            warnings.warn("Parameters 'lr' be overwritten")
            self.lr = lr
        if self.grad_type is not None and grad_type is not None:
            warnings.warn("Parameters 'grad_type' be overwritten")
            self.grad_type = grad_type

    def get_optimizer(self):
        """获取优化器"""
        dict = {'GD': GradientDescent, 'Momentum': Momentum, 'AdaGrad': AdaGrad, 'RMSProp': RMSProp, 'Adam': Adam}
        self.optimizer = dict[self.grad_type](self, self.lr)

    def init_weights(self):
        """初始化参数 (权重)"""
        X_feat = self.X_train.shape[1]
        self.Weights = np.random.uniform(-1, 1, size=(X_feat + 1, 1))

    def train(self, X_train=None, Y_train=None, epochs=None, lr=None, grad_type=None):
        """使用数据集训练模型"""
        self.set_train_data(X_train, Y_train)
        self.set_parameters(epochs, lr, grad_type)
        self.train_grad()

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

    def train_grad(self):
        """使用梯度下降方法进行优化"""
        self.init_weights()
        self.get_optimizer()
        for i in range(self.epochs):
            self.cal_grad()
            self.optimizer.step()
            self.history.append(self.Weights)
            if self.show:
                self.plot_2dim(pause=True, n_iter=i + 1)

    def cal_grad(self):
        """计算梯度值"""
        # 在数据最后一列添加一列单位矩阵作为转置b
        X_B = np.concatenate((self.X_train, np.ones((len(self.X_train), 1))), axis=1)
        ErrorPos = self.Y_train * X_B.dot(self.Weights) < 0
        self.Grad = np.sum((-1 * self.Y_train * X_B)[ErrorPos.flatten()], axis=0).reshape(-1, 1) / len(ErrorPos)

    def plot_2dim(self, X_test=None, Y_test=None, Truth=None, pause=False, n_iter=None):
        """为二维分类数据集和结果画图"""
        plot_2dim_classification(self.X_train, self.Y_train, self.Weights, X_test, Y_test,
                                 Truth=Truth, pause=pause, n_iter=n_iter)


if __name__ == '__main__':
    np.random.seed(100)
    model = Perceptron(epochs=50, lr=0.1, grad_type='Adam')
    run_uniform_classification(model, train_ratio=0.8)
    run_double_classification(model, train_ratio=0.8)
