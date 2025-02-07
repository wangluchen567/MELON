"""
线性回归
Linear Regression
"""
import warnings
import numpy as np
from Models.GradientOptimizer.Optimizer import GradientDescent, Momentum, AdaGrad, RMSProp, Adam
from Models.Utils import plot_2dim_regression, run_uniform_regression, run_contrast_regression


class LinearRegression():
    def __init__(self, X_train=None, Y_train=None, Lambda=0, mode=0, epochs=30, lr=0.01, grad_type='Adam', show=True):
        self.X_train = None  # 训练数据
        self.Y_train = None  # 真实标签
        self.set_train_data(X_train, Y_train)
        self.Lambda = Lambda  # 正则化系数
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
        # 是否展示迭代过程
        self.show = show

    def set_train_data(self, X_train, Y_train):
        """给定训练数据集和标签数据"""
        if any(var is not None for var in [self.X_train, self.Y_train]):
            warnings.warn("Training data will be overwritten")
        if X_train is not None:
            self.X_train = X_train.copy()
        if Y_train is not None:
            self.Y_train = Y_train.copy()

    def set_parameters(self, Lambda, mode, epochs, lr, grad_type):
        """重新修改相关参数"""
        if self.Lambda is not None and Lambda is not None:
            warnings.warn("Parameter 'Lambda' be overwritten")
            self.Lambda = Lambda
        if self.mode is not None and mode is not None:
            warnings.warn("Parameter 'mode' be overwritten")
            self.mode = mode
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

    def train(self, X_train, Y_train, Lambda=None, mode=None, epochs=None, lr=None, grad_type=None):
        """使用数据集训练模型"""
        self.set_train_data(X_train, Y_train)
        self.set_parameters(Lambda, mode, epochs, lr, grad_type)
        if self.mode:
            self.train_grad()
        else:
            self.train_direct()

    def predict(self, X_data):
        """模型对测试集进行预测"""
        if X_data.ndim == 2:
            pass
        elif X_data.ndim == 1:
            X_data = X_data.reshape(1, -1)
        else:
            raise ValueError("Cannot handle data with a shape of 3 dimensions or more")
        X_B = np.concatenate((X_data, np.ones((len(X_data), 1))), axis=1)
        Y_data = X_B.dot(self.Weights)
        return Y_data

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

    def cal_grad(self):
        """计算梯度值"""
        # 在数据最后一列添加一列单位矩阵作为转置b
        X_B = np.concatenate((self.X_train, np.ones((len(self.X_train), 1))), axis=1)
        self.Grad = X_B.T @ (X_B @ self.Weights - self.Y_train) / len(self.X_train)

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

    def plot_2dim(self, X_test=None, Y_test=None, Truth=None, pause=False, n_iter=None):
        """为二维回归数据集和结果画图"""
        plot_2dim_regression(self.X_train, self.Y_train, self.Weights, X_test, Y_test, Truth=Truth,
                             pause=pause, n_iter=n_iter)


if __name__ == '__main__':
    np.random.seed(100)
    model = LinearRegression()
    run_uniform_regression(model)
    run_contrast_regression(model)
