"""
逻辑回归
Logistic Regression
"""
import warnings
import numpy as np
from Models.GradientOptimizer.Optimizer import GradientDescent, Momentum, AdaGrad, RMSProp, Adam
from Models.Utils import plot_2dim_classification, run_uniform_classification, run_double_classification


class LogisticRegression():
    def __init__(self, X_train=None, Y_train=None, epochs=50, lr=0.01, grad_type='Adam'):
        self.X_train = None  # 训练数据
        self.Y_train = None  # 真实标签
        self.Y_train_ = None  # 逻辑回归特殊标签
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
            self.Y_train_ = Y_train.copy()
            # 使用逻辑回归时负类标签为0，在此使用特殊标签
            self.Y_train_[self.Y_train_ == -1] = 0

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
        Y_data[self.sigmoid(X_B.dot(self.Weights)) < 1 / 2] = -1
        return Y_data

    def cal_grad(self):
        """计算梯度值"""
        # 在数据最后一列添加一列单位矩阵作为转置b
        X_B = np.concatenate((self.X_train, np.ones((len(self.X_train), 1))), axis=1)
        # 这里使用的是特殊标签矩阵Y_train_ (0/1)
        self.Grad = X_B.T @ (self.sigmoid(X_B @ self.Weights) - self.Y_train_) / len(self.X_train)

    def train_grad(self):
        """使用梯度下降方法进行优化"""
        self.init_weights()
        self.get_optimizer()
        for i in range(self.epochs):
            self.cal_grad()
            self.optimizer.step()
            self.history.append(self.Weights)
            self.plot_2dim(pause=True, n_iter=i + 1)

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

    def plot_2dim(self, X_data=None, Y_data=None, Truth=None, pause=False, n_iter=None):
        """为二维分类数据集和结果画图"""
        plot_2dim_classification(self.X_train, self.Y_train, self.Weights, X_data, Y_data, Truth=Truth, pause=pause, n_iter=n_iter)


if __name__ == '__main__':
    model = LogisticRegression(epochs=50, lr=0.1, grad_type='Adam')
    run_uniform_classification(model, train_ratio=0.8)
    run_double_classification(model, train_ratio=0.8)
