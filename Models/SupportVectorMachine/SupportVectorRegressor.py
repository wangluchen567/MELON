"""
支持向量机回归模型
Support Vector Regression
"""
import warnings
import numpy as np
from SequentialMinimalOptimization import smo_greedy_step_regression
from Models.Utils import (plot_2dim_regression, run_uniform_regression,
                          plot_2dim_regression_sample, run_circular_regression, run_poly_regression)

class SupportVectorRegressor():
    # 定义核函数类型
    LINEAR = 0
    POLY = 1
    RBF = GAUSSIAN = 2
    SIGMOID = 3

    def __init__(self, X_train=None, Y_train=None, C=10, tol=1.e-4, epsilon=0.6,
                 kernel_type=LINEAR, gamma=None, degree=3, const=1, num_iter=1000):
        self.X_train = None  # 训练数据
        self.Y_train = None  # 真实标签
        self.Y_train_ = None  # 逻辑回归特殊标签
        self.set_train_data(X_train, Y_train)
        self.Weights = None  # 模型参数
        self.kernel_mat = None  # 核函数矩阵
        self.alphas, self.b = None, None  # 乘子参数
        self.C = C  # 惩罚系数
        self.tol = tol  # 残差收敛条件（容忍系数）
        self.epsilon = epsilon  # 误差系数
        self.kernel_type = kernel_type  # 核函数类型
        self.gamma = gamma  # 核函数系数（乘数项）
        self.degree = degree  # 核函数系数（指数项）
        self.const = const  # 核函数系数（常数项）
        self.num_iter = num_iter  # 迭代优化次数

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

    def set_parameters(self, C=None, tol=None, epsilon=None, kernel_type=None,
                       gamma=None, degree=None, const=None, num_iter=None):
        """重新修改相关参数"""
        parameters = ['C', 'tol', 'epsilon', 'kernel_type', 'gamma', 'degree', 'const', 'num_iter']
        values = [C, tol, epsilon, kernel_type, gamma, degree, const, num_iter]
        for param, value in zip(parameters, values):
            if value is not None and getattr(self, param) is not None:
                warnings.warn(f"Parameter '{param}' will be overwritten")
                setattr(self, param, value)

    def cal_kernel_mat(self, X, Y):
        """给定数据计算核函数矩阵"""
        if self.kernel_type == self.LINEAR:
            return self.linear_kernel_mat(X, Y)
        elif self.kernel_type == self.POLY:
            return self.poly_kernel_mat(X, Y, self.gamma, self.degree, self.const)
        elif self.kernel_type == self.RBF or self.kernel_type == self.GAUSSIAN:
            return self.rbf_kernel_mat(X, Y, self.gamma)
        elif self.kernel_type == self.SIGMOID:
            return self.sigmoid_kernel_mat(X, Y, self.const)
        else:
            raise ValueError("There is no such kernel function type")

    @staticmethod
    def linear_kernel_mat(X, Y):
        kernel_mat = X @ Y.T
        return kernel_mat

    @staticmethod
    def poly_kernel_mat(X, Y, gamma=None, degree=3, const=1):
        """计算多项式核函数矩阵"""
        # 如果 gamma未指定，则设置为 1/特征数量
        if gamma is None:
            gamma = 1.0 / X.shape[1]
        # 使用多项式核函数公式计算核函数矩阵
        kernel_mat = (gamma * (X @ Y.T) + const) ** degree
        return kernel_mat

    @staticmethod
    def rbf_kernel_mat(X, Y, gamma=None):
        """计算径向基核（高斯核）函数矩阵"""
        # 如果 gamma 未指定，则设置为 1/特征数量
        if gamma is None:
            gamma = 1.0 / X.shape[1]
        # 使用 NumPy 的广播机制，计算核函数矩阵
        kernel_mat = np.exp(-gamma * np.sum((X[:, np.newaxis, :] - Y[np.newaxis, :, :]) ** 2, axis=2))
        return kernel_mat

    @staticmethod
    def sigmoid_kernel_mat(X, Y, gamma=None, const=1):
        """计算sigmoid核函数矩阵"""
        # 如果 gamma 未指定，则设置为 1/特征数量
        if gamma is None:
            gamma = 1.0 / X.shape[1]
        # 使用Sigmoid核函数公式计算核函数矩阵
        kernel_mat = np.tanh(gamma * (X @ Y.T) + const)
        return kernel_mat

    def init_weights(self):
        """初始化参数 (权重)"""
        X_feat = self.X_train.shape[1]
        self.Weights = np.random.uniform(-1, 1, size=(X_feat + 1, 1))

    def train(self, X_train=None, Y_train=None, C=None, tol=None, epsilon=None,
              kernel_type=None, gamma=None, degree=None, const=None, num_iter=None):
        """使用数据集训练模型"""
        self.set_train_data(X_train, Y_train)
        self.set_parameters(C, tol, epsilon, kernel_type, gamma, degree, const, num_iter)
        # 计算核函数矩阵
        self.kernel_mat = self.cal_kernel_mat(self.X_train, self.X_train)
        # 初始化权重参数
        self.init_weights()
        # 使用smo算法得到乘子参数和模型参数
        self.smo_algorithm()

    def predict(self, X_data):
        """模型对测试集进行预测"""
        if X_data.ndim == 2:
            pass
        elif X_data.ndim == 1:
            X_data = X_data.reshape(1, -1)
        else:
            raise ValueError("Cannot handle data with a shape of 3 dimensions or more")
        # 为简化运算，这里使用线性核函数时利用参数直接求结果
        if self.kernel_type == self.LINEAR:
            X_B = np.concatenate((X_data, np.ones((len(X_data), 1))), axis=1)
            Y_data = X_B.dot(self.Weights)
        else:  # 否则通过核函数求结果
            # 先得到非零的位置，以简化运算
            non_zeros = np.nonzero(np.sum(self.alphas, axis=1))[0]
            # 计算预测数据的核函数矩阵
            kernel_predict = self.cal_kernel_mat(self.X_train[non_zeros], X_data)
            Y_data = kernel_predict.T @ (self.alphas[non_zeros, 1] - self.alphas[non_zeros, 0])[:, np.newaxis] + self.b
        return Y_data

    def cal_weights(self):
        """计算参数 (权重)"""
        X_feat = self.X_train.shape[1]
        self.Weights = np.zeros((X_feat + 1, 1))
        self.Weights[:X_feat] = self.X_train.T @ (self.alphas[:, 1] - self.alphas[:, 0])[:, np.newaxis]
        self.Weights[-1] = self.b

    def smo_algorithm(self):
        """SMO算法"""
        # 初始化相关参数
        self.alphas, self.b, optimize_end = np.zeros((len(self.X_train), 2)), 0, False
        for i in range(self.num_iter):
            self.alphas, self.b, optimize_end \
                = smo_greedy_step_regression(self.kernel_mat, self.X_train, self.Y_train,
                                             self.alphas, self.b, self.C, self.epsilon, self.tol)
            # 若优化结束则跳出循环(没有可优化的项了)
            if optimize_end:
                print("The optimization has ended early, "
                      "and the number of iterations for this optimization is {}".format(i))
                break
            self.cal_weights()
            self.plot_2dim(pause=True, n_iter=i + 1)

    def plot_2dim(self, X_test=None, Y_test=None, Truth=None, pause=False, n_iter=None):
        """为二维回归数据集和结果画图"""
        if self.kernel_type == self.LINEAR and Truth is not None:
            plot_2dim_regression(self.X_train, self.Y_train, self.Weights, X_test, Y_test, Truth=Truth,
                                 support=(np.sum(self.alphas, axis=1) != 0.0), pause=pause, n_iter=n_iter)
        else:
            plot_2dim_regression_sample(self, self.X_train, self.Y_train, X_test, Y_test,
                                 support=(np.sum(self.alphas, axis=1) != 0.0), pause=pause, n_iter=n_iter)


if __name__ == '__main__':
    np.random.seed(100)
    model = SupportVectorRegressor(C=10, kernel_type=SupportVectorRegressor.RBF, num_iter=100)
    # run_uniform_regression(model, train_ratio=0.8)
    # run_poly_regression(model, train_ratio=0.8)
    run_circular_regression(model, train_ratio=0.8)

