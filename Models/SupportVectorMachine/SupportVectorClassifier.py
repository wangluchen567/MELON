"""
支持向量机分类模型
Support Vector Classifier
"""
import warnings
import numpy as np
from Models.SupportVectorMachine.SequentialMinimalOptimization import smo_greedy_step, smo_random
from Models.Utils import (sigmoid, plot_2dim_classification, run_uniform_classification, run_double_classification,
                          plot_2dim_classification_sample, run_circle_classification, run_moons_classification)


class SupportVectorClassifier():
    # 定义核函数类型
    LINEAR = 0
    POLY = 1
    RBF = GAUSSIAN = 2
    SIGMOID = 3

    def __init__(self, X_train=None, Y_train=None, C=10, tol=1.e-4,
                 kernel=LINEAR, gamma=None, degree=3.0, const=1.0, num_iter=300, show=True):
        self.X_train = None  # 训练数据
        self.Y_train = None  # 真实标签
        self.set_train_data(X_train, Y_train)
        self.Weights = None  # 模型参数
        self.kernel_mat = None  # 核函数矩阵
        self.alphas, self.b = None, None  # 乘子参数
        self.C = C  # 惩罚系数
        self.tol = tol  # 残差收敛条件（容忍系数）
        self.kernel = kernel  # 核函数类型
        self.gamma = gamma  # 核函数系数（乘数项）
        self.degree = degree  # 核函数系数（指数项）
        self.const = const  # 核函数系数（常数项）
        self.num_iter = num_iter  # 迭代优化次数
        self.show = show  # 是否展示迭代过程

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

    def set_parameters(self, C=None, tol=None, kernel=None,
                       gamma=None, degree=None, const=None, num_iter=None):
        """重新修改相关参数"""
        parameters = ['C', 'tol', 'kernel', 'gamma', 'degree', 'const', 'num_iter']
        values = [C, tol, kernel, gamma, degree, const, num_iter]
        for param, value in zip(parameters, values):
            if value is not None and getattr(self, param) is not None:
                warnings.warn(f"Parameter '{param}' will be overwritten")
                setattr(self, param, value)

    def cal_kernel_mat(self, X, Y):
        """给定数据计算核函数矩阵"""
        if self.kernel == self.LINEAR:
            return self.linear_kernel_mat(X, Y)
        elif self.kernel == self.POLY:
            return self.poly_kernel_mat(X, Y, self.gamma, self.degree, self.const)
        elif self.kernel == self.RBF or self.kernel == self.GAUSSIAN:
            return self.rbf_kernel_mat(X, Y, self.gamma)
        elif self.kernel == self.SIGMOID:
            return self.sigmoid_kernel_mat(X, Y, self.const)
        else:
            raise ValueError("There is no such kernel function type")

    @staticmethod
    def linear_kernel_mat(X, Y):
        kernel_mat = X @ Y.T
        return kernel_mat

    @staticmethod
    def poly_kernel_mat(X, Y, gamma=None, degree=3.0, const=1.0):
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
    def sigmoid_kernel_mat(X, Y, gamma=None, const=1.0):
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

    def train(self, X_train=None, Y_train=None, C=None, tol=None,
              kernel=None, gamma=None, degree=None, const=None, num_iter=None):
        """使用数据集训练模型"""
        self.set_train_data(X_train, Y_train)
        self.set_parameters(C, tol, kernel, gamma, degree, const, num_iter)
        # 计算核函数矩阵
        self.kernel_mat = self.cal_kernel_mat(self.X_train, self.X_train)
        # 初始化权重参数
        self.init_weights()
        # 使用smo算法计算乘子参数
        self.smo_algorithm()
        # 根据得到的参数计算模型权重参数
        self.cal_weights()

    def predict(self, X_data):
        """模型对测试集进行预测"""
        if X_data.ndim == 2:
            pass
        elif X_data.ndim == 1:
            X_data = X_data.reshape(1, -1)
        else:
            raise ValueError("Cannot handle data with a shape of 3 dimensions or more")
        # 为简化运算，这里使用线性核函数时利用参数直接求结果
        if self.kernel == self.LINEAR:
            X_B = np.concatenate((X_data, np.ones((len(X_data), 1))), axis=1)
            Y_data = np.ones((len(X_data), 1), dtype=int)
            Y_data[X_B.dot(self.Weights) < 0] = -1
        else:  # 否则通过核函数求结果
            # 先得到非零的位置，以简化运算
            non_zeros = np.nonzero(self.alphas.flatten())[0]
            # 计算预测数据的核函数矩阵
            kernel_predict = self.cal_kernel_mat(self.X_train[non_zeros], X_data)
            Y_out = kernel_predict.T @ (self.alphas[non_zeros] * self.Y_train[non_zeros]) + self.b
            Y_data = np.ones((len(X_data), 1), dtype=int)
            Y_data[Y_out >= 0] = 1
            Y_data[Y_out < 0] = -1
        return Y_data

    def predict_prob(self, X_data):
        """模型对测试集进行预测(预测得到是正例的概率)"""
        if X_data.ndim == 2:
            pass
        elif X_data.ndim == 1:
            X_data = X_data.reshape(1, -1)
        else:
            raise ValueError("Cannot handle data with a shape of 3 dimensions or more")
        # 为简化运算，这里使用线性核函数时利用参数直接求结果
        if self.kernel == self.LINEAR:
            X_B = np.concatenate((X_data, np.ones((len(X_data), 1))), axis=1)
            Y_data_prob = sigmoid(X_B.dot(self.Weights))
        else:  # 否则通过核函数求结果
            # 先得到非零的位置，以简化运算
            non_zeros = np.nonzero(self.alphas.flatten())[0]
            # 计算预测数据的核函数矩阵
            kernel_predict = self.cal_kernel_mat(self.X_train[non_zeros], X_data)
            Y_data = kernel_predict.T @ (self.alphas[non_zeros] * self.Y_train[non_zeros]) + self.b
            Y_data_prob = sigmoid(Y_data)
        return Y_data_prob

    def cal_weights(self):
        """计算参数 (权重)"""
        X_feat = self.X_train.shape[1]
        self.Weights = np.zeros((X_feat + 1, 1))
        self.Weights[:X_feat] = (np.tile(self.Y_train, self.X_train.shape[1]) * self.X_train).T @ self.alphas
        self.Weights[-1] = self.b

    def smo_algorithm(self):
        """SMO算法"""
        # 初始化相关参数
        self.alphas, self.b, optimize_end = np.zeros((len(self.X_train), 1)), 0, False
        # 记录历史参数以检查变化
        alphas, b = self.alphas.copy(), self.b
        # # 下面这种是随机选择乘子的方法，效果较差
        # self.alphas, self.b = smo_random(self.X_train, self.Y_train, self.C, self.tol, self.num_iter)
        for i in range(self.num_iter):
            self.alphas, self.b, optimize_end = smo_greedy_step(self.kernel_mat, self.X_train, self.Y_train,
                                                                self.alphas, self.b, self.C, self.tol)
            # 检查参数变化是否过小
            variation = np.sum(np.abs(alphas - self.alphas)) + np.abs(b - self.b) < self.tol
            alphas, b = self.alphas.copy(), self.b  # 更新历史参数
            # 若没有可优化的项或者参数变化过小则跳出循环
            if optimize_end or variation:
                print("The optimization has ended early, "
                      "and the number of iterations for this optimization is {}".format(i))
                break
            if self.show:
                self.cal_weights()  # 每步都计算一下权重
                self.plot_2dim(pause=True, n_iter=i + 1)

    def plot_2dim(self, X_test=None, Y_test=None, Truth=None, pause=False, n_iter=None):
        """为二维分类数据集和结果画图"""
        if self.kernel == self.LINEAR:
            plot_2dim_classification(self.X_train, self.Y_train, self.Weights, X_test, Y_test, Truth=Truth,
                                     support=(self.alphas.flatten() != 0.0), pause=pause, n_iter=n_iter)
        else:
            plot_2dim_classification_sample(self, self.X_train, self.Y_train, X_test, Y_test, neg_label=-1,
                                     support=(self.alphas.flatten() != 0.0), pause=pause, n_iter=n_iter)


if __name__ == '__main__':
    np.random.seed(100)
    model = SupportVectorClassifier(C=10, kernel=SupportVectorClassifier.LINEAR, num_iter=100)
    run_uniform_classification(model)
    run_double_classification(model)
    model = SupportVectorClassifier(C=10, kernel=SupportVectorClassifier.RBF, num_iter=100)
    run_circle_classification(model)
    run_moons_classification(model)
