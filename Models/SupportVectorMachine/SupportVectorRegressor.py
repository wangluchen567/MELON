"""
Copyright (c) 2023 LuChen Wang
MELON is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan
PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY
KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
NON-INFRINGEMENT, MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details.
"""
import warnings
import numpy as np
from Models import Model
from Models.SupportVectorMachine.SequentialMinimalOptimization import smo_greedy_step_regression
from Models.Utils import (plot_2dim_regression, run_uniform_regression,
                          plot_2dim_regression_sample, run_circular_regression, run_poly_regression)


class SupportVectorRegressor(Model):
    # 定义核函数类型
    LINEAR = 0
    POLY = 1
    RBF = GAUSSIAN = 2
    SIGMOID = 3

    def __init__(self, X_train=None, Y_train=None, C=10.0, tol=1.e-3, epsilon=0.6,
                 kernel=LINEAR, gamma=None, degree=3.0, const=1.0, max_iter=1000, show=False):
        """
        支持向量机回归模型
        :param X_train: 训练数据
        :param Y_train: 真实标签
        :param C: 惩罚系数
        :param tol: 残差收敛条件（容忍系数）
        :param epsilon: 误差系数
        :param kernel: 核函数类型
        :param gamma: 核函数系数（乘数项）
        :param degree: 核函数系数（指数项）
        :param const: 核函数系数（常数项）
        :param max_iter: 最大迭代优化次数
        :param show: 是否展示迭代过程
        """
        super().__init__(X_train, Y_train)
        self.Weights = None  # 模型参数
        self.kernel_mat = None  # 核函数矩阵
        self.alphas, self.b = None, None  # 乘子参数
        self.C = C  # 惩罚系数
        self.tol = tol  # 残差收敛条件（容忍系数）
        self.epsilon = epsilon  # 误差系数
        self.kernel = kernel  # 核函数类型
        self.gamma = gamma  # 核函数系数（乘数项）
        self.degree = degree  # 核函数系数（指数项）
        self.const = const  # 核函数系数（常数项）
        self.n_iter = 0  # 记录迭代次数
        self.max_iter = max_iter  # 最大迭代优化次数
        if self.max_iter == -1:  # 若设置为-1则直到优化结束停止
            self.max_iter = np.inf
        self.show = show  # 是否展示迭代过程

    def train(self, X_train=None, Y_train=None):
        """使用数据集训练模型"""
        self.set_train_data(X_train, Y_train)
        # 计算核函数矩阵
        self.kernel_mat = self.cal_kernel_mat(self.X_train, self.X_train)
        # 初始化权重参数
        self.init_weights()
        # 使用smo算法得到乘子参数和模型参数
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
            Y_data = X_B.dot(self.Weights)
        else:  # 否则通过核函数求结果
            # 先得到非零的位置，以简化运算
            non_zeros = np.nonzero(np.sum(self.alphas, axis=1))[0]
            # 计算预测数据的核函数矩阵
            kernel_predict = self.cal_kernel_mat(self.X_train[non_zeros], X_data)
            Y_data = kernel_predict.T @ (self.alphas[non_zeros, 1] - self.alphas[non_zeros, 0])[:, np.newaxis] + self.b
        return Y_data

    def init_weights(self):
        """初始化参数 (权重)"""
        X_feat = self.X_train.shape[1]
        self.Weights = np.random.uniform(-1, 1, size=(X_feat + 1, 1))

    def cal_weights(self):
        """计算参数 (权重)"""
        X_feat = self.X_train.shape[1]
        self.Weights = np.zeros((X_feat + 1, 1))
        self.Weights[:X_feat] = self.X_train.T @ (self.alphas[:, 1] - self.alphas[:, 0])[:, np.newaxis]
        self.Weights[-1] = self.b

    def smo_algorithm(self):
        """SMO算法"""
        # 初始化相关参数
        self.n_iter, self.alphas, self.b, optimize_end = 0, np.zeros((len(self.X_train), 2)), 0, False
        while True:
            self.alphas, self.b, optimize_end \
                = smo_greedy_step_regression(self.kernel_mat, self.X_train, self.Y_train,
                                             self.alphas, self.b, self.C, self.epsilon, self.tol)
            self.n_iter += 1
            if optimize_end:
                # 若没有可优化的项则结束优化
                break
            if self.show:
                self.cal_weights()
                self.plot_2dim(pause=True, n_iter=self.n_iter)
            if self.n_iter >= self.max_iter:
                # 受最大迭代次数限制优化提前结束
                warnings.warn(f"Optimizer ended early (max_iter={self.max_iter})")
                break

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

    def plot_2dim(self, X_test=None, Y_test=None, Truth=None, pause=False, n_iter=None):
        """为二维回归数据集和结果画图"""
        if self.kernel == self.LINEAR:
            plot_2dim_regression(self.X_train, self.Y_train, self.Weights, X_test, Y_test, Truth=Truth,
                                 support=(np.sum(self.alphas, axis=1) != 0.0), pause=pause, n_iter=n_iter)
        else:
            plot_2dim_regression_sample(self, self.X_train, self.Y_train, X_test, Y_test,
                                        support=(np.sum(self.alphas, axis=1) != 0.0), pause=pause, n_iter=n_iter)


if __name__ == '__main__':
    np.random.seed(100)
    # 线性回归测试
    model = SupportVectorRegressor(C=0.1, kernel=SupportVectorRegressor.LINEAR, max_iter=100, show=True)
    run_uniform_regression(model)
    # 多项式回归测试
    model = SupportVectorRegressor(C=1, kernel=SupportVectorRegressor.POLY, max_iter=100, show=True)
    run_poly_regression(model)
    # 三角函数(圆函数)回归测试
    model = SupportVectorRegressor(C=100, kernel=SupportVectorRegressor.RBF, max_iter=100, show=True)
    run_circular_regression(model)
