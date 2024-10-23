"""
支持向量机分类器
Support Vector Classifier
"""
import warnings
import numpy as np
from SequentialMinimalOptimization import smo_greedy_step, smo_random
from Models.Utils import plot_2dim_classification, run_uniform_classification, run_double_classification


class SupportVectorClassifier():
    # 定义核函数类型
    LINEAR = 0

    def __init__(self, X_train=None, Y_train=None, C=1.0, tol=1.e-4, num_iter=1000):
        self.X_train = None  # 训练数据
        self.Y_train = None  # 真实标签
        self.Y_train_ = None  # 逻辑回归特殊标签
        self.set_train_data(X_train, Y_train)
        self.Weights = None  # 模型参数
        self.kernel_mat = None  # 核函数矩阵
        self.alphas, self.b = None, None  # 乘子参数
        self.C = C  # 惩罚系数
        self.tol = tol  # 残差收敛条件（容忍系数）
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

    def set_parameters(self, C=None, tol=None, num_iter=None):
        """重新修改相关参数"""
        if self.C is not None and C is not None:
            warnings.warn("Parameter 'C' be overwritten")
            self.C = C
        if self.tol is not None and tol is not None:
            warnings.warn("Parameters 'tol' be overwritten")
            self.tol = tol
        if self.num_iter is not None and num_iter is not None:
            warnings.warn("Parameters 'num_iter' be overwritten")
            self.num_iter = num_iter

    def cal_kernel_mat(self):
        """计算核函数矩阵"""
        self.kernel_mat = self.X_train @ self.X_train.T

    def train(self, X_train=None, Y_train=None, C=None, tol=None, num_iter=None):
        """使用数据集训练模型"""
        self.set_train_data(X_train, Y_train)
        self.set_parameters(C, tol, num_iter)
        # 计算核函数矩阵
        self.cal_kernel_mat()
        # 使用smo算法计算乘子参数
        self.smo_algorithm()
        # 计算模型参数
        self.cal_weights()

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

    def cal_weights(self):
        """计算参数 (权重)"""
        X_feat = self.X_train.shape[1]
        self.Weights = np.zeros((X_feat + 1, 1))
        self.Weights[:2] = (np.tile(self.Y_train, self.X_train.shape[1]) * self.X_train).T @ self.alphas
        self.Weights[-1] = self.b

    def smo_algorithm(self):
        """SMO算法"""
        # 初始化相关参数
        self.alphas, self.b, optimize_end = np.zeros((len(self.X_train), 1)), 0, False
        # # 下面这种是随机选择乘子的方法，效果较差
        # self.alphas, self.b = smo_random(self.X_train, self.Y_train, self.C, self.tol, self.num_iter)
        for i in range(self.num_iter):
            self.alphas, self.b, optimize_end = smo_greedy_step(self.kernel_mat, self.X_train, self.Y_train,
                                                                self.alphas, self.b, self.C, self.tol)
            # 若优化结束则跳出循环(没有可优化的项了)
            if optimize_end:
                print("The optimization has ended early, "
                      "and the number of iterations for this optimization is {}".format(i))
                break
            self.cal_weights()
            self.plot_2dim(pause=True, n_iter=i + 1)

    def plot_2dim(self, X_data=None, Y_data=None, Truth=None, pause=False, n_iter=None):
        """为二维分类数据集和结果画图"""
        plot_2dim_classification(self.X_train, self.Y_train, self.Weights, X_data, Y_data, Truth=Truth,
                                 support=(self.alphas.flatten() != 0.0), pause=pause, n_iter=n_iter)


if __name__ == '__main__':
    np.random.seed(0)
    model = SupportVectorClassifier(C=10, num_iter=1000)
    run_uniform_classification(model, train_ratio=0.8, X_size=100)
    run_double_classification(model, train_ratio=0.8, X_size=100)
