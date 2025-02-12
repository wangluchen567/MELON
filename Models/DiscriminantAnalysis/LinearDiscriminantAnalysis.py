"""
线性判别分析
Linear Discriminant Analysis
"""
import warnings
import numpy as np
from Models.Utils import sigmoid, plot_2dim_classification, run_uniform_classification, run_double_classification


class FisherDiscriminantAnalysis():
    def __init__(self, X_train=None, Y_train=None, priors=None, n_components=None):
        """
        :param X_train: 训练数据
        :param Y_train: 真实标签
        :param priors: 每类数据的先验概率
        :param n_components: 降维后的特征数量
        """
        self.X_train = None  # 训练数据
        self.Y_train = None  # 真实标签
        self.priors = priors
        self.n_components = n_components
        self.set_train_data(X_train, Y_train)
        self.Weights = None  # 模型参数

    def set_train_data(self, X_train, Y_train):
        """给定训练数据集和标签数据"""
        if X_train is not None:
            if self.X_train is not None:
                warnings.warn("Training data will be overwritten")
            self.X_train = X_train.copy()
        if Y_train is not None:
            if self.Y_train is not None:
                warnings.warn("Training label will be overwritten")
            self.Y_train = Y_train.copy()
