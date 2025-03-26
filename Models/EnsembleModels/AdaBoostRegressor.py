"""
Copyright (c) 2023 LuChen Wang
[Software Name] is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan
PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY
KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
NON-INFRINGEMENT, MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details.
"""
import copy
import warnings
import numpy as np
from Models.DecisionTree.DecisionTreeRegressor import DecisionTreeRegressor
from Models.Utils import (run_uniform_regression, plot_2dim_regression_sample,
                          run_circular_regression, run_poly_regression)


class AdaBoostRegressor:
    def __init__(self, estimator=None, X_train=None, Y_train=None, n_estimators=10, learning_rate=1.0, max_depth=3):
        """
        AdaBoost 自适应提升 回归器
        :param estimator: 基础学习器(弱学习器)
        :param X_train: 训练数据
        :param Y_train: 真实标签
        :param n_estimators: 基础学习器数量
        :param learning_rate: 学习率，用于缩放弱学习器的权重
        :param max_depth: 当基础学习器为决策树时决策树最大深度
        """
        self.X_train = None  # 训练数据
        self.Y_train = None  # 真实标签
        self.estimator = estimator  # 基础学习器(弱学习器)
        self.n_estimators = n_estimators  # 基础学习器数量
        self.learning_rate = learning_rate  # 学习率
        self.estimator_models = []  # 初始化基础估计器集合
        self.alphas = [] # 基础学习器权重
        self.errors = []  # 记录加权误差历史
        if self.estimator is None:
            # 默认使用决策树树桩
            self.estimator = DecisionTreeRegressor(max_depth=max_depth)
        self.set_train_data(X_train, Y_train)

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

    def train(self, X_train=None, Y_train=None):
        """训练模型拟合数据"""
        self.set_train_data(X_train, Y_train)
        self.estimator_models = []
        self.alphas, self.errors = [], []
        # 初始化样本权重
        sample_weights = np.ones(len(self.X_train)) / len(self.X_train)
        # 训练每一个弱回归器
        for i in range(self.n_estimators):
            # 创建一个弱回归器
            model = copy.deepcopy(self.estimator)
            # 使用当前样本权重训练弱回归器
            model.train(self.X_train, self.Y_train, sample_weights)
            # 得到该弱回归器的回归值
            Y_predict = model.predict(self.X_train)
            # 计算归一化误差
            abs_error = np.abs(Y_predict - self.Y_train).flatten()
            max_error = np.max(abs_error)
            max_error = 1.e-15 if max_error == 0 else max_error  # 避免除零
            # 得到归一化误差
            err_t = abs_error / max_error
            # 计算全局误差
            epsilon_t = np.sum(sample_weights * err_t)
            self.errors.append(epsilon_t)  # 保存误差值
            if epsilon_t >= 0.5:
                warnings.warn("The weighted error rate exceeds 1 - 1/K, so early stop")
                break  # 提前终止
            # 计算回归器权重（加入学习率）
            alpha = self.learning_rate * np.log((1 - epsilon_t) / epsilon_t)
            self.alphas.append(alpha)  # 记录模型权重
            # 更新样本权重
            sample_weights *= np.exp(-alpha * (1 - err_t))
            sample_weights /= np.sum(sample_weights)
            self.estimator_models.append(model)

    def predict(self, X_data):
        """给定数据预测结果"""
        Y_predicts = np.zeros((len(self.estimator_models), len(X_data)))
        for i, model in enumerate(self.estimator_models):
            Y_predicts[i] = model.predict(X_data).flatten()
        # 得到每个模型的权重
        weights = np.log(1 / np.array(self.alphas))
        # 使用加权中位数计算最终预测结果
        sorted_idx = np.argsort(Y_predicts, axis=0)
        cum_weights = np.cumsum(weights[sorted_idx], axis=0)
        median_idx = np.argmax(cum_weights >= 0.5 * cum_weights[-1], axis=0)
        final_predicts = Y_predicts[sorted_idx[median_idx, np.arange(len(X_data))], np.arange(len(X_data))]
        return final_predicts

    def plot_2dim(self, X_test=None, Y_test=None, Truth=None):
        """为二维回归数据集和结果画图（只能是连续数据）"""
        X_train = np.array(self.X_train)
        Y_train = np.array(self.Y_train)
        plot_2dim_regression_sample(self, X_train, Y_train, X_test, Y_test)

if __name__ == '__main__':
    np.random.seed(100)
    model = AdaBoostRegressor()
    run_uniform_regression(model)
    run_poly_regression(model)
    run_circular_regression(model)