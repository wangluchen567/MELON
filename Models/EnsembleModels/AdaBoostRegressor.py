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
import copy
import warnings
import numpy as np
from Models import Model
from Models.DecisionTree import DecisionTreeRegressor
from Models.Utils import (run_uniform_regression, plot_2dim_regression_sample,
                          run_circular_regression, run_poly_regression)


class AdaBoostRegressor(Model):
    def __init__(self, estimator=None, X_train=None, Y_train=None, n_estimators=10,
                 learning_rate=1.0, max_depth=3, sampling=True):
        """
        AdaBoost 自适应提升 回归器
        :param estimator: 基础学习器(弱学习器)
        :param X_train: 训练数据
        :param Y_train: 真实标签
        :param n_estimators: 基础学习器数量
        :param learning_rate: 学习率，用于缩放弱学习器的权重
        :param max_depth: 当基础学习器为决策树时决策树最大深度
        :param sampling: 是否使用自助采样实现样本不同权重
        """
        super().__init__(X_train, Y_train)
        self.estimator = estimator  # 基础学习器(弱学习器)
        self.n_estimators = n_estimators  # 基础学习器数量
        self.learning_rate = learning_rate  # 学习率
        self.max_depth = max_depth  # 当基础学习器为决策树时决策树最大深度
        self.sampling = sampling  # 是否使用自助采样实现样本不同权重
        self.estimator_models = []  # 初始化基础估计器集合
        self.alphas = []  # 基础学习器权重
        self.errors = []  # 记录加权误差历史
        if self.estimator is None:
            # 默认使用决策树树桩
            self.estimator = DecisionTreeRegressor(max_depth=self.max_depth)

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
            # 判断是否使用采样方式
            if self.sampling:
                # 每次采取自助采样的方式采样数据集
                sample_idx = np.random.choice(len(self.X_train), size=len(self.X_train), p=sample_weights)
            else:
                sample_idx = np.arange(len(self.X_train))
            X_samples = self.X_train[sample_idx]
            Y_samples = self.Y_train[sample_idx]
            # 使用当前样本权重训练弱回归器
            if self.sampling:
                model.train(X_samples, Y_samples)
            else:
                model.train(X_samples, Y_samples, sample_weight=sample_weights)
            # 得到该弱回归器的回归值
            Y_predict = model.predict(X_samples)
            # 计算归一化误差
            abs_error = np.abs(Y_predict - Y_samples).flatten()
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
            beta = epsilon_t / (1 - epsilon_t)
            # 计算回归器权重（加入学习率）
            alpha = self.learning_rate * np.log(1.0 / beta)
            self.alphas.append(alpha)  # 记录模型权重
            # 更新样本权重
            # sample_weights *= np.power(beta, (1 - err_t) * self.learning_rate)
            sample_weights *= np.exp(-alpha * (1 - err_t))
            sample_weights /= np.sum(sample_weights)
            self.estimator_models.append(model)

    def predict(self, X_data):
        """给定数据预测结果"""
        # 得到每个模型输出的结果，形状为 (num_data, num_estimator)
        Y_predicts = np.array([model.predict(X_data).flatten() for model in self.estimator_models]).T
        # 对每个样本的所有基学习器预测值进行升序排序，返回的是排序后的索引值
        sorted_idx = np.argsort(Y_predicts, axis=1)  # 按预测值排序
        # 将基学习器的权重按预测值从小到大排列，并对排序后的权重按行计算累积和
        weight_cdf = np.cumsum(np.array(self.alphas)[sorted_idx], axis=1)
        # 找到每个样本的权重累积和中首次超过总权重(weight_cdf[:, -1])一半的位置
        median_or_above = weight_cdf >= 0.5 * weight_cdf[:, -1][:, np.newaxis]
        # 对每个样本，找到中位数所在的列索引
        median_idx = median_or_above.argmax(axis=1)
        final_predicts = Y_predicts[np.arange(len(X_data)), sorted_idx[np.arange(len(X_data)), median_idx]]
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
