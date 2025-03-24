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
import warnings
import numpy as np
import pandas as pd
from Models.DecisionTree.DecisionTreeRegressor import DecisionTreeRegressor
from Models.Utils import (run_uniform_regression, plot_2dim_regression_sample,
                          run_circular_regression, run_poly_regression)


class RandomForestRegressor():
    def __init__(self, X_train=None, Y_train=None, n_estimators=10, criterion='mse',
                 max_depth=np.inf, max_features='sqrt', bootstrap=True):
        """
        随机森林回归器模型
        :param X_train: 训练数据
        :param Y_train: 真实标签
        :param n_estimators: 基学习器的数量(决策树数量)
        :param criterion: 特征选择标准(mse/mae)
        :param max_depth: 决策树最大深度
        :param max_features: 每次分裂节点时考虑的最大特征数(sqrt/log2)
        :param bootstrap: 是否对样本进行有放回抽样(否则使用原始数据)
        """
        self.X_train = None  # 训练数据
        self.Y_train = None  # 真实标签
        self.train_data = None  # 训练数据集（训练数据和真实标签的整合）
        self.X_columns = None  # 训练数据的列名称
        self.criterion = criterion  # 特征选择标准(mse/mae)
        self.max_depth = max_depth  # 决策树最大深度
        self.n_estimators = n_estimators  # 基学习器的数量
        self.max_features = max_features  # 每次分裂节点时考虑的最大特征数
        self.bootstrap = bootstrap  # 是否对样本进行有放回抽样
        self.random_forest = None  # 生成的随机森林
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

    def set_parameters(self, **kwargs):
        """重新修改相关参数"""
        for param, value in kwargs.items():
            if hasattr(self, param):  # 检查对象是否有该属性
                if getattr(self, param) is not None:
                    warnings.warn(f"Parameter '{param}' will be overwritten")
                setattr(self, param, value)
            else:
                warnings.warn(f"Parameter '{param}' is not a valid parameter for this model")

    def train(self, X_train=None, Y_train=None):
        """使用数据集训练模型"""
        self.set_train_data(X_train, Y_train)
        self.random_forest = []
        for i in range(self.n_estimators):
            # 有放回采样样本(或原始样本)
            X_sample, Y_sample = self.sampling()
            # 使用采样样本构建决策树
            tree = DecisionTreeRegressor(criterion=self.criterion,
                                         max_features=self.max_features,
                                         max_depth=self.max_depth)
            tree.train(X_sample, Y_sample)  # 训练该决策树
            # 将该决策树加入随机森林
            self.random_forest.append(tree)

    def predict(self, X_data):
        """模型对测试集进行预测"""
        Y_predicts = np.zeros((self.n_estimators, len(X_data)))
        for i in range(self.n_estimators):
            Y_predicts[i] = self.random_forest[i].predict(X_data).flatten()
        return Y_predicts.mean(axis=0).reshape(-1, 1)

    def sampling(self):
        """采样样本"""
        if self.bootstrap:
            data_size = len(self.X_train)
            # 对样本进行有放回采样
            indices = np.random.choice(data_size, size=data_size, replace=True)
            return self.fetch(self.X_train, indices), self.fetch(self.Y_train, indices)
        else:
            # 使用原始样本
            return self.X_train, self.Y_train

    @staticmethod
    def fetch(data, indices):
        """根据不同类型提取给定下标的数据"""
        if isinstance(data, pd.DataFrame):
            return data.iloc[indices, :].reset_index(drop=True)
        elif isinstance(data, pd.Series):
            return data[indices].reset_index(drop=True)
        else:
            return data[indices, :]

    def plot_forest(self):
        """绘制随机森林(慎用，数量太多会卡)"""
        for i in range(self.n_estimators):
            self.random_forest[i].plot_tree()

    def plot_2dim(self, X_test=None, Y_test=None, Truth=None):
        """为二维回归数据集和结果画图（只能是连续数据）"""
        X_train = np.array(self.X_train)
        Y_train = np.array(self.Y_train)
        plot_2dim_regression_sample(self, X_train, Y_train, X_test, Y_test)


if __name__ == '__main__':
    np.random.seed(100)
    model = RandomForestRegressor()
    run_uniform_regression(model)
    run_poly_regression(model)
    run_circular_regression(model)
