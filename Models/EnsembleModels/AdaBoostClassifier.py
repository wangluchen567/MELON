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
import pandas as pd
from Models.DecisionTree.DecisionTreeClassifier import DecisionTreeClassifier
from Models.Utils import (calculate_accuracy, run_uniform_classification, run_double_classification,
                          run_circle_classification, run_moons_classification, plot_2dim_classification_sample)


class AdaBoostClassifier:
    def __init__(self, estimator=None, X_train=None, Y_train=None, n_estimators=10,
                 learning_rate=1.0, algorithm='SAMME.R'):
        """
        AdaBoost 自适应提升 集成学习器
        :param estimator: 基础学习器(弱学习器)
        :param X_train: 训练数据
        :param Y_train: 真实标签
        :param n_estimators: 基础学习器数量
        :param learning_rate: 学习率，用于缩放弱学习器的权重
        :param algorithm: 训练算法的类型('SAMME'/'SAMME.R')
        """
        self.X_train = None  # 训练数据
        self.Y_train = None  # 真实标签
        self.estimator = estimator  # 基础学习器(弱学习器)
        self.n_estimators = n_estimators  # 基础学习器数量
        self.learning_rate = learning_rate  # 学习率
        self.algorithm = algorithm  # 训练算法的类型
        self.class_list = None  # 要分类的类别列表
        self.estimator_models = None  # 初始化基础估计器集合
        self.alphas = np.zeros(self.n_estimators)  # 基础学习器权重
        self.errors = np.zeros(self.n_estimators)  # 记录加权错误率历史
        self.losses = np.zeros(self.n_estimators)  # 记录加权损失值历史
        if self.estimator is None:
            # 默认使用决策树树桩
            self.estimator = DecisionTreeClassifier(max_depth=1)
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
        """训练模型拟合数据(SAMME.R)"""
        self.set_train_data(X_train, Y_train)
        # 得到类别标签列表
        self.class_list = np.unique(Y_train)
        # 要分类的类别个数
        K = len(self.class_list)
        # 初始化基础学习器
        self.estimator_models = []
        # 初始化样本权重
        sample_weights = np.ones(len(self.X_train)) / len(self.X_train)
        # 得到每个类别标签在class_list集合中的索引
        y_true_idx = np.searchsorted(self.class_list, self.Y_train.flatten())
        # 训练每一个弱分类器以逐渐实现分类正确
        for i in range(self.n_estimators):
            # 创建一个弱分类器
            model = copy.deepcopy(self.estimator)
            # 使用当前样本权重训练弱分类器
            model.train(X_train, Y_train, sample_weight=sample_weights)
            # 得到该分类器的类别预测概率, 形状必须为(X_train.len, class_list.len)
            Y_prob = model.predict_prob(X_train)
            # 将概率包含零的部分进行裁剪（避免log(0)）
            Y_prob = np.clip(Y_prob, 1e-15, 1 - 1e-15)
            # 计算每个样本的真实类别对应的log概率
            log_prob_true = np.log(Y_prob[np.arange(len(self.X_train)), y_true_idx])
            # 计算权重更新因子：exp(-(K-1)/K * log P(y_true|x))
            weights_update = np.exp(-(K - 1) / K * log_prob_true)
            # 记录损失值
            self.losses[i] = np.sum(sample_weights * weights_update)
            # 并更新样本权重
            sample_weights *= weights_update
            sample_weights /= np.sum(sample_weights)  # 归一化
            # 保存基分类器
            self.estimator_models.append(model)
            # model.plot_2dim()

    def predict(self, X_data):
        """给定数据预测结果(SAMME.R)"""
        # 初始化类别得分
        class_scores = np.zeros((len(X_data), len(self.class_list)))
        # 对每个基分类器，累加其对各类别的log概率
        for model in self.estimator_models:
            prob = np.clip(model.predict_prob(X_data), 1e-15, 1 - 1e-15)
            class_scores += np.log(prob)
        # 选择得分最高的类别
        Y_data = self.class_list[np.argmax(class_scores, axis=1)].reshape(-1, 1)
        return Y_data

    def plot_2dim(self, X_test=None, Y_test=None, Truth=None):
        """为二维分类数据集和结果画图（只能是连续数据）"""
        X_train = np.array(self.X_train)
        Y_train = np.array(self.Y_train)
        plot_2dim_classification_sample(self, X_train, Y_train, X_test, Y_test, neg_label=-1)

    # def train_(self, X_train=None, Y_train=None):
    #     """训练模型拟合数据(SAMME)"""
    #     self.set_train_data(X_train, Y_train)
    #     self.class_list = np.unique(Y_train)
    #     self.estimator_models = []
    #     # 初始化样本权重
    #     sample_weights = np.ones(len(self.X_train)) / len(self.X_train)
    #     # 训练每一个弱分类器以逐渐实现分类正确
    #     for i in range(self.n_estimators):
    #         # 创建一个弱分类器
    #         model = copy.deepcopy(self.estimator)
    #         # 使用当前样本权重训练弱分类器
    #         model.train(self.X_train, self.Y_train, sample_weights)
    #         # 得到该分类器的类别预测
    #         Y_predict = model.predict(self.X_train)
    #         # 得到分类错误的位置mask向量
    #         incorrect = np.array(Y_predict != self.Y_train).flatten()
    #         # 然后计算该弱分类器的加权错误率
    #         err = np.sum(sample_weights * incorrect) / np.sum(sample_weights)
    #         print(err)
    #         self.err_history[i] = err  # 记录错误率
    #         # 计算弱分类器权重（加入多分类修正项log(K-1)）
    #         self.alphas[i] = np.log((1 - err) / err) + np.log(len(self.class_list) - 1)
    #         # 更新样本权重
    #         sample_weights *= np.exp(self.alphas[i] * incorrect)
    #         sample_weights /= np.sum(sample_weights)  # 归一化
    #         # 保存该弱分类器
    #         self.estimator_models.append(model)
    #         # model.plot_2dim()
    #
    # def predict(self, X_data):
    #     """给定数据预测结果(SAMME)"""
    #     probs = np.zeros((len(X_data), len(self.class_list)))
    #     for i in range(self.n_estimators):
    #         Y_predict = self.estimator_models[i].predict(X_data)
    #         probs += self.alphas[i] * np.array(Y_predict == self.class_list)
    #     Y_data = self.class_list[np.argmax(probs, axis=1)].reshape(len(X_data), -1)
    #     return Y_data


def run_watermelon_example():
    model = AdaBoostClassifier(n_estimators=10)
    data = pd.read_csv("../../Dataset/watermelons2.csv")
    X_train = data.iloc[:, 1:-1]
    Y_train = np.array(data.iloc[:, -1]).reshape(-1, 1)
    model.train(X_train, Y_train)
    Y_predict = model.predict(X_train)
    print("准确率为: {:.3f} %".format(calculate_accuracy(Y_train, Y_predict) * 100))
    # model.plot_forest()


if __name__ == '__main__':
    run_watermelon_example()
    np.random.seed(100)
    model = AdaBoostClassifier(n_estimators=10)
    run_uniform_classification(model)
    run_double_classification(model)
    run_circle_classification(model)
    run_moons_classification(model)
