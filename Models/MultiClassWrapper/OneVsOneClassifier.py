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


class OneVsOneClassifier():
    """一对一 分类包装器"""

    def __init__(self, model, X_train=None, Y_train=None):
        """
        一对一 分类包装器
        :param model: 需要包装的模型
        :param X_train: 训练数据
        :param Y_train: 真实标签
        """
        self.model = model
        self.X_train = None  # 训练数据
        self.Y_train = None  # 真实标签
        self.set_train_data(X_train, Y_train)
        self.class_states = None
        self.binary_states = None

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
        """训练模型"""
        self.set_train_data(X_train, Y_train)
        # 得到需要分类的类别情况
        self.class_states = np.unique(self.Y_train)
        # 记录每个分类器是分哪两类
        self.binary_states = []
        # 根据类别情况创建多个 一对一 分类器
        self.classifiers = []
        # 然后使用每个 一对一 分类器学习 一对一 分类情况
        for i in range(len(self.class_states)):
            for j in range(i + 1, len(self.class_states)):
                # 记录当前是分的哪两类
                self.binary_states.append(np.array([i, j]))
                # 从数据集中只选择出现该两类的数据
                binary_pos = np.array(Y_train == self.class_states[i]) + np.array(Y_train == self.class_states[j])
                X_train_binary = X_train[binary_pos.flatten()]
                Y_train_binary = Y_train[binary_pos.flatten()]
                # 将训练数据改为OVO状态
                Y_train_ovo = np.zeros(Y_train_binary.shape, dtype=int)
                # 这里注意，第i个class为反例
                Y_train_ovo[Y_train_binary == self.class_states[i]] = -1
                Y_train_ovo[Y_train_binary == self.class_states[j]] = 1
                # 创建对于该状态的模型
                model_ovo = copy.deepcopy(self.model)
                model_ovo.train(X_train_binary, Y_train_ovo)
                # 保存模型
                self.classifiers.append(model_ovo)

    def predict(self, X_data):
        """预测数据"""
        # 遍历所有的模型，得到每对类别的分类情况
        votes = np.zeros((len(X_data), len(self.class_states)), dtype=int)  # 记录总体投票情况
        for i in range(len(self.classifiers)):
            # 得到该分类模型得到的预测情况
            y_predict = self.classifiers[i].predict(X_data).flatten()
            y_predict[y_predict == -1] = 0  # 调整预测输出
            # 记录当前模型的输出投票情况
            model_vote = np.zeros((len(X_data), len(self.class_states)), dtype=int)
            model_vote[:, self.binary_states[i]] += np.array([1 - y_predict, y_predict]).T
            # 更新总体投票结果
            votes += model_vote
        # 然后选择概率最大的元素作为预测结果
        Y_pred = self.class_states[np.argmax(votes, axis=1)].reshape(len(X_data), -1)
        return Y_pred

    def predict_prob(self, X_data):
        """预测数据(概率预测)"""
        if not hasattr(self.model, 'predict_prob'):
            raise AttributeError('This model does not have a method for predicting probabilities')
        # 遍历所有的模型，得到每对类别的分类情况
        probs = np.zeros((len(X_data), len(self.class_states)))  # 记录概率情况
        for i in range(len(self.classifiers)):
            # 得到该分类模型得到的预测情况
            y_predict = self.classifiers[i].predict_prob(X_data).flatten()
            # 记录当前模型的输出概率情况
            model_prob = np.zeros((len(X_data), len(self.class_states)))
            model_prob[:, self.binary_states[i]] += np.array([1 - y_predict, y_predict]).T
            # 更新总体投票结果
            probs += model_prob
        # 归一化每行得到每个类的预测概率
        Y_data_prob = probs / probs.sum(1).reshape(-1, 1)
        return Y_data_prob
