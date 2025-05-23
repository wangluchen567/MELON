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
import numpy as np
from Models import Model


class OneVsRestClassifier(Model):
    """一对多 分类包装器"""

    def __init__(self, model, X_train=None, Y_train=None):
        """
        一对多 分类包装器
        :param model: 需要包装的模型
        :param X_train: 训练数据
        :param Y_train: 真实标签
        """
        self.model = model
        super().__init__(X_train, Y_train)
        self.class_states = None
        if not hasattr(self.model, 'predict_prob'):
            raise AttributeError('This model does not have a method for predicting probabilities, '
                                 'so OVR multi classification cannot be used')

    def train(self, X_train=None, Y_train=None):
        """训练模型"""
        self.set_train_data(X_train, Y_train)
        # 得到需要分类的类别情况
        self.class_states = np.unique(self.Y_train)
        # 根据类别情况创建多个 一对多 分类器
        self.classifiers = []
        # 然后使用每个 一对多 分类器学习 一对多 分类情况
        for i in range(len(self.class_states)):
            # 将训练数据改为OVR状态
            Y_train_ovr = np.zeros(Y_train.shape, dtype=int)
            Y_train_ovr[Y_train == self.class_states[i]] = 1
            Y_train_ovr[Y_train != self.class_states[i]] = -1
            # 创建对于该状态的模型
            model_ovr = copy.deepcopy(self.model)
            model_ovr.train(self.X_train, Y_train_ovr)
            # 保存模型
            self.classifiers.append(model_ovr)

    def predict(self, X_data):
        """预测数据"""
        # 先得到预测的概率
        Y_data_prob = self.predict_prob(X_data)
        # 然后选择概率最大的元素作为预测结果
        Y_data = self.class_states[np.argmax(Y_data_prob, axis=1)].reshape(len(X_data), -1)
        return Y_data

    def predict_prob(self, X_data):
        """预测数据(概率预测)"""
        # 遍历所有的模型，得到每个数据是各个类别的概率
        probs = np.zeros((len(X_data), len(self.classifiers)))
        for i in range(len(self.classifiers)):
            probs[:, i] = self.classifiers[i].predict_prob(X_data).flatten()
        # 归一化每行得到每个类的预测概率
        Y_data_prob = probs / probs.sum(1).reshape(-1, 1)
        return Y_data_prob
