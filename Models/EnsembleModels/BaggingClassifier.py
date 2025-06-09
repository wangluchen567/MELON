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
import pandas as pd
from Models import Model
from Models.DecisionTree import DecisionTreeClassifier
from Models.SupportVectorMachine import SupportVectorClassifier
from Models.Utils import (run_uniform_classification, run_double_classification,
                          run_circle_classification, run_moons_classification, plot_2dim_classification_sample)


class BaggingClassifier(Model):
    def __init__(self, estimator, X_train=None, Y_train=None, n_estimators=10,
                 max_samples=1.0, max_features=1.0, bootstrap=True, bootstrap_features=False):
        """
        集成学习Bagging分类封装器
        :param estimator: 基础估计器
        :param X_train: 训练数据
        :param Y_train: 真实标签
        :param n_estimators: 基础估计器数量
        :param max_samples: 抽取用于训练每个基础估计器的样本数(int/float)
        :param max_features: 抽取用于训练每个基础估计器的特征数(int/float)(TO DO)
        :param bootstrap: 是否对样本进行有放回抽样
        :param bootstrap_features: 是否对特征进行有放回抽样
        """
        super().__init__(X_train, Y_train)
        self.estimator = estimator  # 基础估计器
        self.n_estimators = n_estimators  # 基础估计器数量
        self.max_samples = max_samples  # 抽取用于训练每个基础估计器的样本数
        self.max_features = max_features  # 抽取用于训练每个基础估计器的特征数
        self.bootstrap = bootstrap  # 是否对样本进行有放回抽样
        self.bootstrap_features = bootstrap_features  # 是否对特征进行有放回抽样(TO DO)
        self.estimator_models = None  # 初始化基础估计器集合
        self.class_list = None  # 要分类的类别列表

    def train(self, X_train=None, Y_train=None):
        """使用数据集训练模型"""
        self.set_train_data(X_train, Y_train)
        self.class_list = np.unique(self.Y_train)
        self.estimator_models = []
        for i in range(self.n_estimators):
            # 有放回采样样本(或原始样本)
            X_sample, Y_sample = self.sampling()
            # 使用采样的样本构建基础估计器
            base_model = copy.deepcopy(self.estimator)
            # 使用训练数据进行训练
            base_model.train(X_sample, Y_sample)
            # 将模型加入集合
            self.estimator_models.append(base_model)

    def predict(self, X_data):
        """模型对测试集进行预测"""
        Y_predicts = np.zeros((self.n_estimators, len(X_data)), dtype=self.class_list.dtype)
        for i in range(self.n_estimators):
            Y_predicts[i] = self.estimator_models[i].predict(X_data).flatten()
        # 初始化一个数组来存储每个类别的票数
        votes = np.zeros((len(X_data), len(self.class_list)), dtype=int)
        # 统计每个类别的票数
        for i, cls in enumerate(self.class_list):
            votes[:, i] = np.sum(Y_predicts == cls, axis=0)
        # 找到每行中票数最多的类别索引
        max_votes_indices = np.argmax(votes, axis=1)
        # 根据索引获取最终预测结果
        final_predict = self.class_list[max_votes_indices].reshape(-1, 1)
        return final_predict

    def sampling(self):
        """采样样本"""
        if self.bootstrap:
            data_size = len(self.X_train)
            # 采样数据大小
            sample_size = self.max_samples if isinstance(self.max_samples, int) \
                else int(data_size * self.max_samples)
            # 对样本进行有放回采样
            indices = np.random.choice(data_size, size=sample_size, replace=True)
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

    def plot_2dim(self, X_test=None, Y_test=None, Truth=None):
        """为二维分类数据集和结果画图（只能是连续数据）"""
        X_train = np.array(self.X_train)
        Y_train = np.array(self.Y_train)
        plot_2dim_classification_sample(self, X_train, Y_train, X_test, Y_test, neg_label=-1)


if __name__ == '__main__':
    np.random.seed(100)
    model = BaggingClassifier(DecisionTreeClassifier(), n_estimators=10)
    run_uniform_classification(model)
    run_double_classification(model)
    model = BaggingClassifier(SupportVectorClassifier(kernel=SupportVectorClassifier.RBF), n_estimators=100)
    run_circle_classification(model)
    run_moons_classification(model)
