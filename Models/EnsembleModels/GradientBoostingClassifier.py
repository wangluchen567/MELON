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
from Models.Utils import sigmoid, softmax
from Models.DecisionTree import DecisionTreeRegressor
from Models.Utils import plot_changes, plot_2dim_classification_sample
from Models.Utils import (run_uniform_classification, run_double_classification,
                          run_circle_classification, run_moons_classification)


class GradientBoostingClassifier(Model):
    def __init__(self, estimator=None, X_train=None, Y_train=None, n_estimators=10,
                 learning_rate=0.1, subsample=1.0, max_depth=3):
        """
        梯度提升分类器
        :param estimator: 基础学习器(弱学习器)
        :param X_train: 训练数据
        :param Y_train: 真实标签
        :param n_estimators: 基础学习器数量
        :param learning_rate: 学习率，每步拟合残差的学习率
        :param subsample: 子采样率，用于创建随机梯度提升树，取值范围为 (0, 1]
        :param max_depth: 当基础学习器为决策树时决策树最大深度
        """
        super().__init__(X_train, Y_train)
        self.estimator = estimator  # 基础学习器(弱学习器)
        self.n_estimators = n_estimators  # 基础学习器数量
        self.learning_rate = learning_rate  # 学习率
        self.subsample = subsample  # 子采样率
        assert 0.0 < self.subsample <= 1.0
        self.max_depth = max_depth  # 当基础学习器为决策树时决策树最大深度
        self.estimator_models = None  # 初始化基础估计器集合
        self.losses = None  # 记录损失值历史
        self.initials = None  # 初始化初始预测
        self.classes = None  # 要分类的类别情况
        self.num_class = None  # 要分类的类别个数
        self.Y_binary = None  # 二分类使用0/1标签
        self.Y_multi = None  # 多分类使用的0/1/2...标签
        self.Y_onehot = None  # 多分类标签对应的one-hot编码
        if self.estimator is None:
            # 默认使用决策树模型
            self.estimator = DecisionTreeRegressor(max_depth=self.max_depth)
        if not isinstance(self.estimator, Model):
            raise ValueError("The base estimator must be a subclass of the Model")

    def train(self, X_train=None, Y_train=None):
        """训练模型拟合数据"""
        self.set_train_data(X_train, Y_train)
        self.losses = []  # 原损失值历史置空
        self.estimator_models = []
        # 要分类的类别与个数
        self.classes, self.Y_multi = np.unique(self.Y_train, return_inverse=True)
        self.num_class = len(self.classes)
        # 二分类使用二分类0/1标签
        self.Y_binary = self.Y_train.copy()
        # 使用二分类时负类标签为0
        self.Y_binary[self.Y_binary == -1] = 0
        # 将标签转换为one-hot编码
        self.Y_onehot = np.eye(self.num_class)[self.Y_multi.flatten()]
        # 不同类型的分类问题用不同模型
        if self.num_class == 2:
            # 若是二分类则训练二分类模型
            self.train_binary()
        elif self.num_class > 2:
            # 若是多分类则训练多分类模型
            self.train_multi()
        else:
            raise ValueError("The num of class cannot be less than 2")

    def train_binary(self):
        """训练二分类模型"""
        # 初始化初始预测
        self.initials = self.get_binary_initials()
        F_train = np.full_like(self.Y_binary, self.initials, dtype=float)
        # 记录损失值
        self.losses.append(self.cal_binary_loss(F_train))
        # 训练每一个弱回归器以拟合残差
        for i in range(self.n_estimators):
            # 若使用子采样则对样本进行无放回采样
            if self.subsample < 1.0:
                sample_idx = np.random.choice(len(self.X_train), size=len(self.X_train), replace=False)
                X_samples = self.X_train[sample_idx]
                Y_samples = self.Y_binary[sample_idx]
                F_samples = F_train[sample_idx]
            else:
                X_samples = self.X_train
                Y_samples = self.Y_binary
                F_samples = F_train
            # 计算伪残差(负梯度)
            residuals = Y_samples - sigmoid(F_samples)
            # 创建一个基础学习器
            base_model = copy.deepcopy(self.estimator)
            # 训练基础学习器拟合残差
            base_model.train(X_samples, residuals)
            # 更新模型预测(全量样本)
            F_train += self.learning_rate * base_model.predict(self.X_train)
            # 记录损失值
            self.losses.append(self.cal_binary_loss(F_train))
            # 保存基础学习器
            self.estimator_models.append(base_model)

    def train_multi(self):
        """训练多分类模型"""
        # 初始化初始预测
        self.initials = self.get_multi_initials()
        F_train = np.log(self.initials + 1e-15)
        F_train = np.tile(F_train, (len(self.X_train), 1))
        # 记录损失值
        self.losses.append(self.cal_multi_loss(F_train))
        # 训练每一个弱回归器以拟合残差
        for i in range(self.n_estimators):
            # 若使用子采样则对样本进行无放回采样
            if self.subsample < 1.0:
                sample_idx = np.random.choice(len(self.X_train), size=len(self.X_train), replace=False)
                X_samples = self.X_train[sample_idx]
                # Y_samples = self.Y_train[sample_idx]
                Y_onehot_samples = self.Y_onehot[sample_idx]
                F_samples = F_train[sample_idx]

            else:
                X_samples = self.X_train
                # Y_samples = self.Y_train
                Y_onehot_samples = self.Y_onehot
                F_samples = F_train
            # 计算伪残差(负梯度)
            residuals = Y_onehot_samples - softmax(F_samples)
            # 为每一个类别训练一个基础学习器
            models_per_round = []
            for k in range(self.num_class):
                # 创建一个基础学习器
                base_model = copy.deepcopy(self.estimator)
                # 训练基础学习器拟合残差(拟合第k类的残差)
                base_model.train(X_samples, residuals[:, k].reshape(-1, 1))
                # 保存基础学习器
                models_per_round.append(base_model)
                # 更新模型预测
                F_train[:, k] += self.learning_rate * base_model.predict(X_samples).flatten()
            # 记录损失值
            self.losses.append(self.cal_multi_loss(F_train))
            # 保存基础学习器
            self.estimator_models.append(models_per_round)

    def predict(self, X_data):
        """给定数据预测结果"""
        if self.num_class == 2:
            # 若是二分类则预测二分类模型
            return self.predict_binary(X_data)
        elif self.num_class > 2:
            # 若是多分类则预测多分类模型
            return self.predict_multi(X_data)
        else:
            raise ValueError("The num of class cannot be less than 2")

    def predict_binary(self, X_data):
        """对二分类问题进行预测"""
        Y_data = np.ones((len(X_data), 1), dtype=int)
        Y_data[self.predict_binary_prob(X_data) < 0.5] = -1
        return Y_data

    def predict_multi(self, X_data):
        Y_prob = self.predict_multi_prob(X_data)
        Y_data = self.classes[np.argmax(Y_prob, axis=1)]
        return Y_data

    def predict_prob(self, X_data):
        """给定数据预测结果(预测概率)"""
        if self.num_class == 2:
            # 若是二分类则预测二分类模型
            return self.predict_binary_prob(X_data)
        elif self.num_class > 2:
            # 若是多分类则预测多分类模型
            return self.train_multi()
        else:
            raise ValueError("The num of class cannot be less than 2")

    def predict_binary_prob(self, X_data):
        """对二分类问题进行概率预测"""
        F_predict = np.full((len(X_data), 1), self.initials)
        for model in self.estimator_models:
            F_predict += self.learning_rate * model.predict(X_data)
        Y_prob = sigmoid(F_predict)
        return Y_prob

    def predict_multi_prob(self, X_data):
        """对多分类问题进行概率预测"""
        F_predict = np.tile(np.log(self.initials + 1e-15), (len(X_data), 1))
        for models_per_round in self.estimator_models:
            for k, models in enumerate(models_per_round):
                F_predict[:, k] += self.learning_rate * models.predict(X_data).flatten()
        Y_prob = softmax(F_predict)
        return Y_prob

    def get_binary_initials(self):
        """初始化初始预测(二分类)"""
        return np.log(np.mean(self.Y_binary) / (1 - np.mean(self.Y_binary)))

    def get_multi_initials(self):
        """初始化初始预测(多分类)"""
        return np.array([np.mean(self.Y_train == c) for c in self.classes])

    def cal_binary_loss(self, F_train):
        """计算二分类损失"""
        # 计算二元交叉熵
        return -np.mean(self.Y_binary * np.log(sigmoid(F_train)) + (1 - self.Y_binary) * np.log(1 - sigmoid(F_train)))

    def cal_multi_loss(self, F_train):
        """计算多分类损失"""
        # 若是多分类问题则计算交叉熵
        return -np.mean(self.Y_onehot * np.log(softmax(F_train)))

    def plot_2dim(self, X_test=None, Y_test=None, Truth=None):
        """为二维分类数据集和结果画图（只能是连续数据）"""
        X_train = np.array(self.X_train)
        Y_train = np.array(self.Y_train)
        plot_2dim_classification_sample(self, X_train, Y_train, X_test, Y_test, neg_label=-1)

    def plot_loss_change(self):
        model_name = self.__class__.__name__
        plot_changes(self.losses, title=model_name, x_label='n_iter', y_label='train loss')


if __name__ == '__main__':
    np.random.seed(100)
    model = GradientBoostingClassifier(n_estimators=100)
    # 不再展示全样本结果
    run_uniform_classification(model, show=False)
    model.plot_loss_change()
    run_double_classification(model, show=False)
    model.plot_loss_change()
    run_circle_classification(model, show=False)
    model.plot_loss_change()
    run_moons_classification(model, show=False)
    model.plot_loss_change()
