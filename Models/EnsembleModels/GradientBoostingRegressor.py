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
from Models.DecisionTree import DecisionTreeRegressor
from Models.Utils import plot_changes, plot_2dim_regression_sample
from Models.Utils import run_uniform_regression, run_circular_regression, run_poly_regression


class GradientBoostingRegressor(Model):
    def __init__(self, estimator=None, X_train=None, Y_train=None, n_estimators=100,
                 loss_type='mse', learning_rate=0.1, subsample=1.0, max_depth=3):
        """
        梯度提升回归器
        :param estimator: 基础学习器(弱学习器)
        :param X_train: 训练数据
        :param Y_train: 真实标签
        :param n_estimators: 基础学习器数量
        :param loss_type: 损失函数类型('mse'/'mae')
        :param learning_rate: 学习率，每步拟合残差的学习率
        :param subsample: 子采样率，用于创建随机梯度提升树，取值范围为 (0, 1]
        :param max_depth: 当基础学习器为决策树时决策树最大深度
        """
        super().__init__(X_train, Y_train)
        self.estimator = estimator  # 基础学习器(弱学习器)
        self.n_estimators = n_estimators  # 基础学习器数量
        self.loss_type = loss_type  # 损失函数类型
        self.learning_rate = learning_rate  # 学习率
        self.subsample = subsample  # 子采样率
        assert 0.0 < self.subsample <= 1.0
        self.max_depth = max_depth  # 当基础学习器为决策树时决策树最大深度
        self.estimator_models = None  # 初始化基础估计器集合
        self.losses = None  # 记录损失值历史
        self.initials = None  # 初始化初始预测
        if self.estimator is None:
            # 默认使用决策树模型
            self.estimator = DecisionTreeRegressor(max_depth=self.max_depth)

    def train(self, X_train=None, Y_train=None):
        """训练模型拟合数据"""
        self.set_train_data(X_train, Y_train)
        self.estimator_models = []
        self.losses = []  # 原损失值历史置空
        # 初始化初始预测
        self.initials = self.get_initials()
        F_train = np.full_like(self.Y_train, self.initials, dtype=float)
        # 记录损失值
        self.losses.append(self.cal_loss(F_train))
        # 训练每一个弱回归器以拟合残差
        for i in range(self.n_estimators):
            # 若使用子采样则对样本进行无放回采样
            if self.subsample < 1.0:
                sample_idx = np.random.choice(len(self.X_train), size=len(self.X_train), replace=False)
                X_samples = self.X_train[sample_idx]
                Y_samples = self.Y_train[sample_idx]
                F_samples = F_train[sample_idx]
            else:
                X_samples = self.X_train
                Y_samples = self.Y_train
                F_samples = F_train
            # 计算伪残差(负梯度)
            residuals = self.cal_residual(Y_samples, F_samples)
            # 创建一个基础学习器
            base_model = copy.deepcopy(self.estimator)
            # 训练基础学习器拟合残差
            base_model.train(X_samples, residuals)
            # 更新模型预测(全量样本)
            F_train += self.learning_rate * base_model.predict(self.X_train)
            # 记录损失值
            self.losses.append(self.cal_loss(F_train))
            # 保存基础学习器
            self.estimator_models.append(base_model)

    def predict(self, X_data):
        """给定数据预测结果"""
        Y_data = np.full((len(X_data), 1), self.initials)
        for base_model in self.estimator_models:
            Y_data += self.learning_rate * base_model.predict(X_data)
        return Y_data

    def get_initials(self):
        """初始化预测值"""
        if self.loss_type.lower() == 'mse':
            return np.mean(self.Y_train)
        elif self.loss_type.lower() == 'mae':
            return np.median(self.Y_train)
        else:
            raise ValueError(f"There is no such loss function type: {self.loss_type}")

    def cal_loss(self, F_train):
        """计算损失值"""
        if self.loss_type.lower() == 'mse':
            return np.mean((self.Y_train.flatten() - F_train.flatten()) ** 2)
        elif self.loss_type.lower() == 'mae':
            return np.mean(np.abs(self.Y_train.flatten() - F_train.flatten()))
        else:
            raise ValueError(f"There is no such loss function type: {self.loss_type}")

    def cal_residual(self, Y, F):
        """计算伪残差(负梯度)"""
        if self.loss_type.lower() == 'mse':
            return Y - F
        elif self.loss_type.lower() == 'mae':
            return np.sign(Y - F)
        else:
            raise ValueError(f"There is no such loss function type: {self.loss_type}")

    def plot_2dim(self, X_test=None, Y_test=None, Truth=None):
        """为二维回归数据集和结果画图（只能是连续数据）"""
        X_train = np.array(self.X_train)
        Y_train = np.array(self.Y_train)
        plot_2dim_regression_sample(self, X_train, Y_train, X_test, Y_test)

    def plot_loss_change(self):
        model_name = self.__class__.__name__
        plot_changes(self.losses, title=model_name, x_label='n_iter', y_label='train loss')


if __name__ == '__main__':
    np.random.seed(100)
    model = GradientBoostingRegressor()
    run_uniform_regression(model)
    run_poly_regression(model)
    run_circular_regression(model)
    model.plot_loss_change()
