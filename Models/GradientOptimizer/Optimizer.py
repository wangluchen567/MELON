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
import numpy as np


class Optimizer:
    def __init__(self, model, learning_rate):
        """
        梯度下降优化器
        :param model: 需要优化的模型
        :param learning_rate: 学习率
        """
        self.model = model  # 需要优化的模型
        # PS: 模型必须提供要优化的参数和求得的梯度
        assert learning_rate >= 0
        self.learning_rate = learning_rate  # 学习率
        # 统计优化步数
        self.steps = 0

    def step(self):
        """更新一次权重参数"""
        raise NotImplementedError

    def update(self):
        """更新梯度的更新速度"""
        raise NotImplementedError


class GradientDescent(Optimizer):
    def __init__(self, model, learning_rate=0.01):
        super(GradientDescent, self).__init__(model, learning_rate)
        # 记录梯度更新速度
        self.v = 0

    def step(self):
        """更新一次权重参数"""
        self.steps += 1  # 更新步数
        # 先更新梯度更新速度v
        self.update_v()
        # 再更新权重
        self.update()

    def update_v(self):
        """更新梯度更新速度"""
        self.v = self.learning_rate * self.model.Grad

    def update(self):
        """更新权重"""
        self.model.Weights = self.model.Weights - self.v


class Momentum(Optimizer):
    def __init__(self, model, learning_rate=0.01, momentum=0.9):
        super(Momentum, self).__init__(model, learning_rate)
        self.momentum = momentum
        # 记录梯度更新速度
        self.v = 0

    def step(self):
        """每层网络更新一次权重"""
        self.steps += 1  # 更新步数
        # 先更新梯度更新速度v
        self.update_v()
        # 再更新权重
        self.update()

    def update_v(self):
        """更新梯度更新速度"""
        self.v = self.momentum * self.v + self.learning_rate * self.model.Grad

    def update(self):
        """更新权重"""
        self.model.Weights = self.model.Weights - self.v


class AdaGrad(Optimizer):
    def __init__(self, model, learning_rate=0.01):
        super(AdaGrad, self).__init__(model, learning_rate)
        # 记录梯度各分量的平方
        self.s = 0

    def step(self):
        """每层网络更新一次权重"""
        self.steps += 1  # 更新步数
        # 先更新梯度各分量的平方s
        self.update_s()
        # 再更新权重
        self.update()

    def update_s(self):
        """更新梯度各分量平方更新速度"""
        self.s = self.s + self.model.Grad * self.model.Grad

    def update(self):
        """更新权重"""
        self.model.Weights = self.model.Weights - self.learning_rate * self.model.Grad / np.sqrt(self.s + 1e-8)


class RMSProp(Optimizer):
    def __init__(self, model, learning_rate=0.01, beta=0.99):
        super(RMSProp, self).__init__(model, learning_rate)
        # 衰减系数
        assert 0.0 < beta < 1.0
        self.beta = beta
        # 记录梯度各分量的平方
        self.s = 0

    def step(self):
        """每层网络更新一次权重"""
        self.steps += 1  # 更新步数
        # 先更新梯度各分量的平方s
        self.update_s()
        # 再更新权重
        self.update()

    def update_s(self):
        """更新梯度各分量平方更新速度"""
        self.s = self.beta * self.s + (1 - self.beta) * self.model.Grad * self.model.Grad

    def update(self):
        """更新权重"""
        self.model.Weights = self.model.Weights - self.learning_rate * self.model.Grad / np.sqrt(self.s + 1e-8)


class Adam(Optimizer):
    def __init__(self, model, learning_rate=0.01, beta_1=0.9, beta_2=0.999):
        super(Adam, self).__init__(model, learning_rate)
        # 历史梯度衰减系数
        assert 0.0 < beta_1 < 1.0
        self.beta_1 = beta_1
        # 历史梯度各分量平方衰减系数
        assert 0.0 < beta_2 < 1.0
        self.beta_2 = beta_2
        # 历史梯度累积
        self.v = 0
        # 梯度各分量的平方累积
        self.s = 0

    def step(self):
        """每层网络更新一次权重"""
        self.steps += 1  # 更新步数
        # 先更新梯度更新速度v
        self.update_v()
        # 再更新梯度各分量的平方s
        self.update_s()
        # 再更新权重
        self.update()

    def update_v(self):
        """更新梯度更新速度"""
        self.v = self.beta_1 * self.v + (1 - self.beta_1) * self.model.Grad

    def update_s(self):
        """更新梯度各分量平方更新速度"""
        self.s = self.beta_2 * self.s + (1 - self.beta_2) * self.model.Grad * self.model.Grad

    def update(self):
        """更新权重"""
        # 进行偏差校正
        v_cor = self.v / (1 - self.beta_1 ** self.steps)
        s_cor = self.s / (1 - self.beta_2 ** self.steps)
        self.model.Weights = self.model.Weights - self.learning_rate * v_cor / np.sqrt(s_cor + 1e-8)
