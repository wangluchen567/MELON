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
import numpy as np
from Models import Model
from Models.Utils import softmax, plot_2dim_classification, run_uniform_classification, run_double_classification


class GaussianNaiveBayes(Model):
    def __init__(self, X_train=None, Y_train=None, priors=None, var_smoothing=1.e-9):
        """
        高斯朴素贝叶斯
        :param X_train: 训练数据
        :param Y_train: 真实标签
        :param priors: 每类数据的先验概率
        :param var_smoothing: 方差平滑值，避免方差过小
        """
        super().__init__(X_train, Y_train)
        self.priors = priors  # 每类数据的先验概率
        self.var_smoothing = var_smoothing  # 方差平滑值，避免方差过小
        self.have_priors = False if self.priors is None else True
        self.classes = None  # 保存训练数据的类别
        self.mean = None  # 保存训练数据的均值
        self.var = None  # 保存训练数据的方差

    def train(self, X_train=None, Y_train=None):
        """使用数据集训练模型"""
        # 设置数据集和参数
        self.set_train_data(X_train, Y_train)
        self.classes = np.unique(self.Y_train)  # 获取类别
        num_class = len(self.classes)  # 类别数量
        num_data, num_feat = self.X_train.shape  # 数据与特征数量
        # 初始化训练数据的均值和方差以及先验概率
        self.mean = np.zeros((num_class, num_feat))
        self.var = np.zeros((num_class, num_feat))
        if not self.have_priors:
            self.priors = np.zeros(num_class)
        # 计算每类数据的均值和方差以及先验概率
        for idx, k in enumerate(self.classes):
            X_k = self.X_train[(self.Y_train == k).flatten()]
            self.mean[idx, :] = np.mean(X_k, axis=0)
            self.var[idx, :] = np.var(X_k, axis=0)
            if not self.have_priors:
                self.priors[idx] = len(X_k) / num_data
        # 为了提高方差估计的稳定性，将所有特征的最大方差的一部分
        # 添加到每个特征的方差中，从而避免方差过小导致的数值计算问题
        max_variance = np.max(self.var)
        self.var += self.var_smoothing * max_variance

    def predict(self, X_data):
        """模型对测试集进行预测"""
        if X_data.ndim == 2:
            pass
        elif X_data.ndim == 1:
            X_data = X_data.reshape(1, -1)
        else:
            raise ValueError("Cannot handle data with a shape of 3 dimensions or more")
        # 对测试数据进行预测概率
        Y_data_prob = self.predict_prob(X_data)
        # 选择概率最大的类别作为输出类别
        Y_data = self.classes[np.argmax(Y_data_prob, axis=1)]
        Y_data = Y_data.reshape(-1, 1)
        return Y_data

    def predict_prob(self, X_data):
        """模型对测试集进行预测(概率)"""
        if X_data.ndim == 2:
            pass
        elif X_data.ndim == 1:
            X_data = X_data.reshape(1, -1)
        else:
            raise ValueError("Cannot handle data with a shape of 3 dimensions or more")
        # 计算对数先验概率
        log_priors = np.log(self.priors)
        # 计算测试集的对数似然
        log_likelihoods = self.cal_log_likelihood(X_data)
        # 计算测试集的后验概率
        log_posteriors = log_likelihoods + log_priors
        # 使用softmax归一化后验概率
        Y_data_prob = softmax(log_posteriors)
        return Y_data_prob

    def cal_log_likelihood(self, X_data):
        """批量计算每个样本在每个类别下的条件概率的对数(对数似然)"""
        num_class = len(self.classes)  # 类别数量
        num_data, num_feat = X_data.shape  # 数据与特征数量
        log_likelihoods = np.zeros((num_data, num_class))
        # 计算每个类别下的对数似然
        for idx in range(num_class):
            mean = self.mean[idx]
            var = self.var[idx]
            log_likelihoods[:, idx] = -0.5 * (np.sum(((X_data - mean) ** 2) / var, axis=1) +
                                              np.sum(np.log(2 * np.pi * var)))
        return log_likelihoods

    def plot_2dim(self, X_test=None, Y_test=None, Truth=None, pause=False):
        """为二维分类数据集和结果画图"""
        plot_2dim_classification(self.X_train, self.Y_train, None, X_test, Y_test, Truth=Truth, pause=pause)


if __name__ == '__main__':
    np.random.seed(100)
    model = GaussianNaiveBayes()
    run_uniform_classification(model)
    run_double_classification(model)
