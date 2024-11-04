"""
一对多 分类包装器
One-Vs-Rest Classifier Wrapper
"""
import copy
import warnings
import numpy as np


class OneVsRestClassifier():
    """一对多 分类包装器"""
    def __init__(self, model, X_train=None, Y_train=None):
        self.model = model
        self.X_train = None  # 训练数据
        self.Y_train = None  # 真实标签
        self.set_train_data(X_train, Y_train)
        self.class_states = None
        if not hasattr(self.model, 'predict_prob'):
            raise AttributeError('This model does not have a method for predicting probabilities, '
                                 'so OVR multi classification cannot be used')

    def set_train_data(self, X_train, Y_train):
        """给定训练数据集和标签数据"""
        if X_train is not None:
            if self.X_train is not None:
                warnings.warn("Training data will be overwritten")
            self.X_train = X_train.copy()
        if Y_train is not None:
            if self.Y_train is not None:
                warnings.warn("Training data will be overwritten")
            self.Y_train = Y_train.copy()

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
            if hasattr(model_ovr, 'show'):
                model_ovr.show = False
            model_ovr.train(self.X_train, Y_train_ovr)
            # 保存模型
            self.classifiers.append(model_ovr)

    def predict(self, X_data):
        """预测数据"""
        # 遍历所有的模型，得到每个数据是各个类别的概率
        probs = np.zeros((len(X_data), len(self.classifiers)))
        for i in range(len(self.classifiers)):
            probs[:, i] = self.classifiers[i].predict_prob(X_data).flatten()
        # 然后选择概率最大的元素作为预测结果
        Y_pred = self.class_states[np.argmax(probs, axis=1)].reshape(len(X_data), -1)
        return Y_pred