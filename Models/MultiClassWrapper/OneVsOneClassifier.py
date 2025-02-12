"""
一对一 分类包装器
One-Vs-One Classifier Wrapper
"""
import copy
import warnings
import numpy as np


class OneVsOneClassifier():
    """一对一 分类包装器"""

    def __init__(self, model, X_train=None, Y_train=None):
        """
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
                if hasattr(model_ovo, 'show'):
                    model_ovo.show = False
                model_ovo.train(X_train_binary, Y_train_ovo)
                # 保存模型
                self.classifiers.append(model_ovo)

    def predict(self, X_data):
        """预测数据"""
        # 遍历所有的模型，得到每对类别的分类情况
        votes = np.zeros((len(X_data), len(self.class_states)))  # 记录投票情况
        for i in range(len(self.classifiers)):
            y_predict = self.classifiers[i].predict(X_data).flatten()
            y_predict[y_predict == -1] = 0
            predict = self.binary_states[i][y_predict]
            # 使用 np.add.at 对选中的那类进行投票
            row_indices = np.arange(predict.shape[0])
            col_indices = np.array(predict)
            # 要添加的值，这里是加1
            values = np.ones(len(row_indices))
            # 使用 np.add.at 更新投票情况
            np.add.at(votes, (row_indices, col_indices), values)
        # 然后选择概率最大的元素作为预测结果
        Y_pred = self.class_states[np.argmax(votes, axis=1)].reshape(len(X_data), -1)
        return Y_pred
