import warnings
import numpy as np
class Model():
    def __init__(self, X_train=None, Y_train=None):
        """
        模型父类(暂未耦合)(输出为绝大多数类别)
        :param X_train: 训练数据
        :param Y_train: 真实标签
        """
        self.X_train = None  # 训练数据
        self.Y_train = None  # 真实标签
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

    def set_parameters(self, **kwargs):
        """设置或修改相关参数"""
        for param, value in kwargs.items():
            if hasattr(self, param):  # 检查对象是否有该属性
                if getattr(self, param) is not None:
                    warnings.warn(f"Parameter '{param}' will be overwritten")
                setattr(self, param, value)
            else:
                warnings.warn(f"Parameter '{param}' is not a valid parameter for this model")

    def train(self, *args, **kwargs):
        raise NotImplementedError

    def predict(self, X):
        # 使用 np.unique 统计每个元素的出现次数
        uniques, counts = np.unique(self.Y_train, return_counts=True)
        # 找到出现次数最多的元素
        most_element = uniques[np.argmax(counts)]
        # 生成一个与输入数组形状相同的数组，其中每个元素都是出现次数最多的元素
        result = np.full((len(X), 1), most_element)
        return result

    def predict_prob(self):
        pass

    def plot_2dim(self):
        raise NotImplementedError
