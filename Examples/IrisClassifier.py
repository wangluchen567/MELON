import numpy as np
import pandas as pd

from Models.LinearClassifier.Perceptron import Perceptron
from Models.LinearClassifier.LogisticRegression import LogisticRegression
from Models.LinearClassifier.GaussianDiscriminant import GaussianDiscriminant
from Models.LinearClassifier.FisherLinearDiscriminant import FisherLinearDiscriminant


def Normalize(data, min_value=0, max_value=1):
    data_max = np.max(data)
    data_min = np.min(data)
    if data_max == data_min:
        return data
    else:
        return (data - data_min) * (max_value - min_value) / (data_max - data_min) + min_value


def load_iris_data():
    # 读取鸢尾花数据集
    data = pd.read_csv("../Dataset/Iris.csv")
    # 将数据集中的每种花换成整数0, 1, 2
    data.iloc[np.where(data['Species'] == 'Iris-setosa')[0], -1] = 0
    data.iloc[np.where(data['Species'] == 'Iris-versicolor')[0], -1] = 1
    data.iloc[np.where(data['Species'] == 'Iris-virginica')[0], -1] = 2
    # 将Species列的数据设置类型为int
    data['Species'] = data['Species'].astype(int)
    # 数据集特征
    features = data[['SepalLengthCm',
                     'SepalWidthCm',
                     'PetalLengthCm',
                     'PetalWidthCm']].values
    # 数据集标签
    labels = data[['Species']].values
    return features, labels


def load_classifier_data(feat_pos, chose_label=None, pos_label=1, neg_label=-1, ratio=0.8):
    """
    读取分类数据
    :param feat_pos: 选择的特征
    :param chose_label: 选择的标签
    :param pos_label: 正例的标签
    :param neg_label: 负例的标签
    :param ratio: 训练集的比例
    :return: 训练集和验证集
    """
    # 获取数据集
    features, labels = load_iris_data()
    # 从数据集中选取两类进行分类
    if chose_label is None: chose_label = [0, 1]
    chosen = np.sum(labels == chose_label, axis=1, dtype=bool)
    X_Data = features[chosen, :]
    Y_Data = labels[chosen, :]
    """注意两个类的类别标签需要调整"""
    Y_Data[Y_Data == chose_label[0]] = neg_label
    Y_Data[Y_Data == chose_label[1]] = pos_label
    # 选择一部分特征训练
    X_Data = X_Data[:, feat_pos]
    # 归一化数据集
    # X_Data = Normalize(X_Data, min_value=-1, max_value=1)
    # 打乱数据集
    random_index = np.arange(len(X_Data))
    np.random.shuffle(random_index)
    X_Data = X_Data[random_index]
    Y_Data = Y_Data[random_index]
    # 划分训练集和验证集
    train_size = int(ratio * len(X_Data))
    X_train = X_Data[:train_size]
    Y_train = Y_Data[:train_size]
    X_valid = X_Data[train_size:]
    Y_valid = Y_Data[train_size:]
    return X_train, Y_train, X_valid, Y_valid


def run_iris_classifier(model):
    # 设置正负标签值
    pos_label, neg_label = 1, -1
    model_name = type(model).__name__
    print("Model: ", model_name)
    if model_name == 'LogisticRegression':
        neg_label = 0
    # 获取数据集
    X_train, Y_train, X_valid, Y_valid = load_classifier_data(feat_pos=[0, 1], chose_label=[0, 2], pos_label=pos_label,
                                                              neg_label=neg_label, ratio=0.8)
    # 使用数据集对模型训练
    model.train(X_train, Y_train)
    # 训练后的模型参数
    print("Model Weights: ", model.Weights.flatten())
    # 画图展示效果
    model.plat_2dim()
    # 训练准确率计算
    Y_Pred = model.predict(X_train)
    # 计算训练准确率
    train_accuracy = np.array(Y_Pred == Y_train, dtype=int).sum() / len(Y_train)
    print("Train Accuracy:  {:.3f} %".format(train_accuracy * 100))
    # 对验证集进行预测
    y_predict = model.predict(X_valid)
    print("Predict Labels: ", y_predict.flatten())
    # 计算验证集准确率
    valid_accuracy = np.array(y_predict == Y_valid, dtype=int).sum() / len(Y_valid)
    print("Valid Accuracy:  {:.3f} %".format(valid_accuracy * 100))
    # 画图展示效果
    model.plat_2dim(X_valid, Y_valid)


if __name__ == '__main__':
    models = [FisherLinearDiscriminant(),
              GaussianDiscriminant(),
              LogisticRegression(epochs=100, lr=0.01, grad_type='Adam'),
              Perceptron(epochs=100, lr=0.01, grad_type='Adam')]
    model = models[0]
    run_iris_classifier(model)
