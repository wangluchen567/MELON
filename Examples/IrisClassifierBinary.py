import numpy as np
import pandas as pd
from Models.Utils import normalize
from Models.LinearClassifier.Perceptron import Perceptron
from Models.LinearClassifier.LogisticRegression import LogisticRegression
from Models.DecisionTree.DecisionTreeClassifier import DecisionTreeClassifier
from Models.SupportVectorMachine.SupportVectorClassifier import SupportVectorClassifier
from Models.DiscriminantAnalysis.FisherDiscriminantAnalysis import FisherDiscriminantAnalysis
from Models.DiscriminantAnalysis.GaussianDiscriminantAnalysis import GaussianDiscriminantAnalysis



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
    :return: 训练集和测试集
    """
    # 获取数据集
    features, labels = load_iris_data()
    # 从数据集中选取两类进行分类
    if chose_label is None: chose_label = [0, 1]
    chosen = np.sum(labels == chose_label, axis=1, dtype=bool)
    X_data = features[chosen, :]
    Y_data = labels[chosen, :]
    """注意两个类的类别标签需要调整"""
    Y_data[Y_data == chose_label[0]] = neg_label
    Y_data[Y_data == chose_label[1]] = pos_label
    # 选择一部分特征训练
    X_data = X_data[:, feat_pos]
    # 归一化数据集
    X_data = normalize(X_data, axis=0)
    # 打乱数据集
    random_index = np.arange(len(X_data))
    np.random.shuffle(random_index)
    X_data = X_data[random_index]
    Y_data = Y_data[random_index]
    # 划分训练集和测试集
    train_size = int(ratio * len(X_data))
    X_train, Y_train = X_data[:train_size], Y_data[:train_size]
    X_test, Y_test = X_data[train_size:], Y_data[train_size:]
    return X_train, Y_train, X_test, Y_test


def run_iris_classifier(model):
    model_name = type(model).__name__
    print("Model: ", model_name)
    # 获取数据集
    X_train, Y_train, X_test, Y_test = load_classifier_data(feat_pos=[0, 1], chose_label=[0, 2])
    # 使用数据集对模型训练
    model.train(X_train, Y_train)
    # 训练后的模型参数
    if hasattr(model, 'Weights'):
        print("Model Weights: ", model.Weights.flatten())
    # 画图展示效果
    model.plot_2dim()
    # 训练准确率计算
    Y_train_pred = model.predict(X_train)
    # 计算训练准确率
    train_accuracy = np.array(Y_train_pred == Y_train, dtype=int).sum() / len(Y_train)
    print("Train Accuracy:  {:.3f} %".format(train_accuracy * 100))
    # 对测试集进行预测
    Y_test_pred = model.predict(X_test)
    print("Predict Labels: ", Y_test_pred.flatten())
    # 计算测试集准确率
    test_accuracy = np.array(Y_test_pred == Y_test, dtype=int).sum() / len(Y_test)
    print("Test Accuracy:  {:.3f} %".format(test_accuracy * 100))
    # 画图展示效果
    model.plot_2dim(X_test, Y_test)


if __name__ == '__main__':
    models = [FisherDiscriminantAnalysis(),
              GaussianDiscriminantAnalysis(),
              LogisticRegression(epochs=100, lr=0.01, grad_type='Adam'),
              Perceptron(epochs=100, lr=0.01, grad_type='Adam'),
              DecisionTreeClassifier(),
              SupportVectorClassifier(kernel=SupportVectorClassifier.RBF)]
    model = models[4]
    run_iris_classifier(model)
