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
import pandas as pd
from Models.Utils import normalize, calculate_accuracy
from Models.NaiveBayes import *
from Models.LinearClassifier import *
from Models.MultiClassWrapper import *
from Models.DiscriminantAnalysis import *
from Models.NeighborsBased import KNeighborsClassifier
from Models.DecisionTree import DecisionTreeClassifier
from Models.EnsembleModels import RandomForestClassifier
from Models.SupportVectorMachine import SupportVectorClassifier


def load_iris_data():
    # 读取鸢尾花数据集
    data = pd.read_csv("../Dataset/Iris.csv")
    # 数据集特征
    features = data[['SepalLengthCm',
                     'SepalWidthCm',
                     'PetalLengthCm',
                     'PetalWidthCm']].values
    # 数据集标签
    labels = data[['Species']].values
    return features, labels


def load_classifier_data(feat_pos=None, ratio=0.8):
    """
    读取分类数据
    :param feat_pos: 选择的特征
    :param ratio: 训练集的比例
    :return: 训练集和测试集
    """
    # 获取数据集
    X_data, Y_data = load_iris_data()
    if feat_pos is not None:
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


def run_classifier(model, X_train, Y_train, X_test, Y_test):
    """给定分类器模型运行分类器"""
    if hasattr(model, 'model'):
        model_name = type(model.model).__name__
    else:
        model_name = type(model).__name__
    print("Model: ", model_name)
    # 使用数据集对模型训练
    model.train(X_train, Y_train)
    # 训练准确率计算
    Y_train_pred = model.predict(X_train)
    # 计算训练准确率
    print("Train Accuracy:  {:.3f} %".format(calculate_accuracy(Y_train_pred, Y_train) * 100))
    # 对测试集进行预测
    Y_test_pred = model.predict(X_test)
    # print("Predict Labels: ", Y_test_pred.flatten())
    # 计算测试集准确率
    print("Test Accuracy:  {:.3f} %\n".format(calculate_accuracy(Y_test_pred, Y_test) * 100))


if __name__ == '__main__':
    np.random.seed(0)
    # 获取数据集
    X_train, Y_train, X_test, Y_test = load_classifier_data()
    # 构建模型
    models = [FisherDiscriminantAnalysis(),
              GaussianDiscriminantAnalysis(),
              RidgeClassifier(),
              GDClassifier(tol=1.e-4),
              Perceptron(tol=1.e-4, lr=2),
              LogisticRegression(tol=1.e-4),
              SupportVectorClassifier(kernel=SupportVectorClassifier.LINEAR),
              SupportVectorClassifier(kernel=SupportVectorClassifier.RBF)]
    # 使用 一对一 分类器分类
    print("# One-Vs-One Multi-Classifier #")
    for model in models:
        # 将模型用多分类器封装
        mc_model = OneVsOneClassifier(model)
        run_classifier(mc_model, X_train, Y_train, X_test, Y_test)
    # 使用 一对多 分类器分类
    print("# One-Vs-Rest Multi-Classifier #")
    for model in models:
        # 将模型用多分类器封装
        mc_model = OneVsRestClassifier(model)
        run_classifier(mc_model, X_train, Y_train, X_test, Y_test)
    print("# Direct Multi-Classifier #")
    # 朴素贝叶斯模型可以直接用于多分类
    bayes_model = GaussianNaiveBayes()
    run_classifier(bayes_model, X_train, Y_train, X_test, Y_test)
    # 线性判别分析可以直接用于多分类
    lda_model = LinearDiscriminantAnalysis(n_components=2)
    run_classifier(lda_model, X_train, Y_train, X_test, Y_test)
    # KNN模型可以直接用于多分类
    knn_model = KNeighborsClassifier(weights='distance')
    run_classifier(knn_model, X_train, Y_train, X_test, Y_test)
    # 随机森林模型可以直接用于多分类
    dtc_model = RandomForestClassifier(n_estimators=10)
    run_classifier(dtc_model, X_train, Y_train, X_test, Y_test)
    # 决策树模型可以直接用于多分类
    dtc_model = DecisionTreeClassifier(max_depth=3, criterion='entropy')
    run_classifier(dtc_model, X_train, Y_train, X_test, Y_test)
    dtc_model.plot_tree()
