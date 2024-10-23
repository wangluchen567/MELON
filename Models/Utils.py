import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def calculate_accuracy(Truth, Predict):
    """
    计算分类结果的准确率
    :param Truth: 真实标签
    :param Predict: 预测标签
    :return: 准确率
    """
    if len(Truth.flatten()) != len(Predict.flatten()):
        raise ValueError("The number of real labels and predicted labels does not match")
    accuracy = np.array(Truth.flatten() == Predict.flatten(), dtype=int).sum() / len(Truth.flatten())
    return accuracy


def cal_mse_metrics(Truth, Predict):
    """
    计算回归结果的MSE指标值
    :param Truth: 真实值
    :param Predict: 预测值
    :return: 准确率
    """
    if len(Truth.flatten()) != len(Predict.flatten()):
        raise ValueError("The number of real labels and predicted labels does not match")
    mse = np.mean((Truth.flatten() - Predict.flatten()) ** 2)
    return mse


def random_generate_classification(X_size=100, X_feat=2, X_lower=-1, X_upper=1, lower=-1, upper=1):
    """
    随机生成分类数据集
    :param X_size: 数据集大小
    :param X_feat: 数据集特征数
    :param X_lower: 随机生成的数据的下界
    :param X_upper: 随机生成的数据的上界
    :param lower: 随机生成参数的范围最小值
    :param upper: 随机生成参数的范围最大值
    :return: 训练数据和真实参数
    """
    X_data = np.random.uniform(X_lower, X_upper, size=(X_size, X_feat))
    X_mids = (np.max(X_data, axis=0) + np.min(X_data, axis=0)) / 2
    TruthWeights = np.random.uniform(lower, upper, size=(X_feat, 1))
    Bias = - X_mids.reshape(1, -1) @ TruthWeights
    TruthWeights = np.concatenate((TruthWeights, Bias), axis=0)
    X_B = np.concatenate((X_data, np.ones((len(X_data), 1))), axis=1)
    Y_data = np.ones((len(X_data), 1))
    Y_data[X_B.dot(TruthWeights) < 0] = -1
    return X_data, Y_data, TruthWeights


def random_generate_double(X_size=100, X_feat=2, X_lower=-1, X_upper=1):
    """
    随机生成两点散布的分类数据集
    :param X_size: 数据集大小
    :param X_feat: 数据集特征数
    :param X_lower: 随机生成的数据的下界
    :param X_upper: 随机生成的数据的上界
    :param lower: 随机生成的范围最小值
    :param upper: 随机生成的范围最大值
    :return: 训练数据
    """
    # 确定随机的两个点坐标
    point1 = np.zeros(X_feat)
    # 先随机生成前n-1个变量值
    point1[:-1] = np.random.uniform(X_lower, X_upper, size=(1, X_feat - 1))
    # 得到剩下的一个变量值
    point1[-1] = np.sqrt(1 - np.sum(point1[:-1] ** 2))
    point2 = -point1
    conv = np.eye(len(point1)) * (X_upper - X_lower) * 0.1
    X1 = np.random.multivariate_normal(point1, conv, size=int(X_size / 2))
    X2 = np.random.multivariate_normal(point2, conv, size=int(X_size / 2))
    X_data = np.concatenate((X1, X2))
    Y_data = np.ones((len(X_data), 1))
    Y_data[:len(X1), :] = -1
    rand_index = np.arange(0, len(X_data))
    np.random.shuffle(rand_index)
    X_data = X_data[rand_index, :]
    Y_data = Y_data[rand_index, :]
    return X_data, Y_data


def random_generate_regression(X_size=100, X_feat=1, X_lower=0, X_upper=20, lower=-1, upper=1, loc=0, scale=0.3):
    """
    随机生成回归数据集
    :param X_size: 数据集大小
    :param X_feat: 数据集特征数
    :param X_lower: 随机生成的数据的下界
    :param X_upper: 随机生成的数据的上界
    :param lower: 随机生成的参数范围最小值
    :param upper: 随机生成的参数范围最大值
    :param loc: 扰动期望
    :param scale: 扰动方差
    :return: 训练数据与真实参数
    """
    X_data = np.random.uniform(X_lower, X_upper, size=(X_size, X_feat))
    # 在数据最后一列添加一列单位矩阵作为转置b
    X_B = np.concatenate((X_data, np.ones((len(X_data), 1))), axis=1)
    # 随机生成的模型参数
    Truth_Weights = np.random.uniform(lower, upper, size=(X_feat + 1, 1))
    # 计算输出值
    Y_data = X_B.dot(Truth_Weights)
    # 加入扰动，正态分布扰动
    Y_data += np.random.normal(loc, scale, size=Y_data.shape)
    return X_data, Y_data, Truth_Weights


def random_generate_cluster(X_size=100, X_feat=2, k=3, cluster_std=1.0, lower=-10, upper=10):
    """
    随机生成聚类数据集
    :param X_size: 数据集大小
    :param X_feat: 数据集特征数
    :param k: 数据集分类个数
    :param cluster_std: 随机生成正态分布数据的标准差(宽度)
    :param lower: 聚类中心范围下界
    :param upper: 聚类中心范围上界
    :return: 聚类数据和标签
    """
    # 随机得到聚类中心位置
    centers = np.random.uniform(lower, upper, size=(k, X_feat))
    # 初始化数据集和标签
    X_data = np.zeros((X_size, X_feat))
    Y_data = np.zeros(X_size, dtype=int)
    # 计算每个类中数据点的数量
    num_points = X_size // centers.shape[0]
    extra_points = X_size % centers.shape[0]
    # 为每个中心点划分数据得到聚类数据集
    for i, center in enumerate(centers):
        # 若无法平均划分则需要额外考虑
        n_samples_i = num_points + (1 if i < extra_points else 0)
        # 根据聚类中心随机生成正态分布的数据
        X_data[i * num_points:(i + 1) * num_points, :] = np.random.normal(loc=center, scale=cluster_std,
                                                                          size=(n_samples_i, X_feat))
        Y_data[i * num_points:(i + 1) * num_points] = i
    return X_data, Y_data


def plot_data(X, hold=False):
    """对任意维度数据进行绘图"""
    plt.figure()
    X_dim = X.shape[1]
    if X_dim == 2:
        plt.scatter(X[:, 0], X[:, 1], marker="o", c="blue")
        plt.grid()
    elif X_dim == 3:
        ax = plt.subplot(111, projection='3d')
        ax.scatter(X[:, 0], X[:, 1], X[:, 2], marker="o", c="blue")
        # 设置三维图像角度(仰角方位角)
        # ax.view_init(elev=20, azim=20)
    else:
        for i in range(len(X)):
            plt.plot(np.arange(1, X_dim + 1), X[i, :])
    if not hold:
        plt.show()


def plot_2dim_classification(X_data, Y_data, Weights, X_test=None, Y_test=None, neg_label=-1, Truth=None, support=None,
                             ratio=0.15, pause=False, n_iter=None, pause_time=0.15):
    """
    为二维分类数据集和结果画图 (可动态迭代)
    :param X_data: 训练数据
    :param Y_data: 训练数据的标签
    :param Weights: 训练得到的参数
    :param X_test: 预测数据
    :param Y_test: 预测数据的标签
    :param neg_label: 负标签的值 (-1/0)
    :param Truth: 数据集生成时的真实参数
    :param support: 是否是支持向量
    :param ratio: 设置两边伸展的额外比例
    :param pause: 画图是否暂停 (为实现动态迭代)
    :param n_iter: 当前迭代的代数
    :param pause_time: 迭代过程中暂停的时间间隔
    :return: None
    """
    X = X_data
    Y = Y_data
    Predict = Weights
    if not pause: plt.figure()
    plt.clf()
    plt.scatter(X[Y.flatten() == 1, 0], X[Y.flatten() == 1, 1], c='red')
    plt.scatter(X[Y.flatten() == neg_label, 0], X[Y.flatten() == neg_label, 1], c='blue')
    if X_test is not None and Y_test is not None:  # 用于画预测的点
        plt.scatter(X_test[Y_test.flatten() == 1, 0], X_test[Y_test.flatten() == 1, 1], c='red', marker='*', s=120,
                    edgecolors='black', linewidths=0.5)
        plt.scatter(X_test[Y_test.flatten() == neg_label, 0], X_test[Y_test.flatten() == neg_label, 1], c='blue',
                    marker='*', s=120, edgecolors='black', linewidths=0.5)
    if support is not None:
        plt.scatter(X[support, 0], X[support, 1], s=150, c='none', linewidth=1.5, edgecolor='red')
    if Truth is not None:
        # 绘制真实的参数
        PX, PU = get_PXU_classification(X, Truth)
        plt.plot(PX, PU, c='orange', linewidth=5)
    if Predict is not None:
        # 绘制预测的参数
        PX, PU = get_PXU_classification(X, Predict)
        plt.plot(PX, PU, c='red', linewidth=2)
        # 为了方便展示，两边进行额外延伸
        X0_min, X0_max = np.min(X[:, 0]), np.max(X[:, 0])
        X1_min, X1_max = np.min(X[:, 1]), np.max(X[:, 1])
        X0_gap = (X0_max - X0_min) * ratio
        X1_gap = (X1_max - X1_min) * ratio
        plt.xlim([X0_min - X0_gap, X0_max + X0_gap])
        plt.ylim([X1_min - X1_gap, X1_max + X1_gap])
    plt.grid()
    if pause:
        if n_iter:
            plt.title("iter: " + str(n_iter))
        plt.pause(pause_time)
    else:
        plt.show()


def get_PXU_classification(X, Weights, ratio=0.3, step=0.1):
    """
    获取分类任务画图使用的X和其他未知变量
    :param X: 要画图的已知变量
    :param Weights: 要画图的权重
    :param ratio: 两边伸展的额外比例
    :param step: 采样频率
    :return:
    """
    gap = max(X[:, 0]) - min(X[:, 0])
    PX = np.arange(min(X[:, 0]) - ratio * gap, max(X[:, 0]) + ratio * gap, step)
    PX_B = np.concatenate((PX.reshape(-1, 1), np.ones((len(PX), 1))), axis=1)
    PW = np.concatenate((Weights[:-2, :], Weights[-1, :].reshape(-1, 1)), axis=1) / -Weights[-2, :]
    PU = PX_B.dot(PW.T)
    return PX, PU


def plot_2dim_regression(X_data, Y_data, Weights, X_test=None, Y_test=None, Truth=None, ratio=0.15, pause=False,
                         n_iter=None, pause_time=0.15):
    """
    为二维回归数据集和结果画图(可动态迭代)
    :param X_data: 训练数据
    :param Y_data: 训练数据的标签
    :param Weights: 训练得到的参数
    :param X_test: 预测数据
    :param Y_test: 预测数据的标签
    :param Truth: 数据集生成时的真实参数
    :param ratio: 设置两边伸展的额外比例
    :param pause: 画图是否暂停 (为实现动态迭代)
    :param n_iter: 当前迭代的代数
    :param pause_time: 迭代过程中暂停的时间间隔
    :return: None
    """
    X = X_data
    Y = Y_data
    Predict = Weights
    if not pause: plt.figure()
    plt.clf()
    plt.scatter(X, Y, c='blue')
    if X_test is not None and Y_test is not None:  # 用于画预测的点
        plt.scatter(X_test, Y_test, c='red', marker='*', s=120, edgecolors='black', linewidths=0.5)
    if Truth is not None:
        # 绘制真实的参数
        PX, PU = get_PXU_regression(X, Truth)
        plt.plot(PX, PU, c='orange', linewidth=5)
    if Predict is not None:
        # 绘制预测的参数
        PX, PU = get_PXU_regression(X, Predict)
        plt.plot(PX, PU, c='red', linewidth=2)
        # 为了方便展示，两边进行额外延伸
        X_min, X_max = np.min(X), np.max(X)
        Y_min, Y_max = np.min(Y), np.max(Y)
        X_gap = (X_max - X_min) * ratio
        Y_gap = (Y_max - Y_min) * ratio
        plt.xlim([X_min - X_gap, X_max + X_gap])
        plt.ylim([Y_min - Y_gap, Y_max + Y_gap])
    if pause:
        if n_iter:
            plt.title("iter: " + str(n_iter))
        plt.pause(pause_time)
    else:
        plt.show()


def get_PXU_regression(X, Weights, ratio=0.3, step=0.1):
    """
    获取画图使用的X和其他未知变量
    :param X: 要画图的已知变量
    :param Weights: 要画图的权重
    :param ratio: 两边伸展的额外比例
    :param step: 采样频率
    :return:
    """
    gap = max(X) - min(X)
    PX = np.arange(min(X) - ratio * gap, max(X) + ratio * gap, step)
    PX_B = np.concatenate((PX.reshape(-1, 1), np.ones((len(PX), 1))), axis=1)
    PU = PX_B.dot(Weights)
    return PX, PU


def plot_cluster(X, labels, centers=None, pause=False, n_iter=None, pause_time=0.15):
    """
    为聚类结果进行画图（可迭代）
    :param X: 数据集
    :param labels: 聚类结果标签
    :param centers: 聚类中心点位置
    :param pause: 画图是否暂停 (为实现动态迭代)
    :param n_iter: 当前迭代的代数
    :param pause_time: 迭代过程中暂停的时间间隔
    :return: None
    """
    if not pause: plt.figure()
    plt.clf()
    X_dim = X.shape[1]
    k = len(np.unique(labels))
    if X_dim == 2:
        for i in range(k):
            plt.scatter(X[labels == i, 0], X[labels == i, 1], marker="o")
        if centers is not None:
            plt.scatter(centers[:, 0], centers[:, 1], c='black', marker="x", s=120)
        plt.grid()
    elif X_dim == 3:
        ax = plt.subplot(111, projection='3d')
        for i in range(k):
            ax.scatter(X[labels == i, 0], X[labels == i, 1], X[labels == i, 2], marker="o")
        if centers is not None:
            ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2], c='black', marker="x", s=120)
        # 设置三维图像角度(仰角方位角)
        # ax.view_init(elev=20, azim=20)
    else:
        raise ValueError("Unable to draw clustering results with a dataset of more than 3 dimensions")
    if pause:
        if n_iter:
            plt.title("iter: " + str(n_iter))
        plt.pause(pause_time)
    else:
        plt.show()


def run_uniform_classification(model, X_size=100, X_feat=2, X_lower=-1, X_upper=1, lower=-1, upper=1, train_ratio=0.8):
    """
    指定模型对均匀数据的分类测试
    :param model: 指定模型
    :param X_size: 随机生成的数据集大小
    :param X_feat: 数据集特征数
    :param X_lower: 随机生成的数据集下界
    :param X_upper: 随机生成的数据集上界
    :param lower: 随机生成参数的范围最小值
    :param upper: 随机生成参数的范围最大值
    :param train_ratio: 训练集所占比例
    :return: None
    """
    # 生成数据集
    X_data, Y_data, Truth_Weights = random_generate_classification(X_size, X_feat, X_lower, X_upper, lower, upper)
    # 划分训练集和测试集
    train_size = int(train_ratio * len(X_data))
    X_train, Y_train = X_data[:train_size], Y_data[:train_size]
    X_test, Y_test = X_data[train_size:], Y_data[train_size:]
    # 使用数据集对模型训练
    model.train(X_train, Y_train)
    print("Truth Weights: ", Truth_Weights.flatten())
    print("Model Weights: ", model.Weights.flatten())
    # 对训练集进行预测
    Y_train_pred = model.predict(X_train)
    # 计算训练准确率
    train_accuracy = calculate_accuracy(Y_train, Y_train_pred)
    print("Train Accuracy:  {:.3f} %".format(train_accuracy * 100))
    # 画图展示效果
    # model.plot_2dim(Truth=Truth_Weights)
    # 对测试集进行预测
    Y_test_pred = model.predict(X_test)
    print("Truth Values: ", Y_test.flatten())
    print("Predict Values: ", Y_test_pred.flatten())
    # 计算测试集准确率
    test_accuracy = calculate_accuracy(Y_test, Y_test_pred)
    print("Test Accuracy:  {:.3f} %".format(test_accuracy * 100))
    # 对结果进行画图
    model.plot_2dim(X_test, Y_test, Truth=Truth_Weights)


def run_double_classification(model, X_size=100, X_feat=2, X_lower=-1, X_upper=1, train_ratio=0.8):
    """
    指定模型对两散点式数据的分类测试
    :param model: 指定模型
    :param X_size: 随机生成的数据集大小
    :param X_feat: 数据集特征数
    :param X_lower: 随机生成的数据集下界
    :param X_upper: 随机生成的数据集上界
    :param train_ratio: 训练集所占比例
    :return: None
    """
    # 生成数据集
    X_data, Y_data = random_generate_double(X_size, X_feat, X_lower, X_upper)
    # 划分训练集和测试集
    train_size = int(train_ratio * len(X_data))
    X_train, Y_train = X_data[:train_size], Y_data[:train_size]
    X_test, Y_test = X_data[train_size:], Y_data[train_size:]
    # 使用数据集对模型训练
    model.train(X_train, Y_train)
    print("Model Weights:", model.Weights.flatten())
    # 对训练集进行预测
    Y_train_pred = model.predict(X_train)
    # 计算训练准确率
    train_accuracy = calculate_accuracy(Y_train, Y_train_pred)
    print("Train Accuracy:  {:.3f} %".format(train_accuracy * 100))
    # 画图展示效果
    # model.plot_2dim()
    # 对测试集进行预测
    Y_test_pred = model.predict(X_test)
    print("Truth Values: ", Y_test.flatten())
    print("Predict Values: ", Y_test_pred.flatten())
    # 计算测试集准确率
    test_accuracy = calculate_accuracy(Y_test, Y_test_pred)
    print("Test Accuracy:  {:.3f} %".format(test_accuracy * 100))
    # 画图展示效果
    model.plot_2dim(X_test, Y_test)


def run_uniform_regression(model, X_size=100, X_feat=1, X_lower=0, X_upper=20, train_ratio=0.8):
    """
    指定模型对随机生成的回归数据进行回归测试
    :param model: 指定模型
    :param X_size: 随机生成的数据集大小
    :param X_feat: 数据集特征数
    :param X_lower: 随机生成的数据集下界
    :param X_upper: 随机生成的数据集上界
    :param train_ratio: 训练集所占比例
    :return: None
    """
    # 生成数据集
    X_data, Y_data, Truth_Weights = random_generate_regression(X_size, X_feat, X_lower=X_lower, X_upper=X_upper)
    # 划分训练集和测试集
    train_size = int(train_ratio * len(X_data))
    X_train, Y_train = X_data[:train_size], Y_data[:train_size]
    X_test, Y_test = X_data[train_size:], Y_data[train_size:]
    # 使用数据集对模型训练
    model.train(X_train, Y_train)
    print("Truth Weights: ", Truth_Weights.flatten())
    print("Model Weights: ", model.Weights.flatten())
    # 对训练集进行预测
    Y_train_pred = model.predict(X_train)
    # 计算训练结果的mse值
    train_mse = cal_mse_metrics(Y_train, Y_train_pred)
    print("Train MSE Metrics:  {:.3f}".format(train_mse))
    # 对结果进行画图
    # model.plot_2dim(Truth=Truth_Weights)
    # 对测试集进行预测
    Y_test_pred = model.predict(X_test)
    print("Truth Values: ", Y_test.flatten())
    print("Predict Values: ", Y_test_pred.flatten())
    # 计算测试结果的mse值
    test_mse = cal_mse_metrics(Y_test, Y_test_pred)
    print("Test MSE Metrics:  {:.3f}".format(test_mse))
    # 对结果进行画图
    model.plot_2dim(X_test, Y_test, Truth=Truth_Weights)


def run_contrast_regression(model, X_size=100, X_feat=1, X_lower=0, X_upper=20, train_ratio=0.8):
    """
    指定模型对随机生成的回归数据进行回归对比测试
    :param model: 指定模型
    :param X_size: 随机生成的数据集大小
    :param X_feat: 数据集特征数
    :param X_lower: 随机生成的数据集下界
    :param X_upper: 随机生成的数据集上界
    :param train_ratio: 训练集所占比例
    :return: None
    """
    # 生成数据集
    X_data, Y_data, Truth_Weights = random_generate_regression(X_size, X_feat, X_lower=X_lower, X_upper=X_upper)

    # 划分训练集和测试集
    train_size = int(train_ratio * len(X_data))
    X_train, Y_train = X_data[:train_size], Y_data[:train_size]
    X_test, Y_test = X_data[train_size:], Y_data[train_size:]

    # 使用直接计算的方法求解
    print("Direct Solve:")
    # 使用数据集对模型训练
    model.train(X_train, Y_train, mode=0)
    print("Truth Weights: ", Truth_Weights.flatten())
    print("Model Weights: ", model.Weights.flatten())
    # 对训练集进行预测
    Y_train_pred = model.predict(X_train)
    # 计算训练结果的mse值
    train_mse = cal_mse_metrics(Y_train, Y_train_pred)
    print("Train MSE Metrics:  {:.3f}".format(train_mse))

    # 使用梯度的方法求解
    print("Gradient Solve:")
    model.train(X_data, Y_data, mode=1, epochs=100, lr=0.05, grad_type='Adam')
    print("Truth Weights: ", Truth_Weights.flatten())
    print("Model Weights: ", model.Weights.flatten())
    # 对训练集进行预测
    Y_train_pred = model.predict(X_train)
    # 计算训练结果的mse值
    train_mse = cal_mse_metrics(Y_train, Y_train_pred)
    print("Train MSE Metrics:  {:.3f}".format(train_mse))

    # 对测试集进行预测
    Y_test_pred = model.predict(X_test)
    print("Truth Values: ", Y_test.flatten())
    print("Predict Values: ", Y_test_pred.flatten())
    # 计算测试结果的mse值
    test_mse = cal_mse_metrics(Y_test, Y_test_pred)
    print("Test MSE Metrics:  {:.3f}".format(test_mse))
    # 对结果进行画图
    model.plot_2dim(X_test, Y_test, Truth=Truth_Weights)
