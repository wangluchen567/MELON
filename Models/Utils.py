import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def normalize(matrix, axis=None):
    """
    对数据矩阵进行归一化
    :param matrix: 数据矩阵
    :param axis: 沿着哪个轴
    :return: 归一化数据
    """
    if axis is None:
        if max(matrix) == min(matrix):
            return matrix
        else:
            return (matrix - min(matrix)) / (max(matrix) - min(matrix))
    elif axis == 0:
        matrix_ = matrix.copy()
        mask = np.max(matrix_, axis=axis) != np.min(matrix_, axis=axis)
        matrix_[:, mask] = ((matrix_[:, mask] - np.min(matrix_, axis=axis)[mask])
                            / (np.max(matrix_, axis=axis)[mask] - np.min(matrix_, axis=axis)[mask]))
        return matrix_
    elif axis == 1:
        mask = np.max(matrix, axis=axis) != np.min(matrix, axis=axis)
        return ((matrix[mask] - np.min(matrix[mask], axis=axis, keepdims=True))
                / (np.max(matrix[mask], axis=axis, keepdims=True) - np.min(matrix[mask], axis=axis, keepdims=True)))
    else:
        raise ValueError("There is currently no axis larger than 1")


def sigmoid(x):
    """sigmoid函数"""
    # 防止指数溢出
    indices_pos = np.nonzero(x >= 0)
    indices_neg = np.nonzero(x < 0)
    y = np.zeros_like(x)
    # y = 1 / (1 + exp(-x)), x >= 0
    # y = exp(x) / (1 + exp(x)), x < 0
    y[indices_pos] = 1 / (1 + np.exp(-x[indices_pos]))
    y[indices_neg] = np.exp(x[indices_neg]) / (1 + np.exp(x[indices_neg]))
    return y


def calculate_accuracy(Truth, Predict):
    """
    计算分类结果的准确率
    :param Truth: 真实标签
    :param Predict: 预测标签
    :return: 准确率
    """
    Truth, Predict = np.array(Truth), np.array(Predict)
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
    :return: 生成的数据集和真实参数
    """
    X_data = np.random.uniform(X_lower, X_upper, size=(X_size, X_feat))
    X_mids = (np.max(X_data, axis=0) + np.min(X_data, axis=0)) / 2
    TruthWeights = np.random.uniform(lower, upper, size=(X_feat, 1))
    Bias = - X_mids.reshape(1, -1) @ TruthWeights
    TruthWeights = np.concatenate((TruthWeights, Bias), axis=0)
    X_B = np.concatenate((X_data, np.ones((len(X_data), 1))), axis=1)
    Y_data = np.ones((len(X_data), 1), dtype=int)
    Y_data[X_B.dot(TruthWeights) < 0] = -1
    return X_data, Y_data, TruthWeights


def random_generate_double(X_size=100, X_feat=2, X_lower=-1, X_upper=1):
    """
    随机生成两点散布的分类数据集
    :param X_size: 数据集大小
    :param X_feat: 数据集特征数
    :param X_lower: 随机生成的数据的下界
    :param X_upper: 随机生成的数据的上界
    :return: 生成的数据集和真实值
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
    Y_data = np.ones((len(X_data), 1), dtype=int)
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
    :return: 生成的数据集与真实参数
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


def random_generate_cluster(X_size=100, X_feat=2, n_clusters=3, cluster_std=1.0, lower=-10, upper=10):
    """
    随机生成聚类数据集
    :param X_size: 数据集大小
    :param X_feat: 数据集特征数
    :param n_clusters: 数据集分类个数
    :param cluster_std: 随机生成正态分布数据的标准差(宽度)
    :param lower: 聚类中心范围下界
    :param upper: 聚类中心范围上界
    :return: 聚类数据集和标签
    """
    # 随机得到聚类中心位置
    centers = np.random.uniform(lower, upper, size=(n_clusters, X_feat))
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
    if not pause: plt.figure()
    plt.clf()
    plt.scatter(X_data[Y_data.flatten() == 1, 0], X_data[Y_data.flatten() == 1, 1], c='red')
    plt.scatter(X_data[Y_data.flatten() == neg_label, 0], X_data[Y_data.flatten() == neg_label, 1], c='blue')
    if X_test is not None and Y_test is not None:  # 用于画预测的点
        plt.scatter(X_test[Y_test.flatten() == 1, 0], X_test[Y_test.flatten() == 1, 1], c='red', marker='*', s=120,
                    edgecolors='black', linewidths=0.5)
        plt.scatter(X_test[Y_test.flatten() == neg_label, 0], X_test[Y_test.flatten() == neg_label, 1], c='blue',
                    marker='*', s=120, edgecolors='black', linewidths=0.5)
    if support is not None:
        plt.scatter(X_data[support, 0], X_data[support, 1], s=150, c='none', linewidth=1.5, edgecolor='red')
    if Truth is not None:
        # 绘制真实的参数
        PX, PU = get_PXU_classification(X_data, Truth)
        plt.plot(PX, PU, c='orange', linewidth=5)
    if Weights is not None:
        # 绘制预测的参数
        PX, PU = get_PXU_classification(X_data, Weights)
        plt.plot(PX, PU, c='red', linewidth=2)
        # 为了方便展示，两边进行额外延伸
        X0_min, X0_max = np.min(X_data[:, 0]), np.max(X_data[:, 0])
        X1_min, X1_max = np.min(X_data[:, 1]), np.max(X_data[:, 1])
        X0_gap = (X0_max - X0_min) * ratio
        X1_gap = (X1_max - X1_min) * ratio
        plt.xlim([X0_min - X0_gap, X0_max + X0_gap])
        plt.ylim([X1_min - X1_gap, X1_max + X1_gap])
    plt.xlabel('x1')
    plt.ylabel('x2')
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
    :return: 绘制分类线的采样点
    """
    gap = max(X[:, 0]) - min(X[:, 0])
    PX = np.arange(min(X[:, 0]) - ratio * gap, max(X[:, 0]) + ratio * gap, step)
    PX_B = np.concatenate((PX.reshape(-1, 1), np.ones((len(PX), 1))), axis=1)
    PW = np.concatenate((Weights[:-2, :], Weights[-1, :].reshape(-1, 1)), axis=1) / -Weights[-2, :]
    PU = PX_B.dot(PW.T)
    return PX, PU


def plot_2dim_regression(X_data, Y_data, Weights, X_test=None, Y_test=None, Truth=None, support=None,
                         ratio=0.15, pause=False, n_iter=None, pause_time=0.15):
    """
    为二维回归数据集和结果画图(可动态迭代)
    :param X_data: 训练数据
    :param Y_data: 训练数据的标签
    :param Weights: 训练得到的参数
    :param X_test: 预测数据
    :param Y_test: 预测数据的标签
    :param Truth: 数据集生成时的真实参数
    :param support: 是否是支持向量
    :param ratio: 设置两边伸展的额外比例
    :param pause: 画图是否暂停 (为实现动态迭代)
    :param n_iter: 当前迭代的代数
    :param pause_time: 迭代过程中暂停的时间间隔
    :return: None
    """
    if not pause: plt.figure()
    plt.clf()
    plt.scatter(X_data, Y_data, c='blue')
    if X_test is not None and Y_test is not None:  # 用于画预测的点
        plt.scatter(X_test, Y_test, c='red', marker='*', s=120, edgecolors='black', linewidths=0.5)
    if support is not None:  # 用于绘制支持向量位置
        plt.scatter(X_data[support], Y_data[support], s=150, c='none', linewidth=1.5, edgecolor='red')
    if Truth is not None:
        # 绘制真实的参数
        PX, PU = get_PXU_regression(X_data, Truth)
        plt.plot(PX, PU, c='orange', linewidth=5)
    if Weights is not None:
        # 绘制预测的参数
        PX, PU = get_PXU_regression(X_data, Weights)
        plt.plot(PX, PU, c='red', linewidth=2)
        # 为了方便展示，两边进行额外延伸
        X_min, X_max = np.min(X_data), np.max(X_data)
        Y_min, Y_max = np.min(Y_data), np.max(Y_data)
        X_gap = (X_max - X_min) * ratio
        Y_gap = (Y_max - Y_min) * ratio
        plt.xlim([X_min - X_gap, X_max + X_gap])
        plt.ylim([Y_min - Y_gap, Y_max + Y_gap])
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid()
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
    :return: 绘制回归线的采样点
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
    unique_labels = np.unique(labels)
    if X_dim == 2:
        for i in unique_labels:
            if i == -1:  # 处理使用密度聚类时的噪声点
                plt.scatter(X[labels == i, 0], X[labels == i, 1], marker="o", c='black', s=10)
            else:
                plt.scatter(X[labels == i, 0], X[labels == i, 1], marker="o")
        if centers is not None:
            plt.scatter(centers[:, 0], centers[:, 1], c='black', marker="x", s=120)
        plt.grid()
    elif X_dim == 3:
        ax = plt.subplot(111, projection='3d')
        for i in unique_labels:
            if i == -1:  # 处理使用密度聚类时的噪声点
                ax.scatter(X[labels == i, 0], X[labels == i, 1], X[labels == i, 2], marker="o", c='black', s=10)
            else:
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
    if hasattr(model, 'Weights'):
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
    if hasattr(model, 'Weights'):
        print("Model Weights: ", model.Weights.flatten())
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
    if hasattr(model, 'Weights'):
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


def plot_2dim_classification_sample(model, X_data, Y_data, X_test=None, Y_test=None, neg_label=-1, support=None,
                                    sample_steps=200, extra=0.05, pause=False, n_iter=None, pause_time=0.15):
    """
    利用采样为二维二分类数据集和结果画图 (可动态迭代)
    :param model: 给定模型
    :param X_data: 训练数据
    :param Y_data: 训练数据的标签
    :param X_test: 预测数据
    :param Y_test: 预测数据的标签
    :param neg_label: 负标签的值 (-1/0)
    :param support: 是否是支持向量
    :param sample_steps: 采样步数
    :param extra: 额外绘制图像的比例
    :param pause: 画图是否暂停 (为实现动态迭代)
    :param n_iter: 当前迭代的代数
    :param pause_time: 迭代过程中暂停的时间间隔
    :return: None
    """
    if not pause: plt.figure()
    plt.clf()
    x1_min, x1_max = np.min(X_data[:, 0], axis=0), np.max(X_data[:, 0], axis=0)  # 第0列数据的范围
    x2_min, x2_max = np.min(X_data[:, 1], axis=0), np.max(X_data[:, 1], axis=0)  # 第1列数据的范围
    # 注意：这里为了更好看一些，采样数据时会多采样一部分
    x1_range, x2_range = (x1_max - x1_min), (x2_max - x2_min)
    x1_min -= extra * x1_range
    x1_max += extra * x1_range
    x2_min -= extra * x2_range
    x2_max += extra * x2_range
    t1 = np.linspace(x1_min, x1_max, sample_steps)
    t2 = np.linspace(x2_min, x2_max, sample_steps)
    x1, x2 = np.meshgrid(t1, t2)  # 生成网格采样点
    x_sample = np.stack((x1.flat, x2.flat), axis=1)  # 生成采样数据
    y_sample = model.predict(x_sample)  # 使用模型得到预测数据
    y_sample = y_sample.reshape(x1.shape)  # 改变形状与采样点相同
    # 为方便绘制图像，这里将y_sample中-1标签转换为1标签
    y_sample[y_sample == -1] = 0
    cm_light = mpl.colors.ListedColormap(['#A0A0FF', '#FF8080'])
    # 绘制采样图像
    plt.pcolormesh(x1, x2, y_sample, cmap=cm_light)
    # 绘制数据集位置点
    plt.scatter(X_data[Y_data.flatten() == 1, 0], X_data[Y_data.flatten() == 1, 1], c='red')
    plt.scatter(X_data[Y_data.flatten() == neg_label, 0], X_data[Y_data.flatten() == neg_label, 1], c='blue')
    if X_test is not None and Y_test is not None:  # 用于画预测的点
        plt.scatter(X_test[Y_test.flatten() == 1, 0], X_test[Y_test.flatten() == 1, 1], c='red', marker='*', s=120,
                    edgecolors='black', linewidths=0.5)
        plt.scatter(X_test[Y_test.flatten() == neg_label, 0], X_test[Y_test.flatten() == neg_label, 1], c='blue',
                    marker='*', s=120, edgecolors='black', linewidths=0.5)
    if support is not None:
        plt.scatter(X_data[support, 0], X_data[support, 1], s=150, c='none', linewidth=1.5, edgecolor='tomato')
    plt.grid()
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.xlim(x1_min, x1_max)
    plt.ylim(x2_min, x2_max)
    if pause:
        if n_iter:
            plt.title("iter: " + str(n_iter))
        plt.pause(pause_time)
    else:
        plt.show()


def random_make_circles(num_samples=100, factor=0.8, noise=0.01, shuffle=True):
    """
    随机创建同心圆数据
    :param num_samples: 采样的数据大小
    :param factor: 内外圆之间的比例因子
    :param noise: 加入噪音程度
    :param shuffle: 是否打乱数据集
    :return: 生成的数据集和真实值
    """
    if factor > 1 or factor < 0:
        raise ValueError("'factor' has to be between 0 and 1.")
    line_space = np.linspace(0, 2 * np.pi, num_samples // 2 + 1)[:-1]
    outer_circ_x = np.cos(line_space)
    outer_circ_y = np.sin(line_space)
    inner_circ_x = outer_circ_x * factor
    inner_circ_y = outer_circ_y * factor
    # 合并数据
    X = np.vstack((np.append(outer_circ_x, inner_circ_x),
                   np.append(outer_circ_y, inner_circ_y))).T
    y = np.hstack([np.zeros(num_samples // 2, dtype=np.intp),
                   np.ones(num_samples // 2, dtype=np.intp)])
    Y = y.reshape(-1, 1)
    if noise is not None:  # 添加噪声
        X += np.random.normal(scale=noise, size=X.shape)
    if shuffle:  # 打乱数据
        random_index = np.arange(num_samples)
        np.random.shuffle(random_index)
        X = X[random_index]
        Y = Y[random_index]
    return X, Y


def random_make_moons(num_samples=100, noise=0.1, shuffle=True):
    """
    随机创建月亮数据(双半圆数据)
    :param num_samples: 采样的数据大小
    :param noise: 加入噪音程度
    :param shuffle: 是否打乱数据集
    :return: 生成的数据集和真实值
    """
    # 生成两个半圆的数据个数
    n_samples_out = num_samples // 2
    n_samples_in = num_samples - n_samples_out
    # 生成外半圆的数据点
    outer_circ_x = np.cos(np.linspace(0, np.pi, n_samples_out))
    outer_circ_y = np.sin(np.linspace(0, np.pi, n_samples_out))
    # 将外半圆向右移动
    outer_circ_x += 15 * np.pi / 24
    # 生成内半圆的数据点
    inner_circ_x = 1 - np.cos(np.linspace(0, np.pi, n_samples_in))
    inner_circ_y = 1 - np.sin(np.linspace(0, np.pi, n_samples_in))
    # 将内半圆向下移动
    inner_circ_y -= 1 * np.pi / 4
    # 合并数据点
    X = np.vstack((np.column_stack((outer_circ_x, outer_circ_y)),
                   np.column_stack((inner_circ_x, inner_circ_y))))
    y = np.hstack([np.zeros(n_samples_out, dtype=int),
                   np.ones(n_samples_in, dtype=int)])
    Y = y.reshape(-1, 1)
    if noise is not None:  # 添加噪声
        X += np.random.normal(scale=noise, size=X.shape)
    if shuffle:  # 打乱数据
        random_index = np.arange(num_samples)
        np.random.shuffle(random_index)
        X = X[random_index]
        Y = Y[random_index]
    return X, Y


def run_circle_classification(model, X_size=100, factor=0.5, noise=0.1, train_ratio=0.8):
    """
    指定模型对同心圆数据的分类测试
    :param model: 指定模型
    :param X_size: 随机生成的数据集大小
    :param factor: 内外圆之间的比例因子
    :param noise: 噪声扰动程度
    :param train_ratio: 训练集所占比例
    :return: None
    """
    # 生成数据集
    X_data, Y_data = random_make_circles(X_size, factor, noise)
    Y_data[Y_data == 0] = -1
    # 划分训练集和测试集
    train_size = int(train_ratio * len(X_data))
    X_train, Y_train = X_data[:train_size], Y_data[:train_size]
    X_test, Y_test = X_data[train_size:], Y_data[train_size:]
    # 使用数据集对模型训练
    model.train(X_train, Y_train)
    if hasattr(model, 'Weights'):
        print("Model Weights: ", model.Weights.flatten())
    # 对训练集进行预测
    Y_train_pred = model.predict(X_train)
    # 计算训练准确率
    train_accuracy = calculate_accuracy(Y_train, Y_train_pred)
    print("Train Accuracy:  {:.3f} %".format(train_accuracy * 100))
    # 对测试集进行预测
    Y_test_pred = model.predict(X_test)
    print("Truth Values: ", Y_test.flatten())
    print("Predict Values: ", Y_test_pred.flatten())
    # 计算测试集准确率
    test_accuracy = calculate_accuracy(Y_test, Y_test_pred)
    print("Test Accuracy:  {:.3f} %".format(test_accuracy * 100))
    # 对结果进行画图
    model.plot_2dim(X_test, Y_test)


def run_moons_classification(model, X_size=100, noise=0.1, train_ratio=0.8):
    """
    指定模型对月亮数据(双半圆数据)的分类测试
    :param model: 指定模型
    :param X_size: 随机生成的数据集大小
    :param noise: 噪声扰动程度
    :param train_ratio: 训练集所占比例
    :return: None
    """
    # 生成数据集
    X_data, Y_data = random_make_moons(X_size, noise)
    Y_data[Y_data == 0] = -1
    # 划分训练集和测试集
    train_size = int(train_ratio * len(X_data))
    X_train, Y_train = X_data[:train_size], Y_data[:train_size]
    X_test, Y_test = X_data[train_size:], Y_data[train_size:]
    # 使用数据集对模型训练
    model.train(X_train, Y_train)
    if hasattr(model, 'Weights'):
        print("Model Weights: ", model.Weights.flatten())
    # 对训练集进行预测
    Y_train_pred = model.predict(X_train)
    # 计算训练准确率
    train_accuracy = calculate_accuracy(Y_train, Y_train_pred)
    print("Train Accuracy:  {:.3f} %".format(train_accuracy * 100))
    # 对测试集进行预测
    Y_test_pred = model.predict(X_test)
    print("Truth Values: ", Y_test.flatten())
    print("Predict Values: ", Y_test_pred.flatten())
    # 计算测试集准确率
    test_accuracy = calculate_accuracy(Y_test, Y_test_pred)
    print("Test Accuracy:  {:.3f} %".format(test_accuracy * 100))
    # 对结果进行画图
    model.plot_2dim(X_test, Y_test)


def plot_2dim_regression_sample(model, X_data, Y_data, X_test=None, Y_test=None, support=None,
                                sample_steps=200, extra=0.05, pause=False, n_iter=None, pause_time=0.15):
    """
    利用采样为二维回归数据集和结果画图 (可动态迭代)
    :param model: 给定模型
    :param X_data: 训练数据
    :param Y_data: 训练数据的标签
    :param X_test: 预测数据
    :param Y_test: 预测数据的标签
    :param support: 是否是支持向量
    :param sample_steps: 采样步数
    :param extra: 额外绘制图像的比例
    :param pause: 画图是否暂停 (为实现动态迭代)
    :param n_iter: 当前迭代的代数
    :param pause_time: 迭代过程中暂停的时间间隔
    :return: None
    """
    if not pause: plt.figure()
    plt.clf()
    x_min, x_max = np.min(X_data, axis=0), np.max(X_data, axis=0)  # 得到数据范围
    y_min, y_max = np.min(Y_data, axis=0), np.max(Y_data, axis=0)  # 得到数据范围
    # 注意：这里为了更好看一些，采样数据时会多采样一部分
    x_range = x_max - x_min
    x_min -= extra * x_range
    x_max += extra * x_range
    y_range = y_max - y_min
    y_min -= extra * y_range
    y_max += extra * y_range
    x_sample = np.linspace(x_min, x_max, sample_steps)
    y_sample = model.predict(x_sample)  # 使用模型得到预测数据
    # 若数据值比较大则使用科学计数法显示
    if np.max(np.abs(Y_data)) >= 1.e+2:
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    # 绘制预测值
    plt.plot(x_sample, y_sample, c='red', linewidth=2)
    # 绘制数据集位置点
    plt.scatter(X_data, Y_data, c='blue', alpha=0.8)
    if X_test is not None and Y_test is not None:  # 用于画预测的点
        plt.scatter(X_test, Y_test, c='red', marker='*', s=120, edgecolors='black', linewidths=0.5)
    if support is not None:  # 用于绘制支持向量位置
        plt.scatter(X_data[support], Y_data[support], s=150, c='none', linewidth=1.5, edgecolor='tomato')
    plt.grid()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    if pause:
        if n_iter:
            plt.title("iter: " + str(n_iter))
        plt.pause(pause_time)
    else:
        plt.show()


def random_make_circular(num_samples=100, lower=0, upper=10, slope=0, bias=0, noise=0.1, shuffle=True):
    """
    随机创建三角函数（又称圆函数）测试数据
    :param num_samples: 数据集大小
    :param lower: 随机生成的数据集下界
    :param upper: 随机生成的数据集上界
    :param slope: 生成的函数斜率
    :param bias: 生成的函数截距（偏置）
    :param noise: 噪声扰动程度
    :param shuffle: 是否打乱数据集
    :return: 生成的数据集和真实值
    """
    X = np.linspace(lower, upper, num_samples)[:, np.newaxis]
    Y = 2 * (slope + 1) * np.sin(X) + slope * X + bias
    if noise is not None:
        Y += np.random.normal(scale=noise, size=Y.shape)
    if shuffle:
        random_index = np.arange(num_samples)
        np.random.shuffle(random_index)
        X = X[random_index]
        Y = Y[random_index]
    return X, Y


def run_circular_regression(model, X_size=100, X_lower=0, X_upper=15, slope=3, bias=3, noise=0.1, train_ratio=0.8):
    """
    指定模型对三角函数(圆函数)数据的回归测试
    :param model: 指定模型
    :param X_size: 生成的数据集大小
    :param X_lower: 随机生成的数据集下界
    :param X_upper: 随机生成的数据集上界
    :param slope: 生成的函数斜率
    :param bias: 生成的函数截距（偏置）
    :param noise: 噪声扰动程度
    :param train_ratio: 训练集所占比例
    :return: None
    """
    # 生成数据集
    X_data, Y_data = random_make_circular(X_size, X_lower, X_upper, slope, bias, noise)
    # 划分训练集和测试集
    train_size = int(train_ratio * len(X_data))
    X_train, Y_train = X_data[:train_size], Y_data[:train_size]
    X_test, Y_test = X_data[train_size:], Y_data[train_size:]
    # 使用数据集对模型训练
    model.train(X_train, Y_train)
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
    model.plot_2dim(X_test, Y_test)


def random_make_poly(num_samples=100, lower=0, upper=10, degree=3, gamma=3, constant=3, noise=0.1, shuffle=True):
    """
    随机创建多项式函数测试数据
    :param num_samples: 数据集大小
    :param lower: 随机生成的数据集下界
    :param upper: 随机生成的数据集上界
    :param degree: 多项式函数的次数
    :param gamma: 高次项前的系数
    :param constant: 常数项值
    :param noise: 噪声扰动程度
    :param shuffle: 是否打乱数据集
    :return: 生成的数据集和真实值
    """
    X = np.linspace(lower, upper, num_samples)[:, np.newaxis]
    Y = (gamma * X + constant) ** degree
    if noise is not None:
        Y += np.random.normal(scale=noise, size=Y.shape)
    if shuffle:
        random_index = np.arange(num_samples)
        np.random.shuffle(random_index)
        X = X[random_index]
        Y = Y[random_index]
    return X, Y


def run_poly_regression(model, X_size=100, X_lower=-5, X_upper=5,
                        degree=3, gamma=3, constant=3, noise=10, train_ratio=0.8):
    """
    指定模型对多项式函数数据的回归测试
    :param model: 指定模型
    :param X_size: 生成的数据集大小
    :param X_lower: 随机生成的数据集下界
    :param X_upper: 随机生成的数据集上界
    :param degree: 多项式函数的次数
    :param gamma: 高次项前的系数
    :param constant: 常数项值
    :param noise: 噪声扰动程度
    :param train_ratio: 训练集所占比例
    :return: None
    """
    # 生成数据集
    X_data, Y_data = random_make_poly(X_size, X_lower, X_upper, degree, gamma, constant, noise)
    # 划分训练集和测试集
    train_size = int(train_ratio * len(X_data))
    X_train, Y_train = X_data[:train_size], Y_data[:train_size]
    X_test, Y_test = X_data[train_size:], Y_data[train_size:]
    # 使用数据集对模型训练
    model.train(X_train, Y_train)
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
    model.plot_2dim(X_test, Y_test)


def run_blobs_cluster(model, X_size=500, X_feat=2, n_clusters=5):
    """
    指定模型对多个点状分布数据的聚类测试
    :param model: 指定模型
    :param X_size: 生成的数据集大小
    :param X_feat: 数据集特征数量
    :param n_clusters: 生成的数据簇个数
    :return: None
    """
    X, Y = random_generate_cluster(X_size, X_feat, n_clusters)
    model.train(X)
    model.plot_cluster()


def run_circle_cluster(model, X_size=500, factor=0.5, noise=0.05):
    """
    指定模型对同心圆分布数据的聚类测试
    :param model: 指定模型
    :param X_size: 生成的数据集大小
    :param factor: 内外圆之间的比例因子
    :param noise: 加入噪音的程度
    :return: None
    """
    X, Y = random_make_circles(X_size, factor, noise)
    model.train(X)
    model.plot_cluster()


def run_moons_cluster(model, X_size=500, noise=0.1):
    """
    指定模型对同心圆分布数据的聚类测试
    :param model: 指定模型
    :param X_size: 生成的数据集大小
    :param noise: 加入噪音的程度
    :return: None
    """
    X, Y = random_make_moons(X_size, noise)
    model.train(X)
    model.plot_cluster()
