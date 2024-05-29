import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def random_generate_classification(X_size, X_feat=2, X_lower=-1, X_upper=1, lower=-1, upper=1):
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
    X_train = np.random.uniform(X_lower, X_upper, size=(X_size, X_feat))
    X_mids = (np.max(X_train, axis=0) + np.min(X_train, axis=0)) / 2
    TruthWeights = np.random.uniform(lower, upper, size=(X_feat, 1))
    Bias = - X_mids.reshape(1, -1) @ TruthWeights
    TruthWeights = np.concatenate((TruthWeights, Bias), axis=0)
    X_B = np.concatenate((X_train, np.ones((len(X_train), 1))), axis=1)
    Y_train = np.ones((len(X_train), 1))
    Y_train[X_B.dot(TruthWeights) < 0] = -1
    return X_train, Y_train, TruthWeights


def random_generate_double(X_size, X_feat=2, X_lower=-1, X_upper=1):
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
    X_train = np.concatenate((X1, X2))
    Y_train = np.ones((len(X_train), 1))
    Y_train[:len(X1), :] = -1
    rand_index = np.arange(0, len(X_train))
    np.random.shuffle(rand_index)
    X_train = X_train[rand_index, :]
    Y_train = Y_train[rand_index, :]
    return X_train, Y_train


def random_generate_regression(X_size, X_feat=1, X_lower=0, X_upper=20, lower=-1, upper=1, loc=0, scale=0.3):
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
    X_train = np.random.uniform(X_lower, X_upper, size=(X_size, X_feat))
    # 在数据最后一列添加一列单位矩阵作为转置b
    X_B = np.concatenate((X_train, np.ones((len(X_train), 1))), axis=1)
    # 随机生成的模型参数
    Truth_Weights = np.random.uniform(lower, upper, size=(X_feat + 1, 1))
    # 计算输出值
    Y_train = X_B.dot(Truth_Weights)
    # 加入扰动，正态分布扰动
    Y_train += np.random.normal(loc, scale, size=Y_train.shape)
    return X_train, Y_train, Truth_Weights


def plat_data(X, hold=False):
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
        x = np.arange(1, X_dim + 1)
        for i in range(len(X)):
            plt.plot(x, X[i, :])
    if not hold:
        plt.show()


def plat_2dim_classification(X_train, Y_train, Weights, X_data=None, Y_data=None, neg_label=-1, Truth=None, ratio=0.15,
                             pause=False, n_iter=None, pause_time=0.15):
    """
    为二维分类数据集和结果画图 (可动态迭代)
    :param X_train: 训练数据
    :param Y_train: 训练数据的标签
    :param Weights: 训练得到的参数
    :param X_data: 预测数据
    :param Y_data: 预测数据的标签
    :param neg_label: 负标签的值 (-1/0)
    :param Truth: 数据集生成时的真实参数
    :param ratio: 设置两边伸展的额外比例
    :param pause: 画图是否暂停 (为实现动态迭代)
    :param n_iter: 当前迭代的代数
    :param pause_time: 迭代过程中暂停的时间间隔
    :return: None
    """
    X = X_train
    Y = Y_train
    Predict = Weights
    if not pause: plt.figure()
    plt.clf()
    plt.scatter(X[Y.flatten() == 1, 0], X[Y.flatten() == 1, 1], c='red')
    plt.scatter(X[Y.flatten() == neg_label, 0], X[Y.flatten() == neg_label, 1], c='blue')
    if X_data is not None and Y_data is not None:  # 用于画预测的点
        plt.scatter(X_data[Y_data.flatten() == 1, 0], X_data[Y_data.flatten() == 1, 1], c='red', marker='*', s=120,
                    edgecolors='black', linewidths=0.5)
        plt.scatter(X_data[Y_data.flatten() == neg_label, 0], X_data[Y_data.flatten() == neg_label, 1], c='blue',
                    marker='*', s=120, edgecolors='black', linewidths=0.5)
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


def plat_2dim_regression(X_train, Y_train, Weights, X_data=None, Y_data=None, Truth=None, ratio=0.15, pause=False,
                         n_iter=None, pause_time=0.15):
    """
    为二维回归数据集和结果画图(可动态迭代)
    :param X_train: 训练数据
    :param Y_train: 训练数据的标签
    :param Weights: 训练得到的参数
    :param X_data: 预测数据
    :param Y_data: 预测数据的标签
    :param Truth: 数据集生成时的真实参数
    :param ratio: 设置两边伸展的额外比例
    :param pause: 画图是否暂停 (为实现动态迭代)
    :param n_iter: 当前迭代的代数
    :param pause_time: 迭代过程中暂停的时间间隔
    :return: None
    """
    X = X_train
    Y = Y_train
    Predict = Weights
    if not pause: plt.figure()
    plt.clf()
    plt.scatter(X, Y, c='blue')
    if X_data is not None and Y_data is not None:  # 用于画预测的点
        plt.scatter(X_data, Y_data, c='red', marker='*', s=120, edgecolors='black', linewidths=0.5)
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


def run_uniform_classification(model, X_size=100, X_feat=2, X_lower=-1, X_upper=1, lower=-1, upper=1, num_predict=1):
    """
    指定模型对均匀数据的分类测试
    :param X_size: 随机生成的数据集大小
    :param X_feat: 数据集特征数
    :param X_lower: 随机生成的数据集下界
    :param X_upper: 随机生成的数据集上界
    :param lower: 随机生成参数的范围最小值
    :param upper: 随机生成参数的范围最大值
    :param num_predict: 测试数据集的数量
    :return: None
    """
    # 生成数据集
    X_train, Y_train, TruthWeights = random_generate_classification(X_size, X_feat, X_lower, X_upper, lower, upper)
    # 使用数据集对模型训练
    model.train(X_train, Y_train)
    print("ModelWeights:", model.Weights.flatten())
    # 画图展示效果
    model.plat_2dim(Truth=TruthWeights)
    # 随机生成数据用于预测
    x_data = np.random.uniform(X_lower, X_upper, size=(num_predict, 2))
    y_data = model.predict(x_data)
    print("predict labels: ", y_data.flatten())
    # 画图展示效果
    model.plat_2dim(x_data, y_data)


def run_double_classification(model, X_size=100, X_feat=2, X_lower=-1, X_upper=1, num_predict=1):
    """
    指定模型对两散点式数据的分类测试
    :param X_size: 随机生成的数据集大小
    :param X_feat: 数据集特征数
    :param X_lower: 随机生成的数据集下界
    :param X_upper: 随机生成的数据集上界
    :param num_predict: 测试数据集的数量
    :return: None
    """
    # 生成数据集
    X_train, Y_train = random_generate_double(X_size, X_feat, X_lower, X_upper)
    # 使用数据集对模型训练
    model.train(X_train, Y_train)
    print("ModelWeights:", model.Weights.flatten())
    # 画图展示效果
    model.plat_2dim()
    # 随机生成数据用于预测
    x_data = np.random.uniform(X_lower, X_upper, size=(num_predict, 2))
    y_data = model.predict(x_data)
    print("predict labels: ", y_data.flatten())
    # 画图展示效果
    model.plat_2dim(x_data, y_data)


def run_uniform_regression(model, X_size=100, X_lower=0, X_upper=20, num_predict=1):
    # 生成数据集
    X_train, Y_train, Truth_Weights = random_generate_regression(X_size, X_lower=X_lower, X_upper=X_upper)
    # 使用直接计算的方法求解
    model.train(X_train, Y_train)
    print("Truth Weights: ", Truth_Weights.flatten())
    print("Predict Weights: ", model.Weights.flatten())
    model.plat_2dim(Truth=Truth_Weights)
    # 随机生成数据用于预测
    x_data = np.random.uniform(X_lower, X_upper, size=(num_predict, 1))
    y_data = model.predict(x_data)
    print("predict values: ", y_data.flatten())
    # 画图展示效果
    model.plat_2dim(x_data, y_data)


def run_contrast_regression(model, X_size=100, X_lower=0, X_upper=20, num_predict=1):
    # 生成数据集
    X_train, Y_train, Truth_Weights = random_generate_regression(X_size, X_lower=X_lower, X_upper=X_upper)
    # 使用直接计算的方法求解
    print("Direct Solve:")
    model.train(X_train, Y_train)
    print("Truth Weights: ", Truth_Weights.flatten())
    print("Predict Weights: ", model.Weights.flatten())
    model.plat_2dim(Truth=Truth_Weights)
    # 使用梯度的方法求解
    print("Gradient Solve:")
    model.train(X_train, Y_train, mode=1, epochs=50, lr=0.05, grad_type='Adam')
    print("Truth_Weights: ", Truth_Weights.flatten())
    print("Predict_Weights: ", model.Weights.flatten())
    model.plat_2dim(Truth=Truth_Weights)
    # 随机生成数据用于预测
    x_data = np.random.uniform(X_lower, X_upper, size=(num_predict, 1))
    y_data = model.predict(x_data)
    print("predict values: ", y_data.flatten())
    # 画图展示效果
    model.plat_2dim(x_data, y_data)
