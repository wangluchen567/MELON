import numpy as np
import matplotlib.pyplot as plt
from Models.LinearClassifier import *
from Models.Utils import random_generate_classification, random_generate_double, get_PXU_classification

if __name__ == '__main__':
    # 创建数据
    np.random.seed(100)
    X_random, Y_random, Truth_Weights = random_generate_classification()
    X_double, Y_double = random_generate_double()
    X_datas = [X_random, X_double]
    Y_datas = [Y_random, Y_double]
    # 创建模型集合
    model_list = [
        LogisticRegression(),
        Perceptron(),
        RidgeClassifier()
    ]
    # 获取结果集合
    weights_list = []
    for i in range(len(X_datas)):
        for model in model_list:
            model.train(X_datas[i], Y_datas[i])
            weights_list.append(model.Weights)
    # 绘图比较结果
    # 设置子图的行数和列数
    idx, rows, cols = 0, 2, 3
    # 创建图形，调整大小
    fig, axes = plt.subplots(rows, cols, figsize=(9, 6))
    # 模型名称
    models = ['LogisticRegression', 'Perceptron', 'RidgeClassifier']
    # 遍历所有子图
    for i in range(rows):
        for j in range(cols):
            ax = axes[i, j]
            X_data = X_datas[i]
            Y_data = Y_datas[i]
            Weights = weights_list[idx]
            # 绘制结果
            ax.scatter(X_data[Y_data.flatten() == 1, 0], X_data[Y_data.flatten() == 1, 1], c='red', s=20)
            ax.scatter(X_data[Y_data.flatten() == -1, 0], X_data[Y_data.flatten() == -1, 1], c='blue', s=20)
            # 绘制预测的参数
            PX, PU = get_PXU_classification(X_data, Weights)
            ax.plot(PX, PU, c='red', linewidth=2)
            # 为了方便展示，两边进行额外延伸
            X0_min, X0_max = np.min(X_data[:, 0]), np.max(X_data[:, 0])
            X1_min, X1_max = np.min(X_data[:, 1]), np.max(X_data[:, 1])
            X0_gap = (X0_max - X0_min) * 0.15
            X1_gap = (X1_max - X1_min) * 0.15
            ax.set_xlim([X0_min - X0_gap, X0_max + X0_gap])
            ax.set_ylim([X1_min - X1_gap, X1_max + X1_gap])
            idx += 1
            # 绘制虚线网格线
            ax.grid(True, linestyle='--', alpha=0.5)
            # 完全隐藏刻度标记(保留刻度线但设为透明)
            ax.tick_params(
                axis='both',
                which='both',
                length=0,  # 刻度线长度设为0
                labelbottom=False,  # 隐藏x轴标签
                labelleft=False  # 隐藏y轴标签
            )
            # 设置每列标题
            if i == 0:
                ax.set_title(models[j], pad=10, fontsize=12)
    # 调整子图之间的间距
    plt.tight_layout(pad=2.0, w_pad=1.5, h_pad=0.5)
    # 显示图形
    plt.show()
