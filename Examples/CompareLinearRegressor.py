import numpy as np
import matplotlib.pyplot as plt
from Models.LinearRegressor import *
from Models.Utils import random_generate_regression, get_PXU_regression

if __name__ == '__main__':
    # 创建数据
    np.random.seed(100)
    X_data, Y_data, Truth_Weights = random_generate_regression()
    # 创建模型集合
    model_list = [
        GDRegressor(),
        LinearRegression(),
        Ridge()
    ]
    # 获取结果集合
    weights_list = []
    for model in model_list:
        model.train(X_data, Y_data)
        weights_list.append(model.Weights)
    # 绘图比较结果
    # 设置子图的行数和列数
    idx, rows, cols = 0, 1, 3
    # 创建图形，调整大小
    fig, axes = plt.subplots(rows, cols, figsize=(9, 3.6))
    # 如果只有一行或一列，axes的维度会不同，这里统一处理
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)
    # 模型名称
    models = ['GDRegressor', 'LinearRegression', 'Ridge']
    # 遍历所有子图
    for i in range(rows):
        for j in range(cols):
            ax = axes[i, j]
            Weights = weights_list[idx]
            # 绘制结果
            ax.scatter(X_data, Y_data, c='blue', s=20)
            # 绘制预测的参数
            PX, PU = get_PXU_regression(X_data, Weights)
            ax.plot(PX, PU, c='red', linewidth=2)
            # 为了方便展示，两边进行额外延伸
            X_min, X_max = np.min(X_data), np.max(X_data)
            Y_min, Y_max = np.min(Y_data), np.max(Y_data)
            X_gap = (X_max - X_min) * 0.15
            Y_gap = (Y_max - Y_min) * 0.15
            ax.set_xlim([X_min - X_gap, X_max + X_gap])
            ax.set_ylim([Y_min - Y_gap, Y_max + Y_gap])
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
