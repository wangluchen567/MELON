import numpy as np
import matplotlib.pyplot as plt
from Models.Clustering import *
from Models.Utils import random_generate_cluster, random_make_circles, random_make_moons

if __name__ == '__main__':
    # 创建数据
    np.random.seed(100)
    X_blobs, _ = random_generate_cluster(n_clusters=5)
    X_circles, _ = random_make_circles()
    X_moons, _ = random_make_moons()
    # 保存数据集合
    X_list = [X_blobs, X_circles, X_moons]
    label_list = []
    # 创建算法集合
    alg_list1 = [
        KMeans(n_clusters=5),
        DBSCAN(eps=1.0, min_samples=10),
        SpectralClustering(n_clusters=5, affinity=SpectralClustering.NEIGHBORS)
    ]
    alg_list2 = [
        KMeans(n_clusters=2),
        DBSCAN(eps=0.2, min_samples=10),
        SpectralClustering(n_clusters=2, affinity=SpectralClustering.NEIGHBORS)
    ]
    # 运行算法得到结果
    for alg in alg_list1:
        labels = alg.train(X_blobs)
        label_list.append(labels)
    for X_train in X_list[1:]:
        for alg in alg_list2:
            labels = alg.train(X_train)
            label_list.append(labels)
    # 绘图比较结果
    # 设置子图的行数和列数
    idx, rows, cols = 0, 3, 3
    # 创建图形，调整大小
    fig, axes = plt.subplots(rows, cols, figsize=(9, 8))
    # 算法名称
    algorithms = ['K-means', 'DBSCAN', 'SpectralClustering']
    # 遍历所有子图
    for i in range(rows):
        for j in range(cols):
            ax = axes[i, j]
            # 获取数据
            X = X_list[i]
            labels = label_list[idx]
            unique_labels = np.unique(labels)
            # 绘制数据
            for k in unique_labels:
                if k == -1:  # 处理使用密度聚类时的噪声点
                    ax.scatter(X[labels == k, 0], X[labels == k, 1], marker="o", c='black', s=10)
                else:
                    ax.scatter(X[labels == k, 0], X[labels == k, 1], marker="o", s=20)
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
                ax.set_title(algorithms[j], pad=10, fontsize=12)
    # 调整子图之间的间距
    plt.tight_layout(pad=2.0, w_pad=1.5, h_pad=0.5)
    # 显示图形
    plt.show()
