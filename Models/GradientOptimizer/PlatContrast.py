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
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def plot_contrast_2D(cal_func, x_range, X_dict, color_dict):
    X = np.arange(x_range[0], x_range[1], step=0.1)
    Y = cal_func(X)
    Y_dict = dict()
    len_history = 0
    for k in X_dict.keys():
        PX = np.array(X_dict[k])
        len_history = len(PX)
        PY = cal_func(PX)
        Y_dict[k] = PY

    plt.figure()
    for i in range(len_history):
        plt.clf()
        plt.plot(X, Y, color='green', zorder=1)
        for k in X_dict.keys():
            PX = np.array(X_dict[k])
            PY = np.array(Y_dict[k])
            plt.plot(PX[:i + 1], PY[:i + 1], color=color_dict[k], zorder=1, label=k)
            plt.scatter(PX[i], PY[i], c=color_dict[k], zorder=2)
        plt.legend()
        plt.title("iter: " + str(i))
        plt.xlabel('x')
        plt.ylabel('y')
        plt.ylim(bottom=np.min(Y) - 1)
        plt.pause(0.1)
    plt.show()


def plot_contrast_contour(cal_func, x_range, y_range, XY_dict, color_dict):
    x = np.linspace(x_range[0], x_range[1], 100)
    y = np.linspace(y_range[0], y_range[1], 100)
    X, Y = np.meshgrid(x, y)
    Z = cal_func(X, Y)
    len_history = 0
    for k in XY_dict.keys():
        PXY = np.array(XY_dict[k])
        len_history = len(PXY)

    plt.figure()
    for i in range(len_history):
        plt.clf()
        contour = plt.contour(X, Y, Z, levels=np.linspace(np.min(Z), np.max(Z), 20))
        plt.clabel(contour, inline=True, fontsize=8)  # 添加等高线标签
        for k in XY_dict.keys():
            PXY = np.array(XY_dict[k])
            plt.plot(PXY[:i + 1, 0], PXY[:i + 1, 1], color=color_dict[k], label=k)
            plt.scatter(PXY[i, 0], PXY[i, 1], c=color_dict[k])

        plt.legend()
        plt.title("iter: " + str(i))
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.xlim(x_range[0], x_range[1])
        plt.ylim(y_range[0], y_range[1])
        plt.pause(0.1)
    plt.show()


def plot_contrast_3D(cal_func, x_range, y_range, XY_dict, color_dict):
    x = np.linspace(x_range[0], x_range[1], 100)
    y = np.linspace(y_range[0], y_range[1], 100)
    X, Y = np.meshgrid(x, y)
    Z = cal_func(X, Y)
    Z_dict = dict()
    len_history = 0
    for k in XY_dict.keys():
        PXY = np.array(XY_dict[k])
        len_history = len(PXY)
        PZ = cal_func(PXY[:, 0], PXY[:, 1])
        Z_dict[k] = PZ

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for i in range(len_history):
        ax.clear()
        ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8, zorder=1)
        for k in XY_dict.keys():
            PXY = np.array(XY_dict[k])
            PZ = np.array(Z_dict[k])
            ax.plot(PXY[:i + 1, 0], PXY[:i + 1, 1], PZ[:i + 1], color=color_dict[k], label=k, zorder=100)
            ax.scatter(PXY[i, 0], PXY[i, 1], PZ[i], c=color_dict[k], zorder=100)
        plt.legend()
        plt.title("iter: " + str(i))
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
        ax.set_zlabel('y')
        ax.set_xlim(x_range[0], x_range[1])
        ax.set_ylim(y_range[0], y_range[1])
        ax.set_zlim(np.min(Z), np.max(Z))
        ax.view_init(elev=60, azim=120)
        plt.pause(0.1)
    plt.show()
