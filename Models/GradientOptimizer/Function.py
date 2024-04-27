import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from Optimizer import *


class Function():
    """可以使用梯度优化的函数类"""
    def __init__(self, cal_func, cal_grad, init_value, grad_type, learning_rate):
        self.cal_func = cal_func  # 求函数值的函数
        self.cal_grad = cal_grad  # 求梯度值的函数
        self.weight = init_value  # 要优化的初始值
        self.grad = 0  # 初始化梯度值
        dict = {'GD': GradientDescent, 'Momentum': Momentum, 'AdaGrad': AdaGrad, 'RMSProp': RMSProp, 'Adam': Adam}
        self.optimizer = dict[grad_type](self, learning_rate)
        self.history = [self.weight]

    def optimize(self, epochs=100):
        """调用则进行优化"""
        for i in range(epochs):
            self.grad = self.cal_grad(self.weight)
            self.optimizer.step()
            # print(self.weight)
            self.history.append(self.weight)

    def plat_2D(self, x_range):
        """画二维图像"""
        X = np.arange(x_range[0], x_range[1], step=0.1)
        Y = self.cal_func(X)
        PX = np.array(self.history)
        PY = self.cal_func(PX)
        plt.figure()
        for i in range(len(PX)):
            plt.clf()
            plt.plot(X, Y, color='green', zorder=1)
            plt.plot(PX[:i+1], PY[:i+1], color='red', zorder=1)
            plt.scatter(PX[i], PY[i], c='red', zorder=2)
            plt.title("iter: " + str(i))
            plt.ylim(bottom=np.min(Y)-1)
            plt.pause(0.1)
        plt.show()

    def plat_contour(self, x_range, y_range):
        """画等高线图像"""
        x = np.linspace(x_range[0], x_range[1], 100)
        y = np.linspace(y_range[0], y_range[1], 100)
        X, Y = np.meshgrid(x, y)
        Z = self.cal_func(X, Y)
        PXY = np.array(self.history)
        plt.figure()
        for i in range(len(PXY)):
            plt.clf()
            contour = plt.contour(X, Y, Z, levels=np.linspace(np.min(Z), np.max(Z), 20))
            plt.clabel(contour, inline=True, fontsize=8) # 添加等高线标签
            plt.plot(PXY[:i+1, 0], PXY[:i+1, 1], color='red')
            plt.scatter(PXY[i, 0], PXY[i, 1], c='red')
            plt.title("iter: " + str(i))
            plt.xlim(x_range[0], x_range[1])
            plt.ylim(y_range[0], y_range[1])
            plt.pause(0.1)
        plt.show()

    def plat_3D(self, x_range, y_range):
        """画三维图像"""
        x = np.linspace(x_range[0], x_range[1], 100)
        y = np.linspace(y_range[0], y_range[1], 100)
        X, Y = np.meshgrid(x, y)
        Z = self.cal_func(X, Y)
        PXY = np.array(self.history)
        PZ = self.cal_func(PXY[:, 0], PXY[:, 1])
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for i in range(len(PXY)):
            ax.clear()
            ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8, zorder=1)
            ax.plot(PXY[:i+1, 0], PXY[:i+1, 1], PZ[:i+1], color='red', zorder=2)
            ax.scatter(PXY[i, 0], PXY[i, 1], PZ[i], c='red', zorder=2)
            plt.title("iter: " + str(i))
            ax.set_xlim(x_range[0], x_range[1])
            ax.set_ylim(y_range[0], y_range[1])
            ax.set_zlim(np.min(Z), np.max(Z))
            plt.pause(0.1)
        plt.show()
