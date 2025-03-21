"""
Copyright (c) 2023 LuChen Wang
[Software Name] is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan
PSL v2.
You may obtain a copy of Mulan PSL v2 at:
         http://license.coscl.org.cn/MulanPSL2
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY
KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
NON-INFRINGEMENT, MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details.
"""
from PlatContrast import *
from Function import Function
from mpl_toolkits.mplot3d import Axes3D


def example_2D(grad_type='Adam', epochs=100):
    def cal_func(x): return x ** 2

    def cal_grad(x): return 2 * x

    func = Function(cal_func, cal_grad, init_value=5, grad_type=grad_type, learning_rate=0.1)
    func.optimize(epochs=epochs)
    func.plot_2D(x_range=[-5, 5])


def example_contour(grad_type='Adam', epochs=100):
    def cal_func(x, y): return x ** 2 + y ** 2

    def cal_grad(x): return 2 * x

    func = Function(cal_func, cal_grad, init_value=np.array([3, 4]), grad_type=grad_type, learning_rate=0.1)
    func.optimize(epochs=epochs)
    func.plot_contour(x_range=[-5, 5], y_range=[-5, 5])


def example_3D(grad_type='Adam', epochs=100):
    def cal_func(x, y): return x ** 2 - y ** 2

    def cal_grad(x): return np.array([2 * x[0], -2 * x[1]])

    func = Function(cal_func, cal_grad, init_value=np.array([4, 0.001]), grad_type=grad_type, learning_rate=0.1)
    func.optimize(epochs=epochs)
    func.plot_3D(x_range=[-5, 5], y_range=[-5, 5])


def contrast_2D():
    def cal_func(x): return x ** 2

    def cal_grad(x): return 2 * x

    X_dict = dict()
    grad_types = ['GD', 'Momentum', 'AdaGrad', 'RMSProp', 'Adam']
    for k in grad_types:
        func = Function(cal_func, cal_grad, init_value=5, grad_type=k, learning_rate=0.1)
        func.optimize(epochs=100)
        X_dict[k] = func.history

    x_range = [-5, 5]
    color_dict = {'GD': 'red', 'Momentum': 'blue', 'AdaGrad': 'orange', 'RMSProp': 'teal', 'Adam': 'purple'}
    plot_contrast_2D(cal_func, x_range, X_dict, color_dict)


def contrast_contour():
    def cal_func(x, y): return x ** 2 + y ** 2

    def cal_grad(x): return 2 * x

    XY_dict = dict()
    grad_types = ['GD', 'Momentum', 'AdaGrad', 'RMSProp', 'Adam']
    for k in grad_types:
        func = Function(cal_func, cal_grad, init_value=np.array([3, 4]), grad_type=k, learning_rate=0.1)
        func.optimize(epochs=100)
        XY_dict[k] = func.history

    x_range = [-5, 5]
    y_range = [-5, 5]
    color_dict = {'GD': 'red', 'Momentum': 'blue', 'AdaGrad': 'orange', 'RMSProp': 'teal', 'Adam': 'purple'}
    plot_contrast_contour(cal_func, x_range, y_range, XY_dict, color_dict)


def contrast_contour2():
    def cal_func(x, y): return x ** 2 - y ** 2

    def cal_grad(x): return np.array([2 * x[0], -2 * x[1]])

    XY_dict = dict()
    grad_types = ['GD', 'Momentum', 'AdaGrad', 'RMSProp', 'Adam']
    for k in grad_types:
        func = Function(cal_func, cal_grad, init_value=np.array([4, 0.001]), grad_type=k, learning_rate=0.1)
        func.optimize(epochs=100)
        XY_dict[k] = func.history

    x_range = [-5, 5]
    y_range = [-5, 5]
    color_dict = {'GD': 'red', 'Momentum': 'blue', 'AdaGrad': 'orange', 'RMSProp': 'teal', 'Adam': 'purple'}
    plot_contrast_contour(cal_func, x_range, y_range, XY_dict, color_dict)


def contrast_3D():
    def cal_func(x, y): return x ** 2 - y ** 2

    def cal_grad(x): return np.array([2 * x[0], -2 * x[1]])

    XY_dict = dict()
    grad_types = ['GD', 'Momentum', 'AdaGrad', 'RMSProp', 'Adam']
    for k in grad_types:
        func = Function(cal_func, cal_grad, init_value=np.array([4, 0.001]), grad_type=k, learning_rate=0.1)
        func.optimize(epochs=100)
        XY_dict[k] = func.history

    x_range = [-5, 5]
    y_range = [-5, 5]
    color_dict = {'GD': 'red', 'Momentum': 'blue', 'AdaGrad': 'orange', 'RMSProp': 'teal', 'Adam': 'purple'}
    plot_contrast_3D(cal_func, x_range, y_range, XY_dict, color_dict)


if __name__ == '__main__':
    # example_2D('Adam', epochs=100)
    # example_contour('Adam', epochs=100)
    # example_3D('Adam', epochs=100)

    # contrast_2D()
    # contrast_contour()
    # contrast_contour2()
    contrast_3D()
