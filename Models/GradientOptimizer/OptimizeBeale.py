from PlatContrast import *
from Function import Function
from mpl_toolkits.mplot3d import Axes3D

def beale(x, y):
    return (1.5 - x + x*y)**2 + (2.25 - x + x*y**2)**2 + (2.625 - x + x*y**3)**2

def beale_grad(param):
    x, y = param
    df_dx = 2*(1.5-x+x*y)*(y-1) + \
            2*(2.25-x+x*y**2)*(y**2-1) + \
            2*(2.625-x+x*y**3)*(y**3-1)
    df_dy = 2*(1.5-x+x*y)*x + \
            2*(2.25-x+x*y**2)*(2*x*y) + \
            2*(2.625-x+x*y**3)*(3*x*y**2)
    return np.array([df_dx, df_dy])

def contrast_contour():

    XY_dict = dict()
    grad_types = ['SGD', 'Momentum', 'AdaGrad', 'RMSProp', 'Adam']
    for k in grad_types:
        func = Function(beale, beale_grad, init_value=np.array([1, 1]), grad_type=k, learning_rate=0.01)
        func.optimize(epochs=500)
        XY_dict[k] = func.history

    x_range = [-4.5, 4.5]
    y_range = [-4.5, 4.5]
    color_dict = {'SGD': 'red', 'Momentum': 'blue', 'AdaGrad': 'orange', 'RMSProp': 'teal', 'Adam': 'purple'}
    plot_contrast_contour_beale(beale, x_range, y_range, XY_dict, color_dict)

if __name__ == '__main__':
    contrast_contour()
