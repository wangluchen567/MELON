"""
逻辑回归(分类器)
Logistic Regression
"""
import warnings
import numpy as np
from Models.GradientOptimizer.Optimizer import GradientDescent, Momentum, AdaGrad, RMSProp, Adam
from Models.Utils import sigmoid, plot_2dim_classification, run_uniform_classification, run_double_classification


class LogisticRegression():
    def __init__(self, X_train=None, Y_train=None, penalty='l2', alpha=1.e-4, l1_ratio=0.15,
                 max_iter=1000, tol=1.e-3, lr=0.06, optim='Adam', num_no_change=6, show=False):
        """
        :param X_train: 训练数据
        :param Y_train: 真实标签
        :param penalty: 要使用的正则化项(None/'l1'/'l2'/'elasticnet')
        :param alpha: 正则化系数(正则化强度)
        :param l1_ratio: 弹性网络正则化中L1和L2的混合比例, penalty='elasticnet'时使用
        :param max_iter: 最大迭代次数
        :param tol: 停止标准。如果连续num_no_change次迭代没有改善则停止训练
        :param lr: 梯度优化时使用的学习率值
        :param optim: 使用的梯度优化器类型
        :param num_no_change: 停止的迭代次数的阈值
        :param show: 是否展示迭代过程
        """
        self.X_train = None  # 训练数据
        self.Y_train = None  # 真实标签
        self.Y_train_ = None  # 逻辑回归特殊标签
        self.set_train_data(X_train, Y_train)
        self.penalty = penalty  # 要使用的正则化项('l1'/'l2'/'elasticnet')
        self.alpha = alpha  # 正则化系数(正则化强度)
        self.l1_ratio = l1_ratio  # 弹性网络正则化中L1和L2的混合比例, penalty='elasticnet'时使用
        self.max_iter = max_iter  # 最大迭代次数
        self.tol = tol  # 停止标准。如果连续num_no_change次迭代没有改善则停止训练
        self.lr = lr  # 梯度优化时使用的学习率值
        self.optim = optim  # 使用的梯度优化器类型
        self.num_no_change = num_no_change  # 停止的迭代次数的阈值
        if self.max_iter == -1:  # 若设置为-1则直到优化结束停止
            self.max_iter = np.inf
        self.n_iter = None  # 迭代次数
        self.Weights = None  # 模型参数
        self.Grad = None  # 模型梯度
        self.Loss = None  # 模型损失
        self.loss_history = []  # 模型损失历史值记录
        self.X_train_B = None  # 训练数据加偏置(避免重复计算)
        self.Y_train_prob = None  # 模型对训练集的预测概率
        self.show = show  # 是否展示迭代过程

    def set_train_data(self, X_train, Y_train):
        """给定训练数据集和标签数据"""
        if X_train is not None:
            if self.X_train is not None:
                warnings.warn("Training data will be overwritten")
            self.X_train = X_train.copy()
        if Y_train is not None:
            if self.Y_train is not None:
                warnings.warn("Training label will be overwritten")
            self.Y_train = Y_train.copy()
            self.Y_train_ = Y_train.copy()
            # 使用逻辑回归时负类标签为0，在此使用特殊标签
            self.Y_train_[self.Y_train_ == -1] = 0

    def set_parameters(self, penalty=None, alpha=None, l1_ratio=None, max_iter=None,
                       tol=None, lr=None, optim=None, num_no_change=None):
        """重新修改相关参数"""
        parameters = ['penalty', 'alpha', 'l1_ratio', 'max_iter', 'tol', 'lr', 'optimizer', 'num_no_change']
        values = [penalty, alpha, l1_ratio, max_iter, tol, lr, optim, num_no_change]
        for param, value in zip(parameters, values):
            if value is not None and getattr(self, param) is not None:
                warnings.warn(f"Parameter '{param}' will be overwritten")
                setattr(self, param, value)

    def init_optimizer(self):
        """初始化优化器"""
        dict = {'GD': GradientDescent, 'Momentum': Momentum, 'AdaGrad': AdaGrad, 'RMSProp': RMSProp, 'Adam': Adam}
        self.optimizer = dict[self.optim](self, self.lr)

    def init_weights(self):
        """初始化参数(权重)"""
        X_feat = self.X_train.shape[1]
        # 正态分布初始化
        self.Weights = np.random.randn(X_feat + 1, 1) * 0.01

    def train(self, X_train=None, Y_train=None, epochs=None, lr=None, grad_type=None):
        """使用数据集训练模型"""
        self.set_train_data(X_train, Y_train)
        self.set_parameters(epochs, lr, grad_type)
        self.init_weights()  # 初始化权重参数
        self.init_optimizer()  # 初始化优化器
        self.n_iter = 0  # 记录迭代次数
        # 在训练数据最后一列添加一列单位矩阵作为偏置b
        self.X_train_B = np.concatenate((self.X_train, np.ones((len(self.X_train), 1))), axis=1)
        while True:
            # 计算训练数据与参数的点积
            self.Y_train_prob = self.predict_prob(self.X_train)
            self.cal_loss()  # 计算损失
            self.cal_grad()  # 计算梯度
            self.optimizer.step()  # 优化器优化一步
            self.n_iter += 1
            if self.no_change():
                # 若优化连续改变量小则停止优化
                break
            if self.n_iter >= self.max_iter:
                # 受最大迭代次数限制优化提前结束
                warnings.warn(f"Optimizer ended early (max_iter={self.max_iter})")
                break
            if self.show:
                self.plot_2dim(pause=True, n_iter=self.n_iter + 1)

    def predict(self, X_data):
        """模型对测试集进行预测"""
        if X_data.ndim == 2:
            pass
        elif X_data.ndim == 1:
            X_data = X_data.reshape(1, -1)
        else:
            raise ValueError("Cannot handle data with a shape of 3 dimensions or more")
        X_B = np.concatenate((X_data, np.ones((len(X_data), 1))), axis=1)
        Y_data = np.ones((len(X_data), 1), dtype=int)
        Y_data[sigmoid(X_B.dot(self.Weights)) < 1 / 2] = -1
        return Y_data

    def predict_prob(self, X_data):
        """模型对测试集进行预测(概率)"""
        if X_data.ndim == 2:
            pass
        elif X_data.ndim == 1:
            X_data = X_data.reshape(1, -1)
        else:
            raise ValueError("Cannot handle data with a shape of 3 dimensions or more")
        X_B = np.concatenate((X_data, np.ones((len(X_data), 1))), axis=1)
        Y_data_prob = sigmoid(X_B.dot(self.Weights))
        return Y_data_prob

    def cal_loss(self):
        """计算损失值"""
        # 计算损失值
        self.Loss = - np.sum(self.Y_train_ * np.log(self.Y_train_prob) +
                             (1 - self.Y_train_) * np.log(1 - self.Y_train_prob)) / len(self.X_train)
        # 加入正则化项
        if self.penalty is None:
            pass
        elif self.penalty == 'l1':
            self.Loss += self.alpha * np.sum(np.abs(self.Weights))
        elif self.penalty == 'l2':
            self.Loss += self.alpha * np.sum(self.Weights ** 2) / 2
        elif self.penalty == 'elasticnet':
            self.Loss += (self.alpha * self.l1_ratio * np.sum(np.abs(self.Weights))
                          + self.alpha * (1 - self.l1_ratio) * np.sum(self.Weights ** 2)) / 2
        else:
            raise ValueError(f"Unsupported penalty: {self.penalty}")
        self.loss_history.append(self.Loss)

    def cal_grad(self):
        """计算梯度值"""
        # 这里使用的是特殊标签矩阵Y_train_ (0/1)
        self.Grad = self.X_train_B.T @ (sigmoid(self.X_train_B @ self.Weights) - self.Y_train_) / len(self.X_train)
        # 加入正则化项
        if self.penalty is None:
            pass
        elif self.penalty == 'l1':
            self.Grad += self.alpha * self.Weights * (self.Weights > 0)
        elif self.penalty == 'l2':
            self.Grad += self.alpha * self.Weights
        elif self.penalty == 'elasticnet':
            self.Grad += (self.alpha * self.l1_ratio * self.Weights * (self.Weights > 0)
                          + self.alpha * (1 - self.l1_ratio) * self.Weights)
        else:
            raise ValueError(f"Unsupported penalty: {self.penalty}")


    def plot_2dim(self, X_test=None, Y_test=None, Truth=None, pause=False, n_iter=None):
        """为二维分类数据集和结果画图"""
        plot_2dim_classification(self.X_train, self.Y_train, self.Weights, X_test, Y_test,
                                 Truth=Truth, pause=pause, n_iter=n_iter)

    def no_change(self):
        """检查连续几次优化改善情况"""
        if len(self.loss_history) >= self.num_no_change:
            his_array = np.array(self.loss_history[-self.num_no_change:])
            # 计算每两个值的绝对差值
            abs_diffs = np.abs(his_array[1:] - his_array[:-1])
            if np.all(abs_diffs < self.tol):
                return True
        return False

if __name__ == '__main__':
    np.random.seed(100)
    model = LogisticRegression(show=True)
    run_uniform_classification(model)
    run_double_classification(model)

