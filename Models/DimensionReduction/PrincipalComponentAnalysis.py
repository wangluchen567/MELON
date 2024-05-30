"""
主成分分析(提取)
Principal Component Analysis
"""
import warnings
import numpy as np
import matplotlib.pyplot as plt
from Models.Utils import plat_data, random_generate_regression


class PrincipalComponentAnalysis():
    def __init__(self, X=None, num_top=None, threshold=None):
        self.X = None  # 需要降维的数据
        self.set_data(X)
        self.X_stand = None  # 标准化后的数据
        self.X_reduced = None  # 降维后的数据
        self.num_top = num_top  # 选择特征值最大的特征数量
        self.threshold = threshold  # 贡献率累计超过的阈值
        self.sorted_index = None  # 排序后的特征下标
        self.sorted_eigen_val = None  # 排序后的特征值
        self.sorted_eigen_vec = None  # 排序后的特征向量
        self.contribution_rate = None  # 特征值贡献率
        self.cum_contribution_rate = None  # 特征值累计贡献率

    def set_data(self, X):
        """给定训练数据"""
        if X is not None:
            if self.X is not None:
                warnings.warn("Training data will be overwritten")
            self.X = X.copy()

    def set_parameters(self, num_top, threshold):
        """重新修改相关参数"""
        if self.num_top is not None and num_top is not None:
            warnings.warn("Parameter 'num_top' be overwritten")
            self.num_top = num_top
        if self.threshold is not None and threshold is not None:
            warnings.warn("Parameters 'threshold' be overwritten")
            self.threshold = threshold

    def train(self, X=None, num_top=None, threshold=0.9):
        """对数据进行降维"""
        self.set_data(X)
        self.set_parameters(num_top, threshold)
        # 标准化数据
        self.X_stand = self.standardize_data(self.X)
        # 计算协方差矩阵
        covar_mat = np.cov(self.X_stand.T)
        # 计算特征值和特征向量
        eigen_val, eigen_vec = np.linalg.eig(covar_mat)
        # 排序特征值和特征向量
        self.sorted_index = np.argsort(eigen_val)[::-1]
        self.sorted_eigen_val = eigen_val[self.sorted_index]
        self.sorted_eigen_vec = eigen_vec[:, self.sorted_index]
        # 计算特征值贡献率和特征值累计贡献率
        self.contribution_rate = self.sorted_eigen_val / sum(eigen_val)
        self.cum_contribution_rate = np.cumsum(self.contribution_rate)
        # 若未指定要选择的特征数量，则根据阈值选择
        if self.num_top is None:
            self.num_top = np.argmax(self.cum_contribution_rate >= self.threshold) + 1
        # 选择前num_top个特征向量
        eigen_vec_top = self.sorted_eigen_vec[:, :self.num_top]
        # 将数据投影到主成分上
        self.X_reduced = np.dot(self.X_stand, eigen_vec_top)
        return self.X_reduced

    @staticmethod
    def standardize_data(X):
        """数据标准化"""
        X_mean = np.mean(X, axis=0)
        X_std = np.std(X, axis=0)
        X_standardized = (X - X_mean) / X_std
        return X_standardized

    def plat_contribution(self):
        """画特征贡献度的结果图"""
        plt.figure()
        plt.plot(range(1, len(self.cum_contribution_rate) + 1), self.cum_contribution_rate, marker='o')
        plt.xlabel('Number of Features')
        plt.ylabel('Cumulative Contribution Rate')
        plt.title('Cumulative Contribution Rate by PCA')
        plt.grid()
        plt.show()


def run_reduce_instance(model, X_size, X_feat):
    # 随机生成回归数据进行PCA降维(方便查看效果)
    X, Y, _ = random_generate_regression(X_size, X_feat, X_lower=0, X_upper=20, loc=0, scale=0.3)
    X_data = np.concatenate((X, Y), axis=1)
    X_reduced = model.train(X_data)
    model.plat_contribution()
    plat_data(X_data, hold=True)
    plat_data(X_reduced)


if __name__ == '__main__':
    model = PrincipalComponentAnalysis(num_top=2)
    run_reduce_instance(model, X_size=100, X_feat=2)
